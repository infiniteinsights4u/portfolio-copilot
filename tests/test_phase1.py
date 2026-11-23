"""Unit and integration tests for Phase 1 components.

Tests:
1. Kite MCP Client tool allowlist and prohibited order blocking (place_order, modify_order, cancel_order).
2. Configuration hierarchy (CLI > Env > .env) and safety defaults (DRY_RUN=True).
3. SQLite Ledger tables (positions_ledger, gtt_audit_log, pending_buy_commits, screener_cache).
4. Trade Audit Logger (JSONL structured events).
5. Streamlit session health badge expiry logic.
"""

import warnings
import pytest

from src.broker.kite_mcp_client import (
    ALLOWED_TOOLS,
    PROHIBITED_TOOLS,
    KiteMCPClient,
    ProhibitedToolError,
)
from src.config.settings import (
    DEFAULT_KITE_MCP_URL,
    Settings,
    load_settings,
)
from src.storage.ledger import SQLiteLedger
from src.audit.trade_logger import TradeAuditLogger
from src.ui.components.login import get_ist_session_expiry_info


class TestKiteMCPAllowlist:
    """Verify strict tool allowlist enforcement and hard prohibition of regular orders."""

    def test_prohibited_tools_raise_error(self):
        client = KiteMCPClient()
        for forbidden in ("place_order", "modify_order", "cancel_order"):
            with pytest.raises(ProhibitedToolError, match="strictly prohibited"):
                client.validate_tool_allowed(forbidden)

    def test_unauthorized_tools_raise_error(self):
        client = KiteMCPClient()
        for unauthorized in ("execute_script", "transfer_funds", "unknown_tool"):
            with pytest.raises(ProhibitedToolError, match="not in the authorized"):
                client.validate_tool_allowed(unauthorized)

    def test_all_allowed_tools_pass(self):
        client = KiteMCPClient()
        assert len(ALLOWED_TOOLS) == 15
        for tool_name in ALLOWED_TOOLS:
            # Should not raise exception
            client.validate_tool_allowed(tool_name)

    def test_prohibited_tools_set_complete(self):
        assert "place_order" in PROHIBITED_TOOLS
        assert "modify_order" in PROHIBITED_TOOLS
        assert "cancel_order" in PROHIBITED_TOOLS
        # Prohibited tools must never intersect with allowed tools
        assert PROHIBITED_TOOLS.isdisjoint(ALLOWED_TOOLS)


class TestSettingsPrecedence:
    """Verify CLI > Env > .env > Defaults hierarchy and safety flags."""

    def test_default_settings(self):
        settings = load_settings(cli_overrides={})
        assert settings.dry_run is True
        assert settings.kite_mcp_url == DEFAULT_KITE_MCP_URL
        assert settings.max_risk_per_trade_pct == 0.01
        assert settings.max_position_size_pct == 0.10
        assert settings.max_sector_size_pct == 0.25

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.setenv("KITE_MCP_URL", "http://localhost:8000/mcp")
        settings = load_settings()
        assert settings.dry_run is False
        assert settings.kite_mcp_url == "http://localhost:8000/mcp"

    def test_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.setenv("KITE_MCP_URL", "http://env-url/mcp")
        settings = load_settings(cli_overrides={"dry_run": True, "kite_mcp_url": "http://cli-url/mcp"})
        assert settings.dry_run is True
        assert settings.kite_mcp_url == "http://cli-url/mcp"

    def test_ddpi_warning_emitted_when_false(self):
        settings = Settings(zerodha_ddpi_enabled=False)
        with pytest.warns(UserWarning, match="ZERODHA_DDPI_ENABLED is False"):
            assert settings.check_ddpi_status() is False

    def test_ddpi_no_warning_when_true(self):
        settings = Settings(zerodha_ddpi_enabled=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert settings.check_ddpi_status() is True


class TestSQLiteLedger:
    """Verify CRUD and business rules on all 4 SQLite tables."""

    @pytest.fixture
    def ledger(self, tmp_path):
        db_path = tmp_path / "test_copilot.db"
        return SQLiteLedger(db_path=db_path)

    def test_positions_ledger_crud(self, ledger):
        pos_id = ledger.record_position(
            tradingsymbol="INFY",
            entry_price=1500.0,
            quantity=50,
            current_stop_loss=1440.0,
            target_price=1650.0,
            isin="INE009A01021",
            thesis="200-DMA pullback support",
        )
        assert pos_id > 0

        open_positions = ledger.get_open_positions()
        assert len(open_positions) == 1
        assert open_positions[0]["tradingsymbol"] == "INFY"
        assert open_positions[0]["quantity"] == 50
        assert open_positions[0]["current_stop_loss"] == 1440.0

        # Update stop-loss
        updated = ledger.update_position_stop(pos_id, 1475.0)
        assert updated is True
        open_positions = ledger.get_open_positions()
        assert open_positions[0]["current_stop_loss"] == 1475.0

        # Close position
        closed = ledger.close_position(pos_id)
        assert closed is True
        assert len(ledger.get_open_positions()) == 0

    def test_gtt_audit_log(self, ledger):
        log_id = ledger.log_gtt_action(
            trigger_id=1234567,
            action="PLACE",
            stage=1,
            tradingsymbol="RELIANCE",
            trigger_price=2450.0,
            limit_price=2462.25,
            quantity=40,
            status="ACTIVE",
            request_payload={"order_type": "LIMIT"},
            response_payload={"trigger_id": 1234567},
        )
        assert log_id > 0

        history = ledger.get_gtt_audit_history("RELIANCE")
        assert len(history) == 1
        assert history[0]["trigger_id"] == 1234567
        assert history[0]["action"] == "PLACE"
        assert isinstance(history[0]["trigger_id"], int)

    def test_pending_buy_cash_reservation(self, ledger):
        assert ledger.get_total_pending_buy_cash() == 0.0

        # Reserve cash for 2 pending buy GTTs
        c1 = ledger.commit_pending_buy_cash("TCS", 75000.0, trigger_id=101)
        ledger.commit_pending_buy_cash("HDFCBANK", 50000.0, trigger_id=102)

        assert ledger.get_total_pending_buy_cash() == 125000.0

        # Release TCS on fill/cancel
        released = ledger.release_pending_buy_cash(commit_id=c1)
        assert released == 1
        assert ledger.get_total_pending_buy_cash() == 50000.0

        # Release HDFCBANK by trigger_id
        released2 = ledger.release_pending_buy_cash(trigger_id=102)
        assert released2 == 1
        assert ledger.get_total_pending_buy_cash() == 0.0

    def test_screener_cache_ttl(self, ledger):
        sample_data = {"pe_ratio": 22.5, "roe": 18.2, "market_cap_cr": 150000}
        ledger.set_screener_cache("TITAN", sample_data, ttl_days=60)

        cached = ledger.get_screener_cache("TITAN")
        assert cached is not None
        assert cached["pe_ratio"] == 22.5

        # Non-existent symbol returns None
        assert ledger.get_screener_cache("UNKNOWN") is None


class TestTradeAuditLogger:
    """Verify structured JSONL trade event logging."""

    def test_trade_logger_events(self, tmp_path):
        log_file = tmp_path / "test_trade_audit.jsonl"
        audit = TradeAuditLogger(log_path=log_file)

        event1 = audit.log_event(
            event_type="GTT_PROPOSED",
            tradingsymbol="LT",
            action="BUY",
            stage=1,
            dry_run=True,
            details={"trigger_price": 3500.0, "qty": 28},
        )
        assert event1["event_type"] == "GTT_PROPOSED"
        assert event1["dry_run"] is True

        event2 = audit.log_event(
            event_type="GTT_SUBMITTED",
            tradingsymbol="LT",
            action="BUY",
            stage=1,
            dry_run=True,
            trigger_id=987654,
            details={"status": "SIMULATED_SUCCESS"},
        )
        assert event2["trigger_id"] == 987654

        recent = audit.read_recent_events(limit=10)
        assert len(recent) == 2
        assert recent[0]["tradingsymbol"] == "LT"
        assert recent[1]["trigger_id"] == 987654


class TestSessionHealthBadge:
    """Verify IST daily 6:00 AM session expiration calculation."""

    def test_expiry_countdown_returns_valid_time(self):
        hours, minutes, expiring_soon = get_ist_session_expiry_info()
        assert 0 <= hours <= 24
        assert 0 <= minutes < 60
        assert isinstance(expiring_soon, bool)
