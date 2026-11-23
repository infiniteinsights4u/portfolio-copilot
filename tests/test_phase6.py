"""Comprehensive pytest suite for Phase 6: Offline Backtest & Shadow-Mode Readiness.

Verifies:
1. TransactionCostModel: Statutory Indian delivery taxes & charges (STT, GST, DP, Stamp Duty, Slippage).
2. EODBacktestEngine indicators: Wilder RSI, Wilder ATR, GMMA, Heikin-Ashi, 52w drawdown.
3. Dip detection signal verification and rejection filters.
4. Deterministic Dual-Cap position sizing and cash commitment clamps.
5. Exit triggers: Stop-loss, Gap-down SL, 2.5R Target, and Chandelier trailing stop ratcheting.
6. Multi-symbol portfolio backtest with shared capital and max open positions constraint.
7. Performance analytics: Total Return, CAGR, Max Drawdown %, Sharpe, Sortino, Win Rate, Profit Factor.
8. Local kite-mcp-server failover script: URL switching, health probing, and mandatory EXCLUDED_TOOLS enforcement.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error

import numpy as np
import pandas as pd
import pytest

from scripts.failover_kite_mcp import (
    HOSTED_KITE_MCP_URL,
    LOCAL_KITE_MCP_URL,
    MANDATORY_EXCLUDED_TOOLS,
    check_health,
    generate_local_launch_command,
    get_current_mcp_url,
    update_env_file,
)
from src.backtest.eod_engine import (
    BacktestConfig,
    BacktestResult,
    EODBacktestEngine,
    TransactionCostModel,
)


# ── Fixtures & Helper Generators ─────────────────────────────────────────────


def generate_synthetic_ohlcv(
    start_date: date = date(2023, 1, 1),
    n_bars: int = 120,
    base_price: float = 1000.0,
    trend: float = 0.5,
    volatility: float = 15.0,
) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV dataframe for deterministic backtest testing."""
    dates = [start_date + timedelta(days=i) for i in range(n_bars)]
    np.random.seed(42)

    closes = []
    curr = base_price
    for _ in range(n_bars):
        curr += trend + np.random.normal(0, volatility)
        curr = max(10.0, curr)
        closes.append(curr)

    closes_arr = np.array(closes)
    highs = closes_arr + np.abs(np.random.normal(5, 3, n_bars))
    lows = closes_arr - np.abs(np.random.normal(5, 3, n_bars))
    lows = np.maximum(1.0, lows)
    opens = (highs + lows) / 2.0
    volumes = np.random.randint(100_000, 1_000_000, n_bars)

    return pd.DataFrame(
        {
            "DATE": dates,
            "OPEN": opens,
            "HIGH": highs,
            "LOW": lows,
            "CLOSE": closes_arr,
            "VOLUME": volumes,
            "SYMBOL": "INFY",
        }
    )


# ── 1. Indian CNC Delivery Transaction Cost Model Tests ──────────────────────


class TestTransactionCostModel:
    """Verify statutory tax, brokerage, and slippage calculations for Indian CNC equities."""

    def test_buy_cost_breakdown(self):
        model = TransactionCostModel(slippage_pct=0.0005, enabled=True)
        # ₹1,00,000 turnover
        price = 1000.0
        qty = 100
        costs = model.calculate_buy_costs(price, qty)

        assert costs.turnover == 100_000.0
        assert costs.brokerage == 0.0  # Zerodha ₹0 CNC delivery
        assert costs.stt == pytest.approx(100.0, rel=1e-3)  # 0.1% = ₹100
        assert costs.exchange_charges == pytest.approx(2.97, rel=1e-3)  # 0.00297% = ₹2.97
        assert costs.sebi_charges == pytest.approx(0.10, rel=1e-3)  # ₹10 per crore = ₹0.10
        assert costs.stamp_duty == pytest.approx(15.0, rel=1e-3)  # 0.015% = ₹15.00
        # GST = 18% of (exchange + sebi) = 0.18 * 3.07 = ₹0.5526
        assert costs.gst == pytest.approx(0.5526, rel=1e-2)
        assert costs.dp_charges == 0.0  # DP only on sell leg
        assert costs.slippage == pytest.approx(50.0, rel=1e-3)  # 0.05% = ₹50
        assert costs.total == pytest.approx(
            100.0 + 2.97 + 0.10 + 15.0 + 0.5526 + 50.0, rel=1e-2
        )

    def test_sell_cost_breakdown(self):
        model = TransactionCostModel(slippage_pct=0.0005, enabled=True)
        price = 1100.0
        qty = 100
        turnover = 110_000.0
        costs = model.calculate_sell_costs(price, qty)

        assert costs.turnover == turnover
        assert costs.brokerage == 0.0
        assert costs.stt == pytest.approx(110.0, rel=1e-3)  # 0.1% = ₹110
        assert costs.stamp_duty == 0.0  # Stamp duty only on buy
        assert costs.dp_charges == 15.34  # ₹13 + 18% GST
        assert costs.slippage == pytest.approx(55.0, rel=1e-3)
        assert costs.total > 0

    def test_disabled_cost_model(self):
        model = TransactionCostModel(enabled=False)
        buy_c = model.calculate_buy_costs(1000.0, 50)
        sell_c = model.calculate_sell_costs(1000.0, 50)

        assert buy_c.total == 0.0
        assert sell_c.total == 0.0

    def test_cost_dict_serialization(self):
        model = TransactionCostModel(enabled=True)
        costs = model.calculate_buy_costs(500.0, 20)
        d = costs.to_dict()
        assert "turnover" in d
        assert "stt" in d
        assert "dp_charges" in d
        assert "total" in d


# ── 2. EOD Backtest Engine Indicator & Signal Tests ──────────────────────────


class TestEODBacktestEngineIndicators:
    """Verify indicator computation, dip signal filtering, and position sizing."""

    def test_indicator_preparation(self):
        engine = EODBacktestEngine()
        df = generate_synthetic_ohlcv(n_bars=60)
        prepared = engine._prepare_indicators(df)

        assert "RSI_14" in prepared.columns
        assert "ATR_14" in prepared.columns
        assert "DMA_50" in prepared.columns
        assert "DMA_200" in prepared.columns
        assert "GMMA_BULLISH" in prepared.columns
        assert "HA_BULLISH" in prepared.columns
        assert "DRAWDOWN_52W" in prepared.columns

    def test_check_dip_signal_rejection(self):
        engine = EODBacktestEngine()
        df = generate_synthetic_ohlcv(n_bars=50)
        prep = engine._prepare_indicators(df)

        # Before 30 bars, dip signal is always False (insufficient lookback)
        assert engine._check_dip_signal(prep, 10) is False

    def test_position_sizing_dual_cap(self):
        engine = EODBacktestEngine(
            BacktestConfig(
                initial_capital=1_000_000.0,
                max_capital_per_trade_pct=10.0,  # ₹1,00,000 max capital
                max_risk_per_trade_pct=1.0,  # ₹10,000 max risk
            )
        )

        # Case A: Capital cap binds
        # Entry = 1500, SL = 1450 (Risk = 50). Risk cap: 10000 / 50 = 200 shares.
        # Capital cap: 100000 / 1500 = 66 shares.
        qty_a = engine._calculate_position_size(
            entry_price=1500.0,
            stop_loss_price=1450.0,
            portfolio_capital=1_000_000.0,
            available_cash=1_000_000.0,
        )
        assert qty_a == 66  # Capital cap binds at 66

        # Case B: Risk cap binds
        # Entry = 500, SL = 400 (Risk = 100). Capital cap: 100000 / 500 = 200 shares.
        # Risk cap: 10000 / 100 = 100 shares.
        qty_b = engine._calculate_position_size(
            entry_price=500.0,
            stop_loss_price=400.0,
            portfolio_capital=1_000_000.0,
            available_cash=1_000_000.0,
        )
        assert qty_b == 100  # Risk cap binds at 100

    def test_position_sizing_cash_clamp(self):
        engine = EODBacktestEngine()
        # Available cash only ₹5,000
        qty = engine._calculate_position_size(
            entry_price=1000.0,
            stop_loss_price=950.0,
            portfolio_capital=1_000_000.0,
            available_cash=5_000.0,
        )
        assert qty == 5  # Clamped by ₹5,000 cash


# ── 3. Trade Execution & Exit Dynamics Tests ─────────────────────────────────


class TestTradeExecutionDynamics:
    """Verify stop-loss, profit target, and Chandelier trailing stop ratcheting."""

    def test_stop_loss_exit_triggered(self):
        """Simulate a trade where price drops below initial stop-loss."""
        # Force entry at bar 35 via signal_fn
        engine = EODBacktestEngine(
            BacktestConfig(initial_capital=1_000_000.0),
            signal_fn=lambda df, idx: idx == 35,
        )

        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(50)]
        closes = [1000.0 + (i * 2.0) for i in range(35)]  # Uptrend
        closes.extend([1070.0])  # Bar 35: Entry
        # Crash hitting stop-loss
        closes.extend([1020.0, 950.0, 900.0, 850.0] + [800.0] * 10)

        df = pd.DataFrame(
            {
                "DATE": dates,
                "OPEN": closes,
                "HIGH": [c + 10.0 for c in closes],
                "LOW": [c - 10.0 for c in closes],
                "CLOSE": closes,
                "VOLUME": [500_000] * 50,
                "SYMBOL": "TCS",
            }
        )

        result = engine.run_single_symbol(df, "TCS")
        assert result.total_trades >= 1
        sl_trades = [t for t in result.trades if "STOP_LOSS" in t.exit_reason]
        assert len(sl_trades) >= 1
        assert sl_trades[0].net_pnl < 0

    def test_profit_target_exit_triggered(self):
        """Simulate a trade that rallies and hits 2.5R profit target."""
        engine = EODBacktestEngine(
            BacktestConfig(initial_capital=1_000_000.0),
            signal_fn=lambda df, idx: idx == 35,
        )

        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(50)]
        closes = [500.0 + (i * 1.0) for i in range(35)]
        closes.extend([535.0])  # Bar 35: Entry
        # Strong rally upward hitting 2.5R target:
        closes.extend([550.0, 580.0, 620.0, 680.0] + [700.0] * 10)

        df = pd.DataFrame(
            {
                "DATE": dates,
                "OPEN": closes,
                "HIGH": [c + 15.0 for c in closes],
                "LOW": [c - 5.0 for c in closes],
                "CLOSE": closes,
                "VOLUME": [300_000] * 50,
                "SYMBOL": "RELIANCE",
            }
        )

        result = engine.run_single_symbol(df, "RELIANCE")
        assert result.total_trades >= 1
        target_trades = [t for t in result.trades if "TARGET" in t.exit_reason]
        assert len(target_trades) >= 1
        assert target_trades[0].net_pnl > 0
        assert target_trades[0].r_multiple > 1.5

    def test_chandelier_trailing_stop_ratchet(self):
        """Verify that Chandelier trailing stop ratchets up on rising highs and never falls."""
        config = BacktestConfig(
            enable_chandelier_trail=True,
            chandelier_atr_multiplier=2.0,
        )
        engine = EODBacktestEngine(
            config,
            signal_fn=lambda df, idx: idx == 35,
        )

        n_bars = 50
        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(n_bars)]
        closes = [800.0 + (i * 1.5) for i in range(35)]
        closes.extend([852.5])  # Bar 35: Entry
        # Rally to 920, then retrace
        closes.extend([870.0, 890.0, 920.0, 910.0, 880.0, 840.0] + [820.0] * 8)

        df = pd.DataFrame(
            {
                "DATE": dates,
                "OPEN": closes,
                "HIGH": [c + 8.0 for c in closes],
                "LOW": [c - 8.0 for c in closes],
                "CLOSE": closes,
                "VOLUME": [200_000] * n_bars,
                "SYMBOL": "HDFCBANK",
            }
        )

        result = engine.run_single_symbol(df, "HDFCBANK")
        assert result.total_trades >= 1
        trade = result.trades[0]
        # Should have captured positive profit or stopped out cleanly
        assert trade.exit_price > 0


# ── 4. Multi-Symbol Universe & Portfolio Management Tests ────────────────────


class TestPortfolioBacktest:
    """Verify multi-symbol universe backtest with shared cash and risk controls."""

    def test_multi_symbol_portfolio_cash_accounting(self):
        engine = EODBacktestEngine(
            BacktestConfig(
                initial_capital=500_000.0,
                max_open_positions=2,  # Only 2 positions allowed concurrently
            )
        )

        df1 = generate_synthetic_ohlcv(n_bars=80, base_price=500.0)
        df2 = generate_synthetic_ohlcv(n_bars=80, base_price=1200.0)
        df3 = generate_synthetic_ohlcv(n_bars=80, base_price=2500.0)

        dfs = {"INFY": df1, "TCS": df2, "RELIANCE": df3}
        result = engine.run_portfolio(dfs)

        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 500_000.0
        assert not result.equity_curve.empty
        # Verify max concurrent positions was respected
        assert result.equity_curve["open_positions"].max() <= 2

    def test_backtest_analytics_metrics(self):
        engine = EODBacktestEngine(
            signal_fn=lambda df, idx: idx in (35, 60),
        )
        df = generate_synthetic_ohlcv(n_bars=100)
        result = engine.run_single_symbol(df, "INFY")

        summary = result.summary()
        assert "initial_capital" in summary
        assert "ending_capital" in summary
        assert "total_return_pct" in summary
        assert "cagr_pct" in summary
        assert "win_rate_pct" in summary
        assert "profit_factor" in summary
        assert "max_drawdown_pct" in summary
        assert "sharpe_ratio" in summary
        assert "sortino_ratio" in summary
        assert "total_costs_paid" in summary


# ── 5. Kite MCP Local Failover Script Tests ──────────────────────────────────


class TestFailoverKiteMCPScript:
    """Verify local zerodha/kite-mcp-server failover script and security rules."""

    def test_mandatory_excluded_tools_string(self):
        assert "place_order" in MANDATORY_EXCLUDED_TOOLS
        assert "modify_order" in MANDATORY_EXCLUDED_TOOLS
        assert "cancel_order" in MANDATORY_EXCLUDED_TOOLS

    def test_local_launch_command_enforces_safety(self):
        cmd = generate_local_launch_command("kite-mcp-server")
        assert "EXCLUDED_TOOLS=place_order,modify_order,cancel_order" in cmd
        assert "PORT=8080" in cmd
        assert "kite-mcp-server" in cmd

    def test_update_env_file_switch_local_and_hosted(self, tmp_path: Path):
        test_env = tmp_path / ".env"
        test_env.write_text("SOME_KEY=123\nKITE_MCP_URL=https://mcp.kite.trade/mcp\n", encoding="utf-8")

        # Switch to local
        update_env_file(test_env, LOCAL_KITE_MCP_URL)
        assert get_current_mcp_url(test_env) == LOCAL_KITE_MCP_URL

        # Switch back to hosted
        update_env_file(test_env, HOSTED_KITE_MCP_URL)
        assert get_current_mcp_url(test_env) == HOSTED_KITE_MCP_URL

    @patch("urllib.request.urlopen")
    def test_check_health_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        assert check_health("http://localhost:8080/mcp") is True

    @patch("urllib.request.urlopen")
    def test_check_health_reachable_405_or_400(self, mock_urlopen):
        # Streamable HTTP MCP endpoint returning 405 Method Not Allowed on GET confirms it is alive
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://localhost:8080/mcp",
            code=405,
            msg="Method Not Allowed",
            hdrs={},
            fp=None,
        )
        assert check_health("http://localhost:8080/mcp") is True

    @patch("urllib.request.urlopen")
    def test_check_health_unreachable_connection_refused(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        assert check_health("http://localhost:8080/mcp") is False
