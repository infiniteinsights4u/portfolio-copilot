"""Exhaustive pytest suite for Phase 4: LangGraph Durable interrupt() HITL & Safe GTT Execution.

Verifies:
1. Durable SQLite checkpointer persistence, thread isolation, and process-restart durability.
2. Safe Place-Verify-Delete GTT modification pattern (place new -> verify in get_gtts -> delete old).
3. Preservation of old GTT protection if verification fails.
4. LangGraph interrupt() approval gate separation from execute_gtt_node (zero broker writes before approval).
5. Human-in-the-Loop Approve, Edit (with RiskGuard re-validation), and Reject branches.
6. RiskGuard pre-approval rejection bypassing human approval gate directly to terminal state.
7. Streamlit Approval Drawer component helpers and input validation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from langgraph.types import Command

from src.agents.workflows.gtt_approval_workflow import (
    create_gtt_approval_workflow,
)
from src.audit.trade_logger import TradeAuditLogger
from src.broker.gtt_executor import (
    GTTExecutor,
    GTTOrderPayload,
    GTTVerificationError,
    extract_trigger_id,
)
from src.config.settings import Settings
from src.storage.ledger import SQLiteLedger
from src.storage.sqlite_checkpointer import DurableSqliteSaver
from src.ui.components.approval_drawer import (
    format_approval_metrics,
    handle_user_decision,
    validate_user_overrides,
)


# ── 1. Durable SQLite Checkpointer Tests ──────────────────────────────────────


class TestDurableSqliteCheckpointer:
    """Verify on-disk SQLite persistence, thread isolation, and restart survival."""

    def test_sqlite_checkpointer_crud(self, tmp_path: Path):
        db_path = tmp_path / "checkpoints.db"
        saver = DurableSqliteSaver(db_path)

        # Test writing and reading tuple
        config = {"configurable": {"thread_id": "thread-1"}}
        checkpoint = {
            "v": 1,
            "ts": "2026-11-23T18:00:00Z",
            "id": "1f1e7681-0000-0000-0000-000000000001",
            "channel_versions": {"cnt": "1"},
            "versions_seen": {},
            "updated_channels": ["cnt"],
            "channel_values": {"cnt": 42},
        }
        metadata = {"step": 1, "source": "test"}

        saved_config = saver.put(config, checkpoint, metadata, {"cnt": "1"})
        assert saved_config["configurable"]["checkpoint_id"] == checkpoint["id"]

        retrieved = saver.get_tuple(config)
        assert retrieved is not None
        assert retrieved.checkpoint["channel_values"]["cnt"] == 42
        assert retrieved.metadata["step"] == 1
        saver.close()

    def test_sqlite_checkpointer_process_restart(self, tmp_path: Path):
        """Simulate process termination and restart using same SQLite file."""
        db_path = tmp_path / "restart_test.db"

        # Session 1: save checkpoint and close
        saver1 = DurableSqliteSaver(db_path)
        config = {"configurable": {"thread_id": "thread-restart"}}
        c1 = {
            "v": 1,
            "ts": "2026-11-23T18:00:00Z",
            "id": "1f1e7681-0000-0000-0000-000000000010",
            "channel_versions": {"symbol": "1"},
            "versions_seen": {},
            "updated_channels": ["symbol"],
            "channel_values": {"symbol": "TCS"},
        }
        saver1.put(config, c1, {"step": 0}, {"symbol": "1"})
        saver1.close()

        # Session 2: re-open from disk and verify data preserved
        saver2 = DurableSqliteSaver(db_path)
        retrieved = saver2.get_tuple(config)
        assert retrieved is not None
        assert retrieved.checkpoint["channel_values"]["symbol"] == "TCS"
        saver2.close()

    def test_sqlite_checkpointer_thread_isolation(self, tmp_path: Path):
        db_path = tmp_path / "isolation.db"
        saver = DurableSqliteSaver(db_path)

        saver.put(
            {"configurable": {"thread_id": "thread-A"}},
            {
                "v": 1, "ts": "2026", "id": "cp-A",
                "channel_versions": {"x": "1"}, "versions_seen": {},
                "updated_channels": ["x"], "channel_values": {"x": "apple"},
            },
            {},
            {"x": "1"},
        )

        saver.put(
            {"configurable": {"thread_id": "thread-B"}},
            {
                "v": 1, "ts": "2026", "id": "cp-B",
                "channel_versions": {"x": "1"}, "versions_seen": {},
                "updated_channels": ["x"], "channel_values": {"x": "banana"},
            },
            {},
            {"x": "1"},
        )

        tA = saver.get_tuple({"configurable": {"thread_id": "thread-A"}})
        tB = saver.get_tuple({"configurable": {"thread_id": "thread-B"}})
        assert tA.checkpoint["channel_values"]["x"] == "apple"
        assert tB.checkpoint["channel_values"]["x"] == "banana"

        saver.delete_thread("thread-A")
        assert saver.get_tuple({"configurable": {"thread_id": "thread-A"}}) is None
        assert saver.get_tuple({"configurable": {"thread_id": "thread-B"}}) is not None
        saver.close()


# ── 2. GTT Executor & Safe Place-Verify-Delete Tests ──────────────────────────


class TestGTTExecutorSafeModifications:
    """Verify Place-Verify-Delete modification pattern and dry-run defaults."""

    @pytest.mark.asyncio
    async def test_gtt_executor_dry_run_default(self, tmp_path: Path):
        db_path = tmp_path / "copilot.db"
        log_path = tmp_path / "trade_audit.jsonl"
        ledger = SQLiteLedger(db_path)
        logger = TradeAuditLogger(log_path)
        settings = Settings(dry_run=True, db_path=db_path, audit_log_path=log_path)

        executor = GTTExecutor(ledger=ledger, trade_logger=logger, settings=settings)
        payload = GTTOrderPayload(
            tradingsymbol="RELIANCE",
            trigger_values=[2400.0],
            limit_prices=[2412.0],
            quantities=[40],
            stage=1,
            thesis="200-DMA test",
            allocated_cash=96480.0,
        )

        result = await executor.place_gtt(payload)
        assert result.success is True
        assert result.dry_run is True
        assert result.status == "DRY_RUN_PLACED"
        assert result.trigger_id > 0

        # Verify ledger recorded cash commitment
        assert ledger.get_total_pending_buy_cash() == 96480.0

    @pytest.mark.asyncio
    async def test_gtt_executor_strict_cnc_enforcement(self, tmp_path: Path):
        executor = GTTExecutor(settings=Settings(dry_run=True))
        payload = GTTOrderPayload(
            tradingsymbol="INFY",
            product="MIS",  # Prohibited intraday product
            trigger_values=[1500.0],
            limit_prices=[1507.0],
            quantities=[10],
        )

        with pytest.raises(ValueError, match="Strict CNC policy"):
            await executor.place_gtt(payload)

    @pytest.mark.asyncio
    async def test_place_verify_delete_modification_success(self, tmp_path: Path):
        mock_client = AsyncMock()
        mock_client.place_gtt_order.return_value = {"trigger_id": 88888888}
        mock_client.get_gtts.return_value = [{"id": 88888888, "status": "active"}]
        mock_client.delete_gtt_order.return_value = {"status": "success"}

        db_path = tmp_path / "pvd.db"
        log_path = tmp_path / "pvd.jsonl"
        ledger = SQLiteLedger(db_path)
        logger = TradeAuditLogger(log_path)
        settings = Settings(dry_run=False, db_path=db_path, audit_log_path=log_path)

        executor = GTTExecutor(
            kite_client=mock_client,
            ledger=ledger,
            trade_logger=logger,
            settings=settings,
        )

        payload = GTTOrderPayload(
            tradingsymbol="HDFCBANK",
            trigger_values=[1600.0],
            limit_prices=[1608.0],
            quantities=[50],
            stage=1,
        )

        res = await executor.modify_gtt_place_verify_delete(old_trigger_id=77777777, payload=payload)
        assert res.success is True
        assert res.action == "MODIFY_PLACE_VERIFY_DELETE"
        assert res.trigger_id == 88888888
        assert res.old_trigger_id == 77777777

        # Assert correct order of operations: place -> verify -> delete
        mock_client.place_gtt_order.assert_called_once()
        mock_client.get_gtts.assert_called_once()
        mock_client.delete_gtt_order.assert_called_once_with(trigger_id=77777777)

    @pytest.mark.asyncio
    async def test_place_verify_delete_preserves_old_on_verification_failure(self, tmp_path: Path):
        mock_client = AsyncMock()
        # Returns new ID on placement
        mock_client.place_gtt_order.return_value = {"trigger_id": 99999999}
        # Verification fails (new ID not found in get_gtts)
        mock_client.get_gtts.return_value = [{"id": 11111111, "status": "active"}]

        executor = GTTExecutor(
            kite_client=mock_client,
            settings=Settings(dry_run=False),
        )

        payload = GTTOrderPayload(
            tradingsymbol="TCS",
            trigger_values=[3500.0],
            limit_prices=[3515.0],
            quantities=[20],
            stage=1,
        )

        with pytest.raises(GTTVerificationError, match="CRITICAL SAFETY ABORT"):
            await executor.modify_gtt_place_verify_delete(old_trigger_id=12345678, payload=payload)

        # Crucial Safety Assertion: delete_gtt_order was NEVER called on old_trigger_id
        mock_client.delete_gtt_order.assert_not_called()

    def test_extract_trigger_id_variants(self):
        assert extract_trigger_id(12345678) == 12345678
        assert extract_trigger_id("12345678") == 12345678
        assert extract_trigger_id('{"trigger_id": 98765432}') == 98765432
        assert extract_trigger_id({"data": {"trigger_id": 55555555}}) == 55555555
        assert extract_trigger_id("GTT order created with ID 77665544 successfully") == 77665544
        assert extract_trigger_id(None) is None


# ── 3. LangGraph HITL Approval Workflow Tests ─────────────────────────────────


class TestLangGraphHITLApprovalWorkflow:
    """Verify interrupt() pause, separate execute_gtt_node, and Approve/Edit/Reject paths."""

    @pytest.mark.asyncio
    async def test_hitl_pauses_before_execute_node(self, tmp_path: Path):
        """Verify graph halts at interrupt() and zero execution actions occur before resume."""
        db_path = tmp_path / "hitl_pause.db"
        saver = DurableSqliteSaver(db_path)
        builder = create_gtt_approval_workflow()
        graph = builder.compile(checkpointer=saver)

        thread_id = "test-pause-1"
        initial_input = {
            "symbol": "INFY",
            "stage": 1,
            "cmp": 1550.0,
            "atr": 25.0,
            "portfolio_capital": 1000000.0,
            "available_cash": 1000000.0,
            "lower_circuit": 1300.0,
            "upper_circuit": 1700.0,
            "buy_trigger": 1540.0,  # Below CMP by >0.25%
            "target_trigger": 1650.0,
        }

        result = await graph.ainvoke(initial_input, config={"configurable": {"thread_id": thread_id}})

        # Assert paused at interrupt
        assert "__interrupt__" in result
        interrupt_val = result["__interrupt__"][0].value
        assert interrupt_val["symbol"] == "INFY"
        assert interrupt_val["metrics"]["quantity"] > 0
        assert "execute_gtt_node" not in result.get("steps_completed", [])
        saver.close()

    @pytest.mark.asyncio
    async def test_hitl_approve_branch(self, tmp_path: Path):
        """Verify resuming with action='approve' executes order and updates ledger."""
        db_path = tmp_path / "hitl_approve.db"
        saver = DurableSqliteSaver(db_path)
        builder = create_gtt_approval_workflow()
        graph = builder.compile(checkpointer=saver)

        thread_id = "test-approve-1"
        initial_input = {
            "symbol": "INFY",
            "stage": 1,
            "cmp": 1550.0,
            "atr": 25.0,
            "portfolio_capital": 1000000.0,
            "available_cash": 1000000.0,
            "lower_circuit": 1300.0,
            "upper_circuit": 1700.0,
            "buy_trigger": 1540.0,
            "target_trigger": 1650.0,
        }

        # 1. Run to pause
        await graph.ainvoke(initial_input, config={"configurable": {"thread_id": thread_id}})

        # 2. Resume with Approve
        decision = {"action": "approve"}
        res2 = await graph.ainvoke(Command(resume=decision), config={"configurable": {"thread_id": thread_id}})

        assert res2["status"] == "APPROVED_AND_EXECUTED"
        exec_res = res2["execution_result"]
        assert exec_res.success is True
        assert exec_res.action == "PLACE"
        assert exec_res.trigger_id > 0
        assert "execute_gtt_node" in res2["steps_completed"]
        saver.close()

    @pytest.mark.asyncio
    async def test_hitl_reject_branch(self, tmp_path: Path):
        """Verify resuming with action='reject' terminates cleanly without broker execution."""
        db_path = tmp_path / "hitl_reject.db"
        saver = DurableSqliteSaver(db_path)
        builder = create_gtt_approval_workflow()
        graph = builder.compile(checkpointer=saver)

        thread_id = "test-reject-1"
        initial_input = {
            "symbol": "INFY",
            "stage": 1,
            "cmp": 1550.0,
            "atr": 25.0,
            "portfolio_capital": 1000000.0,
            "available_cash": 1000000.0,
            "lower_circuit": 1300.0,
            "upper_circuit": 1700.0,
            "buy_trigger": 1540.0,
            "target_trigger": 1650.0,
        }

        # 1. Run to pause
        await graph.ainvoke(initial_input, config={"configurable": {"thread_id": thread_id}})

        # 2. Resume with Reject
        decision = {"action": "reject", "reason": "Expected broader market sell-off"}
        res2 = await graph.ainvoke(Command(resume=decision), config={"configurable": {"thread_id": thread_id}})

        assert res2["status"] == "REJECTED_BY_USER"
        exec_res = res2["execution_result"]
        assert exec_res.action == "REJECTED"
        assert exec_res.status == "REJECTED_BY_USER"
        assert exec_res.trigger_id is None
        saver.close()

    @pytest.mark.asyncio
    async def test_hitl_edit_branch_revalidates_and_executes(self, tmp_path: Path):
        """Verify user edited parameters are re-validated through RiskGuard before execution."""
        db_path = tmp_path / "hitl_edit.db"
        saver = DurableSqliteSaver(db_path)
        builder = create_gtt_approval_workflow()
        graph = builder.compile(checkpointer=saver)

        thread_id = "test-edit-1"
        initial_input = {
            "symbol": "INFY",
            "stage": 1,
            "cmp": 1550.0,
            "atr": 25.0,
            "portfolio_capital": 1000000.0,
            "available_cash": 1000000.0,
            "lower_circuit": 1300.0,
            "upper_circuit": 1700.0,
            "buy_trigger": 1540.0,
            "target_trigger": 1650.0,
        }

        # 1. Run to pause
        await graph.ainvoke(initial_input, config={"configurable": {"thread_id": thread_id}})

        # 2. Resume with valid edit (buy_trigger revised to 1535.0)
        decision = {
            "action": "edit",
            "overrides": {
                "buy_trigger": 1535.0,
                "cmp": 1550.0,
                "target_trigger": 1660.0,
            },
        }
        res2 = await graph.ainvoke(Command(resume=decision), config={"configurable": {"thread_id": thread_id}})

        assert res2["status"] == "APPROVED_AND_EXECUTED"
        exec_res = res2["execution_result"]
        assert exec_res.success is True
        assert exec_res.trigger_id > 0
        saver.close()

    @pytest.mark.asyncio
    async def test_hitl_edit_branch_fails_on_circuit_breach(self, tmp_path: Path):
        """Verify that editing price above upper circuit limit is rejected by RiskGuard."""
        db_path = tmp_path / "hitl_edit_breach.db"
        saver = DurableSqliteSaver(db_path)
        builder = create_gtt_approval_workflow()
        graph = builder.compile(checkpointer=saver)

        thread_id = "test-edit-breach-1"
        initial_input = {
            "symbol": "INFY",
            "stage": 1,
            "cmp": 1550.0,
            "atr": 25.0,
            "portfolio_capital": 1000000.0,
            "available_cash": 1000000.0,
            "lower_circuit": 1300.0,
            "upper_circuit": 1700.0,
            "buy_trigger": 1540.0,
            "target_trigger": 1650.0,
        }

        # 1. Run to pause
        await graph.ainvoke(initial_input, config={"configurable": {"thread_id": thread_id}})

        # 2. Resume with breach edit (buy_trigger set to 1800.0 > upper circuit 1700.0)
        decision = {
            "action": "edit",
            "overrides": {
                "buy_trigger": 1800.0,
                "cmp": 1550.0,
            },
        }
        res2 = await graph.ainvoke(Command(resume=decision), config={"configurable": {"thread_id": thread_id}})

        assert res2["status"] == "EDIT_REJECTED_BY_RISK_GUARD"
        assert "CIRCUIT_LIMIT_BREACH" in res2["error"]
        saver.close()

    @pytest.mark.asyncio
    async def test_risk_guard_pre_approval_rejection_bypasses_interrupt(self, tmp_path: Path):
        """Verify that if candidate violates risk rules initially, graph ends without pause."""
        db_path = tmp_path / "pre_rejection.db"
        saver = DurableSqliteSaver(db_path)
        builder = create_gtt_approval_workflow()
        graph = builder.compile(checkpointer=saver)

        thread_id = "test-pre-reject-1"
        # buy_trigger equal to CMP violates Clause 1.6
        invalid_input = {
            "symbol": "INFY",
            "stage": 1,
            "cmp": 1550.0,
            "atr": 25.0,
            "portfolio_capital": 1000000.0,
            "available_cash": 1000000.0,
            "lower_circuit": 1300.0,
            "upper_circuit": 1700.0,
            "buy_trigger": 1550.0,  # Proximity breach
            "target_trigger": 1650.0,
        }

        result = await graph.ainvoke(invalid_input, config={"configurable": {"thread_id": thread_id}})

        # Graph should NOT pause at interrupt
        assert "__interrupt__" not in result
        assert result["status"] == "REJECTED_BY_RISK_GUARD"
        assert "rejection_node" in result["steps_completed"]
        assert "human_gtt_approval_node" not in result["steps_completed"]
        saver.close()


# ── 4. Streamlit Approval Drawer Component Logic Tests ────────────────────────


class TestStreamlitApprovalDrawerHelpers:
    """Verify formatting, validation, and decision construction logic."""

    def test_format_approval_metrics(self):
        metrics = {
            "quantity": 64,
            "capital_deployed_inr": 99052.8,
            "capital_deployed_pct": 0.0991,
            "worst_case_risk_inr": 5120.0,
            "worst_case_risk_pct": 0.00512,
            "reward_risk_ratio": 1.25,
            "binding_constraint": "CAPITAL_CAP",
        }
        res = format_approval_metrics(metrics)
        assert res["quantity"] == "64 shares"
        assert "₹99,052.80" in res["capital_deployed"]
        assert "(9.9%)" in res["capital_deployed"]
        assert "₹5,120.00" in res["worst_case_loss"]
        assert "(0.51%)" in res["worst_case_loss"]
        assert res["reward_risk"] == "1.25R"
        assert res["binding_constraint"] == "Capital Cap"

    def test_validate_user_overrides_valid(self):
        valid, err = validate_user_overrides({"buy_trigger": 1540.0, "quantity": 50})
        assert valid is True
        assert err is None

    def test_validate_user_overrides_invalid_price(self):
        valid, err = validate_user_overrides({"buy_trigger": -10.0})
        assert valid is False
        assert "positive" in err

    def test_validate_user_overrides_invalid_qty(self):
        valid, err = validate_user_overrides({"quantity": 0})
        assert valid is False
        assert "greater than zero" in err

    def test_handle_user_decision_payloads(self):
        d_app = handle_user_decision("approve")
        assert d_app == {"action": "approve"}

        d_rej = handle_user_decision("reject", reason="Risk too high")
        assert d_rej == {"action": "reject", "reason": "Risk too high"}

        d_edit = handle_user_decision("edit", overrides={"buy_trigger": 1530.0})
        assert d_edit == {"action": "edit", "overrides": {"buy_trigger": 1530.0}}

        with pytest.raises(ValueError, match="Invalid decision action"):
            handle_user_decision("unknown_action")
