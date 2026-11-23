---
name: build-dip-copilot
description: Guides phased implementation, refactoring, and testing of the forked portfolio-copilot Zerodha CNC dip-buying and GTT stop-loss platform. Use when building Phases 1 through 6, adding RiskGuard/LangGraph HITL nodes, or verifying tests.
---

# Phased Build Protocol for Dip-Investing Copilot

Always start each phase by inspecting existing files in `src/` and `tests/`, producing a concise Implementation Plan artifact for user approval before editing code, and running `pytest` + linting before closing the phase. Execute only the requested phase.

## Phase 1: Kite MCP Hardening, Config & SQLite Ledger

1. Reconcile `README.md` license notice with Apache-2.0 `LICENSE`; ensure `pyproject.toml` / `poetry.lock` are clean on Python 3.12+.
2. Upgrade `src/broker/kite_mcp_client.py` to `langchain-mcp-adapters` Streamable HTTP (`transport="http"`) with `KITE_MCP_URL` (default `https://mcp.kite.trade/mcp`), persistent `async with client.session("kite") as session:`, allowlist filtering, and `ToolCallInterceptor` retry backoff.
3. Implement `src/config/settings.py` with precedence `CLI > Env > .env`, `DRY_RUN=true` default, and `ZERODHA_DDPI_ENABLED` check.
4. Build `src/storage/ledger.py` (SQLite) with tables `positions_ledger`, `gtt_audit_log`, `pending_buy_commits`, `screener_cache`, plus `src/audit/trade_logger.py` (`chmod 600` JSONL mirror).
5. Add a Kite MCP session health badge (warning before 6:00 AM IST expiry) in the Streamlit sidebar.
6. Verify: unit tests for allowlist blocking `place_order`, config precedence, and SQLite CRUD.

## Phase 2: Compliant Data Layer & Clean-Room Technical Dip Engine

1. Build `src/data/bhavcopy.py` (NSE `sec_bhavdata_full` + UDiFF ISIN join + `DATE1` validation) and `src/data/instruments_cache.py` (daily post-8:30 AM `/instruments` CSV keyed by `(exchange, tradingsymbol)`).
2. Wrap Screener.in and BSE corporate events behind provider interfaces; enforce SQLite quarterly cache, `concurrency=1`, `>=2.5s` pacing, and 403/429 backoff on Screener.
3. Build `src/technicals/dip_scanner.py` using `TA-Lib` or `ta_py`: daily/weekly RSI(14) pullback-curl (30–42), 14-day Wilder ATR, 50/200-DMA, GMMA (3–15 vs 30–60 EMA), Heikin-Ashi, 52-week drawdown, and 20-day delivery % check.
4. Add `TechnicalDipAgent` node to the LangGraph research workflow and wire a split-adjusted `yfinance` (`.NS`) loader for offline backtesting only.
5. Verify: unit tests on Bhavcopy `DATE1` mismatch rejection, Screener cache hit/backoff, and indicator outputs against known fixture bars.

## Phase 3: Deterministic RiskGuard & Dual-Cap Position Sizer

1. Implement `src/risk/risk_guard.py` enforcing dual-cap sizing, dynamic NSE tick rounding, Clause 1.6 minimum distance, circuit clamps, 10% single-stock / 25% sector caps, pending-Buy-GTT cash reservation, 30s duplicate suppression, BSE corporate-action blackout, and Stage-2 holdings verification (`quantity >= sell_qty`).
2. Verify: exhaustive `pytest` suite covering every rejection reason and both worked examples (Case A capital-cap binds at 68 shares on ₹0.10 tick; Case B risk-cap binds at 158 shares on ₹0.05 tick for ₹10,00,000 portfolio).

## Phase 4: LangGraph Durable `interrupt()` HITL & Streamlit Approval UI

1. Configure a durable SQLite/Postgres LangGraph checkpointer (`AsyncSqliteSaver` or Postgres) replacing any `InMemorySaver`.
2. Wire `Research Agents -> RiskGuardNode -> human_gtt_approval_node (interrupt()) -> execute_gtt_node`.
3. Implement safe GTT edits as `place_gtt_order` (`product="CNC"`) -> verify `get_gtts` -> `delete_gtt_order` (integer `trigger_id`). Respect `DRY_RUN=true` inside `execute_gtt_node`.
4. Build the Streamlit approval drawer handling `stream.interrupted` with Approve / Edit / Reject controls displaying thesis, fundamentals, BSE events, exact prices, qty, capital %, and worst-case ₹ loss.
5. Verify: `pytest` HITL tests with fresh `thread_id` asserting `__interrupt__` pauses with zero MCP writes before `Command(resume=...)`, and verifying Approve, Edit, and Reject branches.

## Phase 5: Scheduled Positional Jobs & EOD Reconciliation

1. Build `src/scheduler/jobs.py` with the four IST schedules: 08:45 AM pre-market, 04:05 PM post-CAS reconciliation + Stage-2 Sell OCO queuing + gap-down unfilled SL alert + Chandelier/ATR trailing-stop proposals, 06:15 PM EOD Bhavcopy dip scan, and Friday 04:15 PM weekly digest.
2. Add optional Slack/Telegram webhook alerting.
3. Verify: dry-run tests simulating a filled Stage-1 Buy GTT queuing a Stage-2 Sell OCO approval and an unfilled triggered Sell GTT firing an urgent alert.

## Phase 6: Offline Backtest & Shadow-Mode Readiness

1. Build `src/backtest/eod_engine.py` on split-adjusted `.NS` EOD history testing the dip entry, 2x ATR stop, and 2.5R / Chandelier trail with realistic Indian brokerage/STT/slippage assumptions.
2. Produce a 2-week `DRY_RUN=true` shadow-mode checklist and failover runbook for switching `KITE_MCP_URL` to a locally compiled `zerodha/kite-mcp-server` binary (`EXCLUDED_TOOLS=place_order,modify_order,cancel_order`).
