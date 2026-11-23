# Project Context: Zerodha Dip-Buying & Stop-Loss Investing Copilot

This repository is a fork of `mohdasif2294/portfolio-copilot` (Python 3.12+, Poetry, LangGraph, ChromaDB, Streamlit, Rich CLI) being evolved into a personal short-term Indian equity (`CNC` delivery) dip-investing assistant.

## Core Architecture

* **Probabilistic AI vs. Deterministic Execution:** LLM agents (Ollama local default, Anthropic Claude, or configured cloud models) handle research synthesis, fundamental scoring, news RAG, corporate-event checks, and thesis generation ONLY.
* **Deterministic RiskGuard:** Pure Python (`src/risk/risk_guard.py`) computes every share quantity, NSE tick-aligned price, circuit clamp, and GTT rule check. Never let an LLM calculate share quantities or final order prices.
* **Human-in-the-Loop (HITL) Gate:** Every GTT write (`place_gtt_order`, `modify_gtt_order`, `delete_gtt_order`) MUST pause at a LangGraph `interrupt()` node backed by a durable SQLite/Postgres checkpointer (never `InMemorySaver` in production). The actual MCP write MUST live in a separate downstream node after `interrupt()` so graph resumes never trigger duplicate orders.
* **Brokerage Backend:** Connect to official `zerodha/kite-mcp-server` as an external runtime dependency over Streamable HTTP (`transport="http"` via `langchain-mcp-adapters`), defaulting to `https://mcp.kite.trade/mcp` with one-line env failover (`KITE_MCP_URL`) to a local self-hosted Go binary.

## Hard Prohibitions (Never Violate)

1. **GTT-Only Brokerage Scope:** NEVER expose `place_order`, `modify_order`, or `cancel_order` to any LLM agent or tool allowlist. Restricting writes strictly to GTT orders keeps this workflow exempt from SEBI's April 1, 2026 static IP and exchange Algo ID rules.
2. **No Automated TOTP Login:** NEVER implement headless/automated TOTP login (`pyotp` / `KITE_TOTP_SECRET`) or store Zerodha login passwords in code or `.env`. Use Zerodha's official manual daily 2FA OAuth flow (tokens expire daily at 6:00 AM IST).
3. **Clean-Room Technical Indicators Only:** NEVER copy code from `ashishkumar30/Stock_Market_Live_Trading_using_AI` (no OSI open-source license). Re-implement standard technical indicators cleanly from scratch.
4. **No Legacy `pandas-ta`:** NEVER install the unmaintained `pandas-ta` package (breaks on NumPy >= 2.0). Use `TA-Lib` (>=0.8.3 prebuilt wheels), pure-Python `ta_py`, or `pandas-ta-classic`.
5. **No Permanent Kite Data Storage:** NEVER permanently store Kite Connect quotes, candles, or instrument dumps for historical replay or live-tick paper trading. Build permanent EOD storage and offline backtests strictly from official NSE Bhavcopy archives and split-adjusted `yfinance` (`.NS`).
6. **No Stateless MCP Calls or Legacy SSE:** NEVER call `MultiServerMCPClient.get_tools()` statelessly per prompt (which drops `mcp.kite.trade` session binding) and NEVER use deprecated SSE transport. Always hold a single persistent session via `async with client.session("kite") as session:`.
7. **Default Safety Lock:** Keep `DRY_RUN=true` enabled by default across config and tests. Never submit live GTTs in automated tests.
8. **Licensing & Dependencies:** Keep the authoritative Apache-2.0 `LICENSE` file, align `README.md` to Apache-2.0, and pin all dependencies via `poetry.lock`.
9. **No Extra Agent Frameworks:** Keep LangGraph as the sole orchestration framework (do not add Agno or Google ADK).

## Engineering Standards

* Enforce config precedence: `CLI flags > Environment variables > .env`.
* Write `pytest` unit tests for every `RiskGuard` rule and LangGraph `interrupt()` pause/resume path using fresh `thread_id` values.
* Maintain structured logging (`structlog`) and a `chmod 600` JSONL trade audit log mirrored to SQLite.
* Never print logs to `stdout` in any `stdio` MCP subprocess; log to `stderr` or files.
