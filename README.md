# Zerodha CNC Dip-Buying & GTT Stop-Loss Copilot

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-white.svg)](https://ollama.ai/)
[![Anthropic Claude](https://img.shields.io/badge/LLM-Claude-orange.svg)](https://www.anthropic.com/)
[![Zerodha Kite MCP](https://img.shields.io/badge/Broker-Zerodha%20Kite%20MCP-red.svg)](https://kite.zerodha.com/)

> A personal algorithmic investing assistant for Indian equity (`CNC` cash delivery) dip-buying and Stage-2 Sell OCO stop-loss protection via Zerodha Kite MCP and LangGraph Human-in-the-Loop (HITL).
> *Originally forked from [mohdasif2294/portfolio-copilot](https://github.com/mohdasif2294/portfolio-copilot).*

---

## Core Architecture & Safety Rules

1. **Probabilistic Research vs. Deterministic Execution:** LLMs (Ollama local default or Anthropic Claude) synthesize research, fundamentals, and thesis generation. All share quantities, NSE tick-size roundings, Clause 1.6 trigger distances, and circuit clamps are calculated strictly in pure Python by `RiskGuard`.
2. **Strict GTT-Only Brokerage Scope:** Writing is restricted strictly to GTT orders (`place_gtt_order`, `modify_gtt_order`, `delete_gtt_order`). `place_order`, `modify_order`, and `cancel_order` are prohibited at the MCP client layer, keeping workflows compliant and exempt from SEBI April 1, 2026 algo registration rules.
3. **LangGraph Human-in-the-Loop (HITL) Gate:** Every proposed GTT order halts at an `interrupt()` barrier backed by a durable SQLite checkpointer (`DurableSqliteSaver` in WAL mode with `busy_timeout`). Live order execution is quarantined to a downstream node that executes only upon explicit user approval.
4. **Safety Defaults:** `DRY_RUN=true` is enabled by default across all configs and test suites. Orders are validated and simulated safely without placing live orders on Zerodha.

---

## 6-Phase Architecture

| Phase | Package | Description |
|---|---|---|
| **Phase 1: Kite MCP & Ledger** | `src/broker/`, `src/config/`, `src/storage/` | Streamable HTTP Kite MCP client with allowlist and backoff; `Settings` with `CLI > Env > .env` precedence; SQLite positions ledger & JSONL trade audit. |
| **Phase 2: Compliant Data Layer** | `src/data/`, `src/technicals/` | NSE Bhavcopy join (`sec_bhavdata_full` + UDiFF); cached/paced Screener & BSE adapters; clean-room technical indicators (Wilder's RSI, 14-day ATR, 50/200-DMA, GMMA). |
| **Phase 3: Deterministic RiskGuard** | `src/risk/` | Dual-cap position sizer (1% risk cap & 10% capital cap), Clause 1.6 distance check, dynamic NSE tick alignment, circuit clamps, and sector caps. |
| **Phase 4: Durable HITL Workflow** | `src/agents/workflows/`, `src/ui/components/` | LangGraph workflow pausing at `interrupt()`; Streamlit interactive approval drawer supporting Approve / Edit / Reject. |
| **Phase 5: IST Positional Scheduler** | `src/scheduler/` | Four IST schedules: 08:45 AM pre-market, 04:05 PM post-CAS reconciliation & Stage-2 OCO queuing, 06:15 PM EOD Bhavcopy dip scan, and Friday 04:15 PM digest. |
| **Phase 6: EOD Backtest & Shadow Mode** | `src/backtest/`, `scripts/` | Split-adjusted offline backtest engine; Kite MCP local Go failover script & 2-week shadow-mode checklist. |

---

## Quick Start

```bash
# 1. Install dependencies
poetry install

# 2. Configure environment
cp .env.example .env
# DRY_RUN=true is enabled by default

# 3. Run Web Interface
make web        # Streamlit web interface at http://localhost:8501

# or Run CLI
make cli        # Terminal interface with Rich formatting
```

### Key Make Commands

| Command | Description |
|---|---|
| `make web` | Start Streamlit web interface |
| `make cli` | Start interactive Rich CLI interface |
| `make scheduler` | Run the IST market scheduler daemon |
| `make backtest` | Run split-adjusted EOD dip-buying backtest engine |
| `make failover` | Run Kite MCP failover diagnostics and Go server switcher |
| `make test` | Run pytest unit test suite across all modules |
| `make check` | Run linter, typechecker, and unit tests |

---

## CLI Flags

```bash
# Run CLI with custom LLM provider
python -m src.ui.cli --provider claude --model claude-sonnet-4-20250514

# Run CLI with live mode (requires manual 2FA OAuth)
python -m src.ui.cli --no-dry-run

# Run CLI with custom Kite MCP URL
python -m src.ui.cli --kite-mcp-url https://mcp.kite.trade/mcp
```

---

## License

This project is licensed under the **Apache-2.0** License. See the [LICENSE](LICENSE) file for details.
