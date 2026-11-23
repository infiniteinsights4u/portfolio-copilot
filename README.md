# Portfolio Copilot

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-white.svg)](https://ollama.ai/)
[![Anthropic Claude](https://img.shields.io/badge/LLM-Claude-orange.svg)](https://www.anthropic.com/)
[![Zerodha Kite](https://img.shields.io/badge/Broker-Zerodha%20Kite-red.svg)](https://kite.zerodha.com/)

> Ask questions about your portfolio in plain English. Get intelligent analysis powered by multiple LLM providers.

![Web UI Screenshot](docs/images/web-ui.png)

## Features

| | Feature | Description |
|---|---------|-------------|
| 📊 | **Portfolio Analysis** | Identify your best/worst performers with news context |
| 🔍 | **Stock Research** | Get comprehensive research on any stock |
| 📈 | **Market Context** | Understand why your portfolio moved today |
| 💰 | **Fundamental Analysis** | Buy/hold/sell recommendations using screener.in data |
| 👀 | **Watchlist Suggestions** | Discover stocks based on your investment style |
| 📅 | **Corporate Events** | Board meetings, dividends, acquisitions, earnings from BSE |

## Quick Start

```bash
# Clone and install
git clone https://github.com/mohdasif2294/portfolio-copilot.git
cd portfolio-copilot
poetry install

# Configure
cp .env.example .env
# Edit .env to set your preferred LLM provider (see below)

# Run
make web    # Web interface at http://localhost:8501
# or
make cli    # Terminal interface
```

**Prerequisites:** Python 3.12+, [Poetry](https://python-poetry.org/docs/#installation)

### LLM Provider Setup

Choose between **Ollama** (local, free) or **Claude** (API, paid):

#### Option 1: Ollama (Default - Local & Free)

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3

# Run (uses Ollama by default)
make cli
```

#### Option 2: Claude (Anthropic API)

```bash
# Set in .env
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-xxx

# Or use CLI flag
python -m src.ui.cli --provider claude
```

#### CLI Options

```bash
# Use Ollama with a specific model
python -m src.ui.cli --provider ollama --model llama3.1

# Use Claude
python -m src.ui.cli --provider claude

# Override model
python -m src.ui.cli --provider claude --model claude-sonnet-4-20250514
```

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Screenshots](#screenshots)
- [Usage](#usage)
- [Agents](#agents)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [License](#license)

## Screenshots

### Web Interface

![Web UI Screenshot](docs/images/web-ui.png)

*Chat interface with suggested prompts, Kite login sidebar, and real-time responses*

### CLI Interface

![CLI Screenshot](docs/images/cli.png)

*Terminal-based interface with rich formatting and command support*

## Usage

### Natural Language

Just ask questions naturally - the system auto-detects intent:

```
You> What are my worst performing stocks?
You> Tell me about Reliance
You> Why is my portfolio down today?
You> Is TCS a good buy?
You> Suggest stocks for my watchlist
You> Show events for Reliance
You> Any board meetings for Infosys?
```

### Commands

| Command | Description |
|---------|-------------|
| `login` | Connect to your Zerodha account |
| `holdings` | View portfolio table |
| `analyze` | Portfolio analysis |
| `research` | Stock research |
| `fundamentals` | Fundamental analysis |
| `events` | Corporate events from BSE |
| `ingest` | Index news articles |
| `help` | Show all commands |

## Agents

The system uses specialized AI agents for different tasks:

| Agent | What It Does |
|-------|--------------|
| 📊 **Portfolio Analysis** | Analyzes holdings, identifies best/worst performers, fetches relevant news |
| 🔍 **Stock Research** | Researches individual stocks with price data, news, and your position |
| 📈 **Market Context** | Explains why your portfolio moved using market news and trends |
| 👀 **Watchlist Suggestion** | Recommends stocks based on your current sectors and blue chips |
| 💰 **Fundamental Analysis** | Evaluates stocks using screener.in metrics (P/E, ROE, debt) with buy/hold/sell scores |
| 📅 **Corporate Events** | Fetches board meetings, dividends, acquisitions, earnings announcements from BSE India |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              User Interface Layer                           │
│  ┌─────────────────────┐    ┌─────────────────────┐         │
│  │  Web UI (Streamlit) │    │   CLI (Rich)        │         │
│  │  - Chat interface   │    │   - Commands        │         │
│  │  - Suggested prompts│    │   - REPL interface  │         │
│  │  - Login sidebar    │    │                     │         │
│  └─────────────────────┘    └─────────────────────┘         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Agent Orchestrator                        │
│            (Query detection & routing)                      │
└────────┬─────────────┬─────────────┬─────────────┬──────────┘
         │             │             │             │
         ▼             ▼             ▼             ▼
┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
│ Portfolio │ │  Stock    │ │  Market   │ │Fundamental│ │  Stock    │
│ Analysis  │ │ Research  │ │  Context  │ │ Analysis  │ │  Events   │
│  Agent    │ │  Agent    │ │  Agent    │ │  Agent    │ │  Agent    │
└───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘
       │            │             │             │             │
       └────────────┴──────┬──────┴─────────────┴─────────────┘
                           │
       ┌───────────────┬───┼──────────────-----------─┐
       │               │              │               │
       ▼               ▼              ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Kite MCP   │ │ RAG System  │ │ Screener.in │ │  BSE India  │
│  (Holdings, │ │ (ChromaDB + │ │ (Fundamental│ │ (Corporate  │
│   Quotes)   │ │  News)      │ │  Data)      │ │  Events)    │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   LLM Provider  │
                    │  ┌───────────┐  │
                    │  │  Ollama   │  │
                    │  │ (Default) │  │
                    │  └───────────┘  │
                    │  ┌───────────┐  │
                    │  │  Claude   │  │
                    │  │ (Optional)│  │
                    │  └───────────┘  │
                    └─────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| LLM | Ollama (Llama 3) or Claude (Anthropic) |
| Agent Framework | LangGraph |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Broker Integration | Zerodha Kite MCP |
| Web UI | Streamlit |
| CLI | Rich |
| Web Scraping | BeautifulSoup4, httpx |
| Logging | structlog (structured events, console + JSON) |
| Observability | LangSmith (LangGraph tracing), custom tracing decorators |
| Market Data | BSE India API (corporate events) |
| Symbol Resolution | NSE equity list (2200+ stocks, auto-cached) |

## Why This Project?

This is a learning project to understand modern AI systems:
- **MCP (Model Context Protocol)** - Broker data integration
- **RAG (Retrieval Augmented Generation)** - News context
- **LangGraph** - Agentic workflows
- **LLM Orchestration** - Tool calling and synthesis
- **Multi-Provider Support** - Abstraction layer for local (Ollama) and API (Claude) models

---

## License

Apache-2.0 License - see [LICENSE](LICENSE) for details.

**Disclaimer:** This tool is for **educational purposes only**. It is NOT investment advice. Always consult a qualified financial advisor before making investment decisions.
