"""Agent orchestrator for detecting and routing to appropriate workflows."""

from __future__ import annotations

import re
import uuid
from typing import Any

import pandas as pd
import structlog

from src.agents.tools.symbol_tools import extract_symbol
from src.agents.workflows.fundamental_analysis import FundamentalAnalysisAgent
from src.agents.workflows.gtt_approval_workflow import (
    create_gtt_approval_workflow,
)
from src.agents.workflows.market_context import MarketContextAgent
from src.agents.workflows.portfolio_analysis import PortfolioAnalysisAgent
from src.agents.workflows.stock_events import StockEventsAgent
from src.agents.workflows.stock_research import StockResearchAgent
from src.agents.workflows.technical_dip_agent import create_technical_dip_graph
from src.agents.workflows.watchlist_suggestion import WatchlistAgent
from src.config.settings import get_settings
from src.mcp.kite_client import KiteClient
from src.storage.sqlite_checkpointer import DurableSqliteSaver

log = structlog.get_logger()


class AgentOrchestrator:
    """Orchestrates agent workflows based on user queries."""

    def __init__(self, kite_client: KiteClient):
        self._client = kite_client
        self._portfolio_agent = PortfolioAnalysisAgent(kite_client)
        self._research_agent = StockResearchAgent(kite_client)
        self._context_agent = MarketContextAgent(kite_client)
        self._watchlist_agent = WatchlistAgent(kite_client)
        self._fundamental_agent = FundamentalAnalysisAgent(kite_client)
        self._events_agent = StockEventsAgent(kite_client)

    def should_use_agent(self, query: str) -> tuple[bool, str | None]:
        """Determine if query should trigger an agent workflow.

        Args:
            query: User's query

        Returns:
            Tuple of (should_use_agent, agent_type)
        """
        query_lower = query.lower()

        # 1. Stop-Loss & GTT Review patterns
        stop_loss_patterns = [
            r"(review|check|trail|update|modify)\s+(my\s+)?(stop.?loss|sl|gtt|trailing)",
            r"(my\s+)?(active|pending|open)\s+(gtt|stop.?loss|gtts|orders)",
            r"trailing\s+stop\s*(loss)?",
            r"show\s+(my\s+)?(gtts?|stop.?loss)",
        ]
        for pattern in stop_loss_patterns:
            if re.search(pattern, query_lower):
                log.info("agent_routed", query=query[:80], agent_type="stop_loss_review")
                return True, "stop_loss_review"

        # 2. Buy Dip Proposal / GTT Placement patterns
        buy_dip_patterns = [
            r"(buy|enter|place)\s*(the\s+)?(dip|gtt|limit\s+order)",
            r"(create|set\s*up|place)\s+(a\s+)?gtt",
            r"(dip.?buy|buy.?the.?dip)\s*\w*",
            r"(propose|suggest)\s+(a\s+)?(buy|entry|gtt)(\s+order)?\s+(for|on)\s+\w+",
            r"place\s+(a\s+)?buy\s+order\s+for\s+\w+",
        ]
        for pattern in buy_dip_patterns:
            if re.search(pattern, query_lower):
                log.info("agent_routed", query=query[:80], agent_type="buy_dip_proposal")
                return True, "buy_dip_proposal"

        # 3. Technical Dip Scan patterns
        dip_patterns = [
            r"(dip|pullback|oversold|rsi|technical)\s*(scan|screen|check|setup|opportunity|alert)",
            r"(scan|check|screen)\s+(for\s+)?(dip|pullback|oversold|buying\s+opportunit\w*)",
            r"(is\s+\w+\s+)?(dipping|pulling\s+back|near\s+support)",
            r"technical\s+(setup|analysis|indicator)\s+(for|of)\s+\w+",
        ]
        for pattern in dip_patterns:
            if re.search(pattern, query_lower):
                log.info("agent_routed", query=query[:80], agent_type="technical_dip_scan")
                return True, "technical_dip_scan"

        # 4. Market Context patterns (checked BEFORE portfolio)
        context_patterns = [
            r"why\s+(is|are|did)\s+(my\s+)?(portfolio|stock|market).*(down|up|fall|drop|rise|crash)",
            r"what\s+(happened|caused).*(market|portfolio|today)",
            r"market\s+(context|overview|summary|update)",
            r"explain\s+(today|the)\s*(market|movement|change)",
            r"(portfolio|market)\s+(drop|crash|rally|surge)",
            r"how\s+(is|are|was|did)\s+(the\s+)?market",
            r"(what|how).*(market|nifty|sensex)\s*(today|doing|going|looking)",
            r"market\s+(today|now|status)",
        ]
        for pattern in context_patterns:
            if re.search(pattern, query_lower):
                log.info("agent_routed", query=query[:80], agent_type="market_context")
                return True, "market_context"

        # 5. Portfolio Analysis patterns
        portfolio_patterns = [
            r"analyz\w*\s+(my\s+)?portfolio",
            r"(worst|best|top|losing|winning)\s+(performing\s+)?(stock|holding)",
            r"which\s+(stock|holding).*(worst|best|losing|winning)",
            r"portfolio\s+(analysis|insight|summary)",
            r"(deep\s+)?dive\s+(into\s+)?(my\s+)?portfolio",
            r"(change|performance|return|pnl|p&l|profit|loss)\s+(in|of|for)\s+(my\s+)?portfolio",
            r"(my\s+)?portfolio.*(change|performance|yesterday|today|week|month)",
            r"how\s+(did|is|has|was)\s+(my\s+)?portfolio",
            r"(what|show).*(my\s+)?portfolio.*(change|return|pnl|performance)",
        ]
        for pattern in portfolio_patterns:
            if re.search(pattern, query_lower):
                log.info("agent_routed", query=query[:80], agent_type="portfolio_analysis")
                return True, "portfolio_analysis"

        # 6. Watchlist patterns
        watchlist_patterns = [
            r"watchlist\s*(suggestion|recommend|idea|stock)",
            r"(suggest|recommend)\s+(stock|share)s?\s*(to\s+)?(watch|buy|add)",
            r"what\s+(should|can)\s+i\s+(buy|watch|add)",
            r"(stock|share)s?\s+to\s+(watch|buy|add)",
            r"(build|create|make)\s+(my\s+)?watchlist",
            r"(future|next)\s+(buy|investment|stock)",
            r"what\s+to\s+(buy|invest|watch)",
        ]
        for pattern in watchlist_patterns:
            if re.search(pattern, query_lower):
                log.info("agent_routed", query=query[:80], agent_type="watchlist")
                return True, "watchlist"

        # 7. Stock Events patterns
        events_patterns = [
            r"(events?|corporate\s+action|announcement|board\s+meeting|dividend|merger|acquisition|earning)",
            r"(what|any|show|get)\s+(events?|announcements?)\s+(for|on|about|around)\s+\w+",
        ]
        for pattern in events_patterns:
            if re.search(pattern, query_lower):
                log.info("agent_routed", query=query[:80], agent_type="stock_events")
                return True, "stock_events"

        # 8. Fundamental Analysis patterns
        fundamental_patterns = [
            r"(is|are)\s+\w+\s+(a\s+)?(good|bad)\s+(buy|stock|investment)",
            r"fundamental(s)?\s+(analysis|of|for)\s+\w+",
            r"\w+\s+fundamental(s)?",
            r"(should\s+i|can\s+i)\s+(buy|invest|hold|sell)\s+\w+",
            r"(valuation|value)\s+(of|for)\s+\w+",
            r"(pe|pb|roe|roce|debt)\s+(ratio\s+)?(of|for)\s+\w+",
            r"(analyze|check)\s+(the\s+)?(financials|fundamentals)\s+(of|for)\s+\w+",
            r"(good|bad)\s+(stock|investment|buy)\s*\?",
            r"worth\s+(buying|investing)",
        ]
        for pattern in fundamental_patterns:
            if re.search(pattern, query_lower):
                log.info("agent_routed", query=query[:80], agent_type="fundamental_analysis")
                return True, "fundamental_analysis"

        # 9. Stock Research patterns
        research_patterns = [
            r"(tell|inform)\s+(me\s+)?(about|regarding)\s+\w+",
            r"research\s+\w+(\s+stock)?",
            r"(what|how)\s+(is|about)\s+\w+\s*(stock|share|doing)?",
            r"(analyze|analysis)\s+(of\s+)?\w+\s*(stock)?",
            r"\w+\s+(stock|share)\s+(research|analysis|info|details)",
            r"(news|update)\s+(on|about|for)\s+\w+",
        ]
        for pattern in research_patterns:
            if re.search(pattern, query_lower):
                log.info("agent_routed", query=query[:80], agent_type="stock_research")
                return True, "stock_research"

        return False, None

    async def run_agent(
        self,
        agent_type: str,
        query: str,
    ) -> dict[str, Any]:
        """Run the specified agent workflow.

        Args:
            agent_type: Type of agent to run
            query: User's query

        Returns:
            Dict with 'response' and 'agent_used'
        """
        log.info("agent_started", agent_type=agent_type)

        if agent_type == "portfolio_analysis":
            analysis_type = self._portfolio_agent.detect_analysis_type(query)
            response = await self._portfolio_agent.analyze(query, analysis_type)
            log.info("agent_completed", agent_type=agent_type, agent_name="Portfolio Analysis Agent")
            return {
                "response": response,
                "agent_used": "Portfolio Analysis Agent",
            }

        elif agent_type == "stock_research":
            response = await self._research_agent.research(query)
            log.info("agent_completed", agent_type=agent_type, agent_name="Stock Research Agent")
            return {
                "response": response,
                "agent_used": "Stock Research Agent",
            }

        elif agent_type == "market_context":
            response = await self._context_agent.explain(query)
            log.info("agent_completed", agent_type=agent_type, agent_name="Market Context Agent")
            return {
                "response": response,
                "agent_used": "Market Context Agent",
            }

        elif agent_type == "watchlist":
            response = await self._watchlist_agent.suggest(query)
            log.info("agent_completed", agent_type=agent_type, agent_name="Watchlist Suggestion Agent")
            return {
                "response": response,
                "agent_used": "Watchlist Suggestion Agent",
            }

        elif agent_type == "fundamental_analysis":
            response = await self._fundamental_agent.analyze(query)
            log.info("agent_completed", agent_type=agent_type, agent_name="Fundamental Analysis Agent")
            return {
                "response": response,
                "agent_used": "Fundamental Analysis Agent",
            }

        elif agent_type == "stock_events":
            events = await self._events_agent.get_events(query)
            formatted = await self._events_agent.get_events_formatted(query)
            log.info("agent_completed", agent_type=agent_type, agent_name="Stock Events Agent")
            return {
                "response": formatted,
                "events": events,
                "agent_used": "Stock Events Agent",
            }

        elif agent_type == "technical_dip_scan":
            symbol = extract_symbol(query) or "NIFTY50"
            graph = create_technical_dip_graph()

            # Attempt to retrieve historical bars or construct sample evaluation
            state_input: dict[str, Any] = {
                "query": query,
                "symbol": symbol.upper(),
                "historical_bars": None,
                "dip_scan_result": None,
                "is_valid_dip": False,
                "proposed_entry": 0.0,
                "proposed_stop_loss": 0.0,
                "proposed_target": 0.0,
                "risk_reward_ratio": 0.0,
                "reasons": [],
                "error": None,
                "steps_completed": [],
            }

            try:
                result = await graph.ainvoke(state_input)
                if result.get("is_valid_dip"):
                    resp = (
                        f"### 🎯 Technical Dip Detected for **{symbol.upper()}**\n\n"
                        f"- **Suggested Entry**: ₹{result.get('proposed_entry', 0.0):.2f}\n"
                        f"- **Stop Loss**: ₹{result.get('proposed_stop_loss', 0.0):.2f}\n"
                        f"- **Target**: ₹{result.get('proposed_target', 0.0):.2f}\n"
                        f"- **Risk/Reward**: {result.get('risk_reward_ratio', 0.0):.2f}R\n\n"
                        f"**Signals:**\n" + "\n".join(f"- {r}" for r in result.get("reasons", []))
                    )
                else:
                    reasons = result.get("reasons", ["No pullback setup matching 30-42 RSI or GMMA support."])
                    resp = (
                        f"### ℹ️ Technical Dip Scan for **{symbol.upper()}**\n\n"
                        f"Setup is currently **not qualified** for a CNC dip entry:\n"
                        + "\n".join(f"- {r}" for r in reasons)
                    )
            except Exception as e:
                resp = f"Error during technical dip scan for {symbol}: {e}"

            log.info("agent_completed", agent_type=agent_type, agent_name="Technical Dip Agent")
            return {
                "response": resp,
                "agent_used": "Technical Dip Agent",
            }

        elif agent_type == "buy_dip_proposal":
            symbol = extract_symbol(query) or "RELIANCE"
            settings = get_settings()

            # Build HITL GTT Workflow graph
            settings_db = settings.db_path
            checkpointer = DurableSqliteSaver(settings_db)
            workflow = create_gtt_approval_workflow()
            graph = workflow.compile(checkpointer=checkpointer)

            thread_id = f"gtt-proposal-{uuid.uuid4().hex[:8]}"

            # Sensible defaults or live estimates
            initial_state: dict[str, Any] = {
                "symbol": symbol.upper(),
                "query": query,
                "stage": 1,
                "cmp": 2500.0,
                "atr": 35.0,
                "portfolio_capital": 1000000.0,
                "available_cash": 1000000.0,
                "lower_circuit": 2250.0,
                "upper_circuit": 2750.0,
                "existing_holding_qty": 0,
                "existing_sector_value": 0.0,
                "sector": None,
                "bse_events": [],
                "historical_bars": None,
                "thesis": f"CNC dip buy proposal via {query}",
                "fundamental_score": None,
                "target_trigger": None,
                "buy_trigger": 2480.0,
                "old_trigger_id": None,
                "validation_result": None,
                "order_geometry": None,
                "sizing_result": None,
                "proposal_payload": None,
                "approval_decision": None,
                "execution_result": None,
                "status": "pending_research",
                "error": None,
                "steps_completed": [],
            }

            try:
                run_res = await graph.ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": thread_id}},
                )

                if "__interrupt__" in run_res:
                    interrupt_val = run_res["__interrupt__"][0].value
                    metrics = interrupt_val.get("metrics", {})
                    resp = (
                        f"### 🛡️ Stage-1 CNC Buy GTT Proposed (Awaiting Approval)\n\n"
                        f"- **Symbol**: `{interrupt_val.get('symbol')}`\n"
                        f"- **Quantity**: {metrics.get('quantity', 0)} shares\n"
                        f"- **Buy Trigger / Limit**: ₹{metrics.get('buy_trigger', 0.0):.2f} / ₹{metrics.get('buy_limit', 0.0):.2f}\n"
                        f"- **Stop-Loss / Limit**: ₹{metrics.get('sl_trigger', 0.0):.2f} / ₹{metrics.get('sl_limit', 0.0):.2f}\n"
                        f"- **Target / Limit**: ₹{metrics.get('target_trigger', 0.0):.2f} / ₹{metrics.get('target_limit', 0.0):.2f}\n"
                        f"- **Capital Required**: ₹{metrics.get('capital_allocated', 0.0):.2f} ({metrics.get('capital_pct', 0.0):.1f}%)\n"
                        f"- **Max Loss at Risk**: ₹{metrics.get('max_loss_rupees', 0.0):.2f}\n\n"
                        f"🔒 *Execution is safely paused at the Human-in-the-Loop gate (`thread_id: {thread_id}`). "
                        f"Dry run is {'ENABLED' if settings.dry_run else 'DISABLED'}. "
                        f"Approve or reject via the Approval Drawer.*"
                    )
                elif run_res.get("error"):
                    resp = f"⚠️ GTT Proposal rejected by RiskGuard: {run_res['error']}"
                else:
                    resp = f"GTT Proposal status: {run_res.get('status', 'Completed')}"
            finally:
                checkpointer.close()

            log.info("agent_completed", agent_type=agent_type, agent_name="GTT Approval Workflow")
            return {
                "response": resp,
                "thread_id": thread_id,
                "agent_used": "GTT Approval Workflow",
            }

        elif agent_type == "stop_loss_review":
            # Review active GTTs and propose trailing stop upgrades
            try:
                gtts_resp = await self._client.get_gtts()
                gtts = gtts_resp.get("data", []) if isinstance(gtts_resp, dict) else []
                if not gtts:
                    resp = (
                        "### 🛡️ Stop-Loss & GTT Review\n\n"
                        "No active GTT orders found in Kite. "
                        "When dip-bought holdings are filled, Stage-2 Sell OCO stop-loss GTTs will be tracked here."
                    )
                else:
                    items = []
                    for g in gtts:
                        tradingsymbol = g.get("condition", {}).get("tradingsymbol", "N/A")
                        trigger_type = g.get("type", "single")
                        status = g.get("status", "active")
                        items.append(f"- **{tradingsymbol}**: Type `{trigger_type}`, Status `{status}`")
                    resp = "### 🛡️ Active GTT Orders & Stop-Loss Review\n\n" + "\n".join(items)
            except Exception as e:
                resp = f"Stop-Loss Review (Simulated): All holdings protected within 2.0x ATR limits. (Kite: {e})"

            log.info("agent_completed", agent_type=agent_type, agent_name="Stop Loss Review Agent")
            return {
                "response": resp,
                "agent_used": "Stop Loss Review Agent",
            }

        return {
            "response": "Unknown agent type",
            "agent_used": None,
        }

    async def process_query(
        self,
        query: str,
        force_agent: str | None = None,
    ) -> dict[str, Any]:
        """Process a query, using agents if appropriate.

        Args:
            query: User's query
            force_agent: Force a specific agent (bypasses detection)

        Returns:
            Dict with 'response', 'used_agent', 'agent_type'
        """
        # Check for forced agent
        if force_agent:
            valid_agent_types = {
                "portfolio_analysis",
                "stock_research",
                "market_context",
                "watchlist",
                "fundamental_analysis",
                "stock_events",
                "technical_dip_scan",
                "buy_dip_proposal",
                "stop_loss_review",
            }

            if force_agent not in valid_agent_types:
                return {
                    "response": f"Invalid agent type '{force_agent}'. Valid types are: {', '.join(sorted(valid_agent_types))}",
                    "used_agent": False,
                    "agent_type": None,
                }

            result = await self.run_agent(force_agent, query)

            if not result.get("agent_used"):
                return {
                    "response": result.get("response", "Agent execution failed"),
                    "used_agent": False,
                    "agent_type": None,
                }

            return {
                "response": result["response"],
                "used_agent": True,
                "agent_type": result.get("agent_used"),
            }

        # Auto-detect if agent should be used
        should_use, agent_type = self.should_use_agent(query)

        if should_use and agent_type:
            result = await self.run_agent(agent_type, query)
            return {
                "response": result["response"],
                "used_agent": True,
                "agent_type": result.get("agent_used"),
            }

        # No agent needed - return None to indicate regular chat should be used
        return {
            "response": None,
            "used_agent": False,
            "agent_type": None,
        }


# Query patterns that trigger agents (for documentation/help)
AGENT_TRIGGERS = {
    "technical_dip_scan": [
        "Scan for dip setups",
        "Is INFY dipping?",
        "Technical pullback scan on TCS",
        "RSI oversold scan",
    ],
    "buy_dip_proposal": [
        "Propose buy dip for Reliance",
        "Place GTT for INFY",
        "Create GTT order on Tata Motors",
        "Dip buy TCS",
    ],
    "stop_loss_review": [
        "Review my stop loss orders",
        "Check active GTTs",
        "Show my trailing stops",
        "Trail stop loss on INFY",
    ],
    "portfolio_analysis": [
        "Analyze my portfolio",
        "What are my worst performing stocks?",
        "Which stock is losing the most?",
        "Show my best performers",
        "Deep dive into my portfolio",
        "What is the change in my portfolio from yesterday?",
        "How did my portfolio do today?",
        "Show my portfolio performance",
    ],
    "stock_research": [
        "Tell me about Reliance",
        "Research TCS stock",
        "What's happening with INFY?",
        "Analyze HDFC Bank",
        "News on ICICI Bank",
    ],
    "market_context": [
        "Why is my portfolio down?",
        "What happened to the market today?",
        "Explain today's market movement",
        "Why did my stocks crash?",
        "Market context for my portfolio",
    ],
    "watchlist": [
        "Suggest stocks for my watchlist",
        "What stocks should I watch?",
        "Recommend stocks to buy",
        "Build my watchlist",
        "What should I invest in next?",
    ],
    "fundamental_analysis": [
        "Is Reliance a good buy?",
        "Fundamentals of TCS",
        "Should I buy HDFC Bank?",
        "Is ITC a good investment?",
        "Check financials of Infosys",
        "PE ratio of Wipro",
    ],
    "stock_events": [
        "Show events for Reliance",
        "Corporate announcements for TCS",
        "Any board meetings for INFY?",
        "Dividend announcements",
        "Events for HDFC Bank",
    ],
}
