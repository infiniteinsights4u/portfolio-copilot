"""Tests for agent orchestrator pattern matching."""

import pytest
from unittest.mock import MagicMock

from src.agents.orchestrator import AgentOrchestrator


@pytest.fixture
def orchestrator():
    """Create orchestrator with mock client."""
    mock_client = MagicMock()
    return AgentOrchestrator(mock_client)


class TestPortfolioAnalysisPatterns:
    """Test portfolio analysis agent detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "Analyze my portfolio",
            "analyze portfolio",
            "What are my worst performing stocks?",
            "worst stocks in my portfolio",
            "best performing stock",
            "which stocks are performing the best",
            "portfolio analysis",
            "worst holding in portfolio",
        ],
    )
    def test_portfolio_patterns_detected(self, orchestrator, query):
        """Test that portfolio analysis patterns are detected."""
        should_use, agent_type = orchestrator.should_use_agent(query)
        assert should_use is True
        assert agent_type == "portfolio_analysis"


class TestStockResearchPatterns:
    """Test stock research agent detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "Tell me about TCS",
            "tell me about Reliance",
            "Research HDFCBANK",
            "research infosys",
            "news on WIPRO",
            "news about ICICIBANK",
            "what is TCS stock doing",
        ],
    )
    def test_research_patterns_detected(self, orchestrator, query):
        """Test that stock research patterns are detected."""
        should_use, agent_type = orchestrator.should_use_agent(query)
        assert should_use is True
        assert agent_type == "stock_research"


class TestMarketContextPatterns:
    """Test market context agent detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "Why is my portfolio down today?",
            "why did my portfolio drop",
            "what happened to my portfolio today",
            "market context",
            "market overview",
            "portfolio crash today",
        ],
    )
    def test_context_patterns_detected(self, orchestrator, query):
        """Test that market context patterns are detected."""
        should_use, agent_type = orchestrator.should_use_agent(query)
        assert should_use is True
        assert agent_type == "market_context"


class TestWatchlistPatterns:
    """Test watchlist agent detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "Suggest stocks to watch",
            "suggest stocks to buy",
            "what should I buy",
            "What should I add to my watchlist?",
            "watchlist suggestions",
            "recommend stocks to watch",
            "stocks to buy",
            "what to buy",
        ],
    )
    def test_watchlist_patterns_detected(self, orchestrator, query):
        """Test that watchlist patterns are detected."""
        should_use, agent_type = orchestrator.should_use_agent(query)
        assert should_use is True
        assert agent_type == "watchlist"


class TestFundamentalAnalysisPatterns:
    """Test fundamental analysis agent detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "Is TCS a good buy?",
            "is reliance a good stock",
            "Is HDFC Bank a good investment?",
            "Should I buy Infosys?",
            "should i invest in wipro",
            "Fundamentals of TCS",
            "fundamental analysis of reliance",
            "Can I buy ICICI Bank?",
        ],
    )
    def test_fundamental_patterns_detected(self, orchestrator, query):
        """Test that fundamental analysis patterns are detected."""
        should_use, agent_type = orchestrator.should_use_agent(query)
        assert should_use is True
        assert agent_type == "fundamental_analysis"


class TestNoAgentPatterns:
    """Test queries that should not trigger an agent."""

    @pytest.mark.parametrize(
        "query",
        [
            "Hello",
            "Hi there",
            "What time is it?",
            "Help me",
            "Thanks",
            "What can you do?",
        ],
    )
    def test_general_queries_no_agent(self, orchestrator, query):
        """Test that general queries don't trigger an agent."""
        should_use, agent_type = orchestrator.should_use_agent(query)
        assert should_use is False
        assert agent_type is None


class TestPatternPriority:
    """Test that pattern matching has correct priority."""

    def test_fundamental_over_research(self, orchestrator):
        """Test that 'Is X good buy' triggers fundamental, not research."""
        query = "Is TCS a good buy?"
        _, agent_type = orchestrator.should_use_agent(query)
        # Fundamental patterns should match before research patterns
        assert agent_type == "fundamental_analysis"

    def test_portfolio_specific_terms(self, orchestrator):
        """Test portfolio-specific terminology."""
        query = "Which stocks are performing worst?"
        should_use, agent_type = orchestrator.should_use_agent(query)
        assert should_use is True
        assert agent_type == "portfolio_analysis"


class TestTechnicalDipPatterns:
    """Test technical dip agent detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "Scan for dip setups",
            "Is INFY dipping?",
            "check technical dip setup for TCS",
            "rsi oversold scan",
            "technical pullback scan",
        ],
    )
    def test_dip_patterns_detected(self, orchestrator, query):
        should_use, agent_type = orchestrator.should_use_agent(query)
        assert should_use is True
        assert agent_type == "technical_dip_scan"


class TestBuyDipProposalPatterns:
    """Test GTT buy proposal agent detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "propose buy order for RELIANCE",
            "place GTT for INFY",
            "create GTT on TCS",
            "dip buy Tata Motors",
            "place buy order for HDFCBANK",
        ],
    )
    def test_buy_dip_patterns_detected(self, orchestrator, query):
        should_use, agent_type = orchestrator.should_use_agent(query)
        assert should_use is True
        assert agent_type == "buy_dip_proposal"


class TestStopLossReviewPatterns:
    """Test stop-loss review agent detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "review my stop loss",
            "check active GTTs",
            "review active stop loss orders",
            "trail stop loss on INFY",
            "show my open gtts",
        ],
    )
    def test_stop_loss_patterns_detected(self, orchestrator, query):
        should_use, agent_type = orchestrator.should_use_agent(query)
        assert should_use is True
        assert agent_type == "stop_loss_review"


class TestOrchestratorExecution:
    """Test full orchestrator execution flow."""

    @pytest.mark.asyncio
    async def test_orchestrator_dip_to_riskguard_flow(self, orchestrator, tmp_path):
        """Test Orchestrator -> RiskGuard -> interrupt() pause with zero MCP writes in DRY_RUN=true."""
        from src.config.settings import get_settings

        # Enforce isolated test DB and DRY_RUN=true via get_settings overrides
        db_path = tmp_path / "test_orchestrator.db"
        get_settings(force_reload=True, db_path=db_path, dry_run=True)

        query = "Propose buy dip for INFY"
        should_use, agent_type = orchestrator.should_use_agent(query)
        assert should_use is True
        assert agent_type == "buy_dip_proposal"

        result = await orchestrator.run_agent("buy_dip_proposal", query)
        assert result["agent_used"] == "GTT Approval Workflow"
        assert "Stage-1 CNC Buy GTT Proposed" in result["response"]
        assert "thread_id" in result
        assert "paused at the Human-in-the-Loop gate" in result["response"]

        # Assert no MCP order writes occurred on mock_client
        assert not orchestrator._client.place_order.called
        assert not orchestrator._client.place_gtt_order.called

