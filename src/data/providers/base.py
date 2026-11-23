"""Abstract base interfaces for market data and research providers.

Enables seamless future swapping to commercial APIs (e.g., Tapetide, BharatStock)
while keeping clean boundaries across agents.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.data.scrapers.screener import FundamentalData


@runtime_checkable
class FundamentalDataProvider(Protocol):
    """Protocol for fundamental analysis data providers."""

    async def get_fundamentals(self, symbol: str) -> FundamentalData:
        """Fetch fundamental metrics, ratios, and balance sheet metrics for a stock."""
        ...


@runtime_checkable
class CorporateEventsProvider(Protocol):
    """Protocol for exchange corporate actions and events providers."""

    async def get_corporate_events(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch corporate announcements, board meetings, dividends, and splits."""
        ...
