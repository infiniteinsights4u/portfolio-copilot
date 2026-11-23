"""BSE corporate events provider adapter."""

from __future__ import annotations

import dataclasses
from typing import Any

import structlog

from src.data.providers.base import CorporateEventsProvider
from src.data.scrapers.bse import BSEScraper, CorporateEvent

logger = structlog.get_logger(__name__)


class BSEEventsAdapter(CorporateEventsProvider):
    """Adapter wrapping BSE corporate announcements behind CorporateEventsProvider interface."""

    def __init__(self, scraper: BSEScraper | None = None) -> None:
        self.scraper = scraper
        self._owns_scraper = scraper is None

    async def _get_scraper(self) -> BSEScraper:
        if self.scraper is None:
            self.scraper = BSEScraper()
        return self.scraper

    async def close(self) -> None:
        """Close underlying scraper if owned."""
        if self._owns_scraper and self.scraper is not None:
            await self.scraper.close()
            self.scraper = None

    async def get_corporate_events(
        self,
        symbol: str,
        days_back: int = 30,
    ) -> list[dict[str, Any]]:
        """Fetch corporate announcements and events for a stock."""
        clean_symbol = symbol.strip().upper()
        scraper = await self._get_scraper()

        try:
            if hasattr(scraper, "get_corporate_events"):
                events = await scraper.get_corporate_events(clean_symbol, limit=days_back)
            elif hasattr(scraper, "get_announcements"):
                events = await scraper.get_announcements(clean_symbol, days_back=days_back)
            else:
                events = []
            results = []
            for ev in events:
                if dataclasses.is_dataclass(ev):
                    data = dataclasses.asdict(ev)
                elif isinstance(ev, dict):
                    data = dict(ev)
                else:
                    data = {
                        "title": getattr(ev, "title", ""),
                        "category": getattr(ev, "category", ""),
                        "date": getattr(ev, "date", None),
                        "url": getattr(ev, "url", ""),
                        "symbol": getattr(ev, "symbol", clean_symbol),
                    }
                if data.get("date") is not None and hasattr(data["date"], "isoformat"):
                    data["date"] = data["date"].isoformat()
                elif data.get("date") is not None and hasattr(data["date"], "strftime"):
                    data["date"] = data["date"].strftime("%Y-%m-%d")
                results.append(data)
            return results
        except Exception as e:
            logger.error("bse_events_fetch_error", symbol=clean_symbol, error=str(e), exc_info=True)
            return []
