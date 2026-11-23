"""Screener.in provider adapter with quarterly SQLite caching, concurrency=1, and >=2.5s pacing."""

from __future__ import annotations

import asyncio
import dataclasses
import time

import httpx
import structlog

from src.data.providers.base import FundamentalDataProvider
from src.data.scrapers.screener import FundamentalData, ScreenerScraper
from src.storage.ledger import SQLiteLedger

logger = structlog.get_logger(__name__)


class ScreenerRateLimitError(Exception):
    """Raised when Screener.in responds with HTTP 429 or repeated 403."""


class ScreenerAdapter(FundamentalDataProvider):
    """Adapter wrapping Screener.in behind FundamentalDataProvider with rate-limits and caching.

    Enforces:
    1. SQLite quarterly cache lookup first (30-90 day TTL).
    2. Concurrency = 1 via asyncio.Lock.
    3. Pacing >= 2.5s between consecutive outbound requests.
    4. Exponential backoff on HTTP 403 / 429.
    """

    def __init__(
        self,
        ledger: SQLiteLedger | None = None,
        scraper: ScreenerScraper | None = None,
        min_pacing_seconds: float = 2.5,
        default_ttl_days: int = 60,
        max_retries: int = 3,
        initial_backoff_seconds: float = 5.0,
    ) -> None:
        self.ledger = ledger or SQLiteLedger()
        self.scraper = scraper
        self._owns_scraper = scraper is None
        self.min_pacing_seconds = min_pacing_seconds
        self.default_ttl_days = default_ttl_days
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds

        self._lock = asyncio.Lock()
        self._last_request_time: float = 0.0

    async def _get_scraper(self) -> ScreenerScraper:
        if self.scraper is None:
            self.scraper = ScreenerScraper()
        return self.scraper

    async def close(self) -> None:
        """Close underlying scraper if owned."""
        if self._owns_scraper and self.scraper is not None:
            await self.scraper.close()
            self.scraper = None

    async def _enforce_pacing(self) -> None:
        """Ensure at least min_pacing_seconds have elapsed since previous request."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.min_pacing_seconds:
            wait_time = self.min_pacing_seconds - elapsed
            logger.info("screener_pacing_wait", wait_seconds=round(wait_time, 2))
            await asyncio.sleep(wait_time)
        self._last_request_time = time.monotonic()

    async def get_fundamentals(self, symbol: str, force_refresh: bool = False) -> FundamentalData:
        """Fetch fundamental data with cache-first lookup and paced web scraping."""
        clean_symbol = symbol.strip().upper()

        # 1. Check SQLite quarterly cache first
        if not force_refresh:
            cached_dict = self.ledger.get_screener_cache(clean_symbol)
            if cached_dict:
                logger.info("screener_cache_hit", symbol=clean_symbol)
                # Filter dictionary keys to match FundamentalData fields
                valid_fields = {f.name for f in dataclasses.fields(FundamentalData)}
                filtered_data = {k: v for k, v in cached_dict.items() if k in valid_fields}
                return FundamentalData(**filtered_data)

        # 2. Cache miss: serialize outbound calls with concurrency = 1
        async with self._lock:
            # Re-check cache inside lock in case another coroutine just populated it
            if not force_refresh:
                cached_dict = self.ledger.get_screener_cache(clean_symbol)
                if cached_dict:
                    valid_fields = {f.name for f in dataclasses.fields(FundamentalData)}
                    filtered_data = {k: v for k, v in cached_dict.items() if k in valid_fields}
                    return FundamentalData(**filtered_data)

            # Enforce >= 2.5s pacing
            await self._enforce_pacing()

            scraper = await self._get_scraper()
            backoff = self.initial_backoff_seconds

            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.info("screener_scraping_fundamentals", symbol=clean_symbol, attempt=attempt)
                    data = await scraper.get_fundamentals(clean_symbol)

                    # Check for rate limit error in data
                    if data.error and ("429" in data.error or "403" in data.error):
                        if attempt < self.max_retries:
                            logger.warning(
                                "screener_rate_limit_backoff",
                                symbol=clean_symbol,
                                attempt=attempt,
                                backoff_seconds=backoff,
                                error=data.error,
                            )
                            await asyncio.sleep(backoff)
                            backoff *= 2.0
                            await self._enforce_pacing()
                            continue
                        raise ScreenerRateLimitError(f"Screener rate limited on {clean_symbol}: {data.error}")

                    # If successful and no fatal error, store in SQLite cache
                    if not data.error:
                        data_dict = dataclasses.asdict(data)
                        self.ledger.set_screener_cache(
                            symbol=clean_symbol,
                            data=data_dict,
                            ttl_days=self.default_ttl_days,
                        )
                        logger.info("screener_data_cached", symbol=clean_symbol, ttl_days=self.default_ttl_days)

                    return data

                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    status_code = getattr(getattr(exc, "response", None), "status_code", None)
                    if status_code in (403, 429) and attempt < self.max_retries:
                        logger.warning(
                            "screener_http_error_backoff",
                            status=status_code,
                            attempt=attempt,
                            backoff_seconds=backoff,
                        )
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        await self._enforce_pacing()
                        continue
                    raise

            return FundamentalData(symbol=clean_symbol, error=f"Max retries exceeded for {clean_symbol}")
