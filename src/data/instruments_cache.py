"""Zerodha Kite /instruments daily master cache.

Caches Zerodha's /instruments CSV once daily after 8:30 AM IST,
indexed by (exchange, tradingsymbol) rather than numeric instrument_token,
to provide exact live tick_size values (e.g. ₹0.01, ₹0.05, ₹0.10 per NSE/CMTR/67133).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
INSTRUMENTS_URL = "https://api.kite.trade/instruments"
INSTRUMENTS_NSE_URL = "https://api.kite.trade/instruments/NSE"

DEFAULT_TICK_SIZE = 0.05


@dataclass(frozen=True)
class InstrumentInfo:
    """Standardized metadata for a trading instrument."""

    instrument_token: int
    exchange_token: str
    tradingsymbol: str
    name: str
    last_price: float
    tick_size: float
    lot_size: int
    instrument_type: str
    segment: str
    exchange: str


def is_cache_fresh(file_path: Path, current_dt: datetime | None = None) -> bool:
    """Check if the cached instruments file is fresh (post-8:30 AM IST for current cycle).

    Rules:
    1. If file does not exist, return False.
    2. Daily update happens at 8:30 AM IST.
    3. If current time is after 8:30 AM IST, cache must have been updated today after 8:30 AM IST.
    4. If current time is before 8:30 AM IST, cache from yesterday post-8:30 AM IST is still valid.
    """
    if not file_path.exists():
        return False

    now_ist = current_dt.astimezone(IST) if current_dt else datetime.now(IST)
    mtime = file_path.stat().st_mtime
    file_dt_ist = datetime.fromtimestamp(mtime, tz=timezone.utc).astimezone(IST)

    cutoff_today = datetime.combine(now_ist.date(), time(8, 30), tzinfo=IST)

    if now_ist >= cutoff_today:
        # Today is past 8:30 AM, file must be from today post-8:30 AM
        return file_dt_ist >= cutoff_today
    else:
        # Before 8:30 AM today, file must be from yesterday post-8:30 AM
        cutoff_yesterday = cutoff_today - timedelta(days=1)
        return file_dt_ist >= cutoff_yesterday


def parse_instruments_csv(csv_content: str | io.StringIO) -> dict[tuple[str, str], InstrumentInfo]:
    """Parse Kite /instruments CSV content into a dictionary keyed by (exchange, tradingsymbol)."""
    if isinstance(csv_content, str):
        stream = io.StringIO(csv_content)
    else:
        stream = csv_content

    reader = csv.DictReader(stream)
    instruments: dict[tuple[str, str], InstrumentInfo] = {}

    for row in reader:
        try:
            exchange = row.get("exchange", "").strip().upper()
            tradingsymbol = row.get("tradingsymbol", "").strip().upper()
            if not exchange or not tradingsymbol:
                continue

            token = int(row.get("instrument_token", 0))
            exchange_token = row.get("exchange_token", "").strip()
            name = row.get("name", "").strip()
            last_price = float(row.get("last_price", 0.0) or 0.0)
            tick_size = float(row.get("tick_size", DEFAULT_TICK_SIZE) or DEFAULT_TICK_SIZE)
            lot_size = int(row.get("lot_size", 1) or 1)
            instrument_type = row.get("instrument_type", "").strip().upper()
            segment = row.get("segment", "").strip().upper()

            info = InstrumentInfo(
                instrument_token=token,
                exchange_token=exchange_token,
                tradingsymbol=tradingsymbol,
                name=name,
                last_price=last_price,
                tick_size=tick_size,
                lot_size=lot_size,
                instrument_type=instrument_type,
                segment=segment,
                exchange=exchange,
            )
            instruments[(exchange, tradingsymbol)] = info
        except (ValueError, KeyError) as e:
            logger.debug("instrument_parse_row_skipped", error=str(e), row=row)
            continue

    return instruments


class InstrumentsCache:
    """Manages downloading, local storage, and fast lookup of the daily instruments master."""

    def __init__(self, cache_file: str | Path = "data/instruments_cache.csv") -> None:
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[tuple[str, str], InstrumentInfo] = {}

    def is_fresh(self, current_dt: datetime | None = None) -> bool:
        """Check if local cache on disk is fresh."""
        return is_cache_fresh(self.cache_file, current_dt)

    async def fetch_and_save(self, url: str = INSTRUMENTS_URL) -> str:
        """Fetch raw CSV from Kite API and save to local disk."""
        logger.info("fetching_instruments_master", url=url)
        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to fetch /instruments: HTTP {resp.status_code}")
            content = resp.text
            self.cache_file.write_text(content, encoding="utf-8")
            return content

    async def ensure_loaded(self, force_refresh: bool = False) -> None:
        """Ensure instruments are loaded in memory, refreshing from remote if stale."""
        if not force_refresh and self._memory_cache and self.is_fresh():
            return

        if not force_refresh and self.cache_file.exists() and self.is_fresh():
            logger.info("loading_instruments_from_disk_cache", path=str(self.cache_file))
            content = self.cache_file.read_text(encoding="utf-8")
            self._memory_cache = parse_instruments_csv(content)
            return

        # Fetch fresh copy
        content = await self.fetch_and_save()
        self._memory_cache = parse_instruments_csv(content)
        logger.info("instruments_master_refreshed", count=len(self._memory_cache))

    def load_from_text(self, csv_text: str) -> None:
        """Directly populate the cache from CSV string (useful in tests or offline mode)."""
        self._memory_cache = parse_instruments_csv(csv_text)
        self.cache_file.write_text(csv_text, encoding="utf-8")

    def get_instrument_info(self, exchange: str, tradingsymbol: str) -> InstrumentInfo | None:
        """Retrieve instrument info by (exchange, tradingsymbol)."""
        key = (exchange.strip().upper(), tradingsymbol.strip().upper())
        return self._memory_cache.get(key)

    def get_tick_size(self, exchange: str, tradingsymbol: str, fallback: float = DEFAULT_TICK_SIZE) -> float:
        """Get exact tick size for an instrument, defaulting to standard 0.05 if not found."""
        info = self.get_instrument_info(exchange, tradingsymbol)
        if info and info.tick_size > 0:
            return info.tick_size
        return fallback

    def get_instrument_token(self, exchange: str, tradingsymbol: str) -> int | None:
        """Get instrument token by (exchange, tradingsymbol)."""
        info = self.get_instrument_info(exchange, tradingsymbol)
        return info.instrument_token if info else None

    def __len__(self) -> int:
        return len(self._memory_cache)
