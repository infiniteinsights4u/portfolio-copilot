"""NSE Bhavcopy and UDiFF ingestion module.

Handles:
1. Downloading and parsing official NSE sec_bhavdata_full (OHLCV + delivery %).
2. Strict internal DATE1 validation against requested trading date (rejecting stale holiday files).
3. Joining with UDiFF Bhavcopy (TckrSymb, SctySrs, ISIN) to map authoritative ISINs.
4. Parallel split/bonus-adjusted yfinance (.NS) loader strictly for offline historical backtesting.
"""

from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path

import httpx
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

SEC_BHAVDATA_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
UDIFF_BHAVDATA_URL = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv"


class BhavcopyDateMismatchError(ValueError):
    """Raised when the internal DATE1 in NSE Bhavcopy does not match the requested date."""


def normalize_date(target_date: date | str) -> date:
    """Normalize input date string or date object into a date instance."""
    if isinstance(target_date, date):
        return target_date
    clean_str = target_date.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d%m%Y", "%d-%b-%Y", "%d-%b-%y"):
        try:
            return datetime.strptime(clean_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date string: {target_date}")


def parse_bhavcopy_csv(csv_content: str | io.StringIO, requested_date: date | str) -> pd.DataFrame:
    """Parse sec_bhavdata_full CSV content and validate DATE1.

    Args:
        csv_content: Raw CSV string or text stream.
        requested_date: Expected trading date.

    Returns:
        pd.DataFrame with standardized uppercase column names and validated DATE1.

    Raises:
        BhavcopyDateMismatchError: If internal DATE1 does not match requested date.
    """
    req_d = normalize_date(requested_date)
    if isinstance(csv_content, str):
        stream = io.StringIO(csv_content)
    else:
        stream = csv_content

    df = pd.read_csv(stream, skipinitialspace=True)
    df.columns = [c.strip().upper() for c in df.columns]

    if "DATE1" not in df.columns:
        raise ValueError("Missing DATE1 column in sec_bhavdata_full CSV")

    if df.empty:
        raise ValueError("sec_bhavdata_full CSV is empty")

    # Extract internal DATE1 from the first row
    first_date_val = str(df["DATE1"].iloc[0]).strip()
    try:
        internal_date = normalize_date(first_date_val)
    except ValueError as exc:
        raise BhavcopyDateMismatchError(f"Cannot parse internal DATE1: {first_date_val}") from exc

    if internal_date != req_d:
        msg = (
            f"Bhavcopy DATE1 mismatch: internal date '{internal_date}' "
            f"does not match requested trading date '{req_d}'"
        )
        logger.error("bhavcopy_date_mismatch", internal=str(internal_date), requested=str(req_d))
        raise BhavcopyDateMismatchError(msg)

    # Clean numeric columns and symbol strings
    if "SYMBOL" in df.columns:
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    if "SERIES" in df.columns:
        df["SERIES"] = df["SERIES"].astype(str).str.strip().str.upper()

    numeric_cols = [
        "PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "LAST_PRICE",
        "CLOSE_PRICE", "AVG_PRICE", "TTL_TRD_QNTY", "TURNOVER_LACS", "NO_OF_TRADES",
        "DELIV_QTY", "DELIV_PER"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce")

    return df


def parse_udiff_bhavcopy_csv(csv_content: str | io.StringIO) -> pd.DataFrame:
    """Parse NSE UDiFF Bhavcopy CSV content (extracting ISIN mapping).

    Expected UDiFF format columns: TckrSymb, SctySrs, ISIN, etc.
    """
    if isinstance(csv_content, str):
        stream = io.StringIO(csv_content)
    else:
        stream = csv_content

    df = pd.read_csv(stream, skipinitialspace=True)
    df.columns = [c.strip().upper() for c in df.columns]

    # Map standardized columns
    col_mapping = {
        "TCKRSYMB": "SYMBOL",
        "SCTYSRS": "SERIES",
        "ISIN": "ISIN",
    }
    rename_dict = {k: v for k, v in col_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    if "SYMBOL" in df.columns:
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    if "SERIES" in df.columns:
        df["SERIES"] = df["SERIES"].astype(str).str.strip().str.upper()
    if "ISIN" in df.columns:
        df["ISIN"] = df["ISIN"].astype(str).str.strip().str.upper()

    return df


def join_bhavcopy_with_udiff(bhavcopy_df: pd.DataFrame, udiff_df: pd.DataFrame) -> pd.DataFrame:
    """Join sec_bhavdata_full with UDiFF Bhavcopy on (SYMBOL, SERIES) to map ISIN."""
    udiff_subset = udiff_df[["SYMBOL", "SERIES", "ISIN"]].drop_duplicates(subset=["SYMBOL", "SERIES"])
    merged = pd.merge(bhavcopy_df, udiff_subset, on=["SYMBOL", "SERIES"], how="left")
    return merged


class BhavcopyLoader:
    """Downloader and loader for daily NSE Bhavcopy files with local caching."""

    def __init__(self, cache_dir: str | Path = "data/bhavcopy") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, target_date: date) -> Path:
        """Get local cache file path for a given date."""
        return self.cache_dir / f"nse_bhav_{target_date.strftime('%Y%m%d')}.parquet"

    async def fetch_bhavcopy_content(self, target_date: date) -> str:
        """Fetch raw sec_bhavdata_full CSV from NSE archives."""
        ddmmyyyy = target_date.strftime("%d%m%Y")
        url = SEC_BHAVDATA_URL.format(ddmmyyyy=ddmmyyyy)

        async with httpx.AsyncClient(headers=NSE_HEADERS, timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Failed to download Bhavcopy for {target_date}: HTTP {resp.status_code}"
                )
            return resp.text

    async def fetch_udiff_content(self, target_date: date) -> str | None:
        """Fetch raw UDiFF Bhavcopy CSV from NSE archives if available."""
        yyyymmdd = target_date.strftime("%Y%m%d")
        url = UDIFF_BHAVDATA_URL.format(yyyymmdd=yyyymmdd)

        async with httpx.AsyncClient(headers=NSE_HEADERS, timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.text
            logger.info("udiff_not_found_or_not_available", date=str(target_date), status=resp.status_code)
            return None

    async def get_eod_bhavcopy(
        self,
        target_date: date | str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Load EOD Bhavcopy for the specified date, utilizing local cache if available."""
        req_d = normalize_date(target_date)
        cache_path = self.get_cache_path(req_d)

        if not force_refresh and cache_path.exists():
            logger.info("loading_cached_bhavcopy", date=str(req_d), path=str(cache_path))
            return pd.read_parquet(cache_path)

        # Download raw sec_bhavdata_full
        csv_text = await self.fetch_bhavcopy_content(req_d)
        df_bhav = parse_bhavcopy_csv(csv_text, requested_date=req_d)

        # Download and join UDiFF if available
        try:
            udiff_text = await self.fetch_udiff_content(req_d)
            if udiff_text:
                df_udiff = parse_udiff_bhavcopy_csv(udiff_text)
                df_bhav = join_bhavcopy_with_udiff(df_bhav, df_udiff)
        except Exception as e:
            logger.warning("udiff_join_skipped", error=str(e))

        # Cache locally in parquet
        try:
            df_bhav.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.warning("failed_to_write_bhavcopy_cache", error=str(e))

        return df_bhav


# =====================================================================
# Parallel Offline Backtesting Historical Loader (split-adjusted yfinance)
# =====================================================================

def fetch_backtest_ohlcv(
    symbol: str,
    start_date: str | date,
    end_date: str | date | None = None,
) -> pd.DataFrame:
    """Fetch split/bonus-adjusted historical OHLCV data strictly for offline backtesting.

    WARNING: NEVER use unadjusted raw exchange feeds for multi-month backtesting
    because corporate splits/bonuses masquerade as 50%+ technical flash dips.
    """
    clean_sym = symbol.strip().upper()
    ticker = f"{clean_sym}.NS" if not clean_sym.endswith(".NS") and not clean_sym.endswith(".BO") else clean_sym

    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is required for offline backtesting. Install via: pip install yfinance"
        ) from e

    logger.info("fetching_offline_backtest_data", ticker=ticker, start=str(start_date), end=str(end_date))
    data = yf.download(ticker, start=str(start_date), end=str(end_date) if end_date else None, auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError(f"No historical data returned for ticker {ticker}")

    # Reset index and standardize columns
    df = data.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].upper() for c in df.columns]
    else:
        df.columns = [str(c).upper() for c in df.columns]

    rename_map = {"DATE": "DATE", "OPEN": "OPEN", "HIGH": "HIGH", "LOW": "LOW", "CLOSE": "CLOSE", "VOLUME": "VOLUME"}
    df = df.rename(columns=rename_map)
    df["SYMBOL"] = clean_sym
    return df
