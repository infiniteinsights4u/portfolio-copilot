"""Unit tests for Phase 2 components.

Verifies:
1. NSE sec_bhavdata_full DATE1 validation, mismatch rejection, and UDiFF ISIN join.
2. Zerodha /instruments daily post-8:30 AM IST cache and (exchange, tradingsymbol) tick-size lookup.
3. ScreenerAdapter quarterly SQLite caching, concurrency=1, >=2.5s pacing, and 429 backoff.
4. Clean-room technical indicators: Wilder RSI(14) pullback-curl, Wilder ATR(14), GMMA ribbons, Heikin-Ashi.
5. TechnicalDipAgent LangGraph node execution and output schema.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.workflows.technical_dip_agent import (
    TechnicalDipState,
    create_technical_dip_graph,
    evaluate_technical_dip_node,
)
from src.data.bhavcopy import (
    BhavcopyDateMismatchError,
    join_bhavcopy_with_udiff,
    normalize_date,
    parse_bhavcopy_csv,
    parse_udiff_bhavcopy_csv,
)
from src.data.instruments_cache import (
    IST,
    DEFAULT_TICK_SIZE,
    InstrumentsCache,
    is_cache_fresh,
    parse_instruments_csv,
)
from src.data.providers.screener_adapter import (
    ScreenerAdapter,
)
from src.data.scrapers.screener import FundamentalData, ScreenerScraper
from src.storage.ledger import SQLiteLedger
from src.technicals.dip_scanner import (
    TechnicalDipScanner,
    calculate_gmma,
    calculate_heikin_ashi,
    calculate_wilder_atr,
    calculate_wilder_rsi,
)


# =====================================================================
# 1. Bhavcopy and UDiFF Tests
# =====================================================================


class TestBhavcopyLoader:
    """Verify Bhavcopy CSV parsing, DATE1 rejection, and UDiFF ISIN join."""

    SAMPLE_BHAVDATA = """SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER
    INFY, EQ, 21-Nov-2024, 1850.00, 1860.00, 1875.00, 1845.00, 1870.00, 1868.50, 1862.10, 1500000, 27931.5, 45000, 950000, 63.33
    TCS, EQ, 21-Nov-2024, 4100.00, 4120.00, 4150.00, 4090.00, 4135.00, 4140.00, 4125.00, 800000, 33000.0, 30000, 520000, 65.00
    """

    SAMPLE_UDIFF = """TckrSymb, SctySrs, ISIN, OpnPric, HghPric, LwPric, ClsPric
    INFY, EQ, INE009A01021, 1860.00, 1875.00, 1845.00, 1868.50
    TCS, EQ, INE467B01029, 4120.00, 4150.00, 4090.00, 4140.00
    """

    def test_date_normalization(self):
        d1 = normalize_date("2024-11-21")
        d2 = normalize_date("21-11-2024")
        d3 = normalize_date("21-Nov-2024")
        assert d1 == date(2024, 11, 21)
        assert d1 == d2 == d3

    def test_bhavcopy_date1_match_success(self):
        df = parse_bhavcopy_csv(self.SAMPLE_BHAVDATA, requested_date="2024-11-21")
        assert len(df) == 2
        assert "SYMBOL" in df.columns
        assert "CLOSE_PRICE" in df.columns
        assert df.loc[df["SYMBOL"] == "INFY", "CLOSE_PRICE"].iloc[0] == 1868.50
        assert df.loc[df["SYMBOL"] == "INFY", "DELIV_PER"].iloc[0] == 63.33

    def test_bhavcopy_date1_mismatch_raises_error(self):
        # Requesting 2024-11-22 when file internal DATE1 is 21-Nov-2024
        with pytest.raises(BhavcopyDateMismatchError, match="Bhavcopy DATE1 mismatch"):
            parse_bhavcopy_csv(self.SAMPLE_BHAVDATA, requested_date="2024-11-22")

    def test_udiff_join_enriches_isin(self):
        bhav_df = parse_bhavcopy_csv(self.SAMPLE_BHAVDATA, requested_date="2024-11-21")
        udiff_df = parse_udiff_bhavcopy_csv(self.SAMPLE_UDIFF)
        merged = join_bhavcopy_with_udiff(bhav_df, udiff_df)

        assert "ISIN" in merged.columns
        infy_isin = merged.loc[merged["SYMBOL"] == "INFY", "ISIN"].iloc[0]
        assert infy_isin == "INE009A01021"
        tcs_isin = merged.loc[merged["SYMBOL"] == "TCS", "ISIN"].iloc[0]
        assert tcs_isin == "INE467B01029"


# =====================================================================
# 2. Instruments Cache Tests
# =====================================================================


class TestInstrumentsCache:
    """Verify daily /instruments master caching and tick-size indexing."""

    SAMPLE_INSTRUMENTS_CSV = """instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,instrument_type,segment,exchange
408065,1594,INFY,INFOSYS,1870.0,,0.0,0.05,1,EQ,NSE,NSE
2953217,11536,TCS,TATA CONSULTANCY SERVICES,4140.0,,0.0,0.05,1,EQ,NSE,NSE
779521,3045,SBIN,STATE BANK OF INDIA,810.0,,0.0,0.10,1,EQ,NSE,NSE
12345,9999,PENNY,PENNY STOCK,24.5,,0.0,0.01,1,EQ,NSE,NSE
"""

    def test_parse_instruments_by_exchange_and_symbol(self):
        mapping = parse_instruments_csv(self.SAMPLE_INSTRUMENTS_CSV)
        assert ("NSE", "INFY") in mapping
        assert ("NSE", "TCS") in mapping
        assert ("NSE", "SBIN") in mapping
        assert ("NSE", "PENNY") in mapping

        assert mapping[("NSE", "INFY")].instrument_token == 408065
        assert mapping[("NSE", "INFY")].tick_size == 0.05
        assert mapping[("NSE", "SBIN")].tick_size == 0.10
        assert mapping[("NSE", "PENNY")].tick_size == 0.01

    def test_cache_freshness_post_830(self, tmp_path):
        cache_file = tmp_path / "instruments.csv"
        cache_file.write_text(self.SAMPLE_INSTRUMENTS_CSV)

        # Case A: Current time is 10:00 AM IST today, file modified at 9:00 AM IST -> Fresh
        now_10am = datetime(2026, 11, 23, 10, 0, tzinfo=IST)
        with patch.object(Path, "stat") as mock_stat:
            # File mtime: 9:00 AM IST today
            mtime_9am = datetime(2026, 11, 23, 9, 0, tzinfo=IST).timestamp()
            mock_stat.return_value.st_mtime = mtime_9am
            assert is_cache_fresh(cache_file, current_dt=now_10am) is True

        # Case B: Current time is 10:00 AM IST today, but file is from 7:00 AM IST today (pre-8:30 AM) -> Stale
        with patch.object(Path, "stat") as mock_stat:
            mtime_7am = datetime(2026, 11, 23, 7, 0, tzinfo=IST).timestamp()
            mock_stat.return_value.st_mtime = mtime_7am
            assert is_cache_fresh(cache_file, current_dt=now_10am) is False

    def test_cache_lookup_helper(self, tmp_path):
        cache = InstrumentsCache(cache_file=tmp_path / "test_instruments.csv")
        cache.load_from_text(self.SAMPLE_INSTRUMENTS_CSV)

        assert cache.get_tick_size("NSE", "INFY") == 0.05
        assert cache.get_tick_size("NSE", "SBIN") == 0.10
        assert cache.get_tick_size("NSE", "PENNY") == 0.01
        # Unknown stock defaults to fallback 0.05
        assert cache.get_tick_size("NSE", "UNKNOWN") == DEFAULT_TICK_SIZE


# =====================================================================
# 3. Screener Adapter & Provider Tests
# =====================================================================


class TestScreenerAdapter:
    """Verify quarterly SQLite cache, concurrency=1, and >=2.5s pacing."""

    @pytest.fixture
    def ledger(self, tmp_path):
        db_path = tmp_path / "screener_test.db"
        return SQLiteLedger(db_path=db_path)

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_http_call(self, ledger):
        # Populate ledger cache
        sample_data = {
            "symbol": "RELIANCE",
            "name": "Reliance Industries Ltd",
            "pe_ratio": 24.5,
            "roe": 14.2,
            "debt_to_equity": 0.42,
        }
        ledger.set_screener_cache("RELIANCE", sample_data, ttl_days=60)

        mock_scraper = AsyncMock(spec=ScreenerScraper)
        adapter = ScreenerAdapter(ledger=ledger, scraper=mock_scraper)

        result = await adapter.get_fundamentals("RELIANCE")
        assert result.symbol == "RELIANCE"
        assert result.pe_ratio == 24.5
        assert result.roe == 14.2
        # Mock scraper should NOT have been called
        mock_scraper.get_fundamentals.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_scrapes_and_caches_in_ledger(self, ledger):
        mock_scraper = AsyncMock(spec=ScreenerScraper)
        mock_scraper.get_fundamentals.return_value = FundamentalData(
            symbol="TCS",
            name="Tata Consultancy Services Ltd",
            pe_ratio=28.0,
            roe=32.0,
            market_cap=1500000,
        )

        adapter = ScreenerAdapter(
            ledger=ledger,
            scraper=mock_scraper,
            min_pacing_seconds=0.01,  # fast for tests
        )

        assert ledger.get_screener_cache("TCS") is None

        result = await adapter.get_fundamentals("TCS")
        assert result.symbol == "TCS"
        assert result.pe_ratio == 28.0
        mock_scraper.get_fundamentals.assert_called_once_with("TCS")

        # Verify it is now in SQLite ledger
        cached = ledger.get_screener_cache("TCS")
        assert cached is not None
        assert cached["pe_ratio"] == 28.0

    @pytest.mark.asyncio
    async def test_pacing_and_concurrency_lock(self, ledger):
        mock_scraper = AsyncMock(spec=ScreenerScraper)
        mock_scraper.get_fundamentals.side_effect = lambda s: FundamentalData(symbol=s, pe_ratio=20.0)

        adapter = ScreenerAdapter(
            ledger=ledger,
            scraper=mock_scraper,
            min_pacing_seconds=0.1,  # 100ms pacing for testing
        )

        t0 = asyncio.get_event_loop().time()
        # Fire 2 concurrent requests
        res1, res2 = await asyncio.gather(
            adapter.get_fundamentals("SYM1"),
            adapter.get_fundamentals("SYM2"),
        )
        elapsed = asyncio.get_event_loop().time() - t0

        assert res1.symbol == "SYM1"
        assert res2.symbol == "SYM2"
        # At least 1 pacing interval must have been enforced
        assert elapsed >= 0.09

    @pytest.mark.asyncio
    async def test_backoff_on_429(self, ledger):
        mock_scraper = AsyncMock(spec=ScreenerScraper)
        # First call fails with 429 error message, second succeeds
        mock_scraper.get_fundamentals.side_effect = [
            FundamentalData(symbol="WIPRO", error="HTTP 429 Too Many Requests"),
            FundamentalData(symbol="WIPRO", pe_ratio=18.5),
        ]

        adapter = ScreenerAdapter(
            ledger=ledger,
            scraper=mock_scraper,
            min_pacing_seconds=0.01,
            initial_backoff_seconds=0.05,
            max_retries=2,
        )

        result = await adapter.get_fundamentals("WIPRO")
        assert result.pe_ratio == 18.5
        assert mock_scraper.get_fundamentals.call_count == 2


# =====================================================================
# 4. Clean-Room Technical Indicator Tests
# =====================================================================


class TestCleanRoomIndicators:
    """Verify RSI pullback-curl, Wilder ATR, GMMA ribbons, and Heikin-Ashi formulas."""

    def test_wilder_rsi_calculation(self):
        # Series with known dip and recovery: 100 -> 80 (oversold) -> 85 (recovery curl)
        prices = [
            100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 86.0, 84.0, 82.0,
            80.0, 78.0, 76.0, 74.0, 72.0, 71.0, 73.0, 76.0
        ]
        s = pd.Series(prices)
        rsi = calculate_wilder_rsi(s, period=14)

        assert len(rsi) == len(prices)
        assert np.isnan(rsi.iloc[13])
        assert not np.isnan(rsi.iloc[14])
        # RSI at index 15 was low, index 16 & 17 curled up
        assert rsi.iloc[-1] > rsi.iloc[-2]
        assert 0.0 <= rsi.iloc[-1] <= 100.0

    def test_rsi_pullback_curl_detection(self):
        scanner = TechnicalDipScanner(rsi_min=30.0, rsi_max=42.0)
        # Create RSI series where last 3 values are: 35.0, 36.5, 38.0 (inside zone, curling up)
        rsi_series = pd.Series([55.0, 50.0, 45.0, 38.0, 35.0, 36.5, 38.0])
        is_curl, msg = scanner.check_rsi_pullback_curl(rsi_series)
        assert is_curl is True
        assert "curling up" in msg

        # Still falling: 38.0, 35.0, 33.0 (inside zone, but NOT curling up)
        rsi_falling = pd.Series([55.0, 50.0, 45.0, 38.0, 35.0, 33.0])
        is_curl_f, msg_f = scanner.check_rsi_pullback_curl(rsi_falling)
        assert is_curl_f is False
        assert "still falling" in msg_f

    def test_wilder_atr_calculation(self):
        # 20 bars of synthetic data
        n = 20
        high = pd.Series([105.0 + i for i in range(n)])
        low = pd.Series([95.0 + i for i in range(n)])
        close = pd.Series([100.0 + i for i in range(n)])

        atr = calculate_wilder_atr(high, low, close, period=14)
        assert len(atr) == n
        assert np.isnan(atr.iloc[13])
        assert not np.isnan(atr.iloc[14])
        # In this steady bar setup, high - low is constantly 10.0
        assert pytest.approx(atr.iloc[-1], abs=0.5) == 10.0

    def test_gmma_ribbons(self):
        prices = pd.Series([100.0 * (1.005 ** i) for i in range(70)])  # upward trend
        gmma = calculate_gmma(prices)
        assert "short" in gmma
        assert "long" in gmma
        assert "EMA_3" in gmma["short"].columns
        assert "EMA_15" in gmma["short"].columns
        assert "EMA_30" in gmma["long"].columns
        assert "EMA_60" in gmma["long"].columns

        # In a steady uptrend, EMA_30 > EMA_60
        assert gmma["long"]["EMA_30"].iloc[-1] > gmma["long"]["EMA_60"].iloc[-1]

    def test_heikin_ashi_conversion(self):
        open_p = pd.Series([100.0, 102.0, 104.0])
        high_p = pd.Series([103.0, 106.0, 108.0])
        low_p = pd.Series([99.0, 101.0, 103.0])
        close_p = pd.Series([102.0, 105.0, 107.0])

        ha = calculate_heikin_ashi(open_p, high_p, low_p, close_p)
        assert "HA_Open" in ha.columns
        assert "HA_Close" in ha.columns
        # HA Close is average of (O + H + L + C)
        expected_close_0 = (100.0 + 103.0 + 99.0 + 102.0) / 4.0
        assert ha["HA_Close"].iloc[0] == expected_close_0
        # Second bar HA Open is average of prior bar HA Open and HA Close
        expected_open_1 = (ha["HA_Open"].iloc[0] + ha["HA_Close"].iloc[0]) / 2.0
        assert ha["HA_Open"].iloc[1] == expected_open_1

    def test_dip_scanner_full_evaluation(self):
        # Create a 60-bar synthetic dataframe simulating a quality dip recovery
        np.random.seed(42)
        n = 60
        # Base steady uptrend to establish 200/50 DMA and GMMA
        base = np.linspace(100.0, 140.0, n)
        # Introduce a controlled pullback into bars 50-57, then curling up on bars 58-59
        base[50:57] = [135, 132, 128, 124, 121, 120, 120.5]
        base[57:] = [121.5, 123.0, 125.0]

        df = pd.DataFrame({
            "OPEN": base - 0.5,
            "HIGH": base + 2.0,
            "LOW": base - 1.5,
            "CLOSE": base,
            "VOLUME": [500000] * n,
            "DELIV_PER": [55.0] * n,
        })

        scanner = TechnicalDipScanner()
        res = scanner.evaluate(df, symbol="TATAMOTORS")

        assert res.symbol == "TATAMOTORS"
        assert res.last_price == 125.0
        assert res.suggested_stop_loss < res.suggested_entry
        assert res.suggested_target > res.suggested_entry
        assert res.risk_reward_ratio >= 2.0


# =====================================================================
# 5. TechnicalDipAgent LangGraph Workflow Tests
# =====================================================================


class TestTechnicalDipAgentWorkflow:
    """Verify LangGraph node and workflow execution."""

    @pytest.mark.asyncio
    async def test_evaluate_technical_dip_node(self):
        n = 50
        base = np.linspace(200.0, 250.0, n)
        df = pd.DataFrame({
            "OPEN": base,
            "HIGH": base + 1.0,
            "LOW": base - 1.0,
            "CLOSE": base,
            "VOLUME": [100000] * n,
        })

        state: TechnicalDipState = {
            "query": "Scan INFY for dip setup",
            "symbol": "INFY",
            "historical_bars": df,
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

        output = await evaluate_technical_dip_node(state, config={})
        assert output["dip_scan_result"] is not None
        assert output["proposed_entry"] == 250.0
        assert output["proposed_stop_loss"] > 0.0
        assert "evaluate_technical_dip" in output["steps_completed"]

    @pytest.mark.asyncio
    async def test_full_langgraph_pipeline(self):
        graph = create_technical_dip_graph()

        n = 45
        base = np.linspace(500.0, 550.0, n)
        df = pd.DataFrame({
            "OPEN": base,
            "HIGH": base + 2.0,
            "LOW": base - 2.0,
            "CLOSE": base,
            "VOLUME": [200000] * n,
        })

        initial_state: TechnicalDipState = {
            "query": "Is RELIANCE currently dipping?",
            "symbol": "",  # extracted by node
            "historical_bars": df,
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

        final_state = await graph.ainvoke(initial_state)
        assert final_state["symbol"] == "RELIANCE"
        assert final_state["dip_scan_result"] is not None
        assert "extract_symbol" in final_state["steps_completed"]
        assert "evaluate_technical_dip" in final_state["steps_completed"]
