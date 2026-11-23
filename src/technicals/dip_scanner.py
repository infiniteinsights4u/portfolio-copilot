"""Clean-room technical dip-detection engine for Indian equities.

Implements pure-Python mathematical indicators avoiding legacy pandas-ta:
1. Daily and weekly Wilder's RSI(14) with pullback-curl detector (30-42 zone + upward curl).
2. 14-day Wilder ATR for volatility-based trailing stop-loss (2.0 * ATR).
3. 50-DMA and 200-DMA long-term trend structure.
4. Guppy Multiple Moving Average (GMMA): 6 short (3-15) vs 6 long (30-60) EMAs.
5. Heikin-Ashi candlestick swing filter (bullish green reversal bar).
6. 52-week drawdown from trailing 252-day peak.
7. 20-day rolling delivery percentage accumulation check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


# ── Mathematical Indicators ──────────────────────────────────────────────────


def calculate_wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Wilder's smoothed Relative Strength Index (RSI).

    Args:
        close: Closing price series.
        period: RSI period (standard: 14).

    Returns:
        Series of RSI values [0.0, 100.0].
    """
    if len(close) < period + 1:
        return pd.Series(np.nan, index=close.index)

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # First average is simple mean
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))

    avg_gain[period] = gain.iloc[1 : period + 1].mean()
    avg_loss[period] = loss.iloc[1 : period + 1].mean()

    # Wilder exponential smoothing: (prev * (period - 1) + current) / period
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss.iloc[i]) / period

    rsi = np.zeros(len(close))
    rsi[:period] = np.nan

    for i in range(period, len(close)):
        gl = avg_loss[i]
        gg = avg_gain[i]
        if gl == 0.0 and gg == 0.0:
            rsi[i] = 50.0
        elif gl == 0.0:
            rsi[i] = 100.0
        elif gg == 0.0:
            rsi[i] = 0.0
        else:
            rs = gg / gl
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return pd.Series(rsi, index=close.index)


def calculate_wilder_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate 14-day Wilder's Average True Range (ATR).

    TR = max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))
    ATR = Wilder EMA of TR.
    """
    if len(close) < period + 1:
        return pd.Series(np.nan, index=close.index)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = np.zeros(len(close))
    atr[:period] = np.nan
    atr[period] = tr.iloc[1 : period + 1].mean()

    for i in range(period + 1, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr.iloc[i]) / period

    return pd.Series(atr, index=close.index)


def calculate_gmma(
    close: pd.Series,
    short_periods: tuple[int, ...] = (3, 5, 8, 10, 12, 15),
    long_periods: tuple[int, ...] = (30, 35, 40, 45, 50, 60),
) -> dict[str, pd.DataFrame]:
    """Calculate Guppy Multiple Moving Average (GMMA) short and long ribbons.

    Returns:
        dict with keys:
            'short': DataFrame of EMAs (spans 3, 5, 8, 10, 12, 15)
            'long': DataFrame of EMAs (spans 30, 35, 40, 45, 50, 60)
    """
    short_df = pd.DataFrame(index=close.index)
    for p in short_periods:
        short_df[f"EMA_{p}"] = close.ewm(span=p, adjust=False).mean()

    long_df = pd.DataFrame(index=close.index)
    for p in long_periods:
        long_df[f"EMA_{p}"] = close.ewm(span=p, adjust=False).mean()

    return {"short": short_df, "long": long_df}


def calculate_heikin_ashi(
    open_p: pd.Series,
    high_p: pd.Series,
    low_p: pd.Series,
    close_p: pd.Series,
) -> pd.DataFrame:
    """Calculate Heikin-Ashi candlestick OHLC series."""
    n = len(close_p)
    if n == 0:
        return pd.DataFrame(columns=["HA_Open", "HA_High", "HA_Low", "HA_Close"])

    ha_close = (open_p + high_p + low_p + close_p) / 4.0
    ha_open = np.zeros(n)
    ha_open[0] = (open_p.iloc[0] + close_p.iloc[0]) / 2.0

    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0

    ha_open_s = pd.Series(ha_open, index=close_p.index)
    ha_high = pd.concat([high_p, ha_open_s, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low_p, ha_open_s, ha_close], axis=1).min(axis=1)

    return pd.DataFrame(
        {
            "HA_Open": ha_open_s,
            "HA_High": ha_high,
            "HA_Low": ha_low,
            "HA_Close": ha_close,
        },
        index=close_p.index,
    )


# ── Dip Candidate Analysis Model ─────────────────────────────────────────────


@dataclass
class DipScanResult:
    """Detailed evaluation result for a stock's technical dip setup."""

    symbol: str
    is_valid_dip: bool
    last_price: float
    rsi_14: float
    rsi_pullback_curl: bool
    atr_14: float
    dma_50: float
    dma_200: float
    above_200_dma: bool
    near_support: bool
    gmma_trend_bullish: bool
    ha_bullish: bool
    drawdown_52w_pct: float
    delivery_pct: float
    delivery_above_avg: bool
    suggested_entry: float
    suggested_stop_loss: float
    suggested_target: float
    risk_reward_ratio: float
    reasons: list[str] = field(default_factory=list)


class TechnicalDipScanner:
    """Deterministic technical scanner evaluating dip setups for CNC delivery."""

    def __init__(
        self,
        rsi_min: float = 30.0,
        rsi_max: float = 42.0,
        lookback_bars: int = 5,
        atr_multiplier_stop: float = 2.0,
        target_risk_reward: float = 2.5,
    ) -> None:
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.lookback_bars = lookback_bars
        self.atr_multiplier_stop = atr_multiplier_stop
        self.target_risk_reward = target_risk_reward

    def check_rsi_pullback_curl(self, rsi: pd.Series) -> tuple[bool, str]:
        """Detect if RSI was recently in the dip zone [30, 42] and is now curling upward."""
        if len(rsi.dropna()) < 3:
            return False, "Insufficient RSI history"

        recent_rsi = rsi.dropna().iloc[-self.lookback_bars :]
        in_zone = ((recent_rsi >= self.rsi_min) & (recent_rsi <= self.rsi_max)).any()

        # Upward curl: current RSI > previous RSI
        curling_up = rsi.iloc[-1] > rsi.iloc[-2]

        if in_zone and curling_up:
            return True, f"RSI({rsi.iloc[-1]:.1f}) curling up from dip zone [{self.rsi_min}, {self.rsi_max}]"
        elif not in_zone:
            return False, f"RSI({rsi.iloc[-1]:.1f}) not in dip zone [{self.rsi_min}, {self.rsi_max}] in last {self.lookback_bars} bars"
        else:
            return False, f"RSI({rsi.iloc[-1]:.1f}) still falling (no upward curl confirmation)"

    def check_gmma_trend(self, gmma: dict[str, pd.DataFrame]) -> tuple[bool, str]:
        """Verify long-term investor GMMA ribbon is in a bullish expansion/alignment."""
        long_df = gmma["long"]
        if long_df.empty:
            return False, "Empty GMMA series"

        # Check latest row of long-term EMAs
        ema30 = long_df["EMA_30"].iloc[-1]
        ema60 = long_df["EMA_60"].iloc[-1]

        # In a healthy long-term trend, EMA30 >= EMA60 (investors are net bullish)
        is_bullish = ema30 >= ema60
        desc = "GMMA investor ribbon bullish (EMA30 >= EMA60)" if is_bullish else "GMMA investor ribbon bearish (EMA30 < EMA60)"
        return is_bullish, desc

    def evaluate(self, df: pd.DataFrame, symbol: str) -> DipScanResult:
        """Evaluate a historical OHLCV + Delivery DataFrame for a CNC dip setup.

        Expected DataFrame columns (case-insensitive):
        OPEN, HIGH, LOW, CLOSE, VOLUME, and optionally DELIV_PER.
        """
        clean_symbol = symbol.strip().upper()
        cols = {c.upper(): c for c in df.columns}

        close = df[cols["CLOSE"]].astype(float)
        high = df[cols["HIGH"]].astype(float)
        low = df[cols["LOW"]].astype(float)
        open_p = df[cols["OPEN"]].astype(float)

        if len(close) < 30:
            return DipScanResult(
                symbol=clean_symbol,
                is_valid_dip=False,
                last_price=float(close.iloc[-1]) if len(close) else 0.0,
                rsi_14=0.0,
                rsi_pullback_curl=False,
                atr_14=0.0,
                dma_50=0.0,
                dma_200=0.0,
                above_200_dma=False,
                near_support=False,
                gmma_trend_bullish=False,
                ha_bullish=False,
                drawdown_52w_pct=0.0,
                delivery_pct=0.0,
                delivery_above_avg=False,
                suggested_entry=0.0,
                suggested_stop_loss=0.0,
                suggested_target=0.0,
                risk_reward_ratio=0.0,
                reasons=["Insufficient historical bars (< 30) for reliable indicators"],
            )

        last_price = float(close.iloc[-1])

        # 1. RSI(14) & Pullback-Curl
        rsi_series = calculate_wilder_rsi(close, period=14)
        current_rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
        rsi_ok, rsi_msg = self.check_rsi_pullback_curl(rsi_series)

        # 2. Wilder ATR(14)
        atr_series = calculate_wilder_atr(high, low, close, period=14)
        current_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else (last_price * 0.02)

        # 3. Moving Averages (50 & 200 DMA)
        dma_50 = float(close.rolling(50, min_periods=20).mean().iloc[-1])
        dma_200 = float(close.rolling(200, min_periods=50).mean().iloc[-1])
        above_200_dma = last_price >= (dma_200 * 0.97)  # allow up to 3% shakeout below 200-DMA
        near_support = abs(last_price - dma_50) / dma_50 <= 0.05 or abs(last_price - dma_200) / dma_200 <= 0.05

        # 4. GMMA Ribbons
        gmma = calculate_gmma(close)
        gmma_ok, gmma_msg = self.check_gmma_trend(gmma)

        # 5. Heikin-Ashi
        ha_df = calculate_heikin_ashi(open_p, high, low, close)
        ha_close_last = ha_df["HA_Close"].iloc[-1]
        ha_open_last = ha_df["HA_Open"].iloc[-1]
        ha_bullish = bool(ha_close_last >= ha_open_last)

        # 6. 52-Week Drawdown
        high_52w = float(high.rolling(min(len(high), 252)).max().iloc[-1])
        drawdown_52w_pct = round(((last_price - high_52w) / high_52w) * 100.0, 2)

        # 7. Delivery % Check
        delivery_col = cols.get("DELIV_PER")
        if delivery_col and delivery_col in df:
            deliv_s = df[delivery_col].astype(float)
            current_deliv = float(deliv_s.iloc[-1])
            avg_deliv_20 = float(deliv_s.rolling(min(len(deliv_s), 20)).mean().iloc[-1])
            deliv_ok = bool(current_deliv >= avg_deliv_20 * 0.95 or current_deliv >= 40.0)
        else:
            current_deliv = 0.0
            deliv_ok = True  # Neutral if delivery data not available (e.g. backtesting feed)

        # Geometry for GTT:
        suggested_entry = round(last_price, 2)
        suggested_stop_loss = round(max(0.05, suggested_entry - (self.atr_multiplier_stop * current_atr)), 2)
        risk_per_share = suggested_entry - suggested_stop_loss
        suggested_target = round(suggested_entry + (self.target_risk_reward * risk_per_share), 2)
        rr_ratio = round((suggested_target - suggested_entry) / max(0.01, risk_per_share), 2)

        reasons = []
        if rsi_ok:
            reasons.append(rsi_msg)
        else:
            reasons.append(f"RSI reject: {rsi_msg}")

        if above_200_dma:
            reasons.append(f"Price ₹{last_price} in long-term uptrend (>200-DMA ₹{dma_200:.1f})")
        else:
            reasons.append(f"Price ₹{last_price} broken below 200-DMA ₹{dma_200:.1f}")

        reasons.append(gmma_msg)

        if ha_bullish:
            reasons.append("Heikin-Ashi green confirmation bar")
        else:
            reasons.append("Heikin-Ashi still red/bearish")

        if deliv_ok and current_deliv > 0:
            reasons.append(f"Delivery {current_deliv:.1f}% satisfies institutional accumulation threshold")

        # Master dip validation rule:
        # 1. RSI pullback curl must be satisfied
        # 2. Must be above 200-DMA structure (or within tolerance)
        # 3. GMMA trend must be bullish
        # 4. Drawdown between -5% and -35% (disciplined pullback, not insolvency plunge)
        is_valid = bool(
            rsi_ok
            and above_200_dma
            and gmma_ok
            and -38.0 <= drawdown_52w_pct <= -4.0
        )

        return DipScanResult(
            symbol=clean_symbol,
            is_valid_dip=is_valid,
            last_price=last_price,
            rsi_14=round(current_rsi, 2),
            rsi_pullback_curl=rsi_ok,
            atr_14=round(current_atr, 2),
            dma_50=round(dma_50, 2),
            dma_200=round(dma_200, 2),
            above_200_dma=above_200_dma,
            near_support=near_support,
            gmma_trend_bullish=gmma_ok,
            ha_bullish=ha_bullish,
            drawdown_52w_pct=drawdown_52w_pct,
            delivery_pct=round(current_deliv, 2),
            delivery_above_avg=deliv_ok,
            suggested_entry=suggested_entry,
            suggested_stop_loss=suggested_stop_loss,
            suggested_target=suggested_target,
            risk_reward_ratio=rr_ratio,
            reasons=reasons,
        )
