"""Deterministic EOD backtesting engine for Zerodha CNC dip-investing.

Simulates the quality-dip entry, 2.0x ATR stop-loss, 2.5R target, and Chandelier
trailing-stop ratcheting on split-adjusted .NS historical data with realistic
Indian equity delivery transaction costs (STT, GST, DP, Stamp Duty, Slippage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import math
from typing import Any, Callable

import pandas as pd
import structlog

from src.data.bhavcopy import fetch_backtest_ohlcv
from src.risk.risk_guard import get_nse_dynamic_tick_size, round_to_tick
from src.technicals.dip_scanner import (
    TechnicalDipScanner,
    calculate_gmma,
    calculate_heikin_ashi,
    calculate_wilder_atr,
    calculate_wilder_rsi,
)

logger = structlog.get_logger(__name__)


# ── 1. Indian CNC Delivery Transaction Cost Model ────────────────────────────


@dataclass
class CostBreakdown:
    """Detailed itemization of statutory and brokerage transaction costs."""

    turnover: float
    brokerage: float
    stt: float
    exchange_charges: float
    sebi_charges: float
    stamp_duty: float
    gst: float
    dp_charges: float
    slippage: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return {
            "turnover": round(self.turnover, 2),
            "brokerage": round(self.brokerage, 2),
            "stt": round(self.stt, 2),
            "exchange_charges": round(self.exchange_charges, 2),
            "sebi_charges": round(self.sebi_charges, 2),
            "stamp_duty": round(self.stamp_duty, 2),
            "gst": round(self.gst, 2),
            "dp_charges": round(self.dp_charges, 2),
            "slippage": round(self.slippage, 2),
            "total": round(self.total, 2),
        }


class TransactionCostModel:
    """Accurate cost calculator for Indian equity delivery (CNC) on Zerodha.

    Statutory rates (as per Zerodha Brokerage & Regulatory schedule):
    - Brokerage: Zero (₹0 for equity delivery).
    - STT (Securities Transaction Tax): 0.1% on buy & 0.1% on sell turnover.
    - NSE Exchange Turnover Charges: 0.00297% (₹297 per crore).
    - SEBI Turnover Charges: 0.0001% (₹10 per crore).
    - Stamp Duty: 0.015% (₹1500 per crore) on buy turnover only.
    - GST: 18% on (Brokerage + Exchange Charges + SEBI Charges).
    - DP Charges: ₹13.00 + 18% GST (₹15.34) per scrip per day on sell leg only.
    - Slippage: Modelled as a configurable percentage of turnover (default: 0.05%).
    """

    STT_RATE: float = 0.001  # 0.1%
    NSE_EXCHANGE_RATE: float = 0.0000297  # 0.00297%
    SEBI_RATE: float = 0.000001  # 0.0001%
    STAMP_DUTY_BUY_RATE: float = 0.00015  # 0.015% on buy
    GST_RATE: float = 0.18  # 18%
    DP_CHARGE_BASE: float = 13.0  # ₹13.00 per scrip per debit
    DP_CHARGE_TOTAL: float = 15.34  # ₹13.00 + 18% GST

    def __init__(
        self,
        slippage_pct: float = 0.0005,
        enabled: bool = True,
    ) -> None:
        self.slippage_pct = slippage_pct
        self.enabled = enabled

    def calculate_buy_costs(self, price: float, quantity: int) -> CostBreakdown:
        """Compute all transaction charges and slippage for a CNC Buy order."""
        turnover = price * quantity
        if not self.enabled or turnover <= 0.0:
            return CostBreakdown(
                turnover=turnover,
                brokerage=0.0,
                stt=0.0,
                exchange_charges=0.0,
                sebi_charges=0.0,
                stamp_duty=0.0,
                gst=0.0,
                dp_charges=0.0,
                slippage=0.0,
                total=0.0,
            )

        brokerage = 0.0  # Zerodha ₹0 CNC delivery
        stt = turnover * self.STT_RATE
        exchange_charges = turnover * self.NSE_EXCHANGE_RATE
        sebi_charges = turnover * self.SEBI_RATE
        stamp_duty = turnover * self.STAMP_DUTY_BUY_RATE
        gstable_amount = brokerage + exchange_charges + sebi_charges
        gst = gstable_amount * self.GST_RATE
        dp_charges = 0.0  # DP charges only apply on delivery sell
        slippage = turnover * self.slippage_pct

        total = brokerage + stt + exchange_charges + sebi_charges + stamp_duty + gst + dp_charges + slippage
        return CostBreakdown(
            turnover=turnover,
            brokerage=brokerage,
            stt=stt,
            exchange_charges=exchange_charges,
            sebi_charges=sebi_charges,
            stamp_duty=stamp_duty,
            gst=gst,
            dp_charges=dp_charges,
            slippage=slippage,
            total=total,
        )

    def calculate_sell_costs(self, price: float, quantity: int) -> CostBreakdown:
        """Compute all transaction charges and slippage for a CNC Sell order."""
        turnover = price * quantity
        if not self.enabled or turnover <= 0.0:
            return CostBreakdown(
                turnover=turnover,
                brokerage=0.0,
                stt=0.0,
                exchange_charges=0.0,
                sebi_charges=0.0,
                stamp_duty=0.0,
                gst=0.0,
                dp_charges=0.0,
                slippage=0.0,
                total=0.0,
            )

        brokerage = 0.0  # Zerodha ₹0 CNC delivery
        stt = turnover * self.STT_RATE
        exchange_charges = turnover * self.NSE_EXCHANGE_RATE
        sebi_charges = turnover * self.SEBI_RATE
        stamp_duty = 0.0  # No stamp duty on sell leg
        gstable_amount = brokerage + exchange_charges + sebi_charges
        gst = gstable_amount * self.GST_RATE
        dp_charges = self.DP_CHARGE_TOTAL  # ₹15.34 (inclusive of 18% GST)
        slippage = turnover * self.slippage_pct

        total = brokerage + stt + exchange_charges + sebi_charges + stamp_duty + gst + dp_charges + slippage
        return CostBreakdown(
            turnover=turnover,
            brokerage=brokerage,
            stt=stt,
            exchange_charges=exchange_charges,
            sebi_charges=sebi_charges,
            stamp_duty=stamp_duty,
            gst=gst,
            dp_charges=dp_charges,
            slippage=slippage,
            total=total,
        )


# ── 2. Trade & Position Data Structures ──────────────────────────────────────


@dataclass
class TradeRecord:
    """Immutable audit record for a completed backtest trade."""

    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    quantity: int
    exit_reason: str
    gross_pnl: float
    net_pnl: float
    total_costs: float
    itemized_costs: dict[str, float]
    holding_days: int
    return_pct: float
    r_multiple: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date,
            "entry_price": round(self.entry_price, 2),
            "exit_date": self.exit_date,
            "exit_price": round(self.exit_price, 2),
            "quantity": self.quantity,
            "exit_reason": self.exit_reason,
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "total_costs": round(self.total_costs, 2),
            "holding_days": self.holding_days,
            "return_pct": round(self.return_pct, 2),
            "r_multiple": round(self.r_multiple, 2),
        }


@dataclass
class ActivePosition:
    """Tracks state of an open positional holding during simulation."""

    symbol: str
    entry_date: str
    entry_index: int
    entry_price: float
    quantity: int
    initial_stop_loss: float
    current_stop_loss: float
    target_price: float
    atr_at_entry: float
    highest_price: float
    buy_costs: CostBreakdown


# ── 3. Backtest Configuration & Results ──────────────────────────────────────


@dataclass
class BacktestConfig:
    """Configurable parameters for EOD backtest simulation."""

    initial_capital: float = 1_000_000.0  # ₹10,00,000 standard portfolio
    max_capital_per_trade_pct: float = 10.0  # 10% capital cap
    max_risk_per_trade_pct: float = 1.0  # 1.0% risk cap
    atr_stop_multiplier: float = 2.0  # 2.0x ATR stop-loss
    target_risk_reward: float = 2.5  # 2.5R target
    enable_chandelier_trail: bool = True  # Enable Chandelier trailing stop
    chandelier_atr_multiplier: float = 2.0  # Highest High - 2.0 * ATR
    slippage_pct: float = 0.0005  # 0.05% per leg
    enable_transaction_costs: bool = True  # Realistic Indian taxes & charges
    max_open_positions: int = 10  # Maximum concurrent positions


@dataclass
class BacktestResult:
    """Comprehensive performance statistics and equity timeseries."""

    initial_capital: float
    ending_capital: float
    total_return_pct: float
    cagr_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_trade_return_pct: float
    avg_holding_days: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    total_costs_paid: float
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> dict[str, Any]:
        """Return high-level backtest summary dictionary."""
        return {
            "initial_capital": round(self.initial_capital, 2),
            "ending_capital": round(self.ending_capital, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "cagr_pct": round(self.cagr_pct, 2),
            "total_trades": self.total_trades,
            "win_rate_pct": round(self.win_rate_pct, 2),
            "profit_factor": round(self.profit_factor, 2),
            "avg_trade_return_pct": round(self.avg_trade_return_pct, 2),
            "avg_holding_days": round(self.avg_holding_days, 1),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "total_costs_paid": round(self.total_costs_paid, 2),
        }


# ── 4. Deterministic Backtest Engine ─────────────────────────────────────────


class EODBacktestEngine:
    """Split-adjusted EOD backtesting engine for Indian equities.

    Simulates the exact rules enforced by TechnicalDipScanner and RiskGuard:
    1. Dip Entry on bar close:
       - Wilder RSI(14) in [30, 42] with upward curl.
       - Price >= 200-DMA * 0.97 (long-term uptrend filter).
       - GMMA Investor ribbon (EMA30 >= EMA60) bullish.
       - 52-week drawdown between -4% and -38%.
       - Heikin-Ashi green bar confirmation.
    2. Sizing:
       - Dual-cap position sizing: min(capital_cap, risk_cap) on worst-case loss.
       - Clamped to available cash in portfolio.
       - Dynamic NSE tick-size rounding.
    3. Exits:
       - Stop-Loss: Entry - (2.0 * ATR(14)), tick-aligned.
       - Target: Entry + (2.5 * Risk_per_share), tick-aligned.
       - Chandelier Trailing Stop: Highest High - (2.0 * ATR(14)), ratchets UP only.
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        signal_fn: Callable[[pd.DataFrame, int], bool] | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.signal_fn = signal_fn or self._check_dip_signal
        self.cost_model = TransactionCostModel(
            slippage_pct=self.config.slippage_pct,
            enabled=self.config.enable_transaction_costs,
        )
        self.scanner = TechnicalDipScanner(
            rsi_min=30.0,
            rsi_max=42.0,
            atr_multiplier_stop=self.config.atr_stop_multiplier,
            target_risk_reward=self.config.target_risk_reward,
        )

    def _prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute all technical indicators across the historical series."""
        data = df.copy()
        cols = {c.upper(): c for c in data.columns}

        close = data[cols["CLOSE"]].astype(float)
        high = data[cols["HIGH"]].astype(float)
        low = data[cols["LOW"]].astype(float)
        open_p = data[cols["OPEN"]].astype(float)

        data["CLOSE_F"] = close
        data["HIGH_F"] = high
        data["LOW_F"] = low
        data["OPEN_F"] = open_p

        # Standard indicators
        data["RSI_14"] = calculate_wilder_rsi(close, period=14)
        data["ATR_14"] = calculate_wilder_atr(high, low, close, period=14)
        data["DMA_50"] = close.rolling(50, min_periods=20).mean()
        data["DMA_200"] = close.rolling(200, min_periods=50).mean()

        # GMMA ribbons
        gmma = calculate_gmma(close)
        data["EMA_30"] = gmma["long"]["EMA_30"]
        data["EMA_60"] = gmma["long"]["EMA_60"]
        data["GMMA_BULLISH"] = data["EMA_30"] >= data["EMA_60"]

        # Heikin-Ashi
        ha_df = calculate_heikin_ashi(open_p, high, low, close)
        data["HA_OPEN"] = ha_df["HA_Open"]
        data["HA_CLOSE"] = ha_df["HA_Close"]
        data["HA_BULLISH"] = data["HA_CLOSE"] >= data["HA_OPEN"]

        # 52-week drawdown
        high_52w = high.rolling(252, min_periods=30).max()
        data["DRAWDOWN_52W"] = ((close - high_52w) / high_52w) * 100.0

        # Date normalization
        if "DATE" in cols:
            data["DATE_DT"] = pd.to_datetime(data[cols["DATE"]]).dt.date
        elif isinstance(data.index, pd.DatetimeIndex):
            data["DATE_DT"] = data.index.date
        else:
            data["DATE_DT"] = [date(2020, 1, 1) for _ in range(len(data))]

        return data

    def _check_dip_signal(self, data: pd.DataFrame, idx: int) -> bool:
        """Check if row at idx satisfies the technical dip entry rule."""
        if idx < 30:
            return False

        row = data.iloc[idx]

        # 1. RSI pullback-curl: in zone [30, 42] in recent 5 bars, curling up
        rsi_window = data["RSI_14"].iloc[max(0, idx - 4) : idx + 1]
        if rsi_window.isna().all():
            return False
        in_zone = ((rsi_window >= 30.0) & (rsi_window <= 42.0)).any()
        curling_up = row["RSI_14"] > data["RSI_14"].iloc[idx - 1]
        rsi_ok = bool(in_zone and curling_up)

        # 2. Trend: Price >= 200-DMA * 0.97
        dma_200 = row["DMA_200"]
        trend_ok = bool(pd.isna(dma_200) or row["CLOSE_F"] >= (dma_200 * 0.97))

        # 3. GMMA Investor ribbon bullish
        gmma_ok = bool(row["GMMA_BULLISH"])

        # 4. Drawdown between -4% and -38%
        dd = row["DRAWDOWN_52W"]
        dd_ok = bool(pd.isna(dd) or (-38.0 <= dd <= -4.0))

        # 5. Heikin-Ashi confirmation
        ha_ok = bool(row["HA_BULLISH"])

        return bool(rsi_ok and trend_ok and gmma_ok and dd_ok and ha_ok)

    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        portfolio_capital: float,
        available_cash: float,
    ) -> int:
        """Dual-cap sizing: min(capital_cap, risk_cap), clamped to cash."""
        risk_per_share = entry_price - stop_loss_price
        if risk_per_share <= 0:
            return 0

        # Cap 1: Max capital allocation per position (e.g. 10%)
        capital_budget = portfolio_capital * (self.config.max_capital_per_trade_pct / 100.0)
        qty_cap = int(capital_budget / entry_price)

        # Cap 2: Max risk per position (e.g. 1%)
        risk_budget = portfolio_capital * (self.config.max_risk_per_trade_pct / 100.0)
        qty_risk = int(risk_budget / risk_per_share)

        qty = min(qty_cap, qty_risk)

        # Cash check
        max_affordable_qty = int(available_cash / entry_price)
        final_qty = min(qty, max_affordable_qty)

        return max(0, final_qty)

    def run_single_symbol(
        self,
        df: pd.DataFrame,
        symbol: str,
        config: BacktestConfig | None = None,
    ) -> BacktestResult:
        """Run backtest simulation for a single scrip."""
        cfg = config or self.config
        return self.run_portfolio({symbol: df}, config=cfg)

    def run_portfolio(
        self,
        dfs: dict[str, pd.DataFrame],
        config: BacktestConfig | None = None,
    ) -> BacktestResult:
        """Run simulation across a universe of scrips with shared cash management."""
        cfg = config or self.config
        cost_model = TransactionCostModel(
            slippage_pct=cfg.slippage_pct,
            enabled=cfg.enable_transaction_costs,
        )

        # Precompute indicators for all symbols
        prepared: dict[str, pd.DataFrame] = {}
        for sym, df in dfs.items():
            if df.empty or len(df) < 30:
                continue
            prepared[sym] = self._prepare_indicators(df)

        if not prepared:
            return BacktestResult(
                initial_capital=cfg.initial_capital,
                ending_capital=cfg.initial_capital,
                total_return_pct=0.0,
                cagr_pct=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate_pct=0.0,
                profit_factor=0.0,
                avg_trade_return_pct=0.0,
                avg_holding_days=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                total_costs_paid=0.0,
            )

        # Create master sorted unique calendar dates across all symbols
        all_dates = sorted(
            {d for p_df in prepared.values() for d in p_df["DATE_DT"].tolist() if pd.notna(d)}
        )

        cash = cfg.initial_capital
        open_positions: dict[str, ActivePosition] = {}
        completed_trades: list[TradeRecord] = []
        equity_records: list[dict[str, Any]] = []
        total_costs_accumulated = 0.0

        for current_date in all_dates:
            # 1. Process exits and updates on open positions for current date
            active_symbols = list(open_positions.keys())
            for sym in active_symbols:
                pos = open_positions[sym]
                sym_df = prepared[sym]
                day_rows = sym_df[sym_df["DATE_DT"] == current_date]
                if day_rows.empty:
                    continue
                day = day_rows.iloc[0]

                open_p = day["OPEN_F"]
                high_p = day["HIGH_F"]
                low_p = day["LOW_F"]
                close_p = day["CLOSE_F"]
                atr_p = day["ATR_14"]
                if pd.isna(atr_p) or atr_p <= 0:
                    atr_p = pos.atr_at_entry

                # Update highest price seen for Chandelier trailing
                if high_p > pos.highest_price:
                    pos.highest_price = high_p

                # Check exit conditions:
                # A. Stop-Loss exit (hit low <= stop_loss)
                # B. Profit Target exit (hit high >= target)
                exit_triggered = False
                exit_price = 0.0
                exit_reason = ""

                # Conservative gap check: if opened below stop-loss
                if open_p <= pos.current_stop_loss:
                    exit_triggered = True
                    exit_price = open_p  # Filled at open gap down
                    exit_reason = "STOP_LOSS_GAP_DOWN"
                elif low_p <= pos.current_stop_loss:
                    exit_triggered = True
                    exit_price = pos.current_stop_loss
                    exit_reason = "STOP_LOSS"
                elif open_p >= pos.target_price:
                    exit_triggered = True
                    exit_price = open_p  # Filled at open gap up
                    exit_reason = "TARGET_GAP_UP"
                elif high_p >= pos.target_price:
                    exit_triggered = True
                    exit_price = pos.target_price
                    exit_reason = "TARGET_2_5R"

                if exit_triggered:
                    # Finalize trade
                    sell_costs = cost_model.calculate_sell_costs(exit_price, pos.quantity)
                    total_costs_accumulated += pos.buy_costs.total + sell_costs.total

                    gross_proceeds = exit_price * pos.quantity
                    net_proceeds = gross_proceeds - sell_costs.total
                    cash += net_proceeds

                    gross_cost = pos.entry_price * pos.quantity
                    gross_pnl = gross_proceeds - gross_cost
                    net_pnl = net_proceeds - (gross_cost + pos.buy_costs.total)
                    return_pct = (net_pnl / (gross_cost + pos.buy_costs.total)) * 100.0

                    initial_risk = max(0.01, (pos.entry_price - pos.initial_stop_loss) * pos.quantity)
                    r_mult = net_pnl / initial_risk

                    entry_dt = (
                        datetime.strptime(pos.entry_date, "%Y-%m-%d").date()
                        if isinstance(pos.entry_date, str)
                        else pos.entry_date
                    )
                    holding_days = max(1, (current_date - entry_dt).days)

                    itemized = {
                        "buy_stt": pos.buy_costs.stt,
                        "sell_stt": sell_costs.stt,
                        "buy_stamp_duty": pos.buy_costs.stamp_duty,
                        "buy_exchange": pos.buy_costs.exchange_charges,
                        "sell_exchange": sell_costs.exchange_charges,
                        "buy_gst": pos.buy_costs.gst,
                        "sell_gst": sell_costs.gst,
                        "dp_charges": sell_costs.dp_charges,
                        "buy_slippage": pos.buy_costs.slippage,
                        "sell_slippage": sell_costs.slippage,
                    }

                    record = TradeRecord(
                        symbol=sym,
                        entry_date=str(pos.entry_date),
                        entry_price=pos.entry_price,
                        exit_date=str(current_date),
                        exit_price=exit_price,
                        quantity=pos.quantity,
                        exit_reason=exit_reason,
                        gross_pnl=gross_pnl,
                        net_pnl=net_pnl,
                        total_costs=pos.buy_costs.total + sell_costs.total,
                        itemized_costs=itemized,
                        holding_days=holding_days,
                        return_pct=return_pct,
                        r_multiple=r_mult,
                    )
                    completed_trades.append(record)
                    del open_positions[sym]
                else:
                    # Trailing stop update at EOD: Chandelier ratchet
                    if cfg.enable_chandelier_trail:
                        tick = get_nse_dynamic_tick_size(close_p)
                        chandelier_stop = pos.highest_price - (cfg.chandelier_atr_multiplier * atr_p)
                        aligned_trail = round_to_tick(chandelier_stop, tick)
                        # Ratchet: only move stop loss UP, never down
                        if aligned_trail > pos.current_stop_loss:
                            pos.current_stop_loss = aligned_trail

            # 2. Check new entries on scrips that are not already held
            if len(open_positions) < cfg.max_open_positions:
                for sym, s_df in prepared.items():
                    if sym in open_positions:
                        continue
                    if len(open_positions) >= cfg.max_open_positions:
                        break

                    day_rows = s_df[s_df["DATE_DT"] == current_date]
                    if day_rows.empty:
                        continue
                    row_idx = day_rows.index[0]

                    # Check signal
                    if self.signal_fn(s_df, row_idx):
                        day = day_rows.iloc[0]
                        cmp = day["CLOSE_F"]
                        atr = day["ATR_14"]
                        if pd.isna(atr) or atr <= 0:
                            atr = cmp * 0.02

                        tick = get_nse_dynamic_tick_size(cmp)
                        entry_price = round_to_tick(cmp, tick)

                        # Stop-loss geometry: 2.0x ATR
                        raw_sl = entry_price - (cfg.atr_stop_multiplier * atr)
                        stop_loss = round_to_tick(max(0.05, raw_sl), tick)

                        # Target geometry: 2.5R
                        risk_per_share = entry_price - stop_loss
                        raw_target = entry_price + (cfg.target_risk_reward * risk_per_share)
                        target = round_to_tick(raw_target, tick)

                        # Position sizing with portfolio cash tracking
                        current_invested = sum(
                            p.entry_price * p.quantity for p in open_positions.values()
                        )
                        portfolio_val = cash + current_invested
                        qty = self._calculate_position_size(
                            entry_price=entry_price,
                            stop_loss_price=stop_loss,
                            portfolio_capital=portfolio_val,
                            available_cash=cash,
                        )

                        if qty > 0:
                            buy_costs = cost_model.calculate_buy_costs(entry_price, qty)
                            total_needed = (entry_price * qty) + buy_costs.total
                            if cash >= total_needed:
                                cash -= total_needed
                                open_positions[sym] = ActivePosition(
                                    symbol=sym,
                                    entry_date=str(current_date),
                                    entry_index=row_idx,
                                    entry_price=entry_price,
                                    quantity=qty,
                                    initial_stop_loss=stop_loss,
                                    current_stop_loss=stop_loss,
                                    target_price=target,
                                    atr_at_entry=atr,
                                    highest_price=entry_price,
                                    buy_costs=buy_costs,
                                )

            # 3. Snapshot portfolio equity for equity curve
            invested_val = 0.0
            for s_sym, pos in open_positions.items():
                s_df = prepared[s_sym]
                rows = s_df[s_df["DATE_DT"] == current_date]
                mark_price = rows.iloc[0]["CLOSE_F"] if not rows.empty else pos.entry_price
                invested_val += mark_price * pos.quantity

            total_portfolio_value = cash + invested_val
            equity_records.append(
                {
                    "date": current_date,
                    "portfolio_value": total_portfolio_value,
                    "cash": cash,
                    "invested_capital": invested_val,
                    "open_positions": len(open_positions),
                }
            )

        # 4. Finalize any remaining open positions at the final bar close
        if open_positions and all_dates:
            final_date = all_dates[-1]
            for sym, pos in list(open_positions.items()):
                s_df = prepared[sym]
                final_row = s_df.iloc[-1]
                exit_price = final_row["CLOSE_F"]
                sell_costs = cost_model.calculate_sell_costs(exit_price, pos.quantity)
                total_costs_accumulated += pos.buy_costs.total + sell_costs.total

                gross_proceeds = exit_price * pos.quantity
                net_proceeds = gross_proceeds - sell_costs.total
                cash += net_proceeds

                gross_cost = pos.entry_price * pos.quantity
                gross_pnl = gross_proceeds - gross_cost
                net_pnl = net_proceeds - (gross_cost + pos.buy_costs.total)
                return_pct = (net_pnl / (gross_cost + pos.buy_costs.total)) * 100.0

                initial_risk = max(0.01, (pos.entry_price - pos.initial_stop_loss) * pos.quantity)
                r_mult = net_pnl / initial_risk

                entry_dt = (
                    datetime.strptime(pos.entry_date, "%Y-%m-%d").date()
                    if isinstance(pos.entry_date, str)
                    else pos.entry_date
                )
                holding_days = max(1, (final_date - entry_dt).days)

                itemized = {
                    "buy_stt": pos.buy_costs.stt,
                    "sell_stt": sell_costs.stt,
                    "buy_stamp_duty": pos.buy_costs.stamp_duty,
                    "buy_exchange": pos.buy_costs.exchange_charges,
                    "sell_exchange": sell_costs.exchange_charges,
                    "buy_gst": pos.buy_costs.gst,
                    "sell_gst": sell_costs.gst,
                    "dp_charges": sell_costs.dp_charges,
                    "buy_slippage": pos.buy_costs.slippage,
                    "sell_slippage": sell_costs.slippage,
                }

                completed_trades.append(
                    TradeRecord(
                        symbol=sym,
                        entry_date=str(pos.entry_date),
                        entry_price=pos.entry_price,
                        exit_date=str(final_date),
                        exit_price=exit_price,
                        quantity=pos.quantity,
                        exit_reason="END_OF_DATA",
                        gross_pnl=gross_pnl,
                        net_pnl=net_pnl,
                        total_costs=pos.buy_costs.total + sell_costs.total,
                        itemized_costs=itemized,
                        holding_days=holding_days,
                        return_pct=return_pct,
                        r_multiple=r_mult,
                    )
                )

        # 5. Compute performance statistics
        equity_df = pd.DataFrame(equity_records)
        ending_capital = cash
        total_ret_pct = ((ending_capital - cfg.initial_capital) / cfg.initial_capital) * 100.0

        # Max drawdown computation on equity curve
        if not equity_df.empty:
            equity_df["peak"] = equity_df["portfolio_value"].cummax()
            equity_df["drawdown"] = (
                (equity_df["portfolio_value"] - equity_df["peak"]) / equity_df["peak"]
            ) * 100.0
            max_dd_pct = abs(float(equity_df["drawdown"].min()))

            # CAGR
            num_days = max(1, (all_dates[-1] - all_dates[0]).days)
            years = num_days / 365.25
            if years >= 0.2 and ending_capital > 0:
                cagr_pct = ((ending_capital / cfg.initial_capital) ** (1.0 / years) - 1.0) * 100.0
            else:
                cagr_pct = total_ret_pct

            # Sharpe & Sortino
            daily_returns = equity_df["portfolio_value"].pct_change().dropna()
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = float((daily_returns.mean() / daily_returns.std()) * math.sqrt(252))
                downside = daily_returns[daily_returns < 0]
                sortino = (
                    float((daily_returns.mean() / downside.std()) * math.sqrt(252))
                    if len(downside) > 1 and downside.std() > 0
                    else sharpe
                )
            else:
                sharpe = 0.0
                sortino = 0.0
        else:
            max_dd_pct = 0.0
            cagr_pct = 0.0
            sharpe = 0.0
            sortino = 0.0

        # Trade metrics
        total_trades = len(completed_trades)
        winning_trades = [t for t in completed_trades if t.net_pnl > 0]
        losing_trades = [t for t in completed_trades if t.net_pnl <= 0]
        win_rate = (len(winning_trades) / total_trades * 100.0) if total_trades else 0.0

        gross_wins = sum(t.net_pnl for t in winning_trades)
        gross_losses = abs(sum(t.net_pnl for t in losing_trades))
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (99.0 if gross_wins > 0 else 0.0)

        avg_return = (sum(t.return_pct for t in completed_trades) / total_trades) if total_trades else 0.0
        avg_holding = (sum(t.holding_days for t in completed_trades) / total_trades) if total_trades else 0.0

        return BacktestResult(
            initial_capital=cfg.initial_capital,
            ending_capital=ending_capital,
            total_return_pct=total_ret_pct,
            cagr_pct=cagr_pct,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            avg_trade_return_pct=avg_return,
            avg_holding_days=avg_holding,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            total_costs_paid=total_costs_accumulated,
            trades=completed_trades,
            equity_curve=equity_df,
        )

    def fetch_and_run(
        self,
        symbols: list[str],
        start_date: str | date,
        end_date: str | date | None = None,
        config: BacktestConfig | None = None,
    ) -> BacktestResult:
        """Fetch split-adjusted .NS data via yfinance and execute the backtest."""
        dfs: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = fetch_backtest_ohlcv(symbol=sym, start_date=start_date, end_date=end_date)
                if not df.empty:
                    dfs[sym] = df
            except Exception as e:
                logger.warning("skipping_backtest_symbol_error", symbol=sym, error=str(e))

        return self.run_portfolio(dfs=dfs, config=config)
