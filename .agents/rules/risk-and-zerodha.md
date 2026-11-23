---
trigger: glob
description: Exact Zerodha Kite MCP, NSE tick-size, GTT Clause 1.6, dual-cap position sizing, and data-pacing rules for India CNC dip investing.
globs:
  - "src/broker/**"
  - "src/risk/**"
  - "src/technicals/**"
  - "src/data/**"
  - "src/agents/workflows/**"
  - "src/scheduler/**"
  - "src/storage/**"
  - "tests/**"
  - "src/config/**"
---

# Zerodha Kite MCP, NSE & RiskGuard Specifications

## 1. Two-Stage CNC GTT Lifecycle & Limit Defaults

* Equity cash (`CNC`) does NOT support Buy OCO GTTs, and Sell OCO GTTs require the shares to exist in holdings at trigger time.
* **Stage 1 (Dip Entry):** Single-Leg `BUY` `CNC` `LIMIT` GTT.
  * `buy_trigger`: support level (50/200-DMA or horizontal support zone).
  * `buy_limit`: `round_to_tick(buy_trigger * 1.005)` (+0.50% fill buffer).
* **Stage 2 (Post-Fill Exit):** Two-Leg `SELL` OCO `CNC` `LIMIT` GTT (Stop-Loss + Target).
  * Reject creation inside `RiskGuard` unless live `get_holdings` confirms `quantity >= requested_sell_qty`.
  * `sl_trigger`: `round_to_tick(buy_trigger - 2.0 * atr_14)`.
  * `sl_limit`: `round_to_tick(sl_trigger * 0.985)` (-1.50% slippage buffer, clamped >= `lower_circuit_limit`).
  * `target_trigger`: `round_to_tick(buy_trigger + 2.5 * (buy_trigger - sl_trigger))`.
  * `target_limit`: `round_to_tick(target_trigger * 0.998)` (-0.20% fill buffer, clamped <= `upper_circuit_limit`).
* **Always use `LIMIT` orders inside GTTs:** Unprotected `MARKET` GTT orders are rejected under April 2026 Market Price Protection (MPP) rules.
* **DDPI Warning:** Display a startup warning unless `ZERODHA_DDPI_ENABLED=true` is confirmed in config (without DDPI/POA, triggered CNC sell GTTs fail daily CDSL TPIN/eDIS authorization).

## 2. Deterministic Dual-Cap Sizing & Validation (`src/risk/risk_guard.py`)

* **Dual-Cap Quantity Formula (worst-case fill-to-SL-limit risk):**
  * `risk_per_share_worst = buy_limit - sl_limit`
  * `qty_by_risk = floor((portfolio_value * max_risk_pct) / risk_per_share_worst)` (default `max_risk_pct = 0.01`)
  * `qty_by_capital = floor((portfolio_value * max_pos_pct) / buy_limit)` (default `max_pos_pct = 0.10`)
  * `final_qty = min(qty_by_risk, qty_by_capital)`
* **NSE Cash Tick Sizes (NSE/CMTR/67133):**
  * Read exact daily `tick_size` from Kite `/instruments` CSV cache (refreshed once daily after 8:30 AM IST, keyed by `(exchange, tradingsymbol)`, never `instrument_token`).
  * Fallback slab validation: `<250: 0.01`, `250-1000: 0.05`, `>1000-5000: 0.10`, `>5000-10000: 0.50`, `>10000-20000: 1.00`, `>20000: 5.00`.
* **Zerodha GTT Clause 1.6 Minimum Trigger Distance:**
  * If live `last_price >= 50`: `abs(trigger - last_price) / last_price >= 0.0025` (0.25%).
  * If live `last_price < 50`: `abs(trigger - last_price) >= 0.09`.
* **Additional Hard Checks:**
  * Sector concentration cap <= 25% of portfolio.
  * Pending Buy GTT cash reservation: `sum(active_buy_gtt_notional) + new_buy_notional <= available_cash` (Zerodha does not block cash on Buy GTT creation).
  * Duplicate order suppression within a 30-second window.
  * BSE corporate-action blackout: block/flag stocks with imminent earnings (<5 trading sessions) or ex-dates for splits, bonuses, rights, amalgamations, or dividends >2% (which auto-cancel GTTs).

## 3. Upstream `zerodha/kite-mcp-server` Workarounds

* **Session Persistence:** Wrap app/client lifecycle in `async with mcp_client.session("kite") as session:` and load tools via `load_mcp_tools(session, handle_tool_errors=True)`.
* **Strict Tool Allowlist & Interceptor:** Allow only `login`, `get_profile`, `get_margins`, `get_holdings`, `get_positions`, `get_orders`, `get_quotes`, `get_ltp`, `get_ohlc`, `get_historical_data`, `search_instruments`, `get_gtts`, `place_gtt_order`, `modify_gtt_order`, `delete_gtt_order`. Override upstream's blanket `"destructive"` tags via `ToolCallInterceptor` with exponential backoff on HTTP 429/522.
* **GTT Modification Workaround (PR #108):** Until upstream PR #108 merges, implement stop-loss updates/trails as `place_gtt_order` (explicitly passing `product="CNC"`) -> verify via `get_gtts` -> `delete_gtt_order` (ensuring `trigger_id` is serialized strictly as an integer).
* **Holdings Price & Buy Dates (Issues #79 & #91):** Refresh holding prices with batched `get_ltp`/`get_quotes`; store confirmed fill dates and entry prices in SQLite `positions_ledger` for STCG/LTCG tracking.

## 4. Data Sources, Pacing & Indian Market Schedule

* **NSE Bhavcopy (`src/data/bhavcopy.py`):** Download daily `sec_bhavdata_full` (for OHLCV + delivery %) joined with UDiFF Bhavcopy (`ISIN`, `TckrSymb`, `SctySrs`). Always send a browser `User-Agent` and drop the file if internal `DATE1` does not match the requested trading date.
* **Screener.in Cache:** Read from SQLite `screener_cache` first (30–90 day quarterly TTL). Cold scrapes must run at `concurrency=1` with `>=2.5s` delay, hit only `/company/<symbol>/consolidated/`, and back off for hours on HTTP 403/429. Hide Screener and BSE clients behind clean provider adapter interfaces.
* **Kite Rate Limits:** `/quote` max 500 symbols/req @ 1.1s pacing; Historical max 3 req/sec @ 0.35s pacing.
* **Scheduled Jobs (IST, aligned with Dec 7, 2026 CAS 3:30–3:40 PM & post-market 3:50–4:00 PM):**
  * `08:45 AM` weekdays: Pre-market OAuth prompt, `/instruments` cache refresh, BSE ex-date GTT check.
  * `04:05 PM` weekdays: Post-CAS reconciliation (diff holdings/orders/GTTs -> queue Stage-2 Sell OCO cards, alert on unfilled gap-down SL triggers, compute Chandelier/ATR `highest_close - 2.5*ATR` trailing-stop upgrades).
  * `06:15 PM` weekdays: NSE Bhavcopy EOD 3-gate dip scan -> batched `get_quotes` on top 5–15 candidates.
  * `04:15 PM` Fridays: Weekly portfolio, sector concentration, STCG/LTCG, and stale Buy GTT (>30d) digest.
