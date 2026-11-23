# 2-Week Shadow-Mode Verification Checklist & Failover Runbook

This checklist establishes a strict **10-trading-day shadow operational protocol** under `DRY_RUN=true` before transitioning to live capital. It ensures full reliability of session persistence, NSE dynamic tick-size enforcement, LangGraph HITL approval workflows, scheduled positional reconciliation, and local Kite MCP failover.

---

## 1. Pre-Flight Configuration & Security Baseline (Day 0)

- [ ] **Authoritative Environment Lock**:
  - Verify `DRY_RUN=true` is set in `.env` and `src/config/settings.py`.
  - Confirm Zerodha DDPI (Demat Debit and Pledge Instruction) is active on console.zerodha.com.
  - Confirm **NO passwords or automated TOTP seeds (`pyotp`)** exist in `.env`, scripts, or code.
- [ ] **Strict GTT Allowlist Enforcement**:
  - Run allowlist verification test asserting that `place_order`, `modify_order`, and `cancel_order` are blocked.
  - Verify that only read endpoints + the four GTT tools (`place_gtt_order`, `modify_gtt_order`, `delete_gtt_order`, `get_gtts`) are accessible.
- [ ] **Database & Log File Permissions**:
  - Verify SQLite ledger file exists at configured path with proper table schemas (`positions_ledger`, `gtt_audit_log`, `pending_buy_commits`, `screener_cache`).
  - Verify JSONL trade audit log has file permission `chmod 600` (read/write by owner only).

---

## 2. Daily Schedule Verification (10 Consecutive Trading Sessions)

Every day across the 2-week shadow period, verify each scheduled job executes cleanly without uncaught exceptions:

### 08:45 AM IST: Pre-Market Sanity Check
- [ ] **Manual Daily 2FA OAuth Sign-In**:
  - Perform Zerodha Kite 2FA login via browser OAuth.
  - Verify Streamlit sidebar session health badge displays green ("Session Active - Expiry at 06:00 AM IST tomorrow").
- [ ] **Instruments Cache Refresh**:
  - Verify `/instruments` master cache refreshes post-8:30 AM IST.
  - Confirm scrip tick sizes are correctly mapped by `(exchange, tradingsymbol)`.
- [ ] **Corporate Action Blackout Check**:
  - Check that candidate stocks with BSE corporate actions (dividends, splits, bonuses, rights, AGMs) with ex-date within ±2 trading days are blacklisted from Buy GTT generation.
- [ ] **Pending GTT Reconciliation**:
  - Verify active Buy GTT notional commitments do not exceed available cash.

### 16:05 PM IST: Post-CAS & Post-Market Reconciliation
- [ ] **Closing Auction Session (CAS) Close Guard**:
  - Routine triggers strictly after 16:00 PM IST (post-CAS and post-market session).
- [ ] **Filled Stage-1 Buy GTT Detection**:
  - Confirm simulated/dry-run fills move scrip from `pending_buy_commits` to `positions_ledger`.
  - Check that an approval card for Stage-2 Two-Leg Sell OCO (Stop-Loss + Target) is properly queued.
- [ ] **Unfilled Gap-Down Sell GTT Detection**:
  - Verify that if a stop-loss trigger was breached without execution (gap-down or lower circuit), an urgent `CRITICAL` alert is dispatched immediately to Telegram/Slack.
- [ ] **Chandelier / ATR Trailing-Stop Ratchet**:
  - Verify winning holdings have their stop-loss ratcheted up (`Highest Close/High - 2.0 * ATR`), rounded to dynamic NSE tick size.
  - Verify the ratchet rule: stop loss **only moves UP, never down**.

### 18:15 PM IST: EOD Bhavcopy Quality-Dip Scan
- [ ] **Bhavcopy Ingestion & DATE1 Validation**:
  - Ingest official NSE `sec_bhavdata_full.csv`.
  - Verify `BhavcopyDateMismatchError` triggers if NSE returns a stale prior-day file.
- [ ] **Clean-Room Technical Dip Funnel**:
  - Compute Wilder RSI(14) pullback-curl (zone 30-42 + curl).
  - Check 50-DMA and 200-DMA trend filter (`price >= 200-DMA * 0.97`).
  - Verify GMMA investor ribbon (EMA 30 >= EMA 60).
  - Verify Heikin-Ashi green confirmation bar.
  - Check 52-week drawdown within [-38%, -4%].
  - Check 20-day delivery % accumulation threshold.
- [ ] **Deterministic RiskGuard Dual-Cap Sizing**:
  - Verify share quantities are calculated as `min(Capital_Cap_Qty, Risk_Cap_Qty)` against worst-case Buy Limit to SL Limit loss.
  - Confirm zero LLM involvement in share counts or limit prices.
  - Verify all prices align with NSE variable tick bands (₹0.01, ₹0.05, ₹0.10, ₹0.50, ₹1.00, ₹5.00).

### Friday 16:15 PM IST: Weekly Portfolio Digest
- [ ] **Tax Classification & Metrics**:
  - Review STCG vs. LTCG duration tracking from SQLite buy dates.
  - Review realized/unrealized R-multiples.
  - Prune or flag stale unfilled Buy GTTs older than 30 calendar days.

---

## 3. LangGraph Durable Interrupt & HITL UI Verification

- [ ] **Durable Checkpointer Persistence**:
  - Stop the process while a GTT approval card is paused at `human_gtt_approval_node` (`interrupt()`).
  - Restart the application and reload the Streamlit UI.
  - Confirm the approval state resumes seamlessly with exact state and zero duplicate orders.
- [ ] **Approval Drawer Action Branches**:
  - Test **Approve**: Resumes graph and routes to `execute_gtt_node` with dry-run logging.
  - Test **Edit**: Allows manual price/qty modifications; validates with `RiskGuard` before queuing.
  - Test **Reject**: Drops proposal with an audit log reason and releases cash reservation.

---

## 4. Local `zerodha/kite-mcp-server` Failover Runbook

If hosted `https://mcp.kite.trade/mcp` experiences rate-limiting, downtime, or OAuth callback anomalies:

### Step 1: Verify Local Go Binary Prerequisites
Ensure Go (>= 1.22) is installed and build the official Zerodha MCP server binary:
```bash
git clone https://github.com/zerodha/kite-mcp-server.git
cd kite-mcp-server
go build -o kite-mcp-server .
```

### Step 2: Start Local Server with Mandatory Safety Exclusion
Run the binary with `EXCLUDED_TOOLS` permanently blocking standard order execution:
```bash
# Linux / macOS
export EXCLUDED_TOOLS="place_order,modify_order,cancel_order"
export PORT=8080
./kite-mcp-server

# Windows PowerShell
$env:EXCLUDED_TOOLS = "place_order,modify_order,cancel_order"
$env:PORT = "8080"
.\kite-mcp-server.exe
```

### Step 3: Switch Copilot Configuration
Use the automated failover script:
```bash
# Check current status
python scripts/failover_kite_mcp.py status

# Switch to local Go server
python scripts/failover_kite_mcp.py switch-local

# Verify server reachability
python scripts/failover_kite_mcp.py check-health
```
*(Or via PowerShell: `.\scripts\failover_kite_mcp.ps1 switch-local`)*

### Step 4: Revert Back to Hosted Server
When the hosted service stabilizes:
```bash
python scripts/failover_kite_mcp.py switch-hosted
```

---

## 5. Live Ramp-Up Gate (Exit Criteria)

Transition from shadow mode to live capital **only** after fulfilling:
- [ ] 10 consecutive trading days completed with zero runtime crashes or state corruption.
- [ ] 100% of generated orders were strictly `CNC` GTTs (`product="CNC"`).
- [ ] Zero Clause 1.6 trigger proximity rejections or off-tick price rejections.
- [ ] Successful drill of the local Kite MCP server failover procedure.
- [ ] Initial live capital deployment restricted to **25% position sizing** on 1-2 liquid Nifty 100 large-caps.
