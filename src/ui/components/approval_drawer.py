"""Streamlit Human-in-the-Loop (HITL) GTT Approval Drawer Component.

Renders an interactive approval interface for reviewing, editing, or rejecting
proposed GTT orders when a LangGraph workflow reaches an interrupt() barrier.
"""

from __future__ import annotations

from typing import Any, Callable
import structlog

logger = structlog.get_logger(__name__)


def format_approval_metrics(metrics: dict[str, Any]) -> dict[str, str]:
    """Format numerical metrics into human-readable INR currency and percentage strings."""
    qty = metrics.get("quantity", 0)
    capital_inr = metrics.get("capital_deployed_inr", 0.0)
    capital_pct = metrics.get("capital_deployed_pct", 0.0)
    risk_inr = metrics.get("worst_case_risk_inr", 0.0)
    risk_pct = metrics.get("worst_case_risk_pct", 0.0)
    rrr = metrics.get("reward_risk_ratio", 0.0)
    constraint = metrics.get("binding_constraint", "N/A")

    return {
        "quantity": f"{qty:,} shares",
        "capital_deployed": f"₹{capital_inr:,.2f} ({capital_pct * 100:.1f}%)",
        "worst_case_loss": f"₹{risk_inr:,.2f} ({risk_pct * 100:.2f}%)",
        "reward_risk": f"{rrr:.2f}R",
        "binding_constraint": str(constraint).replace("_", " ").title(),
    }


def validate_user_overrides(overrides: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate numeric inputs entered by user in Edit form before submission."""
    buy_trigger = overrides.get("buy_trigger")
    if buy_trigger is not None:
        try:
            val = float(buy_trigger)
            if val <= 0:
                return False, "Buy trigger price must be positive."
        except (ValueError, TypeError):
            return False, "Invalid buy trigger price."

    qty = overrides.get("quantity")
    if qty is not None:
        try:
            val = int(qty)
            if val <= 0:
                return False, "Quantity must be greater than zero."
        except (ValueError, TypeError):
            return False, "Invalid share quantity."

    return True, None


def handle_user_decision(
    action: str,
    overrides: dict[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Construct standard LangGraph Command(resume=decision) payload."""
    action_clean = action.strip().lower()
    if action_clean not in ("approve", "edit", "reject"):
        raise ValueError(f"Invalid decision action: {action!r}. Must be approve, edit, or reject.")

    payload: dict[str, Any] = {"action": action_clean}
    if action_clean == "edit":
        payload["overrides"] = dict(overrides or {})
    elif action_clean == "reject":
        payload["reason"] = reason or "User rejected order proposal"

    return payload


def render_approval_drawer(
    interrupt_payload: dict[str, Any],
    on_decision: Callable[[dict[str, Any]], None],
    key_prefix: str = "hitl",
) -> None:
    """Render the full Streamlit approval card and drawer controls."""
    try:
        import streamlit as st
    except ImportError:
        logger.warning("streamlit_not_installed")
        return

    symbol = interrupt_payload.get("symbol", "UNKNOWN").upper()
    stage = interrupt_payload.get("stage", 1)
    stage_name = "Stage 1: CNC Dip Entry" if stage == 1 else "Stage 2: CNC OCO Protection"
    proposal = interrupt_payload.get("proposal", {})
    geom = proposal.get("order_geometry", {})
    metrics = proposal.get("sizing", {})
    thesis = interrupt_payload.get("thesis") or proposal.get("thesis", "Algorithmic Dip Buy")

    formatted = format_approval_metrics(metrics)

    with st.container():
        st.markdown(
            f"""
            <div style="border: 2px solid #ff4b4b; border-radius: 8px; padding: 16px; margin: 12px 0; background-color: rgba(255, 75, 75, 0.05);">
                <h3 style="margin-top:0; color: #ff4b4b;">🚨 Human Approval Required</h3>
                <p><b>{symbol}</b> — <i>{stage_name}</i> (Strict CNC Delivery)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Core Metrics Display
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if stage == 1:
                st.metric("Buy Trigger", f"₹{geom.get('buy_trigger', 0.0):,.2f}")
                st.caption(f"Limit: ₹{geom.get('buy_limit', 0.0):,.2f}")
            else:
                st.metric("SL Trigger", f"₹{geom.get('sl_trigger', 0.0):,.2f}")
                st.caption(f"SL Limit: ₹{geom.get('sl_limit', 0.0):,.2f}")

        with c2:
            st.metric("Sized Qty", formatted["quantity"])
            st.caption(f"Cap: {formatted['binding_constraint']}")

        with c3:
            st.metric("Capital Deployed", formatted["capital_deployed"].split()[0])
            st.caption(formatted["capital_deployed"].split()[1] if " " in formatted["capital_deployed"] else "")

        with c4:
            st.metric("Worst-Case Loss", formatted["worst_case_loss"].split()[0])
            st.caption(f"Reward/Risk: {formatted['reward_risk']}")

        # Research / Thesis expander
        with st.expander("📖 Thesis & RiskGuard Details", expanded=False):
            st.markdown(f"**Investment Thesis:** {thesis}")
            st.markdown(
                f"- **Stop Loss:** Trigger ₹{geom.get('sl_trigger', 0.0):,.2f} | Limit ₹{geom.get('sl_limit', 0.0):,.2f}\n"
                f"- **Target:** Trigger ₹{geom.get('target_trigger', 0.0):,.2f} | Limit ₹{geom.get('target_limit', 0.0):,.2f}\n"
                f"- **NSE Tick Size:** ₹{geom.get('tick_size', 0.05):.2f}\n"
                f"- **SEBI Algo Exemption:** Verified CNC GTT-Only architecture (April 1, 2026 compliant)"
            )

        st.markdown("---")

        # Action Buttons
        b_col1, b_col2, b_col3 = st.columns([1, 1, 1])

        with b_col1:
            if st.button("✅ Approve Order", key=f"{key_prefix}_btn_approve", type="primary", use_container_width=True):
                decision = handle_user_decision("approve")
                on_decision(decision)

        with b_col2:
            edit_modal = st.expander("✏️ Edit Parameters", expanded=False)
            with edit_modal:
                with st.form(key=f"{key_prefix}_edit_form"):
                    default_buy = float(geom.get("buy_trigger", 0.0))
                    default_qty = int(metrics.get("quantity", 1))
                    default_sl = float(geom.get("sl_trigger", 0.0))
                    default_tgt = float(geom.get("target_trigger", 0.0))

                    edit_buy = st.number_input("Trigger Price (₹)", value=default_buy, step=0.05)
                    edit_qty = st.number_input("Quantity", value=default_qty, step=1, min_value=1)
                    edit_sl = st.number_input("Stop Loss (₹)", value=default_sl, step=0.05)
                    edit_tgt = st.number_input("Target Price (₹)", value=default_tgt, step=0.05)

                    submitted = st.form_submit_button("Submit Edited Order")
                    if submitted:
                        overrides = {
                            "buy_trigger": edit_buy,
                            "quantity": edit_qty,
                            "sl_trigger": edit_sl,
                            "target_trigger": edit_tgt,
                        }
                        valid, err = validate_user_overrides(overrides)
                        if valid:
                            decision = handle_user_decision("edit", overrides=overrides)
                            on_decision(decision)
                        else:
                            st.error(err)

        with b_col3:
            reject_expander = st.expander("❌ Reject Order", expanded=False)
            with reject_expander:
                reason = st.text_input("Rejection Reason", value="Market conditions uncertain", key=f"{key_prefix}_reject_reason")
                if st.button("Confirm Rejection", key=f"{key_prefix}_btn_reject_confirm", use_container_width=True):
                    decision = handle_user_decision("reject", reason=reason)
                    on_decision(decision)
