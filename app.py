"""
Smart Payment Retry Agent
--------------------------
An AI-powered assistant that analyzes failed payment transactions and
recommends the optimal retry strategy (timing, payment method, and
customer messaging) to help recover lost revenue.

Run locally:
    pip install -r requirements.txt
    export GEMINI_API_KEY="your-key-here"   # free key from aistudio.google.com/apikey
    streamlit run app.py
"""

import os
import json
import time
import pandas as pd
import streamlit as st
import plotly.express as px
from google import genai
from google.genai import types

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Payment Retry Agent | Razorpay",
    page_icon="💳",
    layout="wide",
)

# ---------------------------------------------------------------------
# Razorpay brand theme (navy #0C2451, dodger blue #3395FF, white)
# ---------------------------------------------------------------------
RAZORPAY_NAVY = "#0C2451"
RAZORPAY_BLUE = "#3395FF"
RAZORPAY_LIGHT = "#F8FAFC"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {RAZORPAY_LIGHT};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {RAZORPAY_NAVY};
    }}
    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}
    h1, h2, h3 {{
        color: {RAZORPAY_NAVY} !important;
    }}
    .stButton > button {{
        background-color: {RAZORPAY_BLUE};
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        background-color: {RAZORPAY_NAVY};
        color: white;
    }}
    div[data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid #E2E8F0;
        border-left: 4px solid {RAZORPAY_BLUE};
        border-radius: 8px;
        padding: 12px 16px;
    }}
    div[data-testid="stMetricValue"] {{
        color: {RAZORPAY_NAVY};
    }}
    .stAlert {{
        border-radius: 8px;
    }}
</style>
""", unsafe_allow_html=True)

# Header banner
st.markdown(f"""
<div style="background-color:{RAZORPAY_NAVY}; padding: 20px 28px; border-radius: 10px; margin-bottom: 20px;">
    <div style="display:flex; align-items:center; gap: 14px;">
        <span style="font-size: 32px;">💳</span>
        <div>
            <span style="color:white; font-size: 26px; font-weight: 700;">Smart Payment Retry Agent</span><br/>
            <span style="color:{RAZORPAY_BLUE}; font-size: 13px; letter-spacing: 1px; font-weight: 600;">
                RAZORPAY AI BUILDER INTERNSHIP 2026 · TRACK 3: AI REVENUE RECOVERY
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Rule-based fallback engine (used if no API key is provided)
# ---------------------------------------------------------------------
FALLBACK_RULES = {
    "insufficient_funds": {
        "retry_window": "24-48 hours (near salary/credit cycle)",
        "alt_method": "Suggest UPI Autopay or alternate saved card",
        "reasoning": "Insufficient funds often resolve after a payday or "
                     "fund transfer cycle. Immediate retries usually fail again.",
        "recovery_likelihood": "Medium",
    },
    "card_expired": {
        "retry_window": "Do not auto-retry",
        "alt_method": "Prompt customer to update card details",
        "reasoning": "An expired card will fail on every retry. The customer "
                     "must update payment details before any retry can succeed.",
        "recovery_likelihood": "Low (without customer action)",
    },
    "bank_server_down": {
        "retry_window": "15-30 minutes",
        "alt_method": "Same method, or offer UPI as a quick alternative",
        "reasoning": "Bank/network outages are typically short-lived. A quick "
                     "retry after the server recovers has a high success rate.",
        "recovery_likelihood": "High",
    },
    "upi_timeout": {
        "retry_window": "5-10 minutes",
        "alt_method": "Same UPI ID, or offer card as backup",
        "reasoning": "UPI timeouts are often transient app/network glitches "
                     "and resolve quickly on retry.",
        "recovery_likelihood": "High",
    },
    "do_not_honor": {
        "retry_window": "Do not auto-retry",
        "alt_method": "Suggest an alternate card or payment method",
        "reasoning": "Generic bank decline often indicates a policy block. "
                     "Repeated retries on the same instrument rarely succeed.",
        "recovery_likelihood": "Low",
    },
    "wallet_balance_low": {
        "retry_window": "24 hours",
        "alt_method": "Prompt wallet top-up or switch to UPI/card",
        "reasoning": "Wallet balance issues need customer action (top-up) "
                     "before a retry can succeed.",
        "recovery_likelihood": "Medium",
    },
    "session_expired": {
        "retry_window": "Immediate (same session-free flow)",
        "alt_method": "Same method via a fresh checkout link",
        "reasoning": "Session expiry is a UX/timing issue, not a funds or "
                     "risk issue, so an immediate retry usually succeeds.",
        "recovery_likelihood": "High",
    },
    "suspected_fraud": {
        "retry_window": "Do not auto-retry",
        "alt_method": "Route to manual review / KYC verification",
        "reasoning": "Fraud-flagged transactions should never be auto-retried. "
                     "They need risk-team review before any further attempt.",
        "recovery_likelihood": "Low (requires manual review)",
    },
}


def fallback_recommendation(row: dict) -> dict:
    rule = FALLBACK_RULES.get(
        row["failure_reason_code"],
        {
            "retry_window": "24 hours",
            "alt_method": "Same method",
            "reasoning": "No specific rule matched; defaulting to a standard "
                         "24-hour retry window.",
            "recovery_likelihood": "Medium",
        },
    )
    return {
        "category": "soft_decline" if rule["recovery_likelihood"] != "Low" else "hard_decline",
        "retry_window": rule["retry_window"],
        "alt_method": rule["alt_method"],
        "reasoning": rule["reasoning"],
        "recovery_likelihood": rule["recovery_likelihood"],
        "source": "rule-based (no API key)",
    }


# ---------------------------------------------------------------------
# AI-powered recommendation (Google Gemini — free tier)
# ---------------------------------------------------------------------
SYSTEM_PROMPT = """You are a payments recovery analyst for a fintech company.
Given details of a single failed transaction, respond ONLY with a JSON object
(no markdown, no prose) with these exact keys:

- "category": either "soft_decline" or "hard_decline"
- "retry_window": a short human-readable recommendation for WHEN to retry
- "alt_method": a short recommendation for which payment method to use on retry
- "reasoning": 1-2 sentences explaining WHY, in plain business language
- "recovery_likelihood": one of "High", "Medium", "Low"

Be concise and practical. Do not include any text outside the JSON object.
"""


def ai_recommendation(row: dict, api_key: str) -> dict:
    client = genai.Client(api_key=api_key)
    user_prompt = f"""Failed transaction details:
- Amount: {row['amount']} {row['currency']}
- Payment method: {row['payment_method']}
- Failure reason code: {row['failure_reason_code']}
- Failure message: {row['failure_message']}
- Card type: {row.get('card_type', 'N/A')}
- Customer type: {row.get('customer_history', 'N/A')}

Return the JSON recommendation now."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    text = response.text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    result = json.loads(text)
    result["source"] = "AI (Gemini 2.5 Flash)"
    return result


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("sample_transactions.csv")


df = load_data()

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
api_key_input = st.sidebar.text_input(
    "Google Gemini API Key (optional)",
    value=os.environ.get("GEMINI_API_KEY", ""),
    type="password",
    help="Free key from aistudio.google.com/apikey. Leave blank to use the "
         "built-in rule-based engine instead of live AI calls.",
)
use_ai = st.sidebar.toggle("Use AI recommendations", value=bool(api_key_input))
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About**\n\n"
    "Smart Payment Retry Agent analyzes failed transactions and recommends "
    "the best retry timing and method to recover lost revenue.\n\n"
    "*Built for Razorpay — Outgrow Ordinary.*"
)

# ---------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------
total_failed = df["amount"].sum()
total_txns = len(df)
top_reason = df["failure_reason_code"].value_counts().idxmax()

col1, col2, col3 = st.columns(3)
col1.metric("Failed Transactions", total_txns)
col2.metric("Revenue at Risk", f"₹{total_failed:,.0f}")
col3.metric("Top Failure Reason", top_reason.replace("_", " ").title())

st.markdown("---")

# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    reason_counts = df["failure_reason_code"].value_counts().reset_index()
    reason_counts.columns = ["reason", "count"]
    fig1 = px.bar(
        reason_counts, x="reason", y="count",
        title="Failures by Reason",
        color_discrete_sequence=[RAZORPAY_BLUE] * len(reason_counts),
    )
    fig1.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    method_counts = df["payment_method"].value_counts().reset_index()
    method_counts.columns = ["method", "count"]
    fig2 = px.pie(
        method_counts, names="method", values="count",
        title="Failures by Payment Method",
        color_discrete_sequence=[RAZORPAY_NAVY, RAZORPAY_BLUE, "#7BB8FF", "#B8D9FF"],
    )
    fig2.update_layout(paper_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------
# Transaction table + recommendation panel
# ---------------------------------------------------------------------
st.subheader("📋 Failed Transactions")
st.dataframe(
    df[["transaction_id", "customer_id", "amount", "payment_method",
        "failure_reason_code", "timestamp"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("🔍 Get a Retry Recommendation")
selected_txn = st.selectbox("Select a transaction", df["transaction_id"])
row = df[df["transaction_id"] == selected_txn].iloc[0].to_dict()

col_left, col_right = st.columns([1, 1])
with col_left:
    st.markdown("**Transaction Details**")
    st.json({
        "amount": f"{row['amount']} {row['currency']}",
        "payment_method": row["payment_method"],
        "failure_reason": row["failure_reason_code"],
        "message": row["failure_message"],
        "customer_type": row.get("customer_history", "N/A"),
    })

with col_right:
    st.markdown("**AI Recommendation**")
    if st.button("Generate Recommendation", type="primary"):
        with st.spinner("Analyzing failure pattern..."):
            try:
                if use_ai and api_key_input:
                    rec = ai_recommendation(row, api_key_input)
                else:
                    time.sleep(0.5)
                    rec = fallback_recommendation(row)
            except Exception as e:
                st.warning(f"AI call failed ({e}), showing rule-based result instead.")
                rec = fallback_recommendation(row)

            st.success(f"Recovery Likelihood: **{rec['recovery_likelihood']}**")
            st.markdown(f"**Category:** {rec['category'].replace('_', ' ').title()}")
            st.markdown(f"**Recommended Retry Window:** {rec['retry_window']}")
            st.markdown(f"**Recommended Method:** {rec['alt_method']}")
            st.markdown(f"**Reasoning:** {rec['reasoning']}")
            st.caption(f"Source: {rec['source']}")

st.markdown("---")
st.caption(
    "Built for the Razorpay AI Builder Internship 2026 — Track 3: AI Revenue Recovery"
)
