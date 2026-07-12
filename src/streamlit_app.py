# src/app/streamlit_app.py
"""
PD Model Dashboard
"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# Session state for routing
if "page" not in st.session_state:
    st.session_state.page = "scorer"

# CSS 
st.markdown("""
<style>
/* ───── Global ───── */
*, *::before, *::after { box-sizing: border-box; }
[data-testid="stAppViewContainer"] { background: #f0f2f5; }
[data-testid="stMainBlockContainer"] { padding-top: 1.75rem; }

/* ───── Sidebar shell ───── */
[data-testid="stSidebar"] {
    background: #0a0f1e !important;
    border-right: 1px solid #1a2035 !important;
    min-width: 240px;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
    height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Hide default Streamlit sidebar collapse button decoration */
[data-testid="stSidebarNav"] { display: none !important; }
button[data-testid="baseButton-headerNoPadding"] { display: none; }

/* ───── App logo area ───── */
.app-logo-wrap {
    padding: 1.5rem 1.25rem 1.25rem;
    border-bottom: 1px solid #1a2035;
    display: flex;
    align-items: center;
    gap: 12px;
}
.app-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(37,99,235,0.4);
}
.app-logo-text { line-height: 1.2; }
.app-logo-name {
    font-size: 15px; font-weight: 700;
    color: #f1f5f9; letter-spacing: -0.01em;
}
.app-logo-sub {
    font-size: 10px; color: #475569;
    margin-top: 2px; letter-spacing: 0.04em;
}

/* Nav section label */
.nav-section {
    font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #334155;
    padding: 1.25rem 1.25rem 0.5rem;
}

/* Nav buttons */
.nav-btn {
    display: flex; align-items: center; gap: 12px;
    width: 100%; padding: 10px 1.25rem;
    font-size: 14px; font-weight: 500;
    color: #64748b; cursor: pointer;
    border: none; background: transparent;
    border-left: 2px solid transparent;
    transition: all 0.15s ease;
    text-align: left;
}
.nav-btn:hover {
    color: #e2e8f0;
    background: rgba(255,255,255,0.04);
    border-left-color: #334155;
}
.nav-btn.active {
    color: #93c5fd;
    background: rgba(37,99,235,0.12);
    border-left-color: #2563eb;
}
.nav-btn .nav-icon {
    font-size: 17px; width: 22px;
    text-align: center; flex-shrink: 0;
}
.nav-btn .nav-label { font-size: 14px; }

/* Social links */
.social-row {
    display: flex; gap: 10px;
    padding: 1rem 1.25rem;
    border-top: 1px solid #1a2035;
    margin-top: auto;
}
.social-link {
    display: flex; align-items: center; justify-content: center;
    width: 36px; height: 36px;
    border-radius: 8px;
    background: rgba(255,255,255,0.06);
    border: 1px solid #1e293b;
    text-decoration: none;
    font-size: 16px;
    transition: all 0.15s ease;
    color: #64748b;
}
.social-link:hover {
    background: rgba(37,99,235,0.2);
    border-color: #2563eb;
    color: #93c5fd;
}

/* Cards */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.card-title {
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.09em;
    color: #94a3b8; margin-bottom: 1.1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #f1f5f9;
}

/* Metric strip */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin-bottom: 1.25rem;
}
.metric-tile {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.metric-tile-label {
    font-size: 11px; color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 6px;
}
.metric-tile-val {
    font-size: 26px; font-weight: 700;
    color: #2563eb; line-height: 1;
}
.metric-tile-sub {
    font-size: 11px; color: #94a3b8; margin-top: 4px;
}

/* Score display */
.score-hero { text-align: center; padding: 1.25rem 1rem 0.75rem; }
.score-num  {
    font-size: 54px; font-weight: 800;
    line-height: 1; letter-spacing: -0.02em;
}
.score-sub  {
    font-size: 11px; color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em; margin-top: 6px;
}
.score-green { color: #059669; }
.score-amber { color: #d97706; }
.score-red   { color: #dc2626; }

/* Risk badges */
.badge {
    display: inline-flex; align-items: center;
    padding: 5px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 700;
    letter-spacing: 0.02em;
}
.badge-very-low, .badge-low   { background:#d1fae5; color:#065f46; }
.badge-medium                 { background:#fef3c7; color:#92400e; }
.badge-high, .badge-very-high { background:#fee2e2; color:#991b1b; }

/* Info rows */
.info-row {
    display: flex; justify-content: space-between;
    align-items: center; padding: 9px 0;
    border-bottom: 1px solid #f8fafc; font-size: 13px;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #64748b; }
.info-val { color: #0f172a; font-weight: 600; }

/* Band table */
.band-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.band-table th {
    text-align: left; padding: 8px 10px;
    font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.07em; color: #94a3b8;
    border-bottom: 1px solid #e2e8f0; font-weight: 600;
}
.band-table td {
    padding: 10px; color: #475569;
    border-bottom: 1px solid #f8fafc;
}
.band-table tr:last-child td { border-bottom: none; }
.mini-bar-bg { width: 80px; height: 4px; background: #f1f5f9; border-radius: 2px; }
.mini-bar    { height: 4px; background: #2563eb; border-radius: 2px; }

/* Method cards */
.method-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.method-head {
    font-size: 14px; font-weight: 700;
    color: #0f172a; margin-bottom: 0.7rem;
    display: flex; align-items: center; gap: 8px;
}
.method-body-list {
    margin: 0; padding-left: 1.25rem;
    color: #475569; font-size: 13px; line-height: 1.8;
}
.tag-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.75rem; }
.tag {
    font-size: 11px; background: #f1f5f9;
    border: 1px solid #e2e8f0; border-radius: 6px;
    padding: 3px 10px; color: #475569; font-weight: 500;
}

/* Page title */
.page-title {
    font-size: 22px; font-weight: 700;
    color: #0f172a; margin-bottom: 4px;
    letter-spacing: -0.02em;
}
.page-sub {
    font-size: 13px; color: #64748b;
    margin-bottom: 1.5rem; line-height: 1.5;
}

/* Empty state */
.empty-state {
    text-align: center; padding: 3rem 1.5rem;
}
.empty-icon { font-size: 42px; margin-bottom: 0.75rem; }
.empty-title {
    font-size: 15px; font-weight: 700;
    color: #0f172a; margin-bottom: 6px;
}
.empty-sub { font-size: 13px; color: #94a3b8; line-height: 1.6; }

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stDecoration"] { display: none !important; }
div[data-testid="stForm"] { border: none !important; padding: 0 !important; }

/* Streamlit button → nav style (used for nav items) */
div[data-testid="stSidebar"] .stButton button {
    background: transparent !important;
    border: none !important;
    border-left: 2px solid transparent !important;
    border-radius: 0 !important;
    color: #64748b !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 1.25rem !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.15s !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    justify-content: flex-start !important;
}
div[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.04) !important;
    color: #e2e8f0 !important;
    border-left-color: #334155 !important;
}
.nav-active div[data-testid="stSidebar"] .stButton button {
    background: rgba(37,99,235,0.12) !important;
    color: #93c5fd !important;
    border-left-color: #2563eb !important;
}
</style>
""", unsafe_allow_html=True)


# SIDEBAR
with st.sidebar:

    # Logo
    st.markdown("""
    <div class="app-logo-wrap">
        <div class="app-logo-icon">🏦</div>
        <div class="app-logo-text">
            <div class="app-logo-name">Default Predictor</div>
            <div class="app-logo-sub">CREDIT RISK · v1.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Nav section
    st.markdown('<div class="nav-section">Navigation</div>',
                unsafe_allow_html=True)

    # Nav items — defined as (label, icon, page_key)
    nav_items = [
        ("Credit Scorer",    "🎯", "scorer"),
        ("Model Performance","📊", "performance"),
        ("Insights",      "📋", "insights"),
    ]

    for label, icon, key in nav_items:
        is_active = st.session_state.page == key
        active_cls = "nav-active" if is_active else ""
        # Wrap in a div with the active class so CSS can target it
        st.markdown(f'<div class="{active_cls}">', unsafe_allow_html=True)
        if st.button(
            f"{icon}   {label}",
            key=f"nav_{key}",
            width="stretch",
        ):
            st.session_state.page = key
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Spacer pushes social links to the bottom
    st.markdown(
        "<div style='flex:1;min-height:40px'></div>",
        unsafe_allow_html=True,
    )

    # Social links
    st.markdown("""
    <div class="social-row">
        <a class="social-link"
           href="https://github.com/husainridwan/ProbabilityOfDefault"
           target="_blank" title="GitHub">
            <svg width="18" height="18" viewBox="0 0 24 24"
                 fill="currentColor">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435
                9.795 8.205 11.385.6.105.825-.255.825-.57
                0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735
                -4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225
                -1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23
                1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305
                .765-1.605-2.67-.3-5.46-1.335-5.46-5.925
                0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12
                -3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405
                3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23
                3.3-1.23.66 1.65.24 2.88.12 3.18.765.84
                1.23 1.905 1.23 3.225 0 4.605-2.805 5.625
                -5.475 5.925.435.375.81 1.095.81 2.22
                0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57
                A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
            </svg>
        </a>
        <a class="social-link"
           href="https://www.linkedin.com/in/ridwanllah-husain-655458195/"
           target="_blank" title="LinkedIn">
            <svg width="18" height="18" viewBox="0 0 24 24"
                 fill="currentColor">
                <path d="M20.447 20.452h-3.554v-5.569c0-1.328
                -.027-3.037-1.852-3.037-1.853 0-2.136 1.445
                -2.136 2.939v5.667H9.351V9h3.414v1.561h.046
                c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267
                2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062
                0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063
                2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225
                0H1.771C.792 0 0 .774 0 1.729v20.542C0
                23.227.792 24 1.771 24h20.451C23.2 24 24
                23.227 24 22.271V1.729C24 .774 23.2 0
                22.222 0h.003z"/>
            </svg>
        </a>
        <div style="font-size:10px;color:#334155;
                    align-self:center;margin-left:auto;
                    line-height:1.4">
            Rho<br>
            <span style="color:#1e3a5f">Portfolio</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ROUTE
pg = st.session_state.page


# PAGE 1 — CREDIT SCORER
if pg == "scorer":
    st.markdown("<div class='page-title'>Credit scorer</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='page-sub'>Enter loan and borrower details to receive "
        "a real-time probability of default score.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        # Loan details
        st.markdown("<div class='card'>"
                    "<div class='card-title'>Loan details</div>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        loan_number      = c1.number_input(
            "Loan number *(1 = first loan)*",
            min_value=1, value=1, step=1)
        principal_amount = c2.number_input(
            "Principal amount (₦)",
            min_value=1000, value=50000, step=1000)
        c3, c4 = st.columns(2)
        credit_limit = c3.number_input(
            "Credit limit (₦)",
            min_value=1000, value=50000, step=1000)
        tenure       = c4.number_input(
            "Tenure (days)", min_value=7, max_value=365, value=30)
        c5, c6 = st.columns(2)

        st.markdown("</div>", unsafe_allow_html=True)

        # Borrower profile
        st.markdown("<div class='card'>"
                    "<div class='card-title'>Borrower profile</div>",
                    unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        monthly_income = b1.number_input(
            "Monthly income (₦)",
            min_value=0, value=150000, step=5000)
        age            = b2.number_input(
            "Age", min_value=18, max_value=75, value=32)
        b3, b4 = st.columns(2)
        state      = b3.selectbox("State of residence", [
            "Lagos", "Abuja", "Rivers", "Oyo", "Kano", "Enugu",
            "Delta", "Anambra", "Ogun", "Osun", "Cross River", "Other"])
        days_since = b4.number_input(
            "Days since last loan *(0 if first)*",
            min_value=0, value=0)
        st.markdown("</div>", unsafe_allow_html=True)

        # Credit history
        st.markdown(
            "<div class='card'>"
            "<div class='card-title'>"
            "Credit history "
            "<span style='font-weight:400;color:#cbd5e1;"
            "font-size:9px;letter-spacing:0'>— leave 0 if unknown</span>"
            "</div>",
            unsafe_allow_html=True)
        h1, h2 = st.columns(2)
        lender_count  = h1.number_input("Lenders borrowed from",
                                         min_value=0, value=0)
        enquiry_count = h2.number_input("Recent credit enquiries",
                                         min_value=0, value=0)
        h3, h4 = st.columns(2)
        monthly_obligations = h3.number_input(
            "Monthly obligations (₦)", min_value=0, value=0, step=1000)
        total_debt          = h4.number_input(
            "Total outstanding debt (₦)", min_value=0, value=0, step=5000)
        h5, h6 = st.columns(2)
        total_accounts  = h5.number_input("Total loan accounts",
                                           min_value=0, value=0)
        has_written_off = h6.selectbox(
            "Written-off account?", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes")
        has_arrears = st.selectbox(
            "Currently in arrears?", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes")
        st.markdown("</div>", unsafe_allow_html=True)

        scored = st.button(
            "Calculate PD score",
            type="primary",
            width="stretch",
        )

    # Result panel
    with right:
        if scored:
            payload = {
                "loan_number":          int(loan_number),
                "principal_amount":     float(principal_amount),
                "tenure":               int(tenure),
                "credit_limit":         float(credit_limit),
                "monthly_income":       float(monthly_income),
                "age":                  int(age),
                "state":                state,
                "product_type":         "Standard",
                "channel_group":        "Organic",
                "lender_count":         int(lender_count),
                "enquiry_count":        int(enquiry_count),
                "total_bureau_accounts":int(total_accounts),
                "monthly_obligations":  float(monthly_obligations),
                "total_debt":           float(total_debt),
                "has_written_off":      int(has_written_off),
                "has_arrears":          int(has_arrears),
                "days_since_last_loan": int(days_since),
            }

            try:
                resp   = requests.post(
                    f"{API_URL}/predict", json=payload, timeout=15)
                result = resp.json()

                if resp.status_code != 200:
                    st.error(
                        f"API error {resp.status_code}: "
                        f"{result.get('detail','Unknown error')}")
                else:
                    prob     = result["probability_of_default"]
                    band     = result["risk_band"]
                    decision = result["decision"]
                    seg      = result["model_used"]

                    score_cls  = ("score-green" if prob < 0.3
                                  else "score-amber" if prob < 0.5
                                  else "score-red")
                    clr        = ("#059669" if prob < 0.3
                                  else "#d97706" if prob < 0.5
                                  else "#dc2626")
                    badge_cls  = "badge-" + band.lower().replace(" ", "-")
                    dec_clr    = ("#059669" if "Approve" in decision
                                  else "#d97706" if decision == "Review"
                                  else "#dc2626")

                    # Score card
                    st.markdown(f"""
                    <div class='card'>
                        <div class='score-hero'>
                            <div class='score-num {score_cls}'>
                                {prob*100:.1f}%
                            </div>
                            <div class='score-sub'>Probability of Default</div>
                        </div>
                        <div style='text-align:center;margin:0.5rem 0 1rem'>
                            <span class='badge {badge_cls}'>{band} risk</span>
                        </div>
                        <div style='display:flex;justify-content:space-between;
                                    padding-top:0.75rem;
                                    border-top:1px solid #f1f5f9;
                                    font-size:13px;align-items:center'>
                            <span style='color:#64748b;font-size:12px'>
                                {seg}
                            </span>
                            <span style='color:{dec_clr};font-weight:700;
                                         font-size:14px'>
                                {decision}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(prob * 100, 1),
                        number={"suffix": "%",
                                "font": {"size": 22, "color": clr}},
                        gauge={
                            "axis": {"range": [0, 100],
                                     "tickfont": {"size": 10},
                                     "tickcolor": "#94a3b8"},
                            "bar":  {"color": clr, "thickness": 0.22},
                            "bgcolor": "white",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0,  25], "color": "#dcfce7"},
                                {"range": [25, 45], "color": "#fef9c3"},
                                {"range": [45, 65], "color": "#ffedd5"},
                                {"range": [65, 100],"color": "#fee2e2"},
                            ],
                        },
                    ))
                    fig.update_layout(
                        height=190,
                        margin=dict(t=20, b=10, l=20, r=20),
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        font={"family": "Inter, sans-serif"},
                    )
                    st.plotly_chart(
                        fig, width="stretch",
                        config={"displayModeBar": False})

                    # Derived metrics
                    inc  = max(float(monthly_income), 1)
                    lti  = float(principal_amount) / inc
                    util = float(principal_amount) / max(float(credit_limit), 1)
                    burd = lti * float(tenure) / 30
                    dsti = float(monthly_obligations) / inc

                    st.markdown(f"""
                    <div class='card'>
                        <div class='card-title'>Derived metrics</div>
                        <div class='info-row'>
                            <span class='info-key'>Loan-to-income ratio</span>
                            <span class='info-val'>{lti:.2f}×</span>
                        </div>
                        <div class='info-row'>
                            <span class='info-key'>Credit utilisation</span>
                            <span class='info-val'>{util*100:.0f}%</span>
                        </div>
                        <div class='info-row'>
                            <span class='info-key'>Repayment burden</span>
                            <span class='info-val'>{burd:.2f}</span>
                        </div>
                        <div class='info-row'>
                            <span class='info-key'>Bureau DSTI</span>
                            <span class='info-val'>{dsti:.2f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.markdown("""
                <div class='card' style='border-color:#fecaca'>
                    <div style='color:#dc2626;font-weight:700;
                                margin-bottom:8px;font-size:14px'>
                        ⚠ API not reachable
                    </div>
                    <div style='font-size:13px;color:#64748b;
                                margin-bottom:12px'>
                        Start the FastAPI server first:
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.code(
                    "uvicorn src.api.main:app --reload",
                    language="bash")
            except Exception as e:
                st.error(f"Error: {e}")

        else:
            st.markdown("""
            <div class='card empty-state'>
                <div class='empty-icon'>🏦</div>
                <div class='empty-title'>Enter loan details</div>
                <div class='empty-sub'>
                    Fill in the form on the left and click<br>
                    <b style='color:#0f172a'>Calculate PD score</b>
                </div>
            </div>
            <div class='card'>
                <div class='card-title'>Model at a glance</div>
                <div class='info-row'>
                    <span class='info-key'>C1 test AUC</span>
                    <span class='info-val' style='color:#2563eb'>0.677</span>
                </div>
                <div class='info-row'>
                    <span class='info-key'>C2+ test AUC</span>
                    <span class='info-val' style='color:#2563eb'>0.757</span>
                </div>
                <div class='info-row'>
                    <span class='info-key'>Training loans</span>
                    <span class='info-val'>167,000</span>
                </div>
                <div class='info-row'>
                    <span class='info-key'>Overall default rate</span>
                    <span class='info-val'>20.79%</span>
                </div>
                <div class='info-row'>
                    <span class='info-key'>Unique borrowers</span>
                    <span class='info-val'>167,990</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# PAGE 2 — MODEL PERFORMANCE
elif pg == "performance":
    st.markdown("<div class='page-title'>Model performance</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='page-sub'>Test-set evaluation — random user split "
        "(70/15/15). Val–test AUC gap under 1pp across all models.</div>",
        unsafe_allow_html=True)

    # Metric strip
    st.markdown("""
    <div class='metric-strip'>
        <div class='metric-tile'>
            <div class='metric-tile-label'>C1 Test AUC</div>
            <div class='metric-tile-val'>0.677</div>
            <div class='metric-tile-sub'>First-time borrowers</div>
        </div>
        <div class='metric-tile'>
            <div class='metric-tile-label'>C1 Test Gini</div>
            <div class='metric-tile-val'>0.355</div>
            <div class='metric-tile-sub'>= 2 × AUC − 1</div>
        </div>
        <div class='metric-tile'>
            <div class='metric-tile-label'>C2+ Test AUC</div>
            <div class='metric-tile-val'>0.757</div>
            <div class='metric-tile-sub'>Returning borrowers</div>
        </div>
        <div class='metric-tile'>
            <div class='metric-tile-label'>C2+ Test Gini</div>
            <div class='metric-tile-val'>0.514</div>
            <div class='metric-tile-sub'>Production-grade</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Segment toggle 
    seg_col1, seg_col2, _ = st.columns([1, 1, 4])
    if "seg" not in st.session_state:
        st.session_state.seg = "C1"

    with seg_col1:
        if st.button("C1 — first-time",
                     type=("primary" if st.session_state.seg == "C1"
                           else "secondary"),
                     width="stretch"):
            st.session_state.seg = "C1"
            st.rerun()
    with seg_col2:
        if st.button("C2+ — returning",
                     type=("primary" if st.session_state.seg == "C2+"
                           else "secondary"),
                     width="stretch"):
            st.session_state.seg = "C2+"
            st.rerun()

    is_c1 = st.session_state.seg == "C1"

    # AUC comparison chart
    models   = ["LR", "RF", "LightGBM", "Ensemble"]
    val_aucs = ([0.663, 0.685, 0.682, 0.685] if is_c1
                else [0.748, 0.761, 0.759, 0.762])
    tst_aucs = ([0.657, 0.679, 0.676, 0.677] if is_c1
                else [0.743, 0.757, 0.756, 0.757])

    fig = go.Figure()
    fig.add_bar(
        name="Val AUC", x=models, y=val_aucs,
        marker_color="#bfdbfe",
        text=[f"{v:.3f}" for v in val_aucs],
        textposition="outside",
        textfont={"size": 11, "color": "#64748b"},
    )
    fig.add_bar(
        name="Test AUC", x=models, y=tst_aucs,
        marker_color="#2563eb",
        text=[f"{v:.3f}" for v in tst_aucs],
        textposition="outside",
        textfont={"size": 11, "color": "#2563eb"},
    )
    fig.add_hline(
        y=0.70, line_dash="dot", line_color="#94a3b8",
        annotation_text=" 0.70 target",
        annotation_font={"size": 11, "color": "#94a3b8"},
    )
    fig.update_layout(
        barmode="group",
        height=300,
        margin=dict(t=40, b=10, l=0, r=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(range=[0.55, 0.82],
                   gridcolor="#f8fafc", tickformat=".3f"),
        xaxis=dict(gridcolor="#f8fafc"),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig, width="stretch",
                    config={"displayModeBar": False})

    # Val / Test detail cards
    results = {
        "C1":  {"val":  {"AUC": 0.685, "Gini": 0.370, "KS": 0.266},
                "test": {"AUC": 0.677, "Gini": 0.355, "KS": 0.256}},
        "C2+": {"val":  {"AUC": 0.762, "Gini": 0.524, "KS": 0.387},
                "test": {"AUC": 0.757, "Gini": 0.514, "KS": 0.382}},
    }
    r  = results["C1" if is_c1 else "C2+"]
    cv, ct = st.columns(2)

    for col, split_key, split_label in [
        (cv, "val",  "Validation results"),
        (ct, "test", "Test results"),
    ]:
        d = r[split_key]
        col.markdown(
            f"<div class='card'>"
            f"<div class='card-title'>{split_label}</div>"
            f"<div class='info-row'><span class='info-key'>AUC</span>"
            f"<span class='info-val' style='color:#2563eb'>"
            f"{d['AUC']:.3f}</span></div>"
            f"<div class='info-row'><span class='info-key'>Gini</span>"
            f"<span class='info-val'>{d['Gini']:.3f}</span></div>"
            f"<div class='info-row'>"
            f"<span class='info-key'>KS statistic</span>"
            f"<span class='info-val'>{d['KS']:.3f}</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Risk band table
    bands_c1 = [
        ("Very Low",  "18.3%", "3,682",  18, "badge-very-low",  "#059669", "Approve"),
        ("Low",       "35.5%", "3,698",  36, "badge-low",       "#059669", "Approve"),
        ("Medium",    "44.7%", "4,586",  45, "badge-medium",    "#d97706", "Review"),
        ("High",      "54.3%", "3,690",  54, "badge-high",      "#dc2626", "Decline"),
        ("Very High", "66.0%", "2,753",  66, "badge-very-high", "#dc2626", "Decline"),
    ]
    bands_c2 = [
        ("Very Low",  "1.8%",  "14,981",  2, "badge-very-low",  "#059669", "Auto-approve"),
        ("Low",       "6.8%",  "14,843",  7, "badge-low",       "#059669", "Approve"),
        ("Medium",    "13.5%", "18,626", 14, "badge-medium",    "#d97706", "Review"),
        ("High",      "23.1%", "14,897", 23, "badge-high",      "#dc2626", "Decline"),
        ("Very High", "37.9%", "11,179", 38, "badge-very-high", "#dc2626", "Decline"),
    ]
    bands = bands_c1 if is_c1 else bands_c2

    rows = []
    for nm, dr, cnt, bw, bc, dc, dec in bands:
        rows.append(
            "<tr>"
            f"<td><span class='badge {bc}' "
            f"style='font-size:11px;padding:2px 10px'>{nm}</span></td>"
            f"<td>{dr}</td><td>{cnt}</td>"
            "<td><div class='mini-bar-bg'>"
            f"<div class='mini-bar' style='width:{bw}%'></div>"
            "</div></td>"
            f"<td style='color:{dc};font-weight:600'>{dec}</td>"
            "</tr>"
        )

    st.markdown(
        "<div class='card'>"
        "<div class='card-title'>Risk band calibration</div>"
        "<table class='band-table'>"
        "<tr><th>Band</th><th>Default rate</th>"
        "<th>Count</th><th>Spread</th><th>Decision</th></tr>"
        + "".join(rows)
        + "</table></div>",
        unsafe_allow_html=True,
    )


# PAGE 3 — EDA Insights & Business Intelligence

elif pg == "insights":  
    st.markdown("<div class='page-title'>📊 EDA Insights & Business Intelligence</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='page-sub'>"
        "A data-driven investigation into default behaviour across 167k+ loans "
        "from a Nigerian digital lender; uncovering the patterns that drive risk, "
        "revenue, and portfolio health."
        "</div>",
        unsafe_allow_html=True)

    # Business problem framing 
    st.markdown("""
    <div class='card' style='border-left:4px solid #2563eb;background:#f0f6ff'>
        <div class='card-title' style='color:#1d4ed8'>The Business Problem</div>
        <p style='font-size:14px;color:#1e3a5f;line-height:1.8;margin:0'>
            A Nigerian digital lender extends short-term installment loans to over
            <b>167 thousand unique borrowers;</b> many of whom have no formal credit
            history. With a portfolio default rate of <b>20.79%</b>, nearly 1 in 5
            loans ends in default. The challenge: <em>who should be approved, at
            what amount, and on what terms?</em> Without a reliable scoring system,
            the lender either leaves money on the table by declining good borrowers,
            or absorbs avoidable losses by approving bad ones. This analysis
            explores the data to find where that boundary lies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Insight 1 
    st.markdown("""
    <div class='card'>
        <div class='card-title'>Insight 1: The First Loan is the Riskiest Bet</div>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0 0 0.75rem 0'>
            The single most predictive feature in the entire dataset, with an Information Value of <b>0.596</b>; is how many loans a borrower has previously taken. 
            First-time borrowers (C1) default at <b>42.9%</b>, nearly <b>8× the rate</b> of borrowers on their 25th+ loan <b>(5.4%)</b>.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0 0 0.75rem 0'>
            This is not just a statistical pattern; it reflects a fundamental selection effect. Borrowers who repay their first loan are approved for a second. 
            Those who repay the second get a third. Each successive loan in a borrower's history is <em>evidence of creditworthiness</em> that no bureau score 
            can fully capture for first-timers.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0'>
            The implication is stark: <b>the entire C1 segment is structurally riskier</b>, and a single model mixing C1 and C2+ borrowers will systematically 
            underestimate risk for newcomers and over-restrict loyal customers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    fig1 = go.Figure()
    groups = ["C1", "C2-C3", "C4-C5", "C6-C11", "C12-C24", "C25+"]
    drs    = [0.4293, 0.2789, 0.1900, 0.1492, 0.0981, 0.0538]
    colors = ["#dc2626" if d > 0.30 else
              "#d97706" if d > 0.15 else "#059669" for d in drs]
    fig1.add_bar(x=groups, y=drs,
                 marker_color=colors,
                 text=[f"{d:.1%}" for d in drs],
                 textposition="outside",
                 textfont={"size": 11})
    fig1.add_hline(y=0.2079, line_dash="dot", line_color="#94a3b8",
                   annotation_text=" Portfolio avg 20.8%",
                   annotation_font={"size": 10})
    fig1.update_layout(
        title="Default Rate by Loan Sequence Group",
        height=260, margin=dict(t=40, b=10, l=0, r=0),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(tickformat=".0%", gridcolor="#f8fafc"),
        xaxis=dict(gridcolor="#f8fafc"),
        font=dict(family="Inter, sans-serif", size=11),
        showlegend=False,
    )
    st.plotly_chart(fig1, width="stretch",
                    config={"displayModeBar": False})
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("""
        <div style='background:#fef9c3;border-radius:8px;
                    padding:0.75rem 1rem;margin-top:0.5rem;
                    border-left:3px solid #ca8a04'>
            <b style='color:#92400e;font-size:12px'>💼 Business recommendation</b>
            <p style='font-size:12px;color:#78350f;margin:4px 0 0;line-height:1.6'>
                Treat C1 borrowers as a separate risk tier. Start with
                conservative limits and short tenures. Use the first loan
                as a qualifying test, not a profit centre. The data shows
                the real returns come from C4+ borrowers who have proven
                themselves. Invest in onboarding quality, not just volume.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Insight 2 
    st.markdown("""
    <div class='card'>
        <div class='card-title'>
            Insight 2: Borrowing at the Limit is a Default Signal
        </div>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0 0 0.75rem 0'>
            <b>73.8% of all loans</b> in the dataset were taken at exactly the borrower's approved credit limit a utilisation rate of 100%. These full-utilisation borrowers 
            default at <b>24.9%</b>, compared to just <b>9.2%</b> for those who borrow below their limit; a <b>2.7× difference</b>.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0 0 0.75rem 0'>
            This connects directly to first-time borrowers being the most likely to take their full limit. A first-time borrower who takes their full limit is in the 
            <b>highest-risk intersection</b> in the dataset.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0'>
            Why? Borrowers who take less than their limit signal financial discipline, they know what they can repay. Borrowers who max out their limit may be cash-stressed, 
            credit-hungry, or simply testing the system.
        </p>
    </div>
    """, unsafe_allow_html=True)

    fig2 = go.Figure(go.Bar(
        x=["Full utilisation<br>(util = 1.0)", "Under limit<br>(util < 1.0)"],
        y=[0.2488, 0.0922],
        marker_color=["#dc2626", "#059669"],
        text=["24.9%", "9.2%"],
        textposition="outside",
        textfont={"size": 13, "color": ["#dc2626", "#059669"]},
        width=0.5,
    ))
    fig2.update_layout(
        title="Default Rate by Credit Utilisation",
        height=260, margin=dict(t=40, b=10, l=0, r=10),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(tickformat=".0%", gridcolor="#f8fafc", range=[0, 0.35]),
        font=dict(family="Inter, sans-serif", size=11),
        showlegend=False,
    )
    st.plotly_chart(fig2, width="stretch",
                    config={"displayModeBar": False})
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("""
        <div style='background:#fef9c3;border-radius:8px;
                    padding:0.75rem 1rem;margin-top:0.5rem;
                    border-left:3px solid #ca8a04'>
            <b style='color:#92400e;font-size:12px'>💼 Business recommendation</b>
            <p style='font-size:12px;color:#78350f;margin:4px 0 0;line-height:1.6'>
                Introduce tiered limits rather than binary approve/decline.
                For first-time borrowers requesting their full limit, offer 70–80%
                of the requested amount. Monitor repayment behaviour before
                granting full access. Borrowers who voluntarily take less
                than their limit on subsequent loans are strong candidates
                for proactive limit increases.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Insight 3 
    st.markdown("""
    <div class='card'>
        <div class='card-title'>
            Insight 3:  Medium-Tenure Loans Have a Hidden Risk Spike
        </div>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0 0 0.75rem 0'>
            Loan tenure follows a <b>non-monotonic</b> relationship with default, not the linear pattern you might expect. Short loans (15 days) default at <b>15.5%</b>, 
            well below the portfolio average. But 31–60 day loans spike to <b>31.2%</b>; the highest of any tenure band before falling back toward 22–24% for longer tenures.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0'>
            This middle-tenure risk spike likely reflects a <em>borrower type mismatch</em>: short loans attract salary-advance borrowers who repay reliably 
            within one pay cycle. Medium-tenure loans attract borrowers who need multiple months to repay but may be overestimating their future income stability;
            a pattern common in emerging market informal employment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tenure_labels = ["15 days", "16–30 days", "31–60 days",
                     "61–90 days", "91–180 days", ">180 days"]
    tenure_drs    = [0.1553, 0.1955, 0.3124, 0.2383, 0.2237, 0.2446]
    tenor_colors  = ["#059669" if d < 0.20
                     else "#dc2626" if d > 0.28
                     else "#d97706" for d in tenure_drs]

    fig3 = go.Figure()
    fig3.add_bar(x=tenure_labels, y=tenure_drs,
                 marker_color=tenor_colors,
                 text=[f"{d:.1%}" for d in tenure_drs],
                 textposition="outside",
                 textfont={"size": 11})
    fig3.add_hline(y=0.2079, line_dash="dot", line_color="#94a3b8",
                   annotation_text=" Portfolio avg",
                   annotation_font={"size": 10})
    fig3.update_layout(
        title="Default Rate by Loan Tenure",
        height=260, margin=dict(t=40, b=10, l=0, r=0),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(tickformat=".0%", gridcolor="#f8fafc", range=[0, 0.40]),
        font=dict(family="Inter, sans-serif", size=11),
        showlegend=False,
    )
    st.plotly_chart(fig3, width="stretch",
                    config={"displayModeBar": False})
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("""
        <div style='background:#fef9c3;border-radius:8px;
                    padding:0.75rem 1rem;margin-top:0.5rem;
                    border-left:3px solid #ca8a04'>
            <b style='color:#92400e;font-size:12px'>💼 Business recommendation</b>
            <p style='font-size:12px;color:#78350f;margin:4px 0 0;line-height:1.6'>
                Apply stricter income verification and lower maximum amounts
                for 31–60 day products. Consider restructuring the product
                mix to push first-time borrowers toward shorter tenures where
                default rates are more manageable. A 30-day loan at
                a lower amount is a better first-loan offering than a
                60-day loan at the full limit.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # Insight 4 
    st.markdown("""
    <div class='card'>
        <div class='card-title'>
            Insight 4: Geography Encodes Structural Economic Risk
        </div>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0 0 0.75rem 0'>
            State of residence carries meaningful predictive power (IV 0.035), not because of geography per se, but because it proxies for
            <b>income stability, employment formality, and economic infrastructure</b>.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0 0 0.75rem 0'>
            Osun (27.8%), Anambra (26.8%), Abia (26.2%), and Cross River (26.1%) have the highest default rates,  all states with high informal sector employment and lower 
            average incomes relative to Lagos. Lagos at 18.6% sits near the portfolio average despite being the largest single state by volume (36% of all loans), 
            and Abuja at 16.1% is the safest, reflecting the civil servant-heavy population with stable government salaries.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0'>
            The insight is that <b>geography is a poverty proxy</b>
            when income data is self-reported and potentially
            unreliable. It adds signal precisely because it is
            harder to misreport than income.
        </p>
    </div>
    """, unsafe_allow_html=True)

    states  = ["Abuja", "Lagos", "Rivers", "Ogun", "Delta",
               "Oyo", "Kano", "Cross River", "Abia",
               "Anambra", "Osun"]
    st_drs  = [0.161, 0.186, 0.193, 0.200, 0.208,
               0.249, 0.223, 0.261, 0.262, 0.268, 0.278]
    st_clrs = ["#059669" if d < 0.19
               else "#dc2626" if d > 0.25
               else "#d97706" for d in st_drs]

    fig6 = go.Figure(go.Bar(
        x=st_drs, y=states, orientation="h",
        marker_color=st_clrs,
        text=[f"{d:.1%}" for d in st_drs],
        textposition="outside",
        textfont={"size": 10},
    ))
    fig6.add_vline(x=0.2079, line_dash="dot", line_color="#94a3b8",
                   annotation_text="Portfolio avg",
                   annotation_position="top",
                   annotation_font={"size": 9})
    fig6.update_layout(
        title="Default Rate — Top States by Volume",
        height=320, margin=dict(t=40, b=10, l=0, r=60),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(tickformat=".0%", gridcolor="#f8fafc",
                   range=[0, 0.34]),
        yaxis=dict(autorange="reversed"),
        font=dict(family="Inter, sans-serif", size=11),
        showlegend=False,
    )
    st.plotly_chart(fig6, width="stretch",
                    config={"displayModeBar": False})
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("""
        <div style='background:#fef9c3;border-radius:8px;
                    padding:0.75rem 1rem;margin-top:0.5rem;
                    border-left:3px solid #ca8a04'>
            <b style='color:#92400e;font-size:12px'>💼 Business recommendation</b>
            <p style='font-size:12px;color:#78350f;margin:4px 0 0;line-height:1.6'>
                Apply state-based limit modifiers: reduce maximum
                loan amounts in high-default states for first-time borrowers
                while keeping terms consistent for proven repeat borrowers.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    
    # Insight 5 
    st.markdown("""
    <div class='card'>
        <div class='card-title'>
            Insight 5: Employment Type Predicts Repayment Reliability
        </div>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0 0 0.75rem 0'>
            Employment status carries a 15% spread in default rates across categories, not because of income level (salary data is self-reported and unreliable)
            but because it proxies for <b>income regularity and predictability</b>.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0 0 0.75rem 0'>
            Retired borrowers default at just <b>9.4%</b>, the lowest of any group; reflecting pension income that is both stable and direct-deposited.
            Public servants (15.6%) and employed workers (18.7%) follow. Self-employed borrowers (24.4%) and students (31.2%) carry the highest risk, 
            with unemployed borrowers at 39.8% representing the most extreme case.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0'>
            This suggests that the key underwriting question is not <em>how much does the borrower earn</em> but <em>how reliably do they earn it</em>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    emp_labels = ["Retired", "Public Servant", "Employed",
                  "Self Employed", "Student", "Unemployed"]
    emp_drs    = [0.094, 0.156, 0.187, 0.244, 0.312, 0.398]
    e_clrs     = ["#059669" if d < 0.17
                  else "#dc2626" if d > 0.28
                  else "#d97706" for d in emp_drs]

    fig8 = go.Figure(go.Bar(
        x=emp_drs, y=emp_labels, orientation="h",
        marker_color=e_clrs,
        text=[f"{d:.1%}" for d in emp_drs],
        textposition="outside",
        textfont={"size": 11},
    ))
    fig8.add_vline(x=0.2079, line_dash="dot", line_color="#94a3b8",
                   annotation_text="Portfolio avg",
                   annotation_position="top right",
                   annotation_font={"size": 9})
    fig8.update_layout(
        title="Default Rate by Employment Status",
        height=260, margin=dict(t=40, b=10, l=0, r=60),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(tickformat=".0%", gridcolor="#f8fafc",
                   range=[0, 0.50]),
        yaxis=dict(autorange="reversed"),
        font=dict(family="Inter, sans-serif", size=11),
        showlegend=False,
    )
    st.plotly_chart(fig8, width="stretch",
                    config={"displayModeBar": False})
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("""
        <div style='background:#fef9c3;border-radius:8px;
                    padding:0.75rem 1rem;margin-top:0.5rem;
                    border-left:3px solid #ca8a04'>
            <b style='color:#92400e;font-size:12px'>💼 Business recommendation</b>
            <p style='font-size:12px;color:#78350f;margin:4px 0 0;line-height:1.6'>
                Weight employment type in underwriting independently
                of declared income. For self-employed borrowers,
                request bank statement verification to confirm
                income regularity; declared monthly income is
                much less reliable than 6 months of transaction
                history. For students, enforce shorter tenures
                and lower limits regardless of declared income.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Summary story 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card' style='border-left:4px solid #059669;
                              background:linear-gradient(135deg,
                              #f0fdf4 0%, #f8fafc 100%)'>
        <div class='card-title' style='color:#065f46'>
            What It All Means Together
        </div>
        <p style='font-size:13px;color:#1e3a5f;line-height:1.85'>
            These five insights are not isolated findings; they form
            a connected narrative about <b>who digital loan borrowers
            are in Nigeria and how they behave</b>.
        </p>
        <p style='font-size:13px;color:#1e3a5f;line-height:1.85'>
            The highest-risk borrower profile emerges clearly from
            the data: a <b>first-time applicant</b> who requests their 
            <b>full credit limit</b> on a <b>31–60 day loan</b>, 
             acquired via a <b>paid search ad</b>, who is
            <b>self-employed or a student</b>, lives in a 
            <b>high-default state</b>, and has either 
            <b>no bureau data or a thin file</b>.
            That borrower is not necessarily a bad person, 
            they may simply be in a genuinely precarious
            financial position.
        </p>
        <p style='font-size:13px;color:#1e3a5f;line-height:1.85'>
            Conversely, the safest borrower is a <b>returning
            high-cardinal customer</b> who borrows <b>below their limit</b>,
            on a <b>short tenure</b>, is <b>retired or a public servant</b>,
            found the product <b>organically or through referral</b>,
            and has a <b>clean multi-lender bureau history</b>.
            These borrowers default at under 5% and represent the
            core of a profitable, sustainable portfolio.
        </p>
        <p style='font-size:13px;color:#1e3a5f;line-height:1.85'>
            The gap between these two profiles is not luck;  it is
            the output of a deliberate lending strategy.
            The PD model built on this data can score any new
            applicant along this risk spectrum in real time,
            enabling a lender to price risk correctly, set
            appropriate limits, and grow the portfolio
            <b>without growing losses</b>.
        </p>
        <div style='display:flex;gap:2rem;margin-top:1rem;
                    padding-top:1rem;border-top:1px solid #d1fae5;
                    flex-wrap:wrap'>
            <div>
                <div style='font-size:11px;color:#6b7280;
                            text-transform:uppercase;letter-spacing:0.06em'>
                    Highest-risk segment
                </div>
                <div style='font-size:16px;font-weight:700;
                            color:#dc2626;margin-top:3px'>
                    C1 · Full util · 31-60d · Unknown channel
                </div>
                <div style='font-size:12px;color:#94a3b8;margin-top:2px'>
                    Estimated DR ~55–65%
                </div>
            </div>
            <div>
                <div style='font-size:11px;color:#6b7280;
                            text-transform:uppercase;letter-spacing:0.06em'>
                    Lowest-risk segment
                </div>
                <div style='font-size:16px;font-weight:700;
                            color:#059669;margin-top:3px'>
                    C25+ · Under limit · ≤30d · Organic / Referral
                </div>
                <div style='font-size:12px;color:#94a3b8;margin-top:2px'>
                    Estimated DR ~2–5%
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Data note 
    st.markdown("""
    <div style='text-align:center;padding:1rem;
                font-size:11px;color:#94a3b8;line-height:1.6'>
        Analysis based on over 167k resolved loans from a Nigerian digital
        lending platform over 2 years.
        All default rates represent resolved loans only
        (status: Paid, Default, or Paid-Default). Active and overdue
        loans excluded to prevent label contamination.
    </div>
    """, unsafe_allow_html=True)