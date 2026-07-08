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
        ("Methodology",      "📋", "methodology"),
    ]

    for label, icon, key in nav_items:
        is_active = st.session_state.page == key
        active_cls = "nav-active" if is_active else ""
        # Wrap in a div with the active class so CSS can target it
        st.markdown(f'<div class="{active_cls}">', unsafe_allow_html=True)
        if st.button(
            f"{icon}   {label}",
            key=f"nav_{key}",
            use_container_width=True,
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
            use_container_width=True,
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
                        fig, use_container_width=True,
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
                    <span class='info-val'>619,655</span>
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
                     use_container_width=True):
            st.session_state.seg = "C1"
            st.rerun()
    with seg_col2:
        if st.button("C2+ — returning",
                     type=("primary" if st.session_state.seg == "C2+"
                           else "secondary"),
                     use_container_width=True):
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
    st.plotly_chart(fig, use_container_width=True,
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


# PAGE 3 — METHODOLOGY
elif pg == "methodology":
    st.markdown("<div class='page-title'>Methodology</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='page-sub'>How the model was built, "
        "the decisions made at each stage, and known limitations.</div>",
        unsafe_allow_html=True)

    sections = [
        ("📦  Data pipeline", [
            "619,655 loans across 167,990 unique borrowers across 2 years.",
            "Three-layer dbt transformation pipeline: staging → intermediate → mart.",
            "DVC used for data versioning. Great Expectations for quality "
        ], ["PostgreSQL", "dbt-duckdb", "DuckDB", "DVC",
            "Great Expectations", "JSONB parsing"]),

        ("🎯  Target definition", [
            "Default (1): the loan is 90+ days past due or formally written off.",
            "Non-default (0): the loan is not in default.",
            "Threshold: 90+ days past due or formal write-off.",
            "Overall default rate: 20.79%.",
        ], []),

        ("✂️  Split strategy", [
            "Random user-level split 70/15/15 — all loans of a user stay "
            "in one split, preventing borrower-level leakage.",
            "Stratified by each user's majority default class to preserve "
            "class balance across splits.",
            "Temporal split was tested but produced a bureau coverage gap "
            "(40% train vs 3% val/test). Root cause: bureau data is indexed "
            "by user creation date, not loan date.",
            "Random split ensures consistent ~40% bureau coverage and ~20% "
            "default rate across all three sets.",
        ], ["StratifiedGroupKFold", "user_id grouping", "PSI monitoring"]),


        ("🤖  Model architecture", [
            "Two segments trained independently: C1 (first-time borrowers, "
            "DR 41%) and C2+ (returning borrowers, DR 15%).",
            "Training sequence: Logistic Regression baseline → Random Forest "
            "→ LightGBM with Optuna (150 trials, TPE sampler) → "
            "Soft-voting ensemble (equal-weight average of LR + RF + LightGBM).",
            "Cross-validation: StratifiedGroupKFold (n_splits=5, "
            "groups=user_id) within the training set only.",
            "Calibration: isotonic regression on the held-out validation set. "
            "All three constituent models calibrated before ensembling.",
            "All experiments tracked in MLflow with parameters, metrics, "
            "and SHAP artifacts.",
        ], ["LightGBM", "Optuna", "MLflow",  "Soft-voting ensemble"]),

        ("🚀  Inference at deployment", [
            "API routes to the C1 model when loan_number = 1, "
            "and to the C2+ model otherwise.",
            "All training transformations replicated at inference using "
            "saved encoding maps in inference_artifacts.json. "
            "Risk bands calibrated from validation-set score percentiles "
            "and stored in the same artifacts file.",
        ], ["FastAPI", "Streamlit", "joblib", "Railway"]),

        ("⚠️  Known limitations", [
            "Prior default rate has IV ≈ 0 due to the business rule "
            "preventing re-borrowing after default. The model uses "
            "prior loan count as a behavioural proxy.",
            "Random split does not simulate time-series deployment. "
            "C1 AUC 0.677 reflects the inherent difficulty of scoring "
            "thin-file first-time applicants with no internal history.",
        ], []),
    ]

    for title, bullets, tags in sections:
        li_items  = "".join(f"<li>{b}</li>" for b in bullets)
        tag_items = "".join(f"<span class='tag'>{t}</span>" for t in tags)
        tag_block = (f"<div class='tag-row'>{tag_items}</div>"
                     if tags else "")
        st.markdown(
            f"<div class='method-card'>"
            f"<div class='method-head'>{title}</div>"
            f"<ul class='method-body-list'>{li_items}</ul>"
            f"{tag_block}"
            f"</div>",
            unsafe_allow_html=True,
        )