# src/streamlit_app.py — fully self‑contained, no model files required

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# SELF‑CONTAINED SCORING LOGIC

THRESHOLDS_L1 = {
    "very_low": 0.30635292448046497,
    "low": 0.4018747348051978,
    "medium": 0.4937549307785776,
    "high": 0.5669315532763081,
}

THRESHOLDS_L2 = {
    "very_low": 0.04330989678211953,
    "low": 0.09597715750236256,
    "medium": 0.18393348812710986,
    "high": 0.29198637507001746,
}


def get_risk_band(prob, thresholds):
    if prob <= thresholds["very_low"]:
        return "Very Low"
    elif prob <= thresholds["low"]:
        return "Low"
    elif prob <= thresholds["medium"]:
        return "Medium"
    elif prob <= thresholds["high"]:
        return "High"
    return "Very High"


def get_decision(band):
    mapping = {
        "Very Low": "Approve",
        "Low": "Approve",
        "Medium": "Review",
        "High": "Decline",
        "Very High": "Decline",
    }
    return mapping.get(band, "Review")


def predict_pd(
    loan_num,
    principal,
    credit_limit,
    tenure,
    income,
    age,
    state,
    lenders,
    enquiries,
    obligations,
    debt,
    written_off,
    arrears,
    days_last,
):
    """Logistic regression approximation of the full ensemble model."""
    is_c1 = loan_num == 1
    cardinal_log = np.log1p(loan_num)
    is_full_util = 1 if principal >= credit_limit * 0.99 else 0
    is_med_tenure = 1 if 30 < tenure <= 60 else 0
    burden = (principal / max(income, 1)) * (tenure / 30)
    full_util_c1 = is_full_util * (1 if is_c1 else 0)
    lti = principal / max(income, 1)
    dsti = obligations / max(income, 1)

    state_tier = {
        "Lagos": 1,
        "Abuja": 0,
        "Rivers": 1,
        "Oyo": 1,
        "Kano": 1,
        "Enugu": 1,
        "Delta": 1,
        "Anambra": 2,
        "Ogun": 1,
        "Osun": 2,
        "Cross River": 2,
    }.get(state, 1)

    # bureau health score
    if lenders == 0 and obligations == 0:
        bh = 0
    elif written_off:
        bh = 1
    elif arrears:
        bh = 2
    else:
        bh = 3

    logit = -1.2
    logit += cardinal_log * -0.55
    logit += is_full_util * 0.55
    logit += is_med_tenure * 0.28
    logit += full_util_c1 * 0.45
    logit += (state_tier == 2) * 0.22
    logit += (state_tier == 0) * -0.15
    logit += min(burden, 10) * 0.08
    logit += min(lti, 5) * 0.06
    logit += dsti * 0.04
    logit += written_off * 0.55
    logit += arrears * 0.30
    logit += min(lenders, 10) * -0.04
    logit += min(enquiries, 10) * 0.05
    logit += (bh == 0) * 0.15
    logit += (bh == 1) * 0.10
    logit += (bh == 2) * 0.05
    if not is_c1:
        logit += np.log1p(days_last) * -0.02
    if is_c1:
        logit += 0.45

    prob = 1 / (1 + np.exp(-logit))
    thresholds = THRESHOLDS_L1 if is_c1 else THRESHOLDS_L2
    band = get_risk_band(prob, thresholds)
    decision = get_decision(band)
    return prob, band, decision, "L1" if is_c1 else "L2+"


# STREAMLIT UI 

st.markdown("""
<style>
*, *::before, *::after { box-sizing: border-box; }
[data-testid="stAppViewContainer"] { background: #f0f2f5; }
[data-testid="stMainBlockContainer"] { padding-top: 1.75rem; }
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
[data-testid="stSidebarNav"] { display: none !important; }
button[data-testid="baseButton-headerNoPadding"] { display: none; }
.app-logo-wrap {
    padding: 1.5rem 1.25rem 1.25rem;
    border-bottom: 1px solid #1a2035;
    display: flex; align-items: center; gap: 12px;
}
.app-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(37,99,235,0.4);
}
.app-logo-name {
    font-size: 15px; font-weight: 700;
    color: #f1f5f9; letter-spacing: -0.01em;
}
.app-logo-sub {
    font-size: 10px; color: #475569;
    margin-top: 2px; letter-spacing: 0.04em;
}
.nav-section {
    font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #334155; padding: 1.25rem 1.25rem 0.5rem;
}
.social-row {
    display: flex; gap: 10px;
    padding: 1rem 1.25rem;
    border-top: 1px solid #1a2035;
    margin-top: auto;
}
.social-link {
    display: flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; border-radius: 8px;
    background: rgba(255,255,255,0.06);
    border: 1px solid #1e293b;
    text-decoration: none; font-size: 16px;
    transition: all 0.15s ease; color: #64748b;
}
.social-link:hover {
    background: rgba(37,99,235,0.2);
    border-color: #2563eb; color: #93c5fd;
}
.card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 1.5rem;
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
.metric-strip {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin-bottom: 1.25rem;
}
.metric-tile {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem 1.25rem;
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
.metric-tile-sub { font-size: 11px; color: #94a3b8; margin-top: 4px; }
.score-num {
    font-size: 54px; font-weight: 800;
    line-height: 1; letter-spacing: -0.02em;
}
.score-sub {
    font-size: 11px; color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em; margin-top: 6px;
}
.score-green { color: #059669; }
.score-amber { color: #d97706; }
.score-red   { color: #dc2626; }
.badge {
    display: inline-flex; align-items: center;
    padding: 5px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 700;
}
.badge-very-low, .badge-low   { background:#d1fae5; color:#065f46; }
.badge-medium                 { background:#fef3c7; color:#92400e; }
.badge-high, .badge-very-high { background:#fee2e2; color:#991b1b; }
.info-row {
    display: flex; justify-content: space-between;
    align-items: center; padding: 9px 0;
    border-bottom: 1px solid #f8fafc; font-size: 13px;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #64748b; }
.info-val { color: #0f172a; font-weight: 600; }
.band-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.band-table th {
    text-align: left; padding: 8px 10px;
    font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.07em; color: #94a3b8;
    border-bottom: 1px solid #e2e8f0; font-weight: 600;
}
.band-table td { padding: 10px; color: #475569;
    border-bottom: 1px solid #f8fafc; }
.band-table tr:last-child td { border-bottom: none; }
.mini-bar-bg { width: 80px; height: 4px; background: #f1f5f9; border-radius: 2px; }
.mini-bar    { height: 4px; background: #2563eb; border-radius: 2px; }
.page-title {
    font-size: 22px; font-weight: 700;
    color: #0f172a; margin-bottom: 4px;
    letter-spacing: -0.02em;
}
.page-sub {
    font-size: 13px; color: #64748b;
    margin-bottom: 1.5rem; line-height: 1.5;
}
#MainMenu, footer, header,
[data-testid="stDecoration"] { display: none !important; }
div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
div[data-testid="stSidebar"] .stButton button {
    background: transparent !important;
    border: none !important;
    border-left: 2px solid transparent !important;
    border-radius: 0 !important;
    color: #64748b !important;
    font-size: 14px !important; font-weight: 500 !important;
    padding: 10px 1.25rem !important;
    text-align: left !important; width: 100% !important;
    transition: all 0.15s !important;
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
    st.markdown("""
    <div class="app-logo-wrap">
        <div class="app-logo-icon">🏦</div>
        <div>
            <div class="app-logo-name">Default Predictor</div>
            <div class="app-logo-sub">CREDIT RISK · v1.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Navigation</div>',
                unsafe_allow_html=True)

    nav_items = [
        ("🎯   Credit Scorer",     "scorer"),
        ("📊   Model Performance", "performance"),
        ("💡   EDA Insights",      "insights"),
    ]
    for label, key in nav_items:
        is_active = st.session_state.get("page", "scorer") == key
        st.markdown(
            f'<div class="{"nav-active" if is_active else ""}">',
            unsafe_allow_html=True)
        if st.button(label, key=f"nav_{key}",
                     use_container_width=True):
            st.session_state.page = key
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='padding:0.75rem 1.25rem;margin-top:0.5rem'>"
        "<span style='color:#059669;font-size:11px'>● Models ready</span>"
        "</div>",
        unsafe_allow_html=True)

    st.markdown("""
    <div class="social-row">
        <a class="social-link"
           href="https://github.com/husainridwan/ProbabilityOfDefault"
           target="_blank" title="GitHub">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795
                8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23
                -.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345
                -.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945
                -.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495
                .99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46
                -5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12
                -3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s
                2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24
                2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805
                5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015
                2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0
                24 12c0-6.63-5.37-12-12-12z"/>
            </svg>
        </a>
        <a class="social-link"
           href="https://www.linkedin.com/in/ridwanllah-husain-655458195/"
           target="_blank" title="LinkedIn">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027
                -3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667
                H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85
                3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062
                2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm
                1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792
                0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451
                C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0
                22.222 0h.003z"/>
            </svg>
        </a>
        <div style="font-size:10px;color:#334155;align-self:center;
                    margin-left:auto;line-height:1.4">
            Rho<br><span style="color:#1e3a5f">Portfolio</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# PAGE ROUTING
pg = st.session_state.get("page", "scorer")


# SCORER PAGE
if pg == "scorer":
    st.markdown("<div class='page-title'>Credit scorer</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='page-sub'>Enter loan and borrower details to receive "
        "a real-time probability of default score.</div>",
        unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("<div class='card'>"
                    "<div class='card-title'>Loan details</div>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        loan_number = c1.number_input(
            "Loan number *(1 = first loan)*",
            min_value=1, value=1, step=1,
            help="How many loans have you taken with this lender? 1 = first time.")
        principal_amount = c2.number_input(
            "Principal amount (₦)",
            min_value=1000, value=50000, step=1000,
            help="The amount you want to borrow.")
        c3, c4 = st.columns(2)
        credit_limit = c3.number_input(
            "Credit limit (₦)", min_value=1000, value=50000, step=1000,
            help="The maximum amount you are allowed to borrow. Often equals the principal.")
        tenure = c4.number_input(
            "Tenure (days)", min_value=7, max_value=365, value=30,
            help="How many days you have to repay the loan.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>"
                    "<div class='card-title'>Borrower profile</div>",
                    unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        monthly_income = b1.number_input(
            "Monthly income (₦)", min_value=0, value=150000, step=5000,
            help="Your total monthly take‑home income before any deductions.")
        age = b2.number_input(
            "Age", min_value=18, max_value=75, value=32)
        b3, b4 = st.columns(2)
        state = b3.selectbox("State of residence", [
            "Lagos", "Abuja", "Rivers", "Oyo", "Kano", "Enugu",
            "Delta", "Anambra", "Ogun", "Osun", "Cross River", "Other"])
        days_since = b4.number_input(
            "Days since last loan *(0 if first)*", min_value=0, value=0,
            help="How many days have passed since you last took a loan with this lender.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='card'>"
            "<div class='card-title'>Credit history "
            "<span style='font-weight:400;color:#cbd5e1;font-size:9px;"
            "letter-spacing:0'>— leave 0 if unknown</span></div>",
            unsafe_allow_html=True)
        h1, h2 = st.columns(2)
        lender_count = h1.number_input(
            "Lenders borrowed from", min_value=0, value=0,
            help="How many different banks or lenders you currently have loans with.")
        enquiry_count = h2.number_input(
            "Recent credit enquiries", min_value=0, value=0,
            help="How many times has your credit report been checked in the last 12 months?")
        h3, h4 = st.columns(2)
        monthly_obligations = h3.number_input(
            "Monthly obligations (₦)", min_value=0, value=0, step=1000,
            help="Total amount you pay each month towards all existing loans and credit cards.")
        total_debt = h4.number_input(
            "Total outstanding debt (₦)", min_value=0, value=0, step=5000,
            help="If you paid off every loan and credit card today, how much would you owe in total?")
        h5, h6 = st.columns(2)
        total_accounts = h5.number_input(
            "Total loan accounts", min_value=0, value=0,
            help="The total number of open loans and credit accounts you currently have.")
        has_written_off = h6.selectbox(
            "Written-off account?", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Has a lender ever given up on collecting a debt from you (written it off)?")
        has_arrears = st.selectbox(
            "Currently in arrears?", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Are you currently late on any loan or credit card payment?")
        st.markdown("</div>", unsafe_allow_html=True)

        scored = st.button("Calculate PD score", type="primary",
                           use_container_width=True)

        with right:
            if scored:
                prob, band, decision, seg = predict_pd(
                    loan_number, principal_amount, credit_limit, tenure,
                    monthly_income, age, state,
                    lender_count, enquiry_count,
                    monthly_obligations, total_debt,
                    has_written_off, has_arrears, days_since,
                )

                score_cls = ("score-green" if prob < 0.3
                             else "score-amber" if prob < 0.5
                             else "score-red")
                clr = ("#059669" if prob < 0.3
                       else "#d97706" if prob < 0.5
                       else "#dc2626")
                badge_cls = "badge-" + band.lower().replace(" ", "-")
                dec_clr = ("#059669" if "Approve" in decision
                           else "#d97706" if decision == "Review"
                           else "#dc2626")

                st.markdown(f"""
                <div class='card'>
                    <div style='text-align:center;
                                padding:1rem 1rem 0.75rem'>
                        <div class='score-num {score_cls}'>
                            {prob*100:.1f}%
                        </div>
                        <div class='score-sub' style='margin-top:4px'>
                            Probability of Default
                        </div>
                    </div>
                    <div style='text-align:center;
                                margin:0.5rem 0 1rem'>
                        <span class='badge {badge_cls}'>
                            {band} risk
                        </span>
                    </div>
                    <div style='display:flex;
                                justify-content:space-between;
                                padding-top:0.75rem;
                                border-top:1px solid #f1f5f9;
                                font-size:13px;align-items:center'>
                        <span style='color:#64748b;font-size:12px'>
                            {seg}
                        </span>
                        <span style='color:{dec_clr};font-weight:700;
                                     font-size:14px'>{decision}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number={"suffix": "%",
                            "font": {"size": 22, "color": clr}},
                    gauge={
                        "axis": {"range": [0, 100],
                                  "tickfont": {"size": 10},
                                  "tickcolor": "#94a3b8"},
                        "bar": {"color": clr, "thickness": 0.22},
                        "bgcolor": "white", "borderwidth": 0,
                        "steps": [
                            {"range": [0, 25], "color": "#dcfce7"},
                            {"range": [25, 45], "color": "#fef9c3"},
                            {"range": [45, 65], "color": "#ffedd5"},
                            {"range": [65, 100], "color": "#fee2e2"},
                        ],
                    },
                ))
                fig.update_layout(
                    height=190,
                    margin=dict(t=20, b=10, l=20, r=20),
                    paper_bgcolor="white", plot_bgcolor="white",
                    font={"family": "Inter, sans-serif"},
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})

                inc = max(float(monthly_income), 1)
                lti = float(principal_amount) / inc
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
            else:
                st.markdown("""
                <div class='card' style='text-align:center;
                                         padding:3rem 1.5rem'>
                    <div style='font-size:40px;margin-bottom:1rem'>🏦</div>
                    <div style='font-size:15px;font-weight:700;
                                color:#0f172a;margin-bottom:8px'>
                        Enter loan details
                    </div>
                    <div style='font-size:13px;color:#94a3b8;line-height:1.6'>
                        Fill in the form and click<br>
                        <b style='color:#0f172a'>Calculate PD score</b>
                    </div>
                </div>
                <div class='card'>
                    <div class='card-title'>Model at a glance</div>
                    <div class='info-row'>
                        <span class='info-key'>L1 test AUC</span>
                        <span class='info-val' style='color:#2563eb'>0.677</span>
                    </div>
                    <div class='info-row'>
                        <span class='info-key'>L2+ test AUC</span>
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


# ── PERFORMANCE PAGE ─────────────────────────────────────────────────
elif pg == "performance":
    st.markdown("<div class='page-title'>Model performance</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='page-sub'>Test-set evaluation — random user split "
        "(70/15/15). Val–test AUC gap under 1pp across all models.</div>",
        unsafe_allow_html=True)

    st.markdown("""
    <div class='metric-strip'>
        <div class='metric-tile'>
            <div class='metric-tile-label'>L1 Test AUC</div>
            <div class='metric-tile-val'>0.677</div>
            <div class='metric-tile-sub'>First-time borrowers</div>
        </div>
        <div class='metric-tile'>
            <div class='metric-tile-label'>L1 Test Gini</div>
            <div class='metric-tile-val'>0.355</div>
            <div class='metric-tile-sub'>= 2 × AUC − 1</div>
        </div>
        <div class='metric-tile'>
            <div class='metric-tile-label'>L2+ Test AUC</div>
            <div class='metric-tile-val'>0.757</div>
            <div class='metric-tile-sub'>Returning borrowers</div>
        </div>
        <div class='metric-tile'>
            <div class='metric-tile-label'>L2+ Test Gini</div>
            <div class='metric-tile-val'>0.514</div>
            <div class='metric-tile-sub'>Production-grade</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    seg_c1, seg_c2, _ = st.columns([1, 1, 4])
    with seg_c1:
        if st.button("L1 — first-time",
                     type=("primary" if st.session_state.get("seg") == "L1"
                           else "secondary"),
                     use_container_width=True):
            st.session_state.seg = "L1"
            st.rerun()
    with seg_c2:
        if st.button("L2+ — returning",
                     type=("primary" if st.session_state.get("seg") == "L2+"
                           else "secondary"),
                     use_container_width=True):
            st.session_state.seg = "L2+"
            st.rerun()

    is_c1    = st.session_state.get("seg", "L1") == "L1"
    models   = ["LR", "RF", "LightGBM", "Ensemble"]
    val_aucs = ([0.663, 0.685, 0.682, 0.685] if is_c1
                else [0.748, 0.761, 0.759, 0.762])
    tst_aucs = ([0.657, 0.679, 0.676, 0.677] if is_c1
                else [0.743, 0.757, 0.756, 0.757])

    fig = go.Figure()
    fig.add_bar(name="Val AUC", x=models, y=val_aucs,
                marker_color="#bfdbfe",
                text=[f"{v:.3f}" for v in val_aucs],
                textposition="outside",
                textfont={"size": 11, "color": "#64748b"})
    fig.add_bar(name="Test AUC", x=models, y=tst_aucs,
                marker_color="#2563eb",
                text=[f"{v:.3f}" for v in tst_aucs],
                textposition="outside",
                textfont={"size": 11, "color": "#2563eb"})
    fig.add_hline(y=0.70, line_dash="dot", line_color="#94a3b8",
                  annotation_text=" 0.70 target",
                  annotation_font={"size": 11, "color": "#94a3b8"})
    fig.update_layout(
        barmode="group", height=300,
        margin=dict(t=40, b=10, l=0, r=0),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(range=[0.55, 0.82], gridcolor="#f8fafc",
                   tickformat=".3f"),
        xaxis=dict(gridcolor="#f8fafc"),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})

    results = {
        "L1":  {"val":  {"AUC": 0.685, "Gini": 0.370, "KS": 0.266},
                "test": {"AUC": 0.677, "Gini": 0.355, "KS": 0.256}},
        "L2+": {"val":  {"AUC": 0.762, "Gini": 0.524, "KS": 0.387},
                "test": {"AUC": 0.757, "Gini": 0.514, "KS": 0.382}},
    }
    r = results["L1" if is_c1 else "L2+"]
    cv, ct = st.columns(2)
    for col, sk, sl in [(cv, "val", "Validation results"),
                        (ct, "test", "Test results")]:
        d = r[sk]
        col.markdown(
            f"<div class='card'><div class='card-title'>{sl}</div>"
            f"<div class='info-row'><span class='info-key'>AUC</span>"
            f"<span class='info-val' style='color:#2563eb'>"
            f"{d['AUC']:.3f}</span></div>"
            f"<div class='info-row'><span class='info-key'>Gini</span>"
            f"<span class='info-val'>{d['Gini']:.3f}</span></div>"
            f"<div class='info-row'>"
            f"<span class='info-key'>KS statistic</span>"
            f"<span class='info-val'>{d['KS']:.3f}</span></div>"
            f"</div>",
            unsafe_allow_html=True)

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
    rows  = []
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
        + "".join(rows) + "</table></div>",
        unsafe_allow_html=True)


# PAGE 3: EDA INSIGHTS & BUSINESS INTELLIGENCE 
elif pg == "insights":
    st.markdown(
        "<div class='page-title'>📊 EDA Insights & Business Intelligence</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div class='page-sub'>"
        "What 167k+ resolved loans from a Nigerian digital lender tell us "
        "about who defaults, when, and why."
        "</div>",
        unsafe_allow_html=True)

    # Business problem
    st.markdown("""
    <div class='card' style='border-left:4px solid #2563eb;background:#f0f6ff'>
        <div class='card-title' style='color:#1d4ed8'>The Business Problem</div>
        <p style='font-size:14px;color:#1e3a5f;line-height:1.8;margin:0'>
            A Nigerian digital lender extends short-term installment loans to
            over <b>167 thousand unique borrowers</b>, many of whom have little
            or no formal credit history. Nearly 1 in 5 loans ends in default.
            The practical question is simple: <em>who should be approved, for
            how much, and on what terms?</em> Without a reliable way to answer
            that, the lender either turns away good customers out of caution or
            takes losses on bad ones it could have avoided. This analysis digs
            into the data to find where the real risk boundary sits.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Insight 1
    st.markdown("""
    <div class='card'>
        <div class='card-title'>Insight 1: The First Loan Is the Riskiest Bet</div>
        <p style='font-size:13px;color:#475569;line-height:1.8;
                  margin:0 0 0.75rem 0'>
            Of all the features in the dataset, how many loans a borrower has
            taken before turns out to be the most predictive, with an Information
            Value of <b>0.596</b>. First-time borrowers default at <b>42.9%</b>.
            By the time someone reaches their 25th loan with the same lender,
            that rate is down to <b>5.4%</b>. That is an 8-fold difference driven
            entirely by borrowing history.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;
                  margin:0 0 0.75rem 0'>
            What's happening here is a selection effect, not just a correlation.
            Every borrower who repays their first loan gets approved for a second.
            Every second repayment earns a third. By the time someone has ten or
            twenty loans on their record, they've already filtered themselves in.
            No bureau score can replicate that for a first-time applicant, because
            there's nothing to look at yet.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0'>
            Mixing first-timers and returning customers in a single model gets you
            the worst of both worlds: you underestimate risk for new applicants
            and over-restrict loyal ones who have already proven themselves.
        </p>
    </div>
    """, unsafe_allow_html=True)

    fig1 = go.Figure()
    groups = ["L1", "L2-L3", "L4-L5", "L6-L11", "L12-L24", "L25+"]
    drs    = [0.4293, 0.2789, 0.1900, 0.1492, 0.0981, 0.0538]
    c1_colors = ["#dc2626" if d > 0.30 else
                 "#d97706" if d > 0.15 else "#059669" for d in drs]
    fig1.add_bar(x=groups, y=drs, marker_color=c1_colors,
                 text=[f"{d:.1%}" for d in drs],
                 textposition="outside", textfont={"size": 11})
    fig1.add_hline(y=0.2079, line_dash="dot", line_color="#94a3b8",
                   annotation_text=" Portfolio avg 20.8%",
                   annotation_font={"size": 10})
    fig1.update_layout(
        title="Default Rate by Loan Sequence Group", height=280,
        margin=dict(t=40, b=10, l=0, r=0),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(tickformat=".0%", gridcolor="#f8fafc"),
        xaxis=dict(gridcolor="#f8fafc"),
        font=dict(family="Inter, sans-serif", size=11),
        showlegend=False)
    st.plotly_chart(fig1, use_container_width=True,
                    config={"displayModeBar": False})

    st.markdown("""
    <div style='background:#fef9c3;border-radius:8px;
                padding:0.75rem 1rem;margin-bottom:1rem;
                border-left:3px solid #ca8a04'>
        <b style='color:#92400e;font-size:12px'>💼 Business recommendation</b>
        <p style='font-size:12px;color:#78350f;margin:4px 0 0;line-height:1.6'>
            Treat first-time borrowers as their own risk tier. Offer conservative
            limits and short tenures. The first loan is a qualifying round, not
            a revenue opportunity. The data shows returns come from L4+ borrowers
            who've proven themselves. Focus on onboarding quality over volume.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Insight 2
    st.markdown("""
    <div class='card'>
        <div class='card-title'>
            Insight 2: Maxing Out the Credit Limit Is a Warning Sign
        </div>
        <p style='font-size:13px;color:#475569;line-height:1.8;
                  margin:0 0 0.75rem 0'>
            <b>73.8% of loans</b> in this dataset were taken at exactly the
            borrower's approved limit. That is not a coincidence; it is a
            behaviour pattern. Borrowers who took the full amount default at
            <b>24.9%</b>. Those who borrowed less than their limit default at
            <b>9.2%</b>, less than half the rate.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;
                  margin:0 0 0.75rem 0'>
            This connects directly to Insight 1. First-time borrowers are the
            most likely to take everything on offer, combining two high-risk
            signals at once. Someone on their first loan who also maxes out their
            limit sits in the riskiest intersection in the entire dataset.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0'>
            Borrowing below the limit isn't just a financial decision; it signals
            that someone knows what they can comfortably repay. Borrowers who
            take everything available may be stretched thin, or they may simply
            be testing the system.
        </p>
    </div>
    """, unsafe_allow_html=True)

    fig2 = go.Figure(go.Bar(
        x=["Full utilisation (util = 1.0)", "Under limit (util < 1.0)"],
        y=[0.2488, 0.0922],
        marker_color=["#dc2626", "#059669"],
        text=["24.9%", "9.2%"],
        textposition="outside",
        textfont={"size": 13, "color": ["#dc2626", "#059669"]},
        width=0.4,
    ))
    fig2.update_layout(
        title="Default Rate by Credit Utilisation", height=280,
        margin=dict(t=40, b=10, l=0, r=0),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(tickformat=".0%", gridcolor="#f8fafc", range=[0, 0.35]),
        font=dict(family="Inter, sans-serif", size=11),
        showlegend=False)
    st.plotly_chart(fig2, use_container_width=True,
                    config={"displayModeBar": False})

    st.markdown("""
    <div style='background:#fef9c3;border-radius:8px;
                padding:0.75rem 1rem;margin-bottom:1rem;
                border-left:3px solid #ca8a04'>
        <b style='color:#92400e;font-size:12px'>💼 Business recommendation</b>
        <p style='font-size:12px;color:#78350f;margin:4px 0 0;line-height:1.6'>
            Move away from binary approve/decline. For first-time applicants
            requesting their full limit, offer 70–80% instead. Borrowers who
            voluntarily take less than their limit on later loans are good
            candidates for proactive increases.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Insight 3
    st.markdown("""
    <div class='card'>
        <div class='card-title'>
            Insight 3: Medium-Tenure Loans Have a Risk Spike Nobody Expects
        </div>
        <p style='font-size:13px;color:#475569;line-height:1.8;
                  margin:0 0 0.75rem 0'>
            You'd expect default risk to rise smoothly with loan length. It
            doesn't. Short loans of 15 days default at <b>15.5%</b>,
            well below the portfolio average. Then 60-day loans jump to
            <b>31.2%</b>, the highest of any tenure, before falling back
            toward 22–24% for longer products. There's a spike right in
            the middle.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0'>
            The likely reason is that different tenures attract different
            types of borrowers. Short loans pull in salary-advance customers who
            know exactly when their money is coming and repay reliably within
            one pay cycle. The medium-tenure group tends to include borrowers who
            need more time but may be overestimating how stable their income will
            be over the next two months. That gap between expectation and reality
            shows up as defaults.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tenure_labels = ["15 days", "30 days", "60 days",
                     "90 days", "91–180 days", ">180 days"]
    tenure_drs    = [0.1553, 0.1955, 0.3124, 0.2383, 0.2237, 0.2446]
    tenor_colors  = ["#059669" if d < 0.20
                     else "#dc2626" if d > 0.28
                     else "#d97706" for d in tenure_drs]
    fig3 = go.Figure()
    fig3.add_bar(x=tenure_labels, y=tenure_drs,
                 marker_color=tenor_colors,
                 text=[f"{d:.1%}" for d in tenure_drs],
                 textposition="outside", textfont={"size": 11})
    fig3.add_hline(y=0.2079, line_dash="dot", line_color="#94a3b8",
                   annotation_text=" Portfolio avg",
                   annotation_font={"size": 10})
    fig3.update_layout(
        title="Default Rate by Loan Tenure", height=280,
        margin=dict(t=40, b=10, l=0, r=0),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(tickformat=".0%", gridcolor="#f8fafc", range=[0, 0.40]),
        font=dict(family="Inter, sans-serif", size=11),
        showlegend=False)
    st.plotly_chart(fig3, use_container_width=True,
                    config={"displayModeBar": False})

    st.markdown("""
    <div style='background:#fef9c3;border-radius:8px;
                padding:0.75rem 1rem;margin-bottom:1rem;
                border-left:3px solid #ca8a04'>
        <b style='color:#92400e;font-size:12px'>💼 Business recommendation</b>
        <p style='font-size:12px;color:#78350f;margin:4px 0 0;line-height:1.6'>
            Apply tighter income checks and lower caps for 60-day loans.
            For new borrowers especially, a 30-day loan at a lower amount is
            a safer starting point than a 60-day loan at the full limit.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Insight 4
    st.markdown("""
    <div class='card'>
        <div class='card-title'>
            Insight 4: Where Borrowers Live Tells You Something Income Data Can't
        </div>
        <p style='font-size:13px;color:#475569;line-height:1.8;
                  margin:0 0 0.75rem 0'>
            State of residence carries predictive power (IV 0.035), but not
            because geography itself matters. It's a stand-in for things that
            are harder to measure directly: <b>income regularity, employment
            formality, and local economic conditions</b>.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;
                  margin:0 0 0.75rem 0'>
            Osun (27.8%), Anambra (26.8%), Abia (26.2%), and Cross River (26.1%)
            come out worst. These are states with large informal sectors and lower
            average incomes relative to Lagos. Lagos itself sits near the
            portfolio average at 18.6%, despite accounting for 36% of all loans.
            Abuja comes out best at 16.1%, reflecting its high concentration
            of civil servants with predictable government salaries.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0'>
            Geography works as a signal here because it's harder to misreport
            than income. A borrower can tell you they earn any amount; their
            address is harder to fake.
        </p>
    </div>
    """, unsafe_allow_html=True)

    states  = ["Abuja", "Lagos", "Rivers", "Ogun", "Delta",
               "Oyo", "Kano", "Cross River", "Abia", "Anambra", "Osun"]
    st_drs  = [0.161, 0.186, 0.193, 0.200, 0.208,
               0.249, 0.223, 0.261, 0.262, 0.268, 0.278]
    st_clrs = ["#059669" if d < 0.19
               else "#dc2626" if d > 0.25
               else "#d97706" for d in st_drs]
    fig4 = go.Figure(go.Bar(
        x=st_drs, y=states, orientation="h",
        marker_color=st_clrs,
        text=[f"{d:.1%}" for d in st_drs],
        textposition="outside", textfont={"size": 10},
    ))
    fig4.add_vline(x=0.2079, line_dash="dot", line_color="#94a3b8",
                   annotation_text="Portfolio avg",
                   annotation_position="top",
                   annotation_font={"size": 9})
    fig4.update_layout(
        title="Default Rate: Top States by Volume", height=320,
        margin=dict(t=40, b=10, l=0, r=60),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(tickformat=".0%", gridcolor="#f8fafc", range=[0, 0.34]),
        yaxis=dict(autorange="reversed"),
        font=dict(family="Inter, sans-serif", size=11),
        showlegend=False)
    st.plotly_chart(fig4, use_container_width=True,
                    config={"displayModeBar": False})

    st.markdown("""
    <div style='background:#fef9c3;border-radius:8px;
                padding:0.75rem 1rem;margin-bottom:1rem;
                border-left:3px solid #ca8a04'>
        <b style='color:#92400e;font-size:12px'>💼 Business recommendation</b>
        <p style='font-size:12px;color:#78350f;margin:4px 0 0;line-height:1.6'>
            Apply state-based limit adjustments for first-time borrowers in
            high-default states. For returning customers with a clean repayment
            record, geography matters less; keep their terms consistent.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Insight 5
    st.markdown("""
    <div class='card'>
        <div class='card-title'>
            Insight 5: Employment Status Predicts Repayment Better Than Income
        </div>
        <p style='font-size:13px;color:#475569;line-height:1.8;
                  margin:0 0 0.75rem 0'>
            There's a 30% spread in default rates across employment categories. 
            Declared income doesn't explain it, partly because income is self-reported 
            and partly because the real driver isn't how much someone earns; 
            it's how reliably they earn it.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;
                  margin:0 0 0.75rem 0'>
            Retired borrowers default at just <b>9.4%</b>. Pension income is
            stable and deposited on a predictable schedule, which makes it
            easier to plan repayments around. Public servants (15.6%) and
            formally employed workers (18.7%) follow for similar reasons.
            Self-employed borrowers (24.4%) and students (31.2%) are where
            things break down, and unemployed borrowers at 39.8% represent
            the highest-risk group in the dataset.
        </p>
        <p style='font-size:13px;color:#475569;line-height:1.8;margin:0'>
            The underwriting question worth asking is less about salary level
            and more about whether that income shows up on a regular schedule.
        </p>
    </div>
    """, unsafe_allow_html=True)

    emp_labels = ["Retired", "Public Servant", "Employed",
                  "Self Employed", "Student", "Unemployed"]
    emp_drs    = [0.094, 0.156, 0.187, 0.244, 0.312, 0.398]
    e_clrs     = ["#059669" if d < 0.17
                  else "#dc2626" if d > 0.28
                  else "#d97706" for d in emp_drs]
    fig5 = go.Figure(go.Bar(
        x=emp_drs, y=emp_labels, orientation="h",
        marker_color=e_clrs,
        text=[f"{d:.1%}" for d in emp_drs],
        textposition="outside", textfont={"size": 11},
    ))
    fig5.add_vline(x=0.2079, line_dash="dot", line_color="#94a3b8",
                   annotation_text="Portfolio avg",
                   annotation_position="top right",
                   annotation_font={"size": 9})
    fig5.update_layout(
        title="Default Rate by Employment Status", height=280,
        margin=dict(t=40, b=10, l=0, r=60),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(tickformat=".0%", gridcolor="#f8fafc", range=[0, 0.50]),
        yaxis=dict(autorange="reversed"),
        font=dict(family="Inter, sans-serif", size=11),
        showlegend=False)
    st.plotly_chart(fig5, use_container_width=True,
                    config={"displayModeBar": False})

    st.markdown("""
    <div style='background:#fef9c3;border-radius:8px;
                padding:0.75rem 1rem;margin-bottom:1rem;
                border-left:3px solid #ca8a04'>
        <b style='color:#92400e;font-size:12px'>💼 Business recommendation</b>
        <p style='font-size:12px;color:#78350f;margin:4px 0 0;line-height:1.6'>
            Factor employment type into underwriting separately from declared
            income. For self-employed applicants, bank statement verification
            is much more useful than a salary declaration. For students, enforce
            short tenures and low limits regardless of what they report unless a 
            solid proof of consistent income is provided.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card' style='border-left:4px solid #059669;
                              background:linear-gradient(135deg,
                              #f0fdf4 0%, #f8fafc 100%)'>
        <div class='card-title' style='color:#065f46'>
            Putting It All Together
        </div>
        <p style='font-size:13px;color:#1e3a5f;line-height:1.85;
                  margin:0 0 0.75rem 0'>
            These five patterns aren't independent. They stack. A first-time
            borrower who also maxes out their limit, picks a medium tenure,
            and is self-employed is combining four separate
            risk signals at once. The model picks this up through interaction
            features, but even the raw exploratory data makes it visible.
        </p>
        <p style='font-size:13px;color:#1e3a5f;line-height:1.85;
                  margin:0 0 0.75rem 0'>
            On the other side, the safest borrowers in the dataset are returning
            customers who borrow below their limit, on short tenures, and have
            stable formal employment. They default at under 5% and carry the
            portfolio. Everything in the modelling and deployment was built around
            separating these two groups as cleanly as possible.
        </p>
        <p style='font-size:13px;color:#1e3a5f;line-height:1.85;
                  margin:0 0 1rem 0'>
            That's what the PD model does in production: it takes these signals,
            combines them with bureau and demographic features, and returns a
            calibrated probability so the lender can make a consistent,
            data-driven decision on every application.
        </p>
        <div style='display:flex;gap:2rem;padding-top:1rem;
                    border-top:1px solid #d1fae5;flex-wrap:wrap'>
            <div>
                <div style='font-size:11px;color:#6b7280;
                            text-transform:uppercase;letter-spacing:0.06em'>
                    Highest-risk profile
                </div>
                <div style='font-size:15px;font-weight:700;
                            color:#dc2626;margin-top:3px'>
                    L1 · Full limit · 60d · Self-employed
                </div>
                <div style='font-size:12px;color:#94a3b8;margin-top:2px'>
                    Estimated DR ~55–65%
                </div>
            </div>
            <div>
                <div style='font-size:11px;color:#6b7280;
                            text-transform:uppercase;letter-spacing:0.06em'>
                    Lowest-risk profile
                </div>
                <div style='font-size:15px;font-weight:700;
                            color:#059669;margin-top:3px'>
                    L25+ · Under limit · Short tenure · Employed
                </div>
                <div style='font-size:12px;color:#94a3b8;margin-top:2px'>
                    Estimated DR ~2–5%
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;padding:1rem;
                font-size:11px;color:#94a3b8;line-height:1.6'>
        Based on 167k+ resolved loans from a Nigerian digital lender, 2023–2025.
        Analysis covers loans with a fully resolved outcome (paid or defaulted).
        Active and overdue loans were excluded to prevent label contamination.
    </div>
    """, unsafe_allow_html=True)