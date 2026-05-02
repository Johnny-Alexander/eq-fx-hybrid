"""Streamlit app for interactive EQ/FX hybrid and double-digital pricing.

Run locally:
    streamlit run app/app.py

Run on local network (accessible from phone on same WiFi):
    streamlit run app/app.py --server.address 0.0.0.0

Deploy to Streamlit Cloud:
    Push to GitHub, then connect at https://share.streamlit.io
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import replace
from scipy.stats import norm

from src.hybrid_pricer import HybridInputs, price_hybrid_call, bivariate_normal_cdf

# Suppress mathtext parsing in matplotlib labels (so $ shows as literal $)
plt.rcParams["text.parse_math"] = False

st.set_page_config(
    page_title="EQ/FX Hybrid Pricer",
    layout="wide",
    initial_sidebar_state="collapsed",  # collapsed by default - better for mobile
)

# Detect approximate viewport via query param hack — fallback to wide layout
# Mobile users can expand the sidebar with the > arrow

# Compact CSS for mobile readability
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.1rem; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem; }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 1.0rem !important; }
</style>
""", unsafe_allow_html=True)

# ----- Sidebar inputs -----
st.sidebar.header("Trade structure")

payoff_type = st.sidebar.radio(
    "Payoff type",
    ["Hybrid: SPX call x 1{X>B}", "Double digital: 1{S>K AND X>B}"],
    index=0,
)

notional = st.sidebar.number_input(
    "Notional ($mm)", min_value=1.0, max_value=1000.0, value=100.0, step=10.0
) * 1e6

st.sidebar.subheader("SPX leg")
S0 = st.sidebar.number_input("SPX spot", min_value=1000.0, max_value=20000.0,
                              value=7200.0, step=50.0)
K = st.sidebar.number_input("SPX strike K", min_value=1000.0, max_value=20000.0,
                             value=7200.0, step=50.0)
sig_S = st.sidebar.slider("SPX vol (sig_S)", 0.05, 0.50, 0.17, 0.005,
                          format="%.3f")
q = st.sidebar.slider("SPX dividend yield (q)", 0.00, 0.05, 0.015, 0.001,
                      format="%.3f")

st.sidebar.subheader("FX leg (USDJPY = JPY per USD)")
X0 = st.sidebar.number_input("USDJPY spot", min_value=50.0, max_value=300.0,
                              value=156.0, step=0.5)
B = st.sidebar.number_input("USDJPY barrier B", min_value=50.0, max_value=300.0,
                             value=156.0, step=0.5)
sig_X = st.sidebar.slider("USDJPY vol (sig_X)", 0.02, 0.30, 0.084, 0.002,
                          format="%.3f")

st.sidebar.subheader("Rates and correlation")
r_USD = st.sidebar.slider("USD rate (r_d)", 0.00, 0.10, 0.036, 0.001,
                          format="%.3f")
r_JPY = st.sidebar.slider("JPY rate (r_f)", 0.00, 0.10, 0.009, 0.001,
                          format="%.3f")
rho = st.sidebar.slider("Correlation (rho)", -0.99, 0.99, 0.30, 0.01)
T = st.sidebar.slider("Tenor T (years)", 0.05, 3.0, 0.5, 0.05)

# ----- Build the trade -----
p = HybridInputs(
    S0=S0, X0=X0, K=K, B=B, T=T,
    r_d=r_USD, r_f=r_JPY, q=q,
    sig_S=sig_S, sig_X=sig_X, rho=rho,
)

is_dd = payoff_type.startswith("Double")


# ----- Pricing functions -----
def price_dd_inputs(pp: HybridInputs, n: float) -> float:
    """Joint cash-or-nothing double digital price. Pays `n` if both conditions met."""
    sqrtT = np.sqrt(pp.T)
    d2S = (np.log(pp.S0/pp.K) + (pp.r_d - pp.q - 0.5*pp.sig_S**2)*pp.T) / (pp.sig_S*sqrtT)
    mu_X = (pp.r_f - pp.r_d) + pp.sig_X**2
    d2X = (np.log(pp.X0/pp.B) + (mu_X - 0.5*pp.sig_X**2)*pp.T) / (pp.sig_X*sqrtT)
    P_joint = bivariate_normal_cdf(d2S, d2X, pp.rho)
    return n * np.exp(-pp.r_d*pp.T) * P_joint


def price_at(S0_v, X0_v, **ov):
    """Generic price at given (S0, X0), optional overrides."""
    pp = replace(p, S0=S0_v, X0=X0_v, **ov)
    if is_dd:
        return price_dd_inputs(pp, notional)
    res = price_hybrid_call(pp)
    units = notional / S0_v
    return res["price"] * units


# ----- Forwards and probabilities -----
F_S = S0 * np.exp((r_USD - q) * T)
mu_X = (r_JPY - r_USD) + sig_X**2
F_X = X0 * np.exp(mu_X * T)

sqrtT = np.sqrt(T)
d2S = (np.log(S0/K) + (r_USD - q - 0.5*sig_S**2)*T) / (sig_S*sqrtT)
d2X = (np.log(X0/B) + (mu_X - 0.5*sig_X**2)*T) / (sig_X*sqrtT)
P_S = norm.cdf(d2S)
P_X = norm.cdf(d2X)
P_joint = bivariate_normal_cdf(d2S, d2X, rho)

# ----- Header -----
st.title("EQ/FX Hybrid & Double Digital Pricer")
st.caption(payoff_type)

# Top row: price + key probabilities
col1, col2, col3, col4 = st.columns(4)
inception_price = price_at(S0, X0)
col1.metric("Premium ($mm)", f"{inception_price/1e6:.3f}",
            help=f"As % of notional: {inception_price/notional:.2%}")
col2.metric("P(joint exercise)", f"{P_joint:.1%}")
col3.metric("P(SPX > K)", f"{P_S:.1%}")
col4.metric("P(USDJPY > B)", f"{P_X:.1%}")

# Forwards row
col1, col2, col3 = st.columns(3)
col1.metric("SPX forward at T", f"{F_S:.1f}",
            f"{(F_S/S0-1)*100:+.2f}% vs spot")
col2.metric("USDJPY forward at T", f"{F_X:.2f}",
            f"{(F_X/X0-1)*100:+.2f}% vs spot")
col3.metric("FX drift (USD measure)", f"{mu_X*100:+.2f}%",
            help="r_JPY - r_USD + sig_X^2 (under USD risk-neutral measure)")

# ----- Greeks -----
st.subheader("Greeks at inception")

def bump_price(field, h):
    return price_at(S0, X0, **{field: getattr(p, field) + h})

# SPX delta: per 1% spot move, then * 100 for notional-equivalent
spx_delta_pct = (price_at(S0*1.01, X0) - price_at(S0*0.99, X0)) / 2
fx_delta_pct = (price_at(S0, X0*1.01) - price_at(S0, X0*0.99)) / 2
vega_S = (bump_price("sig_S", 0.01) - bump_price("sig_S", -0.01)) / 2
vega_X = (bump_price("sig_X", 0.01) - bump_price("sig_X", -0.01)) / 2
cega = (bump_price("rho", 0.01) - bump_price("rho", -0.01)) / 2

# Theta: -dV/dT (price decay per year)
if T > 0.05:
    dT = 1/365  # 1 day
    theta = -(price_at(S0, X0, T=T+dT) - price_at(S0, X0, T=T-dT)) / (2*dT) / 365
else:
    theta = 0.0

g_col1, g_col2, g_col3 = st.columns(3)
with g_col1:
    st.metric("SPX delta (notional-eq)", f"${spx_delta_pct*100/1e6:.2f}mm",
              help=f"${spx_delta_pct/1e3:.1f}k per 1% SPX move")
    st.metric("SPX vega", f"${vega_S/1e3:.1f}k per vol pt")
with g_col2:
    st.metric("FX delta (notional-eq)", f"${fx_delta_pct*100/1e6:.2f}mm",
              help=f"${fx_delta_pct/1e3:.1f}k per 1% USDJPY move")
    st.metric("FX vega", f"${vega_X/1e3:.1f}k per vol pt")
with g_col3:
    st.metric("Cega", f"${cega/1e3:.1f}k per 1% corr")
    st.metric("Theta", f"${theta/1e3:.2f}k per day")


# ----- Plotting helpers -----
def make_grid(center, halfwidth, n=50, kind="abs"):
    """Grid around center spot. kind='abs' for FX (yen), 'pct' for SPX (% spot)."""
    if kind == "pct":
        return np.linspace(center * (1 - halfwidth), center * (1 + halfwidth), n)
    return np.linspace(center - halfwidth, center + halfwidth, n)


# Cache spots/grids that are reused across tabs
spx_grid = make_grid(S0, 0.20, n=40, kind="pct")
fx_grid = make_grid(X0, 20.0, n=40, kind="abs")

# Pick a few spot levels for the fan curves
spx_lvls = [S0 * f for f in [0.92, 0.96, 1.00, 1.04, 1.08]]
fx_lvls = [X0 + d for d in [-8, -4, 0, 4, 8]]

# ----- Tabs -----
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Price surfaces", "Deltas", "Vegas", "Cega", "Forward curves"]
)

with tab1:
    fig1, ax = plt.subplots(figsize=(9, 4.5))
    for x_lvl in fx_lvls:
        prices = [price_at(s, x_lvl)/1e6 for s in spx_grid]
        ax.plot(spx_grid, prices, label=f"USDJPY={x_lvl:.0f}", linewidth=1.8)
    ax.axvline(K, color="gray", ls=":", alpha=0.6)
    ax.axvline(S0, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("SPX spot"); ax.set_ylabel("Price ($mm)")
    ax.set_title("Price vs SPX, by USDJPY level")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(9, 4.5))
    for s_lvl in spx_lvls:
        prices = [price_at(s_lvl, x)/1e6 for x in fx_grid]
        ax.plot(fx_grid, prices, label=f"SPX={s_lvl:.0f}", linewidth=1.8)
    ax.axvline(B, color="gray", ls=":", alpha=0.6)
    ax.axvline(X0, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("USDJPY spot"); ax.set_ylabel("Price ($mm)")
    ax.set_title("Price vs USDJPY, by SPX level")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with tab2:
    fig1, ax = plt.subplots(figsize=(9, 4.5))
    for x_lvl in fx_lvls:
        deltas = []
        for s in spx_grid:
            d = (price_at(s*1.01, x_lvl) - price_at(s*0.99, x_lvl)) / 2 * 100
            deltas.append(d/1e6)
        ax.plot(spx_grid, deltas, label=f"USDJPY={x_lvl:.0f}", linewidth=1.8)
    ax.axvline(K, color="gray", ls=":", alpha=0.6)
    ax.axvline(S0, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("SPX spot"); ax.set_ylabel("SPX delta, notional-eq ($mm)")
    ax.set_title("SPX Delta vs SPX")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(9, 4.5))
    for s_lvl in spx_lvls:
        deltas = []
        for x in fx_grid:
            d = (price_at(s_lvl, x*1.01) - price_at(s_lvl, x*0.99)) / 2 * 100
            deltas.append(d/1e6)
        ax.plot(fx_grid, deltas, label=f"SPX={s_lvl:.0f}", linewidth=1.8)
    ax.axvline(B, color="gray", ls=":", alpha=0.6)
    ax.axvline(X0, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("USDJPY spot"); ax.set_ylabel("FX delta, notional-eq ($mm)")
    ax.set_title("FX Delta vs USDJPY")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with tab3:
    fig1, ax = plt.subplots(figsize=(9, 4.5))
    for x_lvl in fx_lvls:
        vegas = []
        for s in spx_grid:
            up = price_at(s, x_lvl, sig_S=sig_S+0.01)
            dn = price_at(s, x_lvl, sig_S=sig_S-0.01)
            vegas.append((up-dn)/2/1e3)
        ax.plot(spx_grid, vegas, label=f"USDJPY={x_lvl:.0f}", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(K, color="gray", ls=":", alpha=0.6)
    ax.axvline(S0, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("SPX spot"); ax.set_ylabel("SPX vega ($k per vol pt)")
    ax.set_title("SPX Vega vs SPX")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(9, 4.5))
    for s_lvl in spx_lvls:
        vegas = []
        for x in fx_grid:
            up = price_at(s_lvl, x, sig_X=sig_X+0.01)
            dn = price_at(s_lvl, x, sig_X=sig_X-0.01)
            vegas.append((up-dn)/2/1e3)
        ax.plot(fx_grid, vegas, label=f"SPX={s_lvl:.0f}", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(B, color="gray", ls=":", alpha=0.6)
    ax.axvline(X0, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("USDJPY spot"); ax.set_ylabel("FX vega ($k per vol pt)")
    ax.set_title("FX Vega vs USDJPY")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with tab4:
    fig1, ax = plt.subplots(figsize=(9, 4.5))
    for x_lvl in fx_lvls:
        cegas = []
        for s in spx_grid:
            up = price_at(s, x_lvl, rho=rho+0.01)
            dn = price_at(s, x_lvl, rho=rho-0.01)
            cegas.append((up-dn)/2/1e3)
        ax.plot(spx_grid, cegas, label=f"USDJPY={x_lvl:.0f}", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(K, color="gray", ls=":", alpha=0.6)
    ax.axvline(S0, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("SPX spot"); ax.set_ylabel("Cega ($k per 1% corr)")
    ax.set_title("Cega vs SPX")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(9, 4.5))
    for s_lvl in spx_lvls:
        cegas = []
        for x in fx_grid:
            up = price_at(s_lvl, x, rho=rho+0.01)
            dn = price_at(s_lvl, x, rho=rho-0.01)
            cegas.append((up-dn)/2/1e3)
        ax.plot(fx_grid, cegas, label=f"SPX={s_lvl:.0f}", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(B, color="gray", ls=":", alpha=0.6)
    ax.axvline(X0, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("USDJPY spot"); ax.set_ylabel("Cega ($k per 1% corr)")
    ax.set_title("Cega vs USDJPY")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with tab5:
    t_grid = np.linspace(0, max(T, 1.0), 100)

    fig1, ax = plt.subplots(figsize=(9, 4.5))
    F_S_curve = S0 * np.exp((r_USD - q) * t_grid)
    ax.plot(t_grid, F_S_curve, "b-", linewidth=2.5, label="SPX forward")
    ax.axhline(K, color="gray", ls="--", alpha=0.7, label=f"Strike K={K:.0f}")
    ax.axhline(S0, color="red", ls=":", alpha=0.5, label=f"Spot {S0:.0f}")
    ax.axvline(T, color="black", ls="--", alpha=0.5, label=f"Expiry T={T:.2f}y")
    ax.set_xlabel("Time (years)"); ax.set_ylabel("SPX forward")
    ax.set_title(f"SPX Forward (drift = r_USD - q = {(r_USD-q)*100:+.2f}%)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(9, 4.5))
    F_X_curve = X0 * np.exp(mu_X * t_grid)
    ax.plot(t_grid, F_X_curve, "g-", linewidth=2.5, label="USDJPY forward (USD measure)")
    ax.axhline(B, color="gray", ls="--", alpha=0.7, label=f"Barrier B={B:.0f}")
    ax.axhline(X0, color="red", ls=":", alpha=0.5, label=f"Spot {X0:.0f}")
    ax.axvline(T, color="black", ls="--", alpha=0.5, label=f"Expiry T={T:.2f}y")
    ax.set_xlabel("Time (years)"); ax.set_ylabel("USDJPY forward")
    ax.set_title(f"USDJPY Forward (drift = r_JPY - r_USD + sig_X^2 = {mu_X*100:+.2f}%)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ----- Footer -----
st.caption(
    "FX drift convention: under USD risk-neutral measure, USDJPY drifts at "
    "(r_JPY - r_USD + sig_X^2). Negative drift -> JPY appreciation forward, "
    "consistent with covered interest parity."
)
