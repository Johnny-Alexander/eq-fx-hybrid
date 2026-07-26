"""Streamlit app for EQ/FX hybrid and double-digital pricing.

stlite-compatible version for the GitHub Pages deployment. The pricer package
is mirrored into docs/src/ by sync_docs.sh and mounted by index.html at the
same import paths the repo uses, so the imports below match ../app/app.py
exactly rather than being flattened.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import replace
from scipy.stats import norm

from src.hybrid_pricer import HybridInputs, price_hybrid_call, bivariate_normal_cdf

plt.rcParams["text.parse_math"] = False

st.set_page_config(
    page_title="EQ/FX Hybrid Pricer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
S0 = st.sidebar.number_input("SPX spot", min_value=1000.0, max_value=20000.0, value=7200.0, step=50.0)
K = st.sidebar.number_input("SPX strike K", min_value=1000.0, max_value=20000.0, value=7200.0, step=50.0)
sig_S = st.sidebar.slider("SPX vol (sig_S)", 0.05, 0.50, 0.17, 0.005, format="%.3f")
q = st.sidebar.slider("SPX dividend yield (q)", 0.00, 0.05, 0.015, 0.001, format="%.3f")

st.sidebar.subheader("FX leg (USDJPY)")
X0 = st.sidebar.number_input("USDJPY spot", min_value=50.0, max_value=300.0, value=156.0, step=0.5)
B = st.sidebar.number_input("USDJPY barrier B", min_value=50.0, max_value=300.0, value=156.0, step=0.5)
sig_X = st.sidebar.slider("USDJPY vol (sig_X)", 0.02, 0.30, 0.084, 0.002, format="%.3f")

st.sidebar.subheader("Rates and correlation")
r_USD = st.sidebar.slider("USD rate (r_d)", 0.00, 0.10, 0.036, 0.001, format="%.3f")
r_JPY = st.sidebar.slider("JPY rate (r_f)", 0.00, 0.10, 0.009, 0.001, format="%.3f")
rho = st.sidebar.slider("Correlation (rho)", -0.99, 0.99, 0.30, 0.01)
T = st.sidebar.slider("Tenor T (years)", 0.05, 3.0, 0.5, 0.05)

p = HybridInputs(
    S0=S0, X0=X0, K=K, B=B, T=T,
    r_d=r_USD, r_f=r_JPY, q=q,
    sig_S=sig_S, sig_X=sig_X, rho=rho,
)
is_dd = payoff_type.startswith("Double")


def price_dd_inputs(pp: HybridInputs, n: float) -> float:
    sqrtT = np.sqrt(pp.T)
    d2S = (np.log(pp.S0/pp.K) + (pp.r_d - pp.q - 0.5*pp.sig_S**2)*pp.T) / (pp.sig_S*sqrtT)
    mu_X = (pp.r_f - pp.r_d) + pp.sig_X**2
    d2X = (np.log(pp.X0/pp.B) + (mu_X - 0.5*pp.sig_X**2)*pp.T) / (pp.sig_X*sqrtT)
    P_joint = bivariate_normal_cdf(d2S, d2X, pp.rho)
    return n * np.exp(-pp.r_d*pp.T) * P_joint


def price_at(S0_v, X0_v, **ov):
    pp = replace(p, S0=S0_v, X0=X0_v, **ov)
    if is_dd:
        return price_dd_inputs(pp, notional)
    res = price_hybrid_call(pp)
    units = notional / S0_v
    return res["price"] * units


F_S = S0 * np.exp((r_USD - q) * T)
mu_X = (r_JPY - r_USD) + sig_X**2
F_X = X0 * np.exp(mu_X * T)

sqrtT = np.sqrt(T)
d2S = (np.log(S0/K) + (r_USD - q - 0.5*sig_S**2)*T) / (sig_S*sqrtT)
d2X = (np.log(X0/B) + (mu_X - 0.5*sig_X**2)*T) / (sig_X*sqrtT)
P_S = norm.cdf(d2S)
P_X = norm.cdf(d2X)
P_joint = bivariate_normal_cdf(d2S, d2X, rho)

st.title("EQ/FX Hybrid & Double Digital Pricer")
st.caption(payoff_type)

col1, col2, col3, col4 = st.columns(4)
inception_price = price_at(S0, X0)
col1.metric("Premium ($mm)", f"{inception_price/1e6:.3f}",
            help=f"As % of notional: {inception_price/notional:.2%}")
col2.metric("P(joint)", f"{P_joint:.1%}")
col3.metric("P(SPX > K)", f"{P_S:.1%}")
col4.metric("P(USDJPY > B)", f"{P_X:.1%}")

col1, col2, col3 = st.columns(3)
col1.metric("SPX fwd at T", f"{F_S:.1f}", f"{(F_S/S0-1)*100:+.2f}%")
col2.metric("USDJPY fwd at T", f"{F_X:.2f}", f"{(F_X/X0-1)*100:+.2f}%")
col3.metric("FX drift (USD measure)", f"{mu_X*100:+.2f}%",
            help="r_JPY - r_USD + sig_X^2")

st.subheader("Greeks at inception")

def bump_price(field, h):
    return price_at(S0, X0, **{field: getattr(p, field) + h})

spx_delta_pct = (price_at(S0*1.01, X0) - price_at(S0*0.99, X0)) / 2
fx_delta_pct = (price_at(S0, X0*1.01) - price_at(S0, X0*0.99)) / 2
vega_S = (bump_price("sig_S", 0.01) - bump_price("sig_S", -0.01)) / 2
vega_X = (bump_price("sig_X", 0.01) - bump_price("sig_X", -0.01)) / 2
cega = (bump_price("rho", 0.01) - bump_price("rho", -0.01)) / 2

if T > 0.05:
    dT = 1/365
    theta = -(price_at(S0, X0, T=T+dT) - price_at(S0, X0, T=T-dT)) / (2*dT) / 365
else:
    theta = 0.0

g1, g2, g3 = st.columns(3)
with g1:
    st.metric("SPX delta (notional-eq)", f"${spx_delta_pct*100/1e6:.2f}mm",
              help=f"${spx_delta_pct/1e3:.1f}k per 1% SPX move")
    st.metric("SPX vega", f"${vega_S/1e3:.1f}k per vol pt")
with g2:
    st.metric("FX delta (notional-eq)", f"${fx_delta_pct*100/1e6:.2f}mm",
              help=f"${fx_delta_pct/1e3:.1f}k per 1% USDJPY move")
    st.metric("FX vega", f"${vega_X/1e3:.1f}k per vol pt")
with g3:
    st.metric("Cega", f"${cega/1e3:.1f}k per 1% corr")
    st.metric("Theta", f"${theta/1e3:.2f}k per day")

# Smaller grids for browser-side compute
def make_grid(center, halfwidth, n=30, kind="abs"):
    if kind == "pct":
        return np.linspace(center * (1 - halfwidth), center * (1 + halfwidth), n)
    return np.linspace(center - halfwidth, center + halfwidth, n)

spx_grid = make_grid(S0, 0.20, n=30, kind="pct")
fx_grid = make_grid(X0, 20.0, n=30, kind="abs")
spx_lvls = [S0 * f for f in [0.92, 0.96, 1.00, 1.04, 1.08]]
fx_lvls = [X0 + d for d in [-8, -4, 0, 4, 8]]

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Price", "Deltas", "Vegas", "Cega", "Forwards"]
)

def fan_plot(x_grid, y_func, x_label, y_label, title, ref_lines, legend_fmt):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for lvl, label in legend_fmt:
        y = [y_func(x, lvl) for x in x_grid]
        ax.plot(x_grid, y, label=label, linewidth=1.8)
    for v, color, ls in ref_lines:
        ax.axvline(v, color=color, ls=ls, alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab1:
    fan_plot(
        spx_grid, lambda s, xl: price_at(s, xl)/1e6,
        "SPX spot", "Price ($mm)", "Price vs SPX, by USDJPY",
        [(K, "gray", ":"), (S0, "red", ":")],
        [(xl, f"USDJPY={xl:.0f}") for xl in fx_lvls],
    )
    fan_plot(
        fx_grid, lambda x, sl: price_at(sl, x)/1e6,
        "USDJPY spot", "Price ($mm)", "Price vs USDJPY, by SPX",
        [(B, "gray", ":"), (X0, "red", ":")],
        [(sl, f"SPX={sl:.0f}") for sl in spx_lvls],
    )

with tab2:
    fan_plot(
        spx_grid,
        lambda s, xl: (price_at(s*1.01, xl) - price_at(s*0.99, xl)) / 2 * 100 / 1e6,
        "SPX spot", "SPX delta, notional-eq ($mm)", "SPX Delta vs SPX",
        [(K, "gray", ":"), (S0, "red", ":")],
        [(xl, f"USDJPY={xl:.0f}") for xl in fx_lvls],
    )
    fan_plot(
        fx_grid,
        lambda x, sl: (price_at(sl, x*1.01) - price_at(sl, x*0.99)) / 2 * 100 / 1e6,
        "USDJPY spot", "FX delta, notional-eq ($mm)", "FX Delta vs USDJPY",
        [(B, "gray", ":"), (X0, "red", ":")],
        [(sl, f"SPX={sl:.0f}") for sl in spx_lvls],
    )

with tab3:
    fan_plot(
        spx_grid,
        lambda s, xl: (price_at(s, xl, sig_S=sig_S+0.01) - price_at(s, xl, sig_S=sig_S-0.01)) / 2 / 1e3,
        "SPX spot", "SPX vega ($k per vol pt)", "SPX Vega vs SPX",
        [(K, "gray", ":"), (S0, "red", ":")],
        [(xl, f"USDJPY={xl:.0f}") for xl in fx_lvls],
    )
    fan_plot(
        fx_grid,
        lambda x, sl: (price_at(sl, x, sig_X=sig_X+0.01) - price_at(sl, x, sig_X=sig_X-0.01)) / 2 / 1e3,
        "USDJPY spot", "FX vega ($k per vol pt)", "FX Vega vs USDJPY",
        [(B, "gray", ":"), (X0, "red", ":")],
        [(sl, f"SPX={sl:.0f}") for sl in spx_lvls],
    )

with tab4:
    fan_plot(
        spx_grid,
        lambda s, xl: (price_at(s, xl, rho=rho+0.01) - price_at(s, xl, rho=rho-0.01)) / 2 / 1e3,
        "SPX spot", "Cega ($k per 1% corr)", "Cega vs SPX",
        [(K, "gray", ":"), (S0, "red", ":")],
        [(xl, f"USDJPY={xl:.0f}") for xl in fx_lvls],
    )
    fan_plot(
        fx_grid,
        lambda x, sl: (price_at(sl, x, rho=rho+0.01) - price_at(sl, x, rho=rho-0.01)) / 2 / 1e3,
        "USDJPY spot", "Cega ($k per 1% corr)", "Cega vs USDJPY",
        [(B, "gray", ":"), (X0, "red", ":")],
        [(sl, f"SPX={sl:.0f}") for sl in spx_lvls],
    )

with tab5:
    t_grid = np.linspace(0, max(T, 1.0), 80)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    F_S_curve = S0 * np.exp((r_USD - q) * t_grid)
    ax.plot(t_grid, F_S_curve, "b-", linewidth=2.5, label="SPX forward")
    ax.axhline(K, color="gray", ls="--", alpha=0.7, label=f"K={K:.0f}")
    ax.axhline(S0, color="red", ls=":", alpha=0.5, label=f"Spot {S0:.0f}")
    ax.axvline(T, color="black", ls="--", alpha=0.5, label=f"T={T:.2f}y")
    ax.set_xlabel("Time (years)"); ax.set_ylabel("SPX forward")
    ax.set_title(f"SPX Forward (drift = {(r_USD-q)*100:+.2f}%)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    F_X_curve = X0 * np.exp(mu_X * t_grid)
    ax.plot(t_grid, F_X_curve, "g-", linewidth=2.5, label="USDJPY fwd (USD measure)")
    ax.axhline(B, color="gray", ls="--", alpha=0.7, label=f"B={B:.0f}")
    ax.axhline(X0, color="red", ls=":", alpha=0.5, label=f"Spot {X0:.0f}")
    ax.axvline(T, color="black", ls="--", alpha=0.5, label=f"T={T:.2f}y")
    ax.set_xlabel("Time (years)"); ax.set_ylabel("USDJPY forward")
    ax.set_title(f"USDJPY Forward (drift = {mu_X*100:+.2f}%)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)

st.caption(
    "Running entirely in your browser via stlite (Pyodide). "
    "FX drift under USD measure: r_JPY - r_USD + sig_X^2."
)
