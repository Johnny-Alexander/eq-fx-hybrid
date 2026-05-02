"""
Closed-form bivariate Black-Scholes pricer for an EQ/FX hybrid:

    Payoff at T = max(S_T - K, 0) * 1{X_T > B}

where S = equity (e.g. SPX), X = FX rate quoted as JPY per USD (e.g. USDJPY),
B = FX barrier, K = equity strike. Settlement currency is USD.

Model (under the USD risk-neutral measure)
------------------------------------------
    dS/S = (r_USD - q) dt + sigma_S dW_S
    dX/X = (r_JPY - r_USD + sigma_X^2) dt + sigma_X dW_X
    corr(dW_S, dW_X) = rho

Note on the FX drift
--------------------
Under the USD-risk-neutral measure, an exchange rate quoted as
'domestic-per-foreign' (JPY per USD) has drift:

    mu_X = (r_JPY - r_USD) + sigma_X^2  (the +sigma^2 is the quanto correction)

This is NOT the Garman-Kohlhagen drift (r_JPY - r_USD), which applies under
the JPY-risk-neutral measure. The extra sigma^2 term arises from changing
numeraire from JPY-cash to USD-cash via Ito's lemma on 1/X.

Equivalently, the USD-measure forward is:
    F(T) = S0 * exp((r_JPY - r_USD + sigma_X^2) * T)

Arbitrage check: USD rates above JPY rates -> r_JPY - r_USD < 0 ->
USDJPY forward generally below spot (USD depreciates forward), as observed
in the FX market.

Closed-form price
-----------------
    V0 = S0 e^{-qT}  M2(d1S, d2X + rho*sigma_S*sqrt(T); rho)
       - K  e^{-r_USD T} M2(d2S, d2X; rho)

with
    d1S = [ln(S0/K) + (r_USD - q + 0.5*sigma_S^2)*T] / (sigma_S*sqrt(T))
    d2S = d1S - sigma_S*sqrt(T)
    d1X = [ln(X0/B) + (mu_X + 0.5*sigma_X^2)*T] / (sigma_X*sqrt(T))
    d2X = d1X - sigma_X*sqrt(T)
where mu_X = r_JPY - r_USD + sigma_X^2.

The first term arises from a change of numeraire to S; the FX argument
shifts by rho*sigma_S*sqrt(T) under that measure.

References
----------
- Heynen & Kat (1994), "Crossing Barriers"
- Haug, "Complete Guide to Option Pricing Formulas", Ch. on multi-asset options
- Hull, "Options, Futures and Other Derivatives", Ch. on quantos
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, multivariate_normal


def bivariate_normal_cdf(a: float, b: float, rho: float) -> float:
    """P(Z1 <= a, Z2 <= b) where (Z1, Z2) is standard bivariate normal with
    correlation rho."""
    return multivariate_normal(mean=[0.0, 0.0],
                               cov=[[1.0, rho], [rho, 1.0]]).cdf([a, b])


@dataclass
class HybridInputs:
    """Inputs for the hybrid pricer.

    Attributes
    ----------
    S0     : Equity spot
    X0     : FX spot (units: domestic per foreign, e.g. USDJPY)
    K      : Equity strike
    B      : FX barrier (option pays only if X_T > B)
    T      : Time to expiry in years
    r_d    : Domestic risk-free rate (continuous)
    r_f    : Foreign risk-free rate (continuous)
    q      : Equity dividend yield (continuous)
    sig_S  : Equity volatility
    sig_X  : FX volatility
    rho    : Correlation between dW_S and dW_X under domestic measure
    """
    S0: float
    X0: float
    K: float
    B: float
    T: float
    r_d: float
    r_f: float
    q: float
    sig_S: float
    sig_X: float
    rho: float


def price_hybrid_call(p: HybridInputs) -> dict:
    """Closed-form price of the SPX-call x FX-digital hybrid.

    Returns
    -------
    dict with keys:
        price                  : option fair value (per equity unit)
        term1, term2           : the two components of the closed form
        P_joint_exercise       : Q-probability of joint event {S_T>K, X_T>B}
        P_FX_above_barrier     : marginal Q-probability of {X_T>B}
        vanilla_SPX_call       : price of the unconditional SPX call (for comparison)
    """
    sqrtT = np.sqrt(p.T)

    d1S = (np.log(p.S0 / p.K) + (p.r_d - p.q + 0.5 * p.sig_S ** 2) * p.T) / (p.sig_S * sqrtT)
    d2S = d1S - p.sig_S * sqrtT

    # FX drift under USD risk-neutral measure:
    #   mu_X = (r_JPY - r_USD) + sigma_X^2
    # The +sigma_X^2 is the quanto correction from changing numeraire JPY -> USD.
    # Note: p.r_d here = USD rate (settlement currency), p.r_f = JPY rate.
    mu_X = (p.r_f - p.r_d) + p.sig_X ** 2

    d1X = (np.log(p.X0 / p.B) + (mu_X + 0.5 * p.sig_X ** 2) * p.T) / (p.sig_X * sqrtT)
    d2X = d1X - p.sig_X * sqrtT

    # Joint event {S_T > K, X_T > B} under each measure.
    # Under Q^USD:  P = M2(d2S, d2X; rho)
    # Under Q^S:    the FX log-mean shifts by rho*sig_S*sig_X*T, so the FX
    #               argument becomes d2X + rho*sig_S*sqrt(T).
    term1 = p.S0 * np.exp(-p.q * p.T) * bivariate_normal_cdf(
        d1S, d2X + p.rho * p.sig_S * sqrtT, p.rho
    )
    term2 = p.K * np.exp(-p.r_d * p.T) * bivariate_normal_cdf(d2S, d2X, p.rho)

    price = term1 - term2

    P_joint = bivariate_normal_cdf(d2S, d2X, p.rho)
    P_fx = norm.cdf(d2X)
    vanilla = (p.S0 * np.exp(-p.q * p.T) * norm.cdf(d1S)
               - p.K * np.exp(-p.r_d * p.T) * norm.cdf(d2S))

    return {
        "price": price,
        "term1": term1,
        "term2": term2,
        "P_joint_exercise": P_joint,
        "P_FX_above_barrier": P_fx,
        "vanilla_SPX_call": vanilla,
    }


def price_hybrid_call_mc(p: HybridInputs, n_paths: int = 2_000_000,
                         seed: int = 42) -> dict:
    """Monte Carlo cross-check using terminal joint lognormal draws."""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_paths)
    z_ind = rng.standard_normal(n_paths)
    z2 = p.rho * z1 + np.sqrt(1.0 - p.rho ** 2) * z_ind

    sqrtT = np.sqrt(p.T)
    S_T = p.S0 * np.exp((p.r_d - p.q - 0.5 * p.sig_S ** 2) * p.T + p.sig_S * sqrtT * z1)
    # FX drift under USD measure: mu_X = (r_JPY - r_USD) + sigma_X^2
    mu_X = (p.r_f - p.r_d) + p.sig_X ** 2
    X_T = p.X0 * np.exp((mu_X - 0.5 * p.sig_X ** 2) * p.T + p.sig_X * sqrtT * z2)

    payoff = np.maximum(S_T - p.K, 0.0) * (X_T > p.B)
    pv = np.exp(-p.r_d * p.T) * payoff
    price = pv.mean()
    se = pv.std(ddof=1) / np.sqrt(n_paths)
    return {
        "price": price,
        "std_error": se,
        "ci95": (price - 1.96 * se, price + 1.96 * se),
    }


def _bump(p: HybridInputs, field: str, h: float) -> float:
    """Return price with one input bumped by h."""
    d = p.__dict__.copy()
    d[field] = d[field] + h
    return price_hybrid_call(HybridInputs(**d))["price"]


def greeks(p: HybridInputs) -> dict:
    """Bump-and-revalue Greeks. Bumps are sized for stability.

    Conventions
    -----------
    - delta_SPX  : dV/dS  (per unit of equity spot)
    - gamma_SPX  : d2V/dS2
    - delta_FX   : price change for a +1% move in FX spot (percentage convention).
                   This matches market FX risk reporting; multiply by NOTIONAL/100
                   to get the dollar impact of a 1% USDJPY move on a hedged book.
    - vega_*     : per 1 vol point (1%)
    - cega_corr  : per 1 correlation point (1%)
    - rho_*      : per 1bp rate move

    Returns
    -------
    dict with delta_SPX, gamma_SPX, delta_FX, vega_SPX, vega_FX, cega_corr,
    rho_USD, rho_JPY.
    """
    base = price_hybrid_call(p)["price"]

    dS = 0.01 * p.S0
    dX_pct = 0.01            # 1% FX move
    dX = dX_pct * p.X0       # absolute FX bump for a 1% move
    dv = 0.01    # 1 vol point
    dr = 1e-4    # 1bp
    drho = 0.01

    return {
        "delta_SPX": (_bump(p, "S0", dS) - _bump(p, "S0", -dS)) / (2 * dS),
        "gamma_SPX": (_bump(p, "S0", dS) - 2 * base + _bump(p, "S0", -dS)) / (dS ** 2),
        # FX delta as price change per 1% FX move:
        # = (V(X*1.01) - V(X*0.99)) / 2  -- already on the desired scale
        "delta_FX": (_bump(p, "X0", dX) - _bump(p, "X0", -dX)) / 2,
        "vega_SPX": (_bump(p, "sig_S", dv) - _bump(p, "sig_S", -dv)) / (2 * dv) / 100,
        "vega_FX":  (_bump(p, "sig_X", dv) - _bump(p, "sig_X", -dv)) / (2 * dv) / 100,
        "cega_corr": (_bump(p, "rho", drho) - _bump(p, "rho", -drho)) / (2 * drho) / 100,
        "rho_USD": (_bump(p, "r_d", dr) - _bump(p, "r_d", -dr)) / (2 * dr) / 10000,
        "rho_JPY": (_bump(p, "r_f", dr) - _bump(p, "r_f", -dr)) / (2 * dr) / 10000,
    }
