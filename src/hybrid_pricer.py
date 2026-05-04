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


def _eta(eq_type: str, fx_dir: str) -> tuple:
    """Decode (eq_type, fx_dir) into (eta_S, eta_X) signs."""
    if eq_type == "call":
        eta_S = 1
    elif eq_type == "put":
        eta_S = -1
    else:
        raise ValueError(f"eq_type must be 'call' or 'put', got {eq_type!r}")
    if fx_dir == "above":
        eta_X = 1
    elif fx_dir == "below":
        eta_X = -1
    else:
        raise ValueError(f"fx_dir must be 'above' or 'below', got {fx_dir!r}")
    return eta_S, eta_X


def price_hybrid(p: HybridInputs, eq_type: str = "call",
                 fx_dir: str = "above") -> dict:
    """Closed-form price of a generalized EQ x FX-digital hybrid.

    Payoff at T:
        max(eta_S * (S_T - K), 0) * 1{eta_X * (X_T - B) > 0}

    where eta_S = +1 for an equity call, -1 for a put;
          eta_X = +1 for "FX above barrier", -1 for "FX below".

    All four combinations have a closed-form bivariate Black-Scholes price.
    Identity: hybrid_above + hybrid_below = vanilla (call or put).

    Returns
    -------
    dict with keys:
        price                  : option fair value (per equity unit)
        term1, term2           : the two components of the closed form
        P_joint_exercise       : Q^USD probability of {payoff > 0}
        P_FX_condition         : marginal Q^USD probability of FX condition
        vanilla_equivalent     : unconditional vanilla call or put (for comparison)
    """
    eta_S, eta_X = _eta(eq_type, fx_dir)
    sqrtT = np.sqrt(p.T)

    d1S = (np.log(p.S0 / p.K) + (p.r_d - p.q + 0.5 * p.sig_S ** 2) * p.T) / (p.sig_S * sqrtT)
    d2S = d1S - p.sig_S * sqrtT

    # FX drift under USD risk-neutral measure: mu_X = (r_JPY - r_USD) + sigma_X^2
    # +sigma_X^2 is the quanto correction from changing numeraire JPY -> USD.
    mu_X = (p.r_f - p.r_d) + p.sig_X ** 2

    d1X = (np.log(p.X0 / p.B) + (mu_X + 0.5 * p.sig_X ** 2) * p.T) / (p.sig_X * sqrtT)
    d2X = d1X - p.sig_X * sqrtT

    # Generalized formula: indicator signs flip eta_S, eta_X into the M2 args
    # and into the effective correlation (eta_S * eta_X * rho).
    rho_eff = eta_S * eta_X * p.rho

    term1 = eta_S * p.S0 * np.exp(-p.q * p.T) * bivariate_normal_cdf(
        eta_S * d1S, eta_X * (d2X + p.rho * p.sig_S * sqrtT), rho_eff
    )
    term2 = eta_S * p.K * np.exp(-p.r_d * p.T) * bivariate_normal_cdf(
        eta_S * d2S, eta_X * d2X, rho_eff
    )

    price = term1 - term2

    P_joint = bivariate_normal_cdf(eta_S * d2S, eta_X * d2X, rho_eff)
    P_fx = norm.cdf(eta_X * d2X)

    if eq_type == "call":
        vanilla = (p.S0 * np.exp(-p.q * p.T) * norm.cdf(d1S)
                   - p.K * np.exp(-p.r_d * p.T) * norm.cdf(d2S))
    else:
        vanilla = (p.K * np.exp(-p.r_d * p.T) * norm.cdf(-d2S)
                   - p.S0 * np.exp(-p.q * p.T) * norm.cdf(-d1S))

    return {
        "price": price,
        "term1": term1,
        "term2": term2,
        "P_joint_exercise": P_joint,
        "P_FX_condition": P_fx,
        "vanilla_equivalent": vanilla,
    }


def price_hybrid_mc(p: HybridInputs, eq_type: str = "call",
                    fx_dir: str = "above",
                    n_paths: int = 2_000_000, seed: int = 42) -> dict:
    """Monte Carlo cross-check for the generalized hybrid."""
    eta_S, eta_X = _eta(eq_type, fx_dir)

    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_paths)
    z_ind = rng.standard_normal(n_paths)
    z2 = p.rho * z1 + np.sqrt(1.0 - p.rho ** 2) * z_ind

    sqrtT = np.sqrt(p.T)
    S_T = p.S0 * np.exp((p.r_d - p.q - 0.5 * p.sig_S ** 2) * p.T + p.sig_S * sqrtT * z1)
    mu_X = (p.r_f - p.r_d) + p.sig_X ** 2
    X_T = p.X0 * np.exp((mu_X - 0.5 * p.sig_X ** 2) * p.T + p.sig_X * sqrtT * z2)

    eq_payoff = np.maximum(eta_S * (S_T - p.K), 0.0)
    fx_indicator = (eta_X * (X_T - p.B) > 0).astype(float)

    payoff = eq_payoff * fx_indicator
    pv = np.exp(-p.r_d * p.T) * payoff
    price = pv.mean()
    se = pv.std(ddof=1) / np.sqrt(n_paths)
    return {
        "price": price,
        "std_error": se,
        "ci95": (price - 1.96 * se, price + 1.96 * se),
    }


def price_double_digital(p: HybridInputs, notional: float = 1.0,
                         eq_dir: str = "above",
                         fx_dir: str = "above") -> dict:
    """Joint cash-or-nothing double digital.

    Payoff at T: `notional * 1{eta_S * (S_T - K) > 0 AND eta_X * (X_T - B) > 0}`

    eq_dir = 'above' -> S_T > K (eta_S = +1)
    eq_dir = 'below' -> S_T < K (eta_S = -1)
    fx_dir = 'above' -> X_T > B (eta_X = +1)
    fx_dir = 'below' -> X_T < B (eta_X = -1)

    Returns
    -------
    dict with keys: price, P_joint, P_eq, P_fx, df.
    """
    # Reuse _eta with a placeholder eq_type label; only the sign matters.
    eta_S = +1 if eq_dir == "above" else -1 if eq_dir == "below" else None
    eta_X = +1 if fx_dir == "above" else -1 if fx_dir == "below" else None
    if eta_S is None:
        raise ValueError(f"eq_dir must be 'above' or 'below', got {eq_dir!r}")
    if eta_X is None:
        raise ValueError(f"fx_dir must be 'above' or 'below', got {fx_dir!r}")

    sqrtT = np.sqrt(p.T)
    d2S = (np.log(p.S0 / p.K) + (p.r_d - p.q - 0.5 * p.sig_S ** 2) * p.T) / (p.sig_S * sqrtT)
    mu_X = (p.r_f - p.r_d) + p.sig_X ** 2
    d2X = (np.log(p.X0 / p.B) + (mu_X - 0.5 * p.sig_X ** 2) * p.T) / (p.sig_X * sqrtT)

    rho_eff = eta_S * eta_X * p.rho
    P_joint = bivariate_normal_cdf(eta_S * d2S, eta_X * d2X, rho_eff)
    P_eq = norm.cdf(eta_S * d2S)
    P_fx = norm.cdf(eta_X * d2X)
    df = np.exp(-p.r_d * p.T)

    return {
        "price": notional * df * P_joint,
        "P_joint": P_joint,
        "P_eq": P_eq,
        "P_fx": P_fx,
        "df": df,
    }


def price_double_digital_mc(p: HybridInputs, notional: float = 1.0,
                            eq_dir: str = "above", fx_dir: str = "above",
                            n_paths: int = 2_000_000, seed: int = 42) -> dict:
    """MC cross-check for the double digital."""
    eta_S = +1 if eq_dir == "above" else -1
    eta_X = +1 if fx_dir == "above" else -1

    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_paths)
    z_ind = rng.standard_normal(n_paths)
    z2 = p.rho * z1 + np.sqrt(1.0 - p.rho ** 2) * z_ind

    sqrtT = np.sqrt(p.T)
    S_T = p.S0 * np.exp((p.r_d - p.q - 0.5 * p.sig_S ** 2) * p.T + p.sig_S * sqrtT * z1)
    mu_X = (p.r_f - p.r_d) + p.sig_X ** 2
    X_T = p.X0 * np.exp((mu_X - 0.5 * p.sig_X ** 2) * p.T + p.sig_X * sqrtT * z2)

    indicator = ((eta_S * (S_T - p.K) > 0) & (eta_X * (X_T - p.B) > 0)).astype(float)
    pv = np.exp(-p.r_d * p.T) * notional * indicator
    price = pv.mean()
    se = pv.std(ddof=1) / np.sqrt(n_paths)
    return {"price": price, "std_error": se,
            "ci95": (price - 1.96 * se, price + 1.96 * se)}


def price_hybrid_call(p: HybridInputs) -> dict:
    """Backward-compatible wrapper: SPX call * 1{X > B}.

    Returns the same dict shape as before, including legacy keys
    `vanilla_SPX_call` and `P_FX_above_barrier`.
    """
    res = price_hybrid(p, eq_type="call", fx_dir="above")
    return {
        "price": res["price"],
        "term1": res["term1"],
        "term2": res["term2"],
        "P_joint_exercise": res["P_joint_exercise"],
        "P_FX_above_barrier": res["P_FX_condition"],
        "vanilla_SPX_call": res["vanilla_equivalent"],
    }


def price_hybrid_call_mc(p: HybridInputs, n_paths: int = 2_000_000,
                         seed: int = 42) -> dict:
    """Backward-compatible MC wrapper for the SPX call * 1{X > B} hybrid."""
    return price_hybrid_mc(p, eq_type="call", fx_dir="above",
                           n_paths=n_paths, seed=seed)


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
