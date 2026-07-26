"""Closed-form bivariate Black-Scholes pricer for an EQ/FX hybrid:

    Payoff at T = max(S_T - K, 0) * 1{X_T > B}

where S = equity (e.g. SPX), X = FX rate quoted as JPY per USD (e.g. USDJPY),
B = FX barrier, K = equity strike. Settlement currency is USD.

.. note::
   The implementation now lives in the :mod:`src.hybrid` package, which
   generalises this to any conditioning observable (FX today, CMS rates
   next). This module is a compatibility layer: it keeps the flat
   ``HybridInputs`` interface and the original function names and result-dict
   keys, so existing scripts, notebooks and the Streamlit app are unaffected.
   New code should prefer ``from src.hybrid import ...``.

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
shifts by rho*sigma_S*sqrt(T) under that measure. That shift is the piece
that generalises across asset classes -- see :mod:`src.hybrid.conditions`.

References
----------
- Heynen & Kat (1994), "Crossing Barriers"
- Haug, "Complete Guide to Option Pricing Formulas", Ch. on multi-asset options
- Hull, "Options, Futures and Other Derivatives", Ch. on quantos
"""
from __future__ import annotations

from dataclasses import dataclass

from src.hybrid import (EquityLeg, FXCondition, bivariate_normal_cdf,
                        conditional_european, conditional_european_mc,
                        decode_eta, double_digital, double_digital_mc)
from src.hybrid.greeks import bumped, central_diff, second_diff

__all__ = ["bivariate_normal_cdf", "HybridInputs", "price_hybrid",
           "price_hybrid_mc", "price_double_digital", "price_double_digital_mc",
           "price_hybrid_call", "price_hybrid_call_mc", "greeks"]


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

    def legs(self) -> tuple[EquityLeg, FXCondition]:
        """Split the flat inputs into the equity and conditioning legs.

        Uses the spot-implied constructor, so this flat API keeps its
        original constant-rate convention. Build an
        :meth:`~src.hybrid.EquityLeg.from_market` leg directly to price off
        an observed forward and discount factor instead.
        """
        return (
            EquityLeg.from_spot(S0=self.S0, K=self.K, q=self.q, r_d=self.r_d,
                                sig_S=self.sig_S),
            FXCondition(X0=self.X0, B=self.B, r_d=self.r_d, r_f=self.r_f,
                        sig_X=self.sig_X),
        )


def _eta(eq_type: str, fx_dir: str) -> tuple:
    """Decode (eq_type, fx_dir) into (eta_S, eta_X) signs."""
    return decode_eta(eq_type, fx_dir)


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
    eq, fx = p.legs()
    res = conditional_european(eq, fx, p.T, p.rho, eq_type, fx_dir)
    return {
        "price": res["price"],
        "term1": res["term1"],
        "term2": res["term2"],
        "P_joint_exercise": res["P_joint_exercise"],
        "P_FX_condition": res["P_condition"],
        "vanilla_equivalent": res["vanilla_equivalent"],
    }


def price_hybrid_mc(p: HybridInputs, eq_type: str = "call",
                    fx_dir: str = "above",
                    n_paths: int = 2_000_000, seed: int = 42) -> dict:
    """Monte Carlo cross-check for the generalized hybrid."""
    eq, fx = p.legs()
    return conditional_european_mc(eq, fx, p.T, p.rho, eq_type, fx_dir,
                                   n_paths=n_paths, seed=seed)


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
    eq, fx = p.legs()
    res = double_digital(eq, fx, p.T, p.rho, notional, eq_dir, fx_dir)
    return {
        "price": res["price"],
        "P_joint": res["P_joint"],
        "P_eq": res["P_eq"],
        "P_fx": res["P_condition"],
        "df": res["df"],
    }


def price_double_digital_mc(p: HybridInputs, notional: float = 1.0,
                            eq_dir: str = "above", fx_dir: str = "above",
                            n_paths: int = 2_000_000, seed: int = 42) -> dict:
    """MC cross-check for the double digital."""
    eq, fx = p.legs()
    return double_digital_mc(eq, fx, p.T, p.rho, notional, eq_dir, fx_dir,
                             n_paths=n_paths, seed=seed)


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


def _price(p: HybridInputs) -> float:
    return price_hybrid_call(p)["price"]


def _bump(p: HybridInputs, field: str, h: float) -> float:
    """Return price with one input bumped by h."""
    return bumped(_price, p, field, h)


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
    dS = 0.01 * p.S0
    dX = 0.01 * p.X0         # absolute FX bump for a 1% move
    dv = 0.01                # 1 vol point
    dr = 1e-4                # 1bp
    drho = 0.01

    return {
        "delta_SPX": central_diff(_price, p, "S0", dS),
        "gamma_SPX": second_diff(_price, p, "S0", dS),
        # FX delta as price change per 1% FX move:
        # = (V(X*1.01) - V(X*0.99)) / 2  -- already on the desired scale
        "delta_FX": central_diff(_price, p, "X0", dX) * dX,
        "vega_SPX": central_diff(_price, p, "sig_S", dv) / 100,
        "vega_FX": central_diff(_price, p, "sig_X", dv) / 100,
        "cega_corr": central_diff(_price, p, "rho", drho) / 100,
        "rho_USD": central_diff(_price, p, "r_d", dr) / 10000,
        "rho_JPY": central_diff(_price, p, "r_f", dr) / 10000,
    }
