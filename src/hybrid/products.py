"""Hybrid products, written once against any :class:`ConditionLeg`.

Both products below are asset-class agnostic: swap an :class:`FXCondition`
for a rates condition and the same code prices an EQ/IR hybrid.
"""
from __future__ import annotations

import numpy as np

from .bivariate import bivariate_normal_cdf, norm
from .conditions import ConditionLeg
from .equity import EquityLeg

__all__ = ["decode_eta", "conditional_european", "conditional_european_mc",
           "double_digital", "double_digital_mc"]


def decode_eta(eq_dir: str, cond_dir: str) -> tuple[int, int]:
    """Decode direction labels into (eta_S, eta_R) signs.

    Accepts 'call'/'put' or 'above'/'below' for the equity leg -- for a
    digital the labels mean the same thing, only the sign matters.
    """
    eq_map = {"call": 1, "above": 1, "put": -1, "below": -1}
    cond_map = {"above": 1, "below": -1}
    if eq_dir not in eq_map:
        raise ValueError(
            f"eq direction must be 'call'/'put' or 'above'/'below', got {eq_dir!r}")
    if cond_dir not in cond_map:
        raise ValueError(
            f"condition direction must be 'above' or 'below', got {cond_dir!r}")
    return eq_map[eq_dir], cond_map[cond_dir]


def conditional_european(eq: EquityLeg, cond: ConditionLeg, T: float, rho: float,
                         eq_type: str = "call", cond_dir: str = "above") -> dict:
    """Equity vanilla switched on by a condition on another observable.

    Payoff at T:
        max(eta_S * (S_T - K), 0) * 1{condition holds}

    Identity: cond_above + cond_below = unconditional vanilla.

    Returns
    -------
    dict with price, term1, term2, P_joint_exercise, P_condition,
    vanilla_equivalent.
    """
    eta_S, eta_R = decode_eta(eq_type, cond_dir)

    d1, d2 = eq.d1(T), eq.d2(T)
    h = cond.h(T)
    shift = cond.equity_measure_shift(eq.sig_S, rho, T)
    rho_eff = eta_S * eta_R * rho

    term1 = eta_S * eq.forward_factor(T) * bivariate_normal_cdf(
        eta_S * d1, eta_R * (h + shift), rho_eff)
    term2 = eta_S * eq.K * eq.discount(T) * bivariate_normal_cdf(
        eta_S * d2, eta_R * h, rho_eff)

    return {
        "price": term1 - term2,
        "term1": term1,
        "term2": term2,
        "P_joint_exercise": bivariate_normal_cdf(eta_S * d2, eta_R * h, rho_eff),
        "P_condition": norm.cdf(eta_R * h),
        "vanilla_equivalent": eq.vanilla(T, eta_S),
    }


def double_digital(eq: EquityLeg, cond: ConditionLeg, T: float, rho: float,
                   notional: float = 1.0, eq_dir: str = "above",
                   cond_dir: str = "above") -> dict:
    """Joint cash-or-nothing digital on both legs.

    Payoff at T: notional * 1{eta_S (S_T - K) > 0 AND condition holds}

    No measure change is involved -- the payoff has no S_T factor -- so the
    equity-measure shift does not appear here.
    """
    eta_S, eta_R = decode_eta(eq_dir, cond_dir)

    d2 = eq.d2(T)
    h = cond.h(T)
    rho_eff = eta_S * eta_R * rho

    P_joint = bivariate_normal_cdf(eta_S * d2, eta_R * h, rho_eff)
    df = eq.discount(T)

    return {
        "price": notional * df * P_joint,
        "P_joint": P_joint,
        "P_eq": norm.cdf(eta_S * d2),
        "P_condition": norm.cdf(eta_R * h),
        "df": df,
    }


def _draws(rho: float, n_paths: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_paths)
    z_ind = rng.standard_normal(n_paths)
    z2 = rho * z1 + np.sqrt(1.0 - rho ** 2) * z_ind
    return z1, z2


def _mc_result(pv: np.ndarray) -> dict:
    price = pv.mean()
    se = pv.std(ddof=1) / np.sqrt(pv.size)
    return {"price": price, "std_error": se,
            "ci95": (price - 1.96 * se, price + 1.96 * se)}


def conditional_european_mc(eq: EquityLeg, cond: ConditionLeg, T: float, rho: float,
                            eq_type: str = "call", cond_dir: str = "above",
                            n_paths: int = 2_000_000, seed: int = 42) -> dict:
    """Monte Carlo cross-check for :func:`conditional_european`."""
    eta_S, eta_R = decode_eta(eq_type, cond_dir)
    z1, z2 = _draws(rho, n_paths, seed)

    S_T = eq.simulate(z1, T)
    payoff = np.maximum(eta_S * (S_T - eq.K), 0.0) * cond.indicator(z2, T, eta_R)
    return _mc_result(eq.discount(T) * payoff)


def double_digital_mc(eq: EquityLeg, cond: ConditionLeg, T: float, rho: float,
                      notional: float = 1.0, eq_dir: str = "above",
                      cond_dir: str = "above",
                      n_paths: int = 2_000_000, seed: int = 42) -> dict:
    """Monte Carlo cross-check for :func:`double_digital`."""
    eta_S, eta_R = decode_eta(eq_dir, cond_dir)
    z1, z2 = _draws(rho, n_paths, seed)

    S_T = eq.simulate(z1, T)
    eq_ind = (eta_S * (S_T - eq.K) > 0).astype(float)
    payoff = notional * eq_ind * cond.indicator(z2, T, eta_R)
    return _mc_result(eq.discount(T) * payoff)
