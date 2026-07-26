"""Cross-asset hybrid option pricers.

An equity payoff switched on or off by a condition on another observable.
The conditioning leg is pluggable -- see :mod:`.conditions` for why FX and
rates share one closed form rather than needing separate pricers.

    >>> from src.hybrid import EquityLeg, FXCondition, conditional_european
    >>> eq = EquityLeg(S0=7200, K=7000, q=0.015, r_d=0.045, sig_S=0.16)
    >>> fx = FXCondition(X0=156.0, B=160.0, r_d=0.045, r_f=0.005, sig_X=0.10)
    >>> conditional_european(eq, fx, T=0.5, rho=0.30)["price"]  # doctest: +SKIP
"""
from .bivariate import bivariate_normal_cdf
from .conditions import ConditionLeg, FXCondition
from .equity import EquityLeg
from .greeks import bumped, central_diff, second_diff
from .products import (conditional_european, conditional_european_mc,
                       decode_eta, double_digital, double_digital_mc)

__all__ = [
    "bivariate_normal_cdf",
    "ConditionLeg", "FXCondition",
    "EquityLeg",
    "bumped", "central_diff", "second_diff",
    "conditional_european", "conditional_european_mc",
    "double_digital", "double_digital_mc", "decode_eta",
]
