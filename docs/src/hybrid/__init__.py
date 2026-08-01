"""Cross-asset hybrid option pricers.

An equity payoff switched on or off by a condition on another observable.
The conditioning leg is pluggable -- see :mod:`.conditions` for why FX and
rates share one closed form rather than needing separate pricers.

    >>> from src.hybrid import EquityLeg, FXCondition, conditional_european
    >>> eq = EquityLeg.from_spot(S0=7200, K=7000, q=0.015, r_d=0.045, sig_S=0.16)
    >>> fx = FXCondition(X0=156.0, B=160.0, r_d=0.045, r_f=0.005, sig_X=0.10)
    >>> conditional_european(eq, fx, T=0.5, rho=0.30)["price"]  # doctest: +SKIP

Swapping the conditioning leg is the whole point -- an SPX call contingent
on CMS10 above 4% is the same call with a different leg::

    >>> from src.hybrid import RateCondition
    >>> cms = RateCondition(R_adj=0.0415, B=0.04, sig_R=0.0080)
    >>> conditional_european(eq, cms, T=2.0, rho=-0.30)["price"]  # doctest: +SKIP
"""
from .bivariate import bivariate_normal_cdf, bivariate_normal_cdf_vec
from .cms import adjusted_cms_rate, convexity_adjustment
from .conditions import (ConditionLeg, FXCondition, RateCondition,
                         ShiftedLognormalRateCondition)
from .eqir import EQIRTrade
from .equity import EquityLeg
from .greeks import bumped, central_diff, second_diff
from .products import (conditional_european, conditional_european_mc,
                       decode_eta, double_digital, double_digital_mc)
from .surface import (SurfaceGrid, correlation_schedule,
                      dual_digital_value_grid, hedged_short_pnl_grid,
                      inception_deltas, inception_premium,
                      marginal_probabilities, short_pnl_grid,
                      time_to_expiry_schedule)

__all__ = [
    "bivariate_normal_cdf", "bivariate_normal_cdf_vec",
    "ConditionLeg", "FXCondition", "RateCondition",
    "ShiftedLognormalRateCondition",
    "adjusted_cms_rate", "convexity_adjustment",
    "EQIRTrade",
    "EquityLeg",
    "bumped", "central_diff", "second_diff",
    "conditional_european", "conditional_european_mc",
    "double_digital", "double_digital_mc", "decode_eta",
    "SurfaceGrid", "dual_digital_value_grid", "short_pnl_grid",
    "hedged_short_pnl_grid", "inception_deltas", "inception_premium",
    "marginal_probabilities", "time_to_expiry_schedule",
    "correlation_schedule",
]
