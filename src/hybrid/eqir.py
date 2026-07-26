"""EQ/IR trades: input container and risk in rates conventions.

The pricing core is shared with EQ/FX -- see :mod:`.products`. What is
genuinely asset-class specific is how risk is *quoted*: an FX desk reports
delta per 1% spot move, a rates desk reports DV01 per basis point and vega
per basis point of normal vol. This module supplies that layer, and nothing
else.

:class:`EQIRTrade` holds the primitive market observables rather than derived
ones. In particular it stores ``R_0``, the *unadjusted* forward swap rate, so
that bumping it re-derives the convexity adjustment -- a DV01 computed off a
frozen ``R_adj`` would miss the (small) convexity feedback and, worse, invite
someone to bump the adjusted rate directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conditions import RateCondition
from .equity import EquityLeg
from .greeks import central_diff, second_diff
from .products import conditional_european, double_digital

__all__ = ["EQIRTrade", "greeks"]

BP = 1e-4


@dataclass
class EQIRTrade:
    """An equity payoff conditioned on a CMS fixing.

    Attributes
    ----------
    F        : equity forward to T
    K        : equity strike
    P0T      : discount factor to T
    sig_S    : volatility of the equity forward (see :mod:`.equity`)
    R_0      : forward swap rate, UNADJUSTED -- convexity is applied internally
    B        : rate barrier, decimal (0.04 = 4%)
    sig_R    : swap-rate normal vol, absolute per sqrt(year) (0.0080 = 80bp)
    T        : time to expiry / CMS fixing, in years
    rho      : correlation between the equity forward and the swap rate
    tenor    : underlying swap tenor in years (10 for CMS10)
    freq     : underlying swap coupon frequency per year
    eq_type  : 'call' or 'put'
    cond_dir : 'above' or 'below' the rate barrier
    """

    F: float
    K: float
    P0T: float
    sig_S: float
    R_0: float
    B: float
    sig_R: float
    T: float
    rho: float
    tenor: float = 10.0
    freq: int = 2
    eq_type: str = "call"
    cond_dir: str = "above"

    def legs(self) -> tuple[EquityLeg, RateCondition]:
        return (
            EquityLeg.from_market(F=self.F, K=self.K, P0T=self.P0T,
                                  sig_S=self.sig_S, T=self.T),
            RateCondition.from_forward_swap(
                R_0=self.R_0, B=self.B, sig_R=self.sig_R, T=self.T,
                tenor=self.tenor, freq=self.freq),
        )

    @property
    def R_adj(self) -> float:
        """The convexity-adjusted rate actually used for pricing."""
        return self.legs()[1].R_adj

    def result(self) -> dict:
        eq, cms = self.legs()
        return conditional_european(eq, cms, self.T, self.rho,
                                    self.eq_type, self.cond_dir)

    def price(self) -> float:
        return self.result()["price"]

    def digital(self, notional: float = 1.0) -> dict:
        """The matching double digital on the same two conditions."""
        eq, cms = self.legs()
        eq_dir = "above" if self.eq_type == "call" else "below"
        return double_digital(eq, cms, self.T, self.rho, notional,
                              eq_dir, self.cond_dir)


def _price(t: EQIRTrade) -> float:
    return t.price()


def greeks(t: EQIRTrade) -> dict:
    """Bump-and-revalue risk, in the conventions each desk actually uses.

    Conventions
    -----------
    - ``delta_eq``    : dV/dF, per unit of equity forward
    - ``gamma_eq``    : d2V/dF2
    - ``vega_eq``     : per 1 equity vol point (1%)
    - ``dv01``        : per +1bp on the forward swap rate. Convexity is
                        re-derived inside the bump.
    - ``rate_vega``   : per +1bp of swap-rate *normal* vol. Note this moves
                        the convexity adjustment as well as the distribution
                        -- both effects are included, as they should be.
    - ``cega``        : per 1 correlation point (0.01)
    - ``discount_dv01``: per +1bp parallel shift of the discount curve,
                        applied as P(0,T) -> P(0,T) exp(-1bp * T)

    Partial derivatives, not curve scenarios
    ----------------------------------------
    ``dv01`` moves the CMS forward while holding the equity forward and the
    discount factor fixed, and ``discount_dv01`` moves the discount factor
    alone. A real parallel shift of the curve would move all three together;
    these are the decomposed sensitivities, which is what a risk report wants
    but not what a scenario P&L wants. Sum them only if you mean to.
    """
    dF = 0.01 * t.F        # 1% of forward
    dv = 0.01              # 1 equity vol point
    dR = 5 * BP            # 5bp on the swap rate, for central-difference noise
    dsig = 5 * BP          # 5bp of normal vol
    drho = 0.01

    # A +1bp parallel curve shift scales the discount factor by exp(-1bp*T).
    dP = t.P0T * (np.exp(-BP * t.T) - 1.0)

    return {
        "delta_eq": central_diff(_price, t, "F", dF),
        "gamma_eq": second_diff(_price, t, "F", dF),
        "vega_eq": central_diff(_price, t, "sig_S", dv) * dv,
        "dv01": central_diff(_price, t, "R_0", dR) * BP,
        "rate_vega": central_diff(_price, t, "sig_R", dsig) * BP,
        "cega": central_diff(_price, t, "rho", drho) * drho,
        "discount_dv01": central_diff(_price, t, "P0T", dP) * dP,
    }
