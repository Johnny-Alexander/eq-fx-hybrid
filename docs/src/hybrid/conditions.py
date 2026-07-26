"""Conditioning legs for hybrid payoffs.

A hybrid here is an equity payoff switched on or off by an indicator on some
*other* observable -- an FX rate today, a CMS rate next. The pricing insight
that makes one codebase cover both is that the conditioning leg enters the
closed form through exactly two scalars:

    h(T)                      standardised threshold, so that
                              P(condition holds) = Phi(eta_R * h)

    equity_measure_shift(..)  what h shifts by when the numeraire changes
                              from cash to the equity

and the shift is ``rho * sig_S * sqrt(T)`` *regardless of the leg's dynamics*.
It is a shift on the standard normal driving the condition, so it does not
care whether that normal drives a lognormal FX rate or a normal CMS rate.
Only ``h`` differs between asset classes:

    FX  (lognormal)   h = [ln(X0/B) + (mu_X - 0.5 sig_X^2) T] / (sig_X sqrt(T))
    CMS (normal)      h = (R_adj - B) / (sig_R sqrt(T))          <- no log

Subclasses therefore implement ``h`` and inherit everything else.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = ["ConditionLeg", "FXCondition", "RateCondition",
           "ShiftedLognormalRateCondition"]


class ConditionLeg(ABC):
    """A binary condition on a non-equity observable, priced at expiry T."""

    #: Human-readable name for the observable, used in result dict keys.
    label: str = "condition"

    @abstractmethod
    def h(self, T: float) -> float:
        """Standardised threshold: P(condition) = Phi(eta_R * h(T))."""

    def equity_measure_shift(self, sig_S: float, rho: float, T: float) -> float:
        """Shift applied to ``h`` under the equity (share) measure.

        Changing numeraire from cash to S translates the standard normal
        driving this leg by ``rho * sig_S * sqrt(T)``. This is dynamics-
        independent, which is why FX and rates share one implementation.
        """
        return rho * sig_S * np.sqrt(T)

    def sigma(self) -> float:
        """Volatility of the conditioning observable, in its own units.

        Lognormal legs report a relative vol; normal legs report an absolute
        one. Used only for Greeks bump sizing, never in the closed form.
        """
        raise NotImplementedError

    @abstractmethod
    def simulate(self, z: np.ndarray, T: float) -> np.ndarray:
        """Terminal level of the observable for standard normal draws ``z``."""

    def indicator(self, z: np.ndarray, T: float, eta_R: int) -> np.ndarray:
        """1 where the condition holds, for MC. Requires a barrier ``B``."""
        return (eta_R * (self.simulate(z, T) - self.B) > 0).astype(float)


@dataclass
class FXCondition(ConditionLeg):
    """Lognormal FX rate, quoted domestic-per-foreign (e.g. USDJPY).

    Under the domestic risk-neutral measure the drift carries a quanto
    correction::

        mu_X = (r_f - r_d) + sig_X^2

    The ``+sig_X^2`` is *not* Garman-Kohlhagen -- it arises from changing
    numeraire from foreign-cash to domestic-cash via Ito on 1/X. See the
    module docstring of :mod:`src.hybrid_pricer` for the full derivation.

    Attributes
    ----------
    X0    : FX spot (domestic per foreign)
    B     : barrier level
    r_d   : domestic risk-free rate (continuous)
    r_f   : foreign risk-free rate (continuous)
    sig_X : FX volatility (relative)
    """

    X0: float
    B: float
    r_d: float
    r_f: float
    sig_X: float

    label: str = "FX"

    def mu(self) -> float:
        """Drift under the domestic risk-neutral measure."""
        return (self.r_f - self.r_d) + self.sig_X ** 2

    def h(self, T: float) -> float:
        return ((np.log(self.X0 / self.B) + (self.mu() - 0.5 * self.sig_X ** 2) * T)
                / (self.sig_X * np.sqrt(T)))

    def sigma(self) -> float:
        return self.sig_X

    def simulate(self, z: np.ndarray, T: float) -> np.ndarray:
        """Terminal FX level for standard normal draws ``z`` (for MC)."""
        return self.X0 * np.exp((self.mu() - 0.5 * self.sig_X ** 2) * T
                                + self.sig_X * np.sqrt(T) * z)


# -- rates ---------------------------------------------------------------
#
# Units, because this is where hybrids go wrong quietly: rates and barriers
# are decimals (0.04 = 4%), and normal vol is in absolute rate units per
# sqrt(year) (0.0080 = 80bp). The guards below reject the common slips --
# passing 4 for 4%, or 80 for 80bp -- rather than returning a plausible
# looking wrong number.

_MAX_PLAUSIBLE_RATE = 1.0      # 100%
_MAX_PLAUSIBLE_NORMAL_VOL = 0.5  # 5000bp/yr


def _check_rate_units(**values: float) -> None:
    for name, v in values.items():
        if abs(v) > _MAX_PLAUSIBLE_RATE:
            raise ValueError(
                f"{name}={v} looks like a percentage, not a decimal rate. "
                f"Pass 0.04 for 4%, not 4.")


@dataclass
class RateCondition(ConditionLeg):
    """Normal (Bachelier) rate, e.g. a 10y CMS fixing.

    Payoff condition: ``1{R_T > B}`` (or ``<``, via the product's direction
    argument), where R is the rate observed at T -- typically a CMS rate.

    The standardised threshold has no log and no ``-0.5 sig^2 T`` term::

        h = (R_adj - B) / (sig_R sqrt(T))

    which is the *only* thing separating this from :class:`FXCondition`. The
    equity-measure shift is inherited unchanged, because it acts on the
    standard normal driving the condition rather than on the rate itself.
    Verified against Monte Carlo across call/put x above/below x rho in
    [-0.6, 0.6]; see tests/test_rates.py.

    Bachelier is the default because it is the post-2015 market convention
    for swaption and CMS vol, and it handles zero and negative rates without
    a displacement. Use :class:`ShiftedLognormalRateCondition` if your vols
    are quoted shifted-lognormal.

    Attributes
    ----------
    R_adj : convexity-adjusted forward rate under Q^T.

            NOT the plain forward swap rate. A CMS rate is not a martingale
            under the T-forward measure, so E^T[R_T] exceeds the forward swap
            rate by a convexity adjustment; there is a further adjustment if
            the fixing and payment dates differ. Computing that adjustment
            (Hagan-style static replication off a swaption cube) is not yet
            implemented -- this leg takes the adjusted rate as an input, and
            passing the unadjusted forward will systematically misprice.
    B     : barrier level, as a decimal (0.04 = 4%)
    sig_R : normal/absolute volatility per sqrt(year) (0.0080 = 80bp)
    """

    R_adj: float
    B: float
    sig_R: float

    label: str = "rate"

    def __post_init__(self) -> None:
        _check_rate_units(R_adj=self.R_adj, B=self.B)
        if not 0 < self.sig_R <= _MAX_PLAUSIBLE_NORMAL_VOL:
            raise ValueError(
                f"sig_R={self.sig_R} is not a plausible normal vol. Pass "
                f"absolute rate units per sqrt(year): 0.0080 for 80bp, not 80.")

    @classmethod
    def from_forward_swap(cls, R_0: float, B: float, sig_R: float, T: float,
                          tenor: float, freq: int = 2) -> "RateCondition":
        """Build from the *unadjusted* forward swap rate, adjusting internally.

        Prefer this over passing ``R_adj`` by hand -- it is the path that
        cannot silently omit the convexity adjustment.

            >>> RateCondition.from_forward_swap(
            ...     R_0=0.0410, B=0.04, sig_R=0.0080, T=2.0, tenor=10)

        See :mod:`src.hybrid.cms` for the approximation used and its limits.
        """
        from .cms import adjusted_cms_rate
        R_adj = adjusted_cms_rate(R_0, T, sig_R, tenor, freq,
                                  vol_type="normal")
        return cls(R_adj=R_adj, B=B, sig_R=sig_R)

    def h(self, T: float) -> float:
        return (self.R_adj - self.B) / (self.sig_R * np.sqrt(T))

    def sigma(self) -> float:
        return self.sig_R

    def simulate(self, z: np.ndarray, T: float) -> np.ndarray:
        return self.R_adj + self.sig_R * np.sqrt(T) * z


@dataclass
class ShiftedLognormalRateCondition(ConditionLeg):
    """Shifted-lognormal rate: ``R + a`` is lognormal, for displacement ``a``.

    Reduces to plain lognormal Black when ``shift = 0``, and permits rates
    down to ``-shift``. Here ``sig_R`` is a *relative* vol on the displaced
    rate, so it is not interchangeable with :class:`RateCondition`'s.

        h = [ln((R_adj + a)/(B + a)) - 0.5 sig_R^2 T] / (sig_R sqrt(T))
    """

    R_adj: float
    B: float
    sig_R: float
    shift: float = 0.0

    label: str = "rate"

    def __post_init__(self) -> None:
        _check_rate_units(R_adj=self.R_adj, B=self.B, shift=self.shift)
        if self.R_adj + self.shift <= 0 or self.B + self.shift <= 0:
            raise ValueError(
                f"shift={self.shift} must make both the rate and the barrier "
                f"positive: R_adj+shift={self.R_adj + self.shift}, "
                f"B+shift={self.B + self.shift}")
        if self.sig_R <= 0:
            raise ValueError(f"sig_R must be positive, got {self.sig_R}")

    def h(self, T: float) -> float:
        return ((np.log((self.R_adj + self.shift) / (self.B + self.shift))
                 - 0.5 * self.sig_R ** 2 * T)
                / (self.sig_R * np.sqrt(T)))

    def sigma(self) -> float:
        return self.sig_R

    def simulate(self, z: np.ndarray, T: float) -> np.ndarray:
        displaced = (self.R_adj + self.shift) * np.exp(
            -0.5 * self.sig_R ** 2 * T + self.sig_R * np.sqrt(T) * z)
        return displaced - self.shift
