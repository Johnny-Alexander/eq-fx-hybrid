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

__all__ = ["ConditionLeg", "FXCondition"]


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
