"""The equity leg of a hybrid payoff.

Phase 1 keeps the original convention -- constant ``r_d`` with the forward
implied as ``S0 exp((r_d - q) T)``. Phase 2 moves to the T-forward measure,
taking the market forward and P(0,T) as inputs instead; that change is
confined to this module and leaves the conditioning legs untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bivariate import norm

__all__ = ["EquityLeg"]


@dataclass
class EquityLeg:
    """Lognormal equity under the domestic risk-neutral measure.

    Attributes
    ----------
    S0    : spot
    K     : strike
    q     : continuous dividend yield
    r_d   : domestic risk-free rate (continuous)
    sig_S : volatility
    """

    S0: float
    K: float
    q: float
    r_d: float
    sig_S: float

    def d1(self, T: float) -> float:
        return ((np.log(self.S0 / self.K) + (self.r_d - self.q + 0.5 * self.sig_S ** 2) * T)
                / (self.sig_S * np.sqrt(T)))

    def d2(self, T: float) -> float:
        return self.d1(T) - self.sig_S * np.sqrt(T)

    def discount(self, T: float) -> float:
        return np.exp(-self.r_d * T)

    def forward_factor(self, T: float) -> float:
        """Coefficient of the first closed-form term: S0 exp(-qT)."""
        return self.S0 * np.exp(-self.q * T)

    def vanilla(self, T: float, eta_S: int = 1) -> float:
        """Unconditional Black-Scholes call (eta_S=+1) or put (eta_S=-1)."""
        d1, d2 = self.d1(T), self.d2(T)
        if eta_S == 1:
            return (self.forward_factor(T) * norm.cdf(d1)
                    - self.K * self.discount(T) * norm.cdf(d2))
        return (self.K * self.discount(T) * norm.cdf(-d2)
                - self.forward_factor(T) * norm.cdf(-d1))

    def simulate(self, z: np.ndarray, T: float) -> np.ndarray:
        """Terminal equity level for standard normal draws ``z`` (for MC)."""
        return self.S0 * np.exp((self.r_d - self.q - 0.5 * self.sig_S ** 2) * T
                                + self.sig_S * np.sqrt(T) * z)
