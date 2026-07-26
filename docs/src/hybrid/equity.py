"""The equity leg of a hybrid payoff, under the T-forward measure.

Why the T-forward measure
-------------------------
The original formulation used a constant short rate ``r_d``: it discounted at
``exp(-r_d T)`` and implied the forward as ``S0 exp((r_d - q) T)``. That is
self-consistent only while rates are deterministic -- which stops being true
the moment the payoff is *conditioned on a rate*, as an EQ/IR hybrid is.

Pricing under the T-forward measure Q^T (numeraire: the zero-coupon bond
P(0,T)) removes the contradiction without complicating the formula:

    V0 = P(0,T) * E^T[ payoff ]

P(0,T) is then a directly observable market discount factor rather than
something derived from a flat rate, and the equity forward F(T) is the market
forward -- which already embeds the whole curve. Stochastic discounting is
absorbed into two observables.

One subtlety worth stating: under Q^T the martingale is the *forward*, so
``sig_S`` is the volatility of F(T), not of spot. The two coincide only when
rates are deterministic. Quoted equity vols are spot vols, so for long-dated
hybrids with material rate vol this is a genuine (if second-order) input
adjustment, not just a relabelling.

Backwards compatibility
-----------------------
:meth:`EquityLeg.from_spot` reproduces the old parameterisation exactly --
``P(0,T) F(T) = S0 exp(-qT)`` and ``ln(F/K) = ln(S0/K) + (r_d - q)T`` are
identities, so prices are unchanged to floating-point noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .bivariate import norm

__all__ = ["EquityLeg"]


@dataclass
class EquityLeg:
    """Lognormal equity forward under the T-forward measure.

    Construct via :meth:`from_spot` (spot, carry and a flat rate -- the
    original convention) or :meth:`from_market` (an observed forward and
    discount factor -- preferred once rates are stochastic).

    Attributes
    ----------
    K          : strike
    sig_S      : volatility of the forward F(T)
    forward_fn : T -> equity forward to T
    discount_fn: T -> P(0,T)
    source     : label describing where forward/discount came from
    """

    K: float
    sig_S: float
    forward_fn: Callable[[float], float] = field(repr=False)
    discount_fn: Callable[[float], float] = field(repr=False)
    source: str = "spot-implied"

    # -- constructors ----------------------------------------------------

    @classmethod
    def from_spot(cls, S0: float, K: float, q: float, r_d: float,
                  sig_S: float) -> "EquityLeg":
        """Original convention: flat rate, forward implied by carry."""
        return cls(
            K=K,
            sig_S=sig_S,
            forward_fn=lambda T: S0 * np.exp((r_d - q) * T),
            discount_fn=lambda T: np.exp(-r_d * T),
            source="spot-implied",
        )

    @classmethod
    def from_market(cls, F: float, K: float, P0T: float, sig_S: float,
                    T: float) -> "EquityLeg":
        """Observed forward and discount factor for a single expiry ``T``.

        The leg is only valid at that expiry -- pricing it at any other
        maturity raises rather than silently extrapolating a constant.
        """
        def _at(name: str, value: float) -> Callable[[float], float]:
            def fn(T_req: float) -> float:
                if not np.isclose(T_req, T):
                    raise ValueError(
                        f"this leg was built from market {name} at T={T}; "
                        f"cannot price it at T={T_req}")
                return value
            return fn

        return cls(K=K, sig_S=sig_S,
                   forward_fn=_at("forward", F),
                   discount_fn=_at("discount factor", P0T),
                   source="market")

    # -- observables -----------------------------------------------------

    def forward(self, T: float) -> float:
        return self.forward_fn(T)

    def discount(self, T: float) -> float:
        return self.discount_fn(T)

    def forward_factor(self, T: float) -> float:
        """Coefficient of the first closed-form term: P(0,T) F(T).

        Equals ``S0 exp(-qT)`` under the spot-implied parameterisation.
        """
        return self.discount(T) * self.forward(T)

    # -- Black-Scholes pieces --------------------------------------------

    def d1(self, T: float) -> float:
        return ((np.log(self.forward(T) / self.K) + 0.5 * self.sig_S ** 2 * T)
                / (self.sig_S * np.sqrt(T)))

    def d2(self, T: float) -> float:
        return self.d1(T) - self.sig_S * np.sqrt(T)

    def vanilla(self, T: float, eta_S: int = 1) -> float:
        """Unconditional Black-Scholes call (eta_S=+1) or put (eta_S=-1)."""
        d1, d2 = self.d1(T), self.d2(T)
        df = self.discount(T)
        if eta_S == 1:
            return df * (self.forward(T) * norm.cdf(d1) - self.K * norm.cdf(d2))
        return df * (self.K * norm.cdf(-d2) - self.forward(T) * norm.cdf(-d1))

    def simulate(self, z: np.ndarray, T: float) -> np.ndarray:
        """Terminal equity level under Q^T for standard normal draws ``z``."""
        return self.forward(T) * np.exp(-0.5 * self.sig_S ** 2 * T
                                        + self.sig_S * np.sqrt(T) * z)
