"""PnL surfaces for the EQ/FX dual digital, priced on a spot grid.

The closed form in :mod:`src.hybrid.products` prices one point. This module
prices a whole ``(S, X)`` plane at once, which is what a risk surface is, and
does it fast enough to animate: the grid pricer below is the same formula
written against :func:`~src.hybrid.bivariate.bivariate_normal_cdf_vec`.

What the surface shows
----------------------
Height is **mark-to-market PnL from the seller's side** -- the bank sold the
structure, collected the premium, and now marks it:

    PnL(S, X, tau) = premium_received - N * P(0,tau) * M2(.)

so the surface is bounded above by the premium (the client never exercises)
and below by ``premium - N`` (full payout). Both bounds are hit exactly at
expiry, where the surface degenerates into a two-level step.

The interesting part is the *shape between them*. At long tau the transition
is a smooth bivariate-normal ridge. As tau -> 0 it collapses onto a corner at
``(K, B)`` -- and a corner, not an edge, is the whole point of a hybrid: the
loss requires both legs to land, so the gradient at the vertex is a
correlation exposure that neither single-asset book carries.

Everything here holds market data flat and moves only time (or only rho). It
is a risk view, not a simulated path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bivariate import bivariate_normal_cdf_vec, norm
from .products import decode_eta

__all__ = ["SurfaceGrid", "dual_digital_value_grid", "short_pnl_grid",
           "inception_premium", "time_to_expiry_schedule", "correlation_schedule"]


@dataclass(frozen=True)
class SurfaceGrid:
    """The ``(S, X)`` plane a surface is evaluated on.

    ``S`` and ``X`` are the 1-D axis vectors; ``Z`` grids returned by this
    module have shape ``(len(X), len(S))`` -- row index is FX, column index is
    equity. That is plotly's convention for ``go.Surface(x=S, y=X, z=Z)``.
    """

    S: np.ndarray
    X: np.ndarray

    @classmethod
    def around(cls, S0: float, X0: float, S_pct: float = 0.16,
               X_pct: float = 0.11, n: int = 140) -> "SurfaceGrid":
        """A grid spanning +-``S_pct`` / +-``X_pct`` around the given spots.

        Defaults are roughly +-1.4 standard deviations of a 6m move in each
        leg at the example trade's vols, which is wide enough to show both
        flat wings and the whole transition region without wasting pixels on
        territory the surface has already gone flat in.
        """
        return cls(S=np.linspace(S0 * (1 - S_pct), S0 * (1 + S_pct), n),
                   X=np.linspace(X0 * (1 - X_pct), X0 * (1 + X_pct), n))

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """``(S_mesh, X_mesh)``, both shaped ``(len(X), len(S))``."""
        return np.meshgrid(self.S, self.X)


def dual_digital_value_grid(grid: SurfaceGrid, p, tau: float,
                            notional: float, rho: float | None = None,
                            eq_dir: str = "above",
                            cond_dir: str = "above") -> np.ndarray:
    """Mark-to-market value of the dual digital across the ``(S, X)`` plane.

    Parameters
    ----------
    p     : :class:`~src.hybrid_pricer.HybridInputs`-shaped params. Only the
            rates, carry, vols and levels are read -- ``S0``/``X0``/``T`` are
            overridden by the grid and by ``tau``.
    tau   : time to expiry for this frame. ``tau <= 0`` returns the terminal
            payoff exactly, rather than dividing by ``sqrt(0)``.
    rho   : override for ``p.rho``, so a correlation sweep does not have to
            rebuild the params object per frame.

    Returns the value grid in the same units as ``notional``.
    """
    eta_S, eta_R = decode_eta(eq_dir, cond_dir)
    rho = p.rho if rho is None else rho
    S, X = grid.mesh()

    if tau <= 0.0:
        both = ((eta_S * (S - p.K) > 0) & (eta_R * (X - p.B) > 0))
        return notional * both.astype(float)

    root_tau = np.sqrt(tau)

    # Equity leg: forward to tau, then the usual d2.
    F = S * np.exp((p.r_d - p.q) * tau)
    d2 = (np.log(F / p.K) - 0.5 * p.sig_S ** 2 * tau) / (p.sig_S * root_tau)

    # FX leg: drift carries the quanto correction (+sig_X^2), matching
    # FXCondition.mu() -- see that class for why it is not Garman-Kohlhagen.
    mu_X = (p.r_f - p.r_d) + p.sig_X ** 2
    h = ((np.log(X / p.B) + (mu_X - 0.5 * p.sig_X ** 2) * tau)
         / (p.sig_X * root_tau))

    P_joint = bivariate_normal_cdf_vec(eta_S * d2, eta_R * h, eta_S * eta_R * rho)
    return notional * np.exp(-p.r_d * tau) * P_joint


def inception_premium(p, notional: float, rho: float | None = None,
                      eq_dir: str = "above", cond_dir: str = "above") -> float:
    """Premium struck at inception: the trade priced at its own spots and T."""
    eta_S, eta_R = decode_eta(eq_dir, cond_dir)
    rho = p.rho if rho is None else rho
    root_T = np.sqrt(p.T)

    F = p.S0 * np.exp((p.r_d - p.q) * p.T)
    d2 = (np.log(F / p.K) - 0.5 * p.sig_S ** 2 * p.T) / (p.sig_S * root_T)
    mu_X = (p.r_f - p.r_d) + p.sig_X ** 2
    h = ((np.log(p.X0 / p.B) + (mu_X - 0.5 * p.sig_X ** 2) * p.T)
         / (p.sig_X * root_T))

    P_joint = float(bivariate_normal_cdf_vec(np.array(eta_S * d2),
                                             np.array(eta_R * h),
                                             eta_S * eta_R * rho))
    return notional * float(np.exp(-p.r_d * p.T)) * P_joint


def short_pnl_grid(grid: SurfaceGrid, p, tau: float, notional: float,
                   premium: float, rho: float | None = None,
                   eq_dir: str = "above", cond_dir: str = "above") -> np.ndarray:
    """PnL of the **short** position: premium kept less what it is worth now.

    Bounded in ``[premium - notional, premium]``.
    """
    value = dual_digital_value_grid(grid, p, tau, notional, rho=rho,
                                    eq_dir=eq_dir, cond_dir=cond_dir)
    return premium - value


def marginal_probabilities(p, tau: float, S: float, X: float,
                           eq_dir: str = "above",
                           cond_dir: str = "above") -> tuple[float, float]:
    """``(P(equity leg lands), P(FX leg lands))`` at a point -- for readouts."""
    eta_S, eta_R = decode_eta(eq_dir, cond_dir)
    if tau <= 0.0:
        return (float(eta_S * (S - p.K) > 0), float(eta_R * (X - p.B) > 0))

    root_tau = np.sqrt(tau)
    F = S * np.exp((p.r_d - p.q) * tau)
    d2 = (np.log(F / p.K) - 0.5 * p.sig_S ** 2 * tau) / (p.sig_S * root_tau)
    mu_X = (p.r_f - p.r_d) + p.sig_X ** 2
    h = ((np.log(X / p.B) + (mu_X - 0.5 * p.sig_X ** 2) * tau)
         / (p.sig_X * root_tau))
    return float(norm.cdf(eta_S * d2)), float(norm.cdf(eta_R * h))


# -- frame schedules -----------------------------------------------------


def time_to_expiry_schedule(T: float, n_frames: int,
                            tau_min_days: float = 0.25) -> np.ndarray:
    """Time-to-expiry for each animation frame, spaced **geometrically**.

    Linear time is the obvious choice and the wrong one. The width of the
    transition region scales with ``sig sqrt(tau)``, so under linear stepping
    the first half of the clip shows almost no change and the collapse
    happens in the last three frames. Geometric spacing -- constant *relative*
    decay in tau -- makes the visible change per frame roughly constant, which
    is what makes the animation read as a smooth morph rather than a jump.

    Runs from ``T`` down to ``tau_min_days`` (a quarter-day by default; the
    surface is already visually a step there). The caller can append a
    ``tau=0`` frame for the exact terminal payoff.
    """
    tau_min = tau_min_days / 365.0
    return T * (tau_min / T) ** (np.linspace(0.0, 1.0, n_frames))


def correlation_schedule(n_frames: int, rho_max: float = 0.9,
                         pingpong: bool = True) -> np.ndarray:
    """Correlation for each frame, sweeping ``-rho_max -> +rho_max``.

    With ``pingpong`` the sweep returns to its start so the clip loops
    seamlessly on a slide that is left running.
    """
    up = np.linspace(-rho_max, rho_max, n_frames)
    if not pingpong:
        return up
    return np.concatenate([up, up[-2:0:-1]])
