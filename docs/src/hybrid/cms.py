"""CMS convexity adjustment (analytic approximation).

Why it is needed
----------------
A forward swap rate is a martingale under the *annuity* measure Q^A, whose
numeraire is the swap's annuity A(t). Our hybrids price under the T-forward
measure Q^T, and a swap rate is *not* a martingale there:

    E^T[R_T]  !=  R_0    (the forward swap rate)

The gap is the convexity adjustment, and it is positive: the annuity is a
convex decreasing function of the rate, so high-rate scenarios carry more
weight under Q^T than under Q^A.

Feeding an unadjusted forward swap rate into
:class:`~src.hybrid.conditions.RateCondition` therefore understates the
probability of an upper barrier being breached, and systematically misprices.

The approximation
-----------------
Write the swap's annuity as a function of a single flat yield y::

    G(y) = sum_{i=1..n} delta / (1 + delta*y)^i        delta = 1/freq

Expanding the measure change to second order (Hull, "Options, Futures and
Other Derivatives", ch. on convexity/timing adjustments; equivalently the
first-order case of Hagan, "Convexity Conundrums", 2003) gives

    E^T[R_T] ~= R_0 - 0.5 * var * T * G''(R_0) / G'(R_0)

with ``var = sigma_N^2`` for a normal/Bachelier vol, or ``sigma_LN^2 R_0^2``
for a lognormal one. Since G' < 0 and G'' > 0, the adjustment is positive.

What this approximation ignores
-------------------------------
- **The smile.** A single vol is used, so this cannot see the skew that full
  static replication (integrating over the swaption strike continuum) would.
  Expect it to differ from a replication-based number by a few tenths of a bp
  to a few bp, growing with maturity, tenor and skew steepness.
- **Higher-order terms.** Second-order Taylor only; error grows with
  ``sigma^2 T``.
- **The shape of the distribution.** Only the *mean* is adjusted -- the leg
  keeps its original vol and its normal (or shifted-lognormal) shape.

There is no payment-delay term here, and for these products there should not
be: the hybrids condition on the CMS *fixing* at T and settle at T, so there
is no CMS coupon paid later to adjust for.
"""
from __future__ import annotations

import numpy as np

__all__ = ["annuity", "annuity_first_derivative", "annuity_second_derivative",
           "convexity_adjustment", "adjusted_cms_rate"]


def _periods(tenor: float, freq: int) -> np.ndarray:
    """Payment indices 1..n for a `tenor`-year swap paying `freq` times a year."""
    if freq <= 0:
        raise ValueError(f"freq must be positive, got {freq}")
    if tenor <= 0:
        raise ValueError(f"tenor must be positive, got {tenor}")
    n_exact = tenor * freq
    n = int(round(n_exact))
    if abs(n_exact - n) > 1e-9:
        raise ValueError(
            f"tenor*freq must be a whole number of periods, got {n_exact}")
    return np.arange(1, n + 1)


def annuity(y: float, tenor: float, freq: int = 2) -> float:
    """G(y): annuity per unit notional at flat yield ``y``."""
    i = _periods(tenor, freq)
    delta = 1.0 / freq
    return float(np.sum(delta / (1.0 + delta * y) ** i))


def annuity_first_derivative(y: float, tenor: float, freq: int = 2) -> float:
    """G'(y). Negative: the annuity falls as the yield rises."""
    i = _periods(tenor, freq)
    delta = 1.0 / freq
    return float(-np.sum(delta ** 2 * i / (1.0 + delta * y) ** (i + 1)))


def annuity_second_derivative(y: float, tenor: float, freq: int = 2) -> float:
    """G''(y). Positive: the annuity is convex in the yield."""
    i = _periods(tenor, freq)
    delta = 1.0 / freq
    return float(np.sum(delta ** 3 * i * (i + 1) / (1.0 + delta * y) ** (i + 2)))


def convexity_adjustment(R_0: float, T: float, sigma: float, tenor: float,
                         freq: int = 2, vol_type: str = "normal") -> float:
    """Additive CMS convexity adjustment, in the rate's own units.

    Parameters
    ----------
    R_0      : forward swap rate (decimal, e.g. 0.041)
    T        : time to the CMS fixing, in years
    sigma    : swap-rate volatility. Absolute/normal (0.0080 = 80bp) when
               ``vol_type='normal'``; relative when ``'lognormal'``.
    tenor    : swap tenor in years (10 for CMS10)
    freq     : coupon frequency of the underlying swap, per year
    vol_type : 'normal' (default, Bachelier) or 'lognormal'

    Returns
    -------
    float : the adjustment to *add* to ``R_0``. Always positive for a
            standard (positive-yield) swap.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    if vol_type == "normal":
        var = sigma ** 2
    elif vol_type == "lognormal":
        var = (sigma * R_0) ** 2
    else:
        raise ValueError(
            f"vol_type must be 'normal' or 'lognormal', got {vol_type!r}")

    g1 = annuity_first_derivative(R_0, tenor, freq)
    g2 = annuity_second_derivative(R_0, tenor, freq)
    return -0.5 * var * T * g2 / g1


def adjusted_cms_rate(R_0: float, T: float, sigma: float, tenor: float,
                      freq: int = 2, vol_type: str = "normal") -> float:
    """Convexity-adjusted CMS rate ``E^T[R_T]``.

    This is what :class:`~src.hybrid.conditions.RateCondition` wants for its
    ``R_adj``; see :meth:`RateCondition.from_forward_swap` for the shortcut.
    """
    return R_0 + convexity_adjustment(R_0, T, sigma, tenor, freq, vol_type)
