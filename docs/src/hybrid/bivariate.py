"""Shared normal-distribution machinery for the hybrid pricers."""
from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal, norm

__all__ = ["bivariate_normal_cdf", "bivariate_normal_cdf_vec", "norm"]

#: Beyond this the standard normal CDF is 0 or 1 to double precision, and the
#: quadrature integrand underflows to zero. Clipping keeps +-inf out of the
#: arithmetic, which matters for surfaces where tau -> 0 sends d2 and h to
#: infinity at every point off the strike/barrier.
_ARG_CLIP = 40.0


def bivariate_normal_cdf(a: float, b: float, rho: float) -> float:
    """P(Z1 <= a, Z2 <= b) where (Z1, Z2) is standard bivariate normal with
    correlation rho."""
    return multivariate_normal(mean=[0.0, 0.0],
                               cov=[[1.0, rho], [rho, 1.0]]).cdf([a, b])


def bivariate_normal_cdf_vec(a, b, rho: float, n_nodes: int = 64):
    """Vectorised :func:`bivariate_normal_cdf`, broadcasting over ``a``/``b``.

    Same quantity, but scipy's ``multivariate_normal.cdf`` is scalar-only and
    costs ~100us a call. A PnL surface is 140x140 points x 300 animation
    frames -- six million evaluations -- so the scalar path is hours and this
    one is seconds.

    Uses the Drezner-Wesolowsky identity in Genz's trigonometric form::

        M2(a,b;rho) = Phi(a)Phi(b)
                      + 1/(2 pi) Int_0^{asin rho}
                            exp( -(a^2 - 2ab sin t + b^2) / (2 cos^2 t) ) dt

    which removes the ``1/sqrt(1-r^2)`` endpoint singularity of the textbook
    form, leaving an analytic integrand that Gauss-Legendre nails. Measured
    agreement with the scalar implementation is at machine precision (max abs
    error ~2e-16) across ``|rho| <= 0.99`` and ``a, b`` in [-4, 4]; see
    ``tests/test_surface.py``, which pins it. Correlation is a scalar: one
    surface, one rho.
    """
    a = np.clip(np.asarray(a, dtype=float), -_ARG_CLIP, _ARG_CLIP)
    b = np.clip(np.asarray(b, dtype=float), -_ARG_CLIP, _ARG_CLIP)
    rho = float(np.clip(rho, -0.999999, 0.999999))

    independent = norm.cdf(a) * norm.cdf(b)
    theta = np.arcsin(rho)
    if theta == 0.0:
        return independent

    # Gauss-Legendre nodes mapped from [-1, 1] onto [0, asin(rho)].
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    t = 0.5 * theta * (x + 1.0)
    sin_t, cos2_t = np.sin(t), np.cos(t) ** 2

    a_, b_ = a[..., None], b[..., None]
    integrand = np.exp(-(a_ ** 2 - 2.0 * a_ * b_ * sin_t + b_ ** 2)
                       / (2.0 * cos2_t))
    integral = 0.5 * theta * np.sum(w * integrand, axis=-1)

    return independent + integral / (2.0 * np.pi)
