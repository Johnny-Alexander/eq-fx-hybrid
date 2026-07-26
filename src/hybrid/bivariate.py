"""Shared normal-distribution machinery for the hybrid pricers."""
from __future__ import annotations

from scipy.stats import multivariate_normal, norm

__all__ = ["bivariate_normal_cdf", "norm"]


def bivariate_normal_cdf(a: float, b: float, rho: float) -> float:
    """P(Z1 <= a, Z2 <= b) where (Z1, Z2) is standard bivariate normal with
    correlation rho."""
    return multivariate_normal(mean=[0.0, 0.0],
                               cov=[[1.0, rho], [rho, 1.0]]).cdf([a, b])
