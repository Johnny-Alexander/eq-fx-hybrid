"""Generic bump-and-revalue risk.

Kept deliberately free of asset-class conventions: these helpers only know
how to perturb a field of a frozen-ish params object and re-price. Scaling
(per vol point, per bp, per 1% spot move) belongs to the caller, because the
conventions differ by asset class -- FX quotes delta per 1% spot move while
rates quote DV01 per basis point.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

__all__ = ["bumped", "central_diff", "second_diff"]

PriceFn = Callable[[Any], float]


def bumped(price_fn: PriceFn, params: Any, field: str, h: float) -> float:
    """Re-price with ``field`` shifted by ``h``."""
    return price_fn(replace(params, **{field: getattr(params, field) + h}))


def central_diff(price_fn: PriceFn, params: Any, field: str, h: float) -> float:
    """Central first derivative w.r.t. ``field``."""
    return (bumped(price_fn, params, field, h)
            - bumped(price_fn, params, field, -h)) / (2 * h)


def second_diff(price_fn: PriceFn, params: Any, field: str, h: float) -> float:
    """Central second derivative w.r.t. ``field``."""
    base = price_fn(params)
    return (bumped(price_fn, params, field, h)
            - 2 * base
            + bumped(price_fn, params, field, -h)) / (h ** 2)
