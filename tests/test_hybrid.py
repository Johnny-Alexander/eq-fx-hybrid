"""Sanity tests: closed form matches Monte Carlo, prices are well-bounded."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace
import numpy as np

from src.hybrid_pricer import price_hybrid_call, price_hybrid_call_mc, greeks
from src.trade_config import EXAMPLE_TRADE as p


def test_closed_form_matches_mc():
    cf = price_hybrid_call(p)
    mc = price_hybrid_call_mc(p, n_paths=500_000, seed=0)
    diff = abs(cf["price"] - mc["price"])
    # Should be within ~3 standard errors
    assert diff < 3 * mc["std_error"], (
        f"closed form {cf['price']:.4f} vs MC {mc['price']:.4f} "
        f"diff {diff:.4f} > 3 SE {3*mc['std_error']:.4f}"
    )


def test_hybrid_below_vanilla():
    # Hybrid <= vanilla because indicator <= 1
    res = price_hybrid_call(p)
    assert res["price"] <= res["vanilla_SPX_call"] + 1e-6


def test_hybrid_nonneg():
    assert price_hybrid_call(p)["price"] >= 0


def test_monotone_in_correlation():
    # For an SPX call * FX digital, price should be increasing in rho:
    # positive rho means the joint event {S>K, X>B} co-occurs more often.
    rhos = [-0.5, 0.0, 0.5, 0.9]
    prices = [price_hybrid_call(replace(p, rho=r))["price"] for r in rhos]
    for i in range(len(prices) - 1):
        assert prices[i] <= prices[i + 1], f"non-monotone in rho: {prices}"


def test_barrier_limits():
    # As B -> 0, hybrid -> vanilla call
    far_below = price_hybrid_call(replace(p, B=1.0))
    assert abs(far_below["price"] - far_below["vanilla_SPX_call"]) < 1e-3
    # As B -> infinity, hybrid -> 0
    far_above = price_hybrid_call(replace(p, B=10000.0))
    assert far_above["price"] < 1e-3


def test_greeks_finite():
    g = greeks(p)
    for k, v in g.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nAll tests passed.")
