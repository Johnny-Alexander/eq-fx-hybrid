"""Tests for the src.hybrid package and its compatibility shim.

The existing tests/test_hybrid.py exercises the legacy flat API. These cover
the layered one, plus the invariants that the ConditionLeg abstraction has to
preserve if a rates leg is going to slot in unchanged.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools

import numpy as np
import pytest

from src.hybrid import (EquityLeg, FXCondition, conditional_european,
                        conditional_european_mc, double_digital)
from src.hybrid_pricer import price_hybrid, price_double_digital
from src.trade_config import EXAMPLE_TRADE as p

DIRS = list(itertools.product(["call", "put"], ["above", "below"]))


def test_shim_matches_package():
    """The legacy flat API must agree with the layered one to ~machine
    precision -- they are the same arithmetic, differently grouped."""
    eq, fx = p.legs()
    for eq_type, fx_dir in DIRS:
        legacy = price_hybrid(p, eq_type, fx_dir)["price"]
        layered = conditional_european(eq, fx, p.T, p.rho, eq_type, fx_dir)["price"]
        assert legacy == pytest.approx(layered, rel=1e-12, abs=1e-12)


def test_condition_direction_parity():
    """above + below = unconditional vanilla, for calls and puts."""
    eq, fx = p.legs()
    for eq_type in ("call", "put"):
        up = conditional_european(eq, fx, p.T, p.rho, eq_type, "above")
        dn = conditional_european(eq, fx, p.T, p.rho, eq_type, "below")
        assert up["price"] + dn["price"] == pytest.approx(
            up["vanilla_equivalent"], rel=1e-12)


def test_double_digital_quadrants_sum_to_df():
    """The four (eq, condition) quadrants partition the sample space, so
    their undiscounted probabilities sum to 1."""
    eq, fx = p.legs()
    total = sum(double_digital(eq, fx, p.T, p.rho, 1.0, ed, cd)["P_joint"]
                for ed, cd in itertools.product(["above", "below"],
                                                ["above", "below"]))
    assert total == pytest.approx(1.0, abs=1e-10)


def test_measure_shift_is_dynamics_independent():
    """The equity-measure shift must be rho*sig_S*sqrt(T) and must not depend
    on the conditioning leg's own dynamics. This is the property that lets a
    normal-rate leg reuse the FX closed form."""
    _, fx = p.legs()
    expected = p.rho * p.sig_S * np.sqrt(p.T)
    assert fx.equity_measure_shift(p.sig_S, p.rho, p.T) == pytest.approx(expected)

    # Changing only the leg's own vol must not move the shift.
    fx_high_vol = FXCondition(X0=fx.X0, B=fx.B, r_d=fx.r_d, r_f=fx.r_f,
                              sig_X=fx.sig_X * 3)
    assert fx_high_vol.equity_measure_shift(p.sig_S, p.rho, p.T) == pytest.approx(expected)


def test_closed_form_matches_mc_all_directions():
    """MC agreement across all four sign combinations, not just the call."""
    eq, fx = p.legs()
    for eq_type, fx_dir in DIRS:
        cf = conditional_european(eq, fx, p.T, p.rho, eq_type, fx_dir)["price"]
        mc = conditional_european_mc(eq, fx, p.T, p.rho, eq_type, fx_dir,
                                     n_paths=400_000, seed=3)
        assert abs(cf - mc["price"]) < 3.5 * mc["std_error"], (
            f"{eq_type}/{fx_dir}: closed form {cf:.4f} vs MC {mc['price']:.4f}")


def test_double_digital_matches_shim():
    eq, fx = p.legs()
    for ed, cd in itertools.product(["above", "below"], ["above", "below"]):
        legacy = price_double_digital(p, 1.0, ed, cd)
        layered = double_digital(eq, fx, p.T, p.rho, 1.0, ed, cd)
        assert legacy["price"] == pytest.approx(layered["price"], rel=1e-12)
        assert legacy["P_fx"] == pytest.approx(layered["P_condition"], rel=1e-12)


def test_bad_direction_labels_raise():
    eq, fx = p.legs()
    with pytest.raises(ValueError, match="eq direction"):
        conditional_european(eq, fx, p.T, p.rho, "banana", "above")
    with pytest.raises(ValueError, match="condition direction"):
        conditional_european(eq, fx, p.T, p.rho, "call", "sideways")
