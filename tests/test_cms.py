"""CMS convexity adjustment.

The adjustment is an approximation, so these tests pin the things that must
hold exactly (derivatives, scaling laws, sign) rather than a golden number
that only one implementation would produce.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.hybrid import EquityLeg, RateCondition, conditional_european
from src.hybrid.cms import (adjusted_cms_rate, annuity,
                            annuity_first_derivative,
                            annuity_second_derivative, convexity_adjustment)

R0, TENOR, FREQ, SIG = 0.041, 10, 2, 0.0080


# -- the annuity function ------------------------------------------------

def test_derivatives_match_finite_differences():
    h = 1e-6
    fd1 = (annuity(R0 + h, TENOR, FREQ) - annuity(R0 - h, TENOR, FREQ)) / (2 * h)
    fd2 = (annuity(R0 + h, TENOR, FREQ) - 2 * annuity(R0, TENOR, FREQ)
           + annuity(R0 - h, TENOR, FREQ)) / h ** 2
    assert annuity_first_derivative(R0, TENOR, FREQ) == pytest.approx(fd1, rel=1e-7)
    assert annuity_second_derivative(R0, TENOR, FREQ) == pytest.approx(fd2, rel=1e-4)


def test_annuity_shape():
    """Decreasing and convex in the yield -- the two facts that make the
    adjustment positive."""
    assert annuity_first_derivative(R0, TENOR, FREQ) < 0
    assert annuity_second_derivative(R0, TENOR, FREQ) > 0
    assert annuity(0.02, TENOR, FREQ) > annuity(0.06, TENOR, FREQ)


def test_annuity_approximates_tenor_at_zero_yield():
    """At zero yield the annuity is just the sum of year fractions."""
    assert annuity(0.0, 10, 2) == pytest.approx(10.0, rel=1e-12)


# -- scaling laws --------------------------------------------------------

def test_adjustment_is_positive():
    assert convexity_adjustment(R0, 2.0, SIG, TENOR, FREQ) > 0


def test_adjustment_linear_in_time():
    a1 = convexity_adjustment(R0, 1.0, SIG, TENOR, FREQ)
    a5 = convexity_adjustment(R0, 5.0, SIG, TENOR, FREQ)
    assert a5 == pytest.approx(5 * a1, rel=1e-12)


def test_adjustment_quadratic_in_vol():
    a1 = convexity_adjustment(R0, 2.0, SIG, TENOR, FREQ)
    a2 = convexity_adjustment(R0, 2.0, 2 * SIG, TENOR, FREQ)
    assert a2 == pytest.approx(4 * a1, rel=1e-12)


def test_adjustment_increases_with_tenor():
    adjustments = [convexity_adjustment(R0, 2.0, SIG, tn, FREQ)
                   for tn in (2, 5, 10, 30)]
    assert adjustments == sorted(adjustments)


def test_normal_and_lognormal_agree_at_matched_vol():
    """sigma_N = sigma_LN * R_0 is the standard vol conversion, so the two
    parameterisations must give the same adjustment."""
    a = convexity_adjustment(R0, 2.0, SIG, TENOR, FREQ, "normal")
    b = convexity_adjustment(R0, 2.0, SIG / R0, TENOR, FREQ, "lognormal")
    assert a == pytest.approx(b, rel=1e-12)


def test_magnitude_is_plausible():
    """A 10y CMS fixing 2y out on 80bp vol should adjust by a few bp -- not a
    fraction of a bp, and not tens of bp. Guards against unit slips."""
    bp = convexity_adjustment(R0, 2.0, SIG, TENOR, FREQ) * 1e4
    assert 1.0 < bp < 20.0, f"{bp:.2f}bp is outside the plausible range"


# -- integration with the pricer ----------------------------------------

def test_from_forward_swap_applies_the_adjustment():
    cond = RateCondition.from_forward_swap(R_0=R0, B=0.04, sig_R=SIG, T=2.0,
                                           tenor=TENOR, freq=FREQ)
    assert cond.R_adj > R0
    assert cond.R_adj == pytest.approx(
        adjusted_cms_rate(R0, 2.0, SIG, TENOR, FREQ), rel=1e-15)


def test_ignoring_convexity_underprices_an_upper_barrier():
    """The whole point: an unadjusted forward makes an upper barrier look
    less likely to be breached, so the structure looks too cheap."""
    eq = EquityLeg.from_market(F=7647.0, K=7000.0, P0T=0.9139, sig_S=0.16, T=2.0)
    naive = RateCondition(R_adj=R0, B=0.04, sig_R=SIG)          # wrong
    correct = RateCondition.from_forward_swap(R_0=R0, B=0.04, sig_R=SIG,
                                              T=2.0, tenor=TENOR, freq=FREQ)
    p_naive = conditional_european(eq, naive, 2.0, -0.3, "call", "above")
    p_correct = conditional_european(eq, correct, 2.0, -0.3, "call", "above")
    assert p_correct["P_condition"] > p_naive["P_condition"]
    assert p_correct["price"] > p_naive["price"]


# -- input validation ----------------------------------------------------

def test_fractional_period_count_rejected():
    with pytest.raises(ValueError, match="whole number of periods"):
        annuity(R0, tenor=10.3, freq=2)


def test_bad_vol_type_rejected():
    with pytest.raises(ValueError, match="vol_type"):
        convexity_adjustment(R0, 2.0, SIG, TENOR, FREQ, vol_type="bachelier")


@pytest.mark.parametrize("kwargs,match", [
    (dict(T=0.0), "T must be positive"),
    (dict(sigma=0.0), "sigma must be positive"),
    (dict(tenor=0), "tenor must be positive"),
    (dict(freq=0), "freq must be positive"),
])
def test_degenerate_inputs_rejected(kwargs, match):
    args = dict(R_0=R0, T=2.0, sigma=SIG, tenor=TENOR, freq=FREQ)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        convexity_adjustment(**args)
