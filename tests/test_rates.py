"""EQ/IR hybrids: an equity payoff conditioned on a rate, e.g. CMS10 > 4%.

The claim these tests defend is that a rates leg needs no new pricing code --
only a different standardised threshold h. If the closed form built for FX
reproduces Monte Carlo for a *normal* rate across every sign combination,
the ConditionLeg abstraction is doing its job.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools

import numpy as np
import pytest

from src.hybrid import (EquityLeg, RateCondition, ShiftedLognormalRateCondition,
                        conditional_european, conditional_european_mc,
                        double_digital, double_digital_mc)

# SPX call contingent on 10y CMS above 4%, two years out.
EQ = EquityLeg.from_market(F=7280.0, K=7000.0, P0T=np.exp(-0.045 * 2.0),
                           sig_S=0.16, T=2.0)
CMS = RateCondition(R_adj=0.0415, B=0.04, sig_R=0.0080)  # 80bp/yr normal vol
T = 2.0

DIRS = list(itertools.product(["call", "put"], ["above", "below"]))
RHOS = (-0.6, -0.3, 0.0, 0.3, 0.6)


@pytest.mark.parametrize("eq_type,cond_dir", DIRS)
@pytest.mark.parametrize("rho", RHOS)
def test_closed_form_matches_mc(eq_type, cond_dir, rho):
    cf = conditional_european(EQ, CMS, T, rho, eq_type, cond_dir)["price"]
    mc = conditional_european_mc(EQ, CMS, T, rho, eq_type, cond_dir,
                                 n_paths=400_000, seed=5)
    assert abs(cf - mc["price"]) < 3.5 * mc["std_error"], (
        f"{eq_type}/{cond_dir} rho={rho}: closed form {cf:.4f} "
        f"vs MC {mc['price']:.4f} +/- {mc['std_error']:.4f}")


@pytest.mark.parametrize("rho", RHOS)
def test_direction_parity(rho):
    """above + below = unconditional vanilla, exactly."""
    for eq_type in ("call", "put"):
        up = conditional_european(EQ, CMS, T, rho, eq_type, "above")
        dn = conditional_european(EQ, CMS, T, rho, eq_type, "below")
        assert up["price"] + dn["price"] == pytest.approx(
            up["vanilla_equivalent"], rel=1e-12)


def test_threshold_has_no_log_or_variance_term():
    """A normal rate's h is a plain standardised distance -- this is the one
    line that differs from FX, so pin it."""
    expected = (CMS.R_adj - CMS.B) / (CMS.sig_R * np.sqrt(T))
    assert CMS.h(T) == pytest.approx(expected, rel=1e-15)


def test_marginal_probability_is_normal():
    """P(R_T > B) must be the Bachelier probability, not a lognormal one."""
    from scipy.stats import norm
    res = conditional_european(EQ, CMS, T, 0.0, "call", "above")
    expected = norm.cdf((CMS.R_adj - CMS.B) / (CMS.sig_R * np.sqrt(T)))
    assert res["P_condition"] == pytest.approx(expected, rel=1e-12)


def test_rate_can_go_negative():
    """Bachelier admits negative rates; a barrier below zero must still price
    and must be nearly certain to be exceeded from a 4.15% forward."""
    deep = RateCondition(R_adj=0.0415, B=-0.01, sig_R=0.0080)
    res = conditional_european(EQ, deep, T, 0.0, "call", "above")
    assert res["P_condition"] > 0.999
    assert res["price"] == pytest.approx(res["vanilla_equivalent"], rel=1e-4)


def test_double_digital_matches_mc():
    for eq_dir, cond_dir in itertools.product(["above", "below"], repeat=2):
        cf = double_digital(EQ, CMS, T, 0.3, 1.0, eq_dir, cond_dir)["price"]
        mc = double_digital_mc(EQ, CMS, T, 0.3, 1.0, eq_dir, cond_dir,
                               n_paths=400_000, seed=9)
        assert abs(cf - mc["price"]) < 3.5 * mc["std_error"], (
            f"{eq_dir}/{cond_dir}: {cf:.6f} vs {mc['price']:.6f}")


def test_shifted_lognormal_matches_mc():
    sln = ShiftedLognormalRateCondition(R_adj=0.0415, B=0.04, sig_R=0.20,
                                        shift=0.02)
    for eq_type, cond_dir in DIRS:
        cf = conditional_european(EQ, sln, T, 0.3, eq_type, cond_dir)["price"]
        mc = conditional_european_mc(EQ, sln, T, 0.3, eq_type, cond_dir,
                                     n_paths=400_000, seed=13)
        assert abs(cf - mc["price"]) < 3.5 * mc["std_error"], (
            f"shifted-LN {eq_type}/{cond_dir}: {cf:.4f} vs {mc['price']:.4f}")


def test_shifted_lognormal_reduces_to_black():
    """With zero shift the leg is plain lognormal Black."""
    sln = ShiftedLognormalRateCondition(R_adj=0.0415, B=0.04, sig_R=0.20)
    expected = ((np.log(0.0415 / 0.04) - 0.5 * 0.20 ** 2 * T)
                / (0.20 * np.sqrt(T)))
    assert sln.h(T) == pytest.approx(expected, rel=1e-14)


# -- unit guards ---------------------------------------------------------

def test_percentage_rate_rejected():
    with pytest.raises(ValueError, match="looks like a percentage"):
        RateCondition(R_adj=4.15, B=4.0, sig_R=0.0080)


def test_basis_point_vol_rejected():
    with pytest.raises(ValueError, match="not a plausible normal vol"):
        RateCondition(R_adj=0.0415, B=0.04, sig_R=80.0)


def test_lognormal_vol_mistaken_for_normal_is_rejected():
    """20% relative vol passed as a normal vol is 2000bp -- implausible."""
    with pytest.raises(ValueError, match="not a plausible normal vol"):
        RateCondition(R_adj=0.0415, B=0.04, sig_R=20.0)


def test_shift_must_make_rate_positive():
    with pytest.raises(ValueError, match="must make both the rate"):
        ShiftedLognormalRateCondition(R_adj=-0.01, B=0.04, sig_R=0.20,
                                      shift=0.005)
