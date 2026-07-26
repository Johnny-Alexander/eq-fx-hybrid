"""EQ/IR trade container and rates-convention risk."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace

import numpy as np
import pytest

from src.hybrid import conditional_european, conditional_european_mc
from src.hybrid.eqir import BP, EQIRTrade, greeks

T = 2.0
TRADE = EQIRTrade(
    F=7200.0 * np.exp((0.045 - 0.015) * T),
    K=7000.0,
    P0T=np.exp(-0.045 * T),
    sig_S=0.16,
    R_0=0.0410,
    B=0.04,
    sig_R=0.0080,
    T=T,
    rho=-0.30,
)


# -- container -----------------------------------------------------------

def test_price_matches_direct_leg_construction():
    eq, cms = TRADE.legs()
    direct = conditional_european(eq, cms, TRADE.T, TRADE.rho,
                                  TRADE.eq_type, TRADE.cond_dir)["price"]
    assert TRADE.price() == pytest.approx(direct, rel=1e-15)


def test_stored_rate_is_unadjusted():
    """R_0 is the raw forward swap rate; R_adj is derived and strictly above it."""
    assert TRADE.R_adj > TRADE.R_0
    assert TRADE.R_adj - TRADE.R_0 == pytest.approx(4.45e-4, abs=0.5e-4)


def test_matches_monte_carlo():
    eq, cms = TRADE.legs()
    mc = conditional_european_mc(eq, cms, TRADE.T, TRADE.rho, TRADE.eq_type,
                                 TRADE.cond_dir, n_paths=500_000, seed=21)
    assert abs(TRADE.price() - mc["price"]) < 3.5 * mc["std_error"]


# -- risk signs ----------------------------------------------------------

def test_equity_greek_signs():
    g = greeks(TRADE)
    assert g["delta_eq"] > 0        # call is long the equity forward
    assert g["gamma_eq"] > 0        # long optionality
    assert g["vega_eq"] > 0


def test_dv01_positive_for_rate_above_barrier():
    """Higher rates make an upper barrier more likely, so a call x 1{R>B}
    gains value."""
    assert greeks(TRADE)["dv01"] > 0


def test_dv01_flips_with_condition_direction():
    below = replace(TRADE, cond_dir="below")
    assert greeks(below)["dv01"] == pytest.approx(-greeks(TRADE)["dv01"],
                                                  rel=1e-6)


def test_discount_dv01_negative():
    """A higher discount rate lowers the present value."""
    assert greeks(TRADE)["discount_dv01"] < 0


def test_cega_positive_for_call_above():
    """Positive correlation makes {equity up, rates up} co-occur more often."""
    assert greeks(TRADE)["cega"] > 0


def test_rate_vega_negative_when_condition_is_in_the_money():
    """R_adj (4.14%) sits above the 4% barrier, so P(cond) > 50%. More rate
    dispersion pulls it back toward 50% and the structure loses value --
    even though convexity raises R_adj at the same time."""
    g = greeks(TRADE)
    assert TRADE.R_adj > TRADE.B
    assert g["rate_vega"] < 0


# -- the thing a frozen R_adj would get wrong ----------------------------

def test_dv01_includes_convexity_feedback():
    """Bumping R_0 must re-derive the convexity adjustment. If R_adj were
    frozen, d(R_adj)/d(R_0) would be exactly 1; the annuity derivatives
    depend on R_0, so it is close to but not equal to 1."""
    dR = 5 * BP
    up = replace(TRADE, R_0=TRADE.R_0 + dR).R_adj
    dn = replace(TRADE, R_0=TRADE.R_0 - dR).R_adj
    slope = (up - dn) / (2 * dR)
    assert slope != pytest.approx(1.0, abs=1e-6)
    assert slope == pytest.approx(1.0, abs=1e-2)   # but only just


def test_rate_vega_captures_both_effects():
    """Raising vol both widens the distribution and raises R_adj. Confirm the
    second effect is really present, so rate_vega is not a pure dispersion
    number."""
    lo = replace(TRADE, sig_R=0.0070).R_adj
    hi = replace(TRADE, sig_R=0.0090).R_adj
    assert hi > lo


# -- digital -------------------------------------------------------------

def test_digital_probability_bounds():
    d = TRADE.digital(notional=1.0)
    assert 0.0 < d["P_joint"] < min(d["P_eq"], d["P_condition"])
    assert d["price"] == pytest.approx(TRADE.P0T * d["P_joint"], rel=1e-12)
