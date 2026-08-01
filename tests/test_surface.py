"""The surface machinery must agree with the scalar pricer it replaces.

The grid pricer in :mod:`src.hybrid.surface` re-implements the double-digital
closed form in vectorised form for speed. Re-implementation is where a
visualisation quietly stops matching the model it claims to show, so these
tests pin the fast path against the slow one point by point.
"""
import numpy as np
import pytest

from src.hybrid import EquityLeg, FXCondition, double_digital
from src.hybrid.bivariate import bivariate_normal_cdf, bivariate_normal_cdf_vec
from src.hybrid.surface import (SurfaceGrid, correlation_schedule,
                                dual_digital_value_grid, hedged_short_pnl_grid,
                                inception_deltas, inception_premium,
                                short_pnl_grid, time_to_expiry_schedule)
from src.trade_config import EXAMPLE_TRADE as P, NOTIONAL


# -- the vectorised bivariate normal CDF ---------------------------------

@pytest.mark.parametrize("rho", [-0.99, -0.9, -0.6, -0.3, 0.0,
                                 0.3, 0.6, 0.9, 0.99])
def test_vectorised_cdf_matches_scipy(rho):
    """Machine precision against scipy's scalar implementation."""
    pts = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
    a, b = np.meshgrid(pts, pts)

    fast = bivariate_normal_cdf_vec(a, b, rho)
    slow = np.array([[bivariate_normal_cdf(ai, bi, rho) for ai in pts]
                     for bi in pts])

    assert np.max(np.abs(fast - slow)) < 1e-13


def test_vectorised_cdf_edge_cases():
    """Independence at rho=0, and the marginals in the limits."""
    a = np.array([-1.0, 0.0, 1.5])
    b = np.array([0.5, -0.5, 2.0])
    from scipy.stats import norm

    assert np.allclose(bivariate_normal_cdf_vec(a, b, 0.0),
                       norm.cdf(a) * norm.cdf(b))
    # b -> +inf recovers Phi(a); the +-40 clip is deliberately inside the
    # range where the normal CDF is 1 to double precision.
    assert np.allclose(bivariate_normal_cdf_vec(a, np.full_like(a, 50.0), 0.6),
                       norm.cdf(a), atol=1e-12)
    assert np.allclose(bivariate_normal_cdf_vec(a, np.full_like(a, -50.0), 0.6),
                       0.0, atol=1e-12)


def test_vectorised_cdf_broadcasts():
    """Grid in, grid out, with shape preserved."""
    a = np.random.default_rng(0).standard_normal((7, 11))
    b = np.random.default_rng(1).standard_normal((7, 11))
    assert bivariate_normal_cdf_vec(a, b, 0.4).shape == (7, 11)


# -- the grid pricer vs the scalar product -------------------------------

def _scalar_price(S0, X0, tau, rho, eq_dir="above", cond_dir="above"):
    eq = EquityLeg.from_spot(S0=S0, K=P.K, q=P.q, r_d=P.r_d, sig_S=P.sig_S)
    fx = FXCondition(X0=X0, B=P.B, r_d=P.r_d, r_f=P.r_f, sig_X=P.sig_X)
    return double_digital(eq, fx, T=tau, rho=rho, notional=NOTIONAL,
                          eq_dir=eq_dir, cond_dir=cond_dir)["price"]


@pytest.mark.parametrize("tau", [0.5, 0.25, 0.05, 0.01])
@pytest.mark.parametrize("eq_dir,cond_dir", [("above", "above"),
                                             ("above", "below"),
                                             ("below", "above"),
                                             ("below", "below")])
def test_grid_matches_double_digital(tau, eq_dir, cond_dir):
    """Every grid node equals the scalar closed form at that node."""
    grid = SurfaceGrid(S=np.array([6600.0, 7000.0, 7400.0]),
                       X=np.array([150.0, 160.0, 168.0]))

    Z = dual_digital_value_grid(grid, P, tau, NOTIONAL,
                                eq_dir=eq_dir, cond_dir=cond_dir)

    for i, X in enumerate(grid.X):
        for j, S in enumerate(grid.S):
            expected = _scalar_price(S, X, tau, P.rho, eq_dir, cond_dir)
            assert Z[i, j] == pytest.approx(expected, rel=1e-11, abs=1e-6)


def test_grid_row_column_orientation():
    """z[i, j] indexes FX by row and equity by column, as plotly expects.

    A transposed surface still looks plausible, so this is worth asserting:
    raising the equity level must move along the column axis.
    """
    grid = SurfaceGrid(S=np.array([6000.0, 8000.0]), X=np.array([140.0, 175.0]))
    Z = dual_digital_value_grid(grid, P, 0.25, NOTIONAL)

    # Both legs 'above': value increases left-to-right (equity up) and
    # bottom-to-top (FX up).
    assert Z[0, 1] > Z[0, 0]
    assert Z[1, 0] > Z[0, 0]


def test_rho_override_beats_params():
    grid = SurfaceGrid.around(P.S0, P.X0, n=8)
    base = dual_digital_value_grid(grid, P, 0.3, NOTIONAL)
    high = dual_digital_value_grid(grid, P, 0.3, NOTIONAL, rho=0.85)
    assert not np.allclose(base, high)
    # Positive correlation makes {equity up AND fx up} more likely.
    assert high.mean() > base.mean()


# -- the terminal limit --------------------------------------------------

def test_terminal_payoff_is_exact_step():
    """tau <= 0 returns the payoff itself, with no division by sqrt(0)."""
    grid = SurfaceGrid(S=np.array([6900.0, 7100.0]), X=np.array([155.0, 165.0]))
    Z = dual_digital_value_grid(grid, P, 0.0, NOTIONAL)

    assert Z[0, 0] == 0.0            # S < K, X < B
    assert Z[0, 1] == 0.0            # S > K, X < B
    assert Z[1, 0] == 0.0            # S < K, X > B
    assert Z[1, 1] == NOTIONAL       # both land


def test_surface_converges_to_step_as_tau_shrinks():
    """The visual claim of the animation, asserted."""
    grid = SurfaceGrid(S=np.array([6900.0, 7100.0]), X=np.array([155.0, 165.0]))
    terminal = dual_digital_value_grid(grid, P, 0.0, NOTIONAL)

    errors = [np.max(np.abs(dual_digital_value_grid(grid, P, tau, NOTIONAL)
                            - terminal))
              for tau in (0.5, 0.1, 0.01, 0.0005)]

    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 0.02 * NOTIONAL


# -- PnL framing ---------------------------------------------------------

def test_inception_premium_matches_scalar_pricer():
    expected = _scalar_price(P.S0, P.X0, P.T, P.rho)
    assert inception_premium(P, NOTIONAL) == pytest.approx(expected, rel=1e-11)


def test_short_pnl_is_bounded_by_premium_and_payout():
    """Selling a digital: keep the premium at best, pay notional at worst."""
    premium = inception_premium(P, NOTIONAL)
    grid = SurfaceGrid.around(P.S0, P.X0, n=40)

    for tau in (P.T, 0.1, 0.001, 0.0):
        pnl = short_pnl_grid(grid, P, tau, NOTIONAL, premium)
        assert pnl.max() <= premium + 1e-6
        assert pnl.min() >= premium - NOTIONAL - 1e-6

    # Both bounds are actually attained at expiry over a wide enough grid.
    terminal = short_pnl_grid(grid, P, 0.0, NOTIONAL, premium)
    assert terminal.max() == pytest.approx(premium)
    assert terminal.min() == pytest.approx(premium - NOTIONAL)


# -- frame schedules -----------------------------------------------------

def test_time_schedule_is_monotone_and_spans_the_trade():
    tau = time_to_expiry_schedule(P.T, n_frames=50, tau_min_days=0.25)

    assert len(tau) == 50
    assert tau[0] == pytest.approx(P.T)
    assert tau[-1] == pytest.approx(0.25 / 365.0)
    assert np.all(np.diff(tau) < 0)


def test_time_schedule_is_geometric_not_linear():
    """Constant relative decay -- the reason the animation reads smoothly."""
    tau = time_to_expiry_schedule(P.T, n_frames=30)
    ratios = tau[1:] / tau[:-1]
    assert np.allclose(ratios, ratios[0])


def test_correlation_schedule_pingpongs():
    rho = correlation_schedule(n_frames=20, rho_max=0.9)
    assert rho[0] == pytest.approx(-0.9)
    assert rho.max() == pytest.approx(0.9)
    assert rho[-1] == pytest.approx(rho[1])       # loops back, no repeat frame
    assert np.all(np.abs(rho) <= 0.9)


# -- the delta hedge -----------------------------------------------------

def test_inception_deltas_are_positive_and_sized_right():
    """Short a dual digital, hedge long both legs."""
    delta_S, delta_X = inception_deltas(P, NOTIONAL)

    assert delta_S > 0 and delta_X > 0
    # Sanity on magnitude: the FX leg is the barrier leg and carries far more
    # delta per unit of spot than the equity leg -- around 3x notional in
    # equivalent terms, which is the headline risk number for this trade.
    assert 2.5 < delta_X * P.X0 / NOTIONAL < 4.0
    assert 0.5 < delta_S * P.S0 / NOTIONAL < 1.2


def test_hedge_is_flat_at_spot_by_construction():
    """The hedge contributes exactly zero where it was struck."""
    premium = inception_premium(P, NOTIONAL)
    deltas = inception_deltas(P, NOTIONAL)
    point = SurfaceGrid(S=np.array([P.S0]), X=np.array([P.X0]))

    for tau in (P.T, 0.1, 0.0):
        raw = short_pnl_grid(point, P, tau, NOTIONAL, premium)[0, 0]
        hedged = hedged_short_pnl_grid(point, P, tau, NOTIONAL, premium,
                                       deltas)[0, 0]
        assert hedged == pytest.approx(raw)


def test_hedge_removes_the_first_order_slope():
    """Near spot the hedged surface is flat to first order; raw is not."""
    premium = inception_premium(P, NOTIONAL)
    deltas = inception_deltas(P, NOTIONAL)
    h = 5.0
    near = SurfaceGrid(S=np.array([P.S0 - h, P.S0 + h]), X=np.array([P.X0]))

    raw = short_pnl_grid(near, P, P.T, NOTIONAL, premium)[0]
    hedged = hedged_short_pnl_grid(near, P, P.T, NOTIONAL, premium, deltas)[0]

    raw_slope = abs(raw[1] - raw[0]) / (2 * h)
    hedged_slope = abs(hedged[1] - hedged[0]) / (2 * h)
    assert hedged_slope < 0.01 * raw_slope


def test_hedged_surface_equals_raw_plus_linear_hedge():
    """No hidden terms: it is exactly the tangent-plane subtraction."""
    premium = inception_premium(P, NOTIONAL)
    deltas = inception_deltas(P, NOTIONAL)
    grid = SurfaceGrid.around(P.S0, P.X0, n=25)
    S, X = grid.mesh()

    raw = short_pnl_grid(grid, P, 0.2, NOTIONAL, premium)
    hedged = hedged_short_pnl_grid(grid, P, 0.2, NOTIONAL, premium, deltas)

    expected = raw + deltas[0] * (S - P.S0) + deltas[1] * (X - P.X0)
    assert np.allclose(hedged, expected)


def test_hedged_surface_is_unbounded_unlike_the_raw_one():
    """A linear hedge against a step payoff has no floor. Worth asserting."""
    premium = inception_premium(P, NOTIONAL)
    deltas = inception_deltas(P, NOTIONAL)

    narrow = SurfaceGrid.around(P.S0, P.X0, S_pct=0.16, X_pct=0.11, n=30)
    wide = SurfaceGrid.around(P.S0, P.X0, S_pct=0.32, X_pct=0.22, n=30)

    assert (hedged_short_pnl_grid(wide, P, 0.0, NOTIONAL, premium, deltas).min()
            < hedged_short_pnl_grid(narrow, P, 0.0, NOTIONAL, premium,
                                    deltas).min())
    # ...whereas the raw short PnL is floored at premium - notional whatever
    # the grid does.
    assert short_pnl_grid(wide, P, 0.0, NOTIONAL, premium).min() == pytest.approx(
        short_pnl_grid(narrow, P, 0.0, NOTIONAL, premium).min())
