"""Tests for the presentation layer.

Only the parts that can be wrong *numerically* -- a colour scale whose neutral
drifts off zero, or a frame that shows a different surface from the one the
caption claims. How it looks is not testable here; that it is honest is.
"""
import numpy as np
import pytest

from src.hybrid.surface import SurfaceGrid, inception_premium
from src.trade_config import EXAMPLE_TRADE as P, NOTIONAL
from src.viz.scenes import correlation_scene, time_decay_scene
from src.viz.surface3d import (build_animation, diverging_pnl_colorscale,
                               format_time_to_expiry)


# -- colour --------------------------------------------------------------

def test_colorscale_neutral_sits_at_zero_pnl():
    """The grey must land on zero, not on the middle of the data."""
    vmin, vmax = -79.7, 20.3
    scale = diverging_pnl_colorscale(vmin, vmax)

    expected = (0.0 - vmin) / (vmax - vmin)
    neutral = [pos for pos, colour in scale if colour == "#383835"]

    assert len(neutral) == 1
    assert neutral[0] == pytest.approx(expected)
    # Sanity: on this trade zero is four-fifths of the way up the ramp, so a
    # naive symmetric scale would be badly wrong.
    assert neutral[0] > 0.75


def test_colorscale_is_monotone_and_spans_zero_to_one():
    scale = diverging_pnl_colorscale(-79.7, 20.3)
    positions = [pos for pos, _ in scale]

    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(1.0)
    assert positions == sorted(positions)


def test_colorscale_rejects_a_range_that_does_not_straddle_zero():
    with pytest.raises(ValueError, match="straddle zero"):
        diverging_pnl_colorscale(5.0, 20.0)


# -- captions ------------------------------------------------------------

@pytest.mark.parametrize("tau,expected", [
    (0.5, "6.0 months to expiry"),
    (0.25, "3.0 months to expiry"),
    (10 / 365, "10 days to expiry"),
    (12 / 8760, "12 hours to expiry"),
    (0.0, "EXPIRY"),
])
def test_time_captions(tau, expected):
    assert format_time_to_expiry(tau) == expected


# -- frames say what they show -------------------------------------------

def test_frame_surface_matches_its_caption():
    """Frame i must carry the surface for tau_i, not a neighbour's."""
    anim = time_decay_scene(P, NOTIONAL, n_grid=20, n_frames=12)
    premium = inception_premium(P, NOTIONAL)

    from src.hybrid.surface import (short_pnl_grid, time_to_expiry_schedule)
    taus = time_to_expiry_schedule(P.T, 12, tau_min_days=0.25)
    grid = SurfaceGrid.around(P.S0, P.X0, n=20)

    for i in (0, 5, 11):
        expected = short_pnl_grid(grid, P, taus[i], NOTIONAL, premium) / 1e6
        assert np.allclose(anim.frames[i].data[0].z, expected, atol=1e-4)
        assert anim.captions[i] == format_time_to_expiry(taus[i])


def test_time_and_rho_scenes_share_the_frozen_colour_range():
    """Both clips must encode PnL identically, or the pair mislead side by side."""
    time_anim = time_decay_scene(P, NOTIONAL, n_grid=16, n_frames=6)
    rho_anim = correlation_scene(P, NOTIONAL, n_grid=16, n_frames=6)

    t_surface, r_surface = time_anim.figure.data[0], rho_anim.figure.data[0]
    assert t_surface.cmin == pytest.approx(r_surface.cmin)
    assert t_surface.cmax == pytest.approx(r_surface.cmax)

    premium = inception_premium(P, NOTIONAL)
    assert t_surface.cmax == pytest.approx(premium / 1e6)
    assert t_surface.cmin == pytest.approx((premium - NOTIONAL) / 1e6)


def test_frame_figure_carries_no_playback_controls():
    """The MP4 and poster paths must not bake Play/Pause into the pixels."""
    from src.viz.surface3d import add_playback_controls

    anim = time_decay_scene(P, NOTIONAL, n_grid=16, n_frames=5)
    add_playback_controls(anim)                  # mutates anim.figure
    assert anim.figure.layout.updatemenus        # controls are on the live one

    still = anim.frame_figure(2)
    assert not still.layout.updatemenus
    assert not still.layout.sliders


def test_build_animation_rejects_mismatched_schedules():
    grid = SurfaceGrid.around(P.S0, P.X0, n=8)
    with pytest.raises(ValueError, match="equal length"):
        build_animation(P, NOTIONAL, inception_premium(P, NOTIONAL), grid,
                        taus=np.array([0.5, 0.25]), rhos=np.array([0.3]),
                        sweep="time", title="t", subtitle="s")


def test_correlation_scene_holds_time_still_and_moves_rho():
    anim = correlation_scene(P, NOTIONAL, n_grid=16, n_frames=8)
    captions = anim.captions

    assert all(c.startswith("ρ =") for c in captions)
    assert captions[0] == "ρ = -0.90"
    assert max(captions) != min(captions)
    # Ping-pong: ends where it began, so the clip loops on a running slide.
    assert captions[-1] == captions[1]
