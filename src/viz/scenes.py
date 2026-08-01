"""The two animations, defined once and rendered to both HTML and MP4.

Grid and frame counts differ by target and that is deliberate: the interactive
page ships every frame's z-grid as JSON to the browser, so it runs a coarser
grid, while the MP4 rasterises server-side and can afford the fine one.
"""
from __future__ import annotations

import numpy as np

from ..hybrid.surface import (SurfaceGrid, correlation_schedule,
                              inception_deltas, inception_premium,
                              time_to_expiry_schedule)
from .surface3d import DEFAULT_THEME, SurfaceAnimation, build_animation

__all__ = ["time_decay_scene", "correlation_scene", "hedged_time_decay_scene",
           "SCENES"]

#: Plotly hands text containing a *pair* of dollar signs to MathJax. Subtitles
#: quote money more than once, so they use the entity throughout.
USD = "&#36;"

#: Interactive pages ship every frame's z-grid to the browser, so page weight
#: is roughly ``n_frames * n_grid^2 * 5.3`` bytes (base64 float32). These
#: values land the two-panel page around 5 MB; the video path rasterises
#: server-side and is not constrained.
WEB_GRID, WEB_FRAMES, WEB_RHO_FRAMES = 68, 72, 45
VIDEO_GRID, VIDEO_FRAMES = 140, 300


def time_decay_scene(p, notional: float, n_grid: int = VIDEO_GRID,
                     n_frames: int = VIDEO_FRAMES,
                     theme=DEFAULT_THEME) -> SurfaceAnimation:
    """Time to expiry collapsing the surface onto the strike/barrier corner."""
    premium = inception_premium(p, notional)
    grid = SurfaceGrid.around(p.S0, p.X0, n=n_grid)
    taus = time_to_expiry_schedule(p.T, n_frames, tau_min_days=0.25)

    return build_animation(
        p, notional, premium, grid,
        taus=taus, rhos=np.full(n_frames, p.rho), sweep="time",
        title="EQ/FX dual digital — mark-to-market PnL",
        subtitle=(f"Short {USD}{notional / 1e6:,.0f}mm  ·  pays if SPX &gt; "
                  f"{p.K:,.0f} <b>and</b> USDJPY &gt; {p.B:.0f} at expiry  "
                  f"·  ρ = {p.rho:+.2f}"),
        theme=theme,
    )


def hedged_time_decay_scene(p, notional: float, n_grid: int = VIDEO_GRID,
                            n_frames: int = VIDEO_FRAMES,
                            theme=DEFAULT_THEME) -> SurfaceAnimation:
    """The same decay, after subtracting the inception delta hedge.

    The hedge is struck once and held -- see
    :func:`~src.hybrid.surface.inception_deltas` for why a re-hedged book has
    no single surface. The subtitle states that up front, because the static
    assumption is the first thing a risk audience will reach for.
    """
    premium = inception_premium(p, notional)
    grid = SurfaceGrid.around(p.S0, p.X0, n=n_grid)
    taus = time_to_expiry_schedule(p.T, n_frames, tau_min_days=0.25)
    deltas = inception_deltas(p, notional)

    return build_animation(
        p, notional, premium, grid,
        taus=taus, rhos=np.full(n_frames, p.rho), sweep="time",
        title="EQ/FX dual digital — delta-hedged PnL",
        subtitle=(f"Short {USD}{notional / 1e6:,.0f}mm, hedged once at "
                  f"inception and held: long {USD}{deltas[0] * p.S0 / 1e6:,.0f}mm "
                  f"SPX, {USD}{deltas[1] * p.X0 / 1e6:,.0f}mm USDJPY  ·  "
                  f"ρ = {p.rho:+.2f}"),
        theme=theme, deltas=deltas,
    )


def correlation_scene(p, notional: float, n_grid: int = VIDEO_GRID,
                      n_frames: int = 150, tau: float | None = None,
                      rho_max: float = 0.9,
                      theme=DEFAULT_THEME) -> SurfaceAnimation:
    """Correlation sweeping the surface with spot, vol and time held still.

    The premium stays the one struck at inception, so the height of the
    surface is the mark-to-market consequence of correlation alone -- the
    exposure a single-asset book does not have.
    """
    premium = inception_premium(p, notional)
    grid = SurfaceGrid.around(p.S0, p.X0, n=n_grid)
    rhos = correlation_schedule(n_frames, rho_max=rho_max, pingpong=True)
    tau = p.T / 2.0 if tau is None else tau

    return build_animation(
        p, notional, premium, grid,
        taus=np.full(len(rhos), tau), rhos=rhos, sweep="rho",
        title="EQ/FX dual digital — the correlation exposure",
        subtitle=(f"Short {USD}{notional / 1e6:,.0f}mm  ·  spot, vols and time "
                  f"held fixed at {tau * 12:.0f}m to expiry  ·  "
                  f"only ρ moves"),
        theme=theme,
    )


SCENES = {
    "time": time_decay_scene,
    "rho": correlation_scene,
    "hedged": hedged_time_decay_scene,
}
