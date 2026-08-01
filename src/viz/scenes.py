"""The two animations, defined once and rendered to both HTML and MP4.

Grid and frame counts differ by target and that is deliberate: the interactive
page ships every frame's z-grid as JSON to the browser, so it runs a coarser
grid, while the MP4 rasterises server-side and can afford the fine one.
"""
from __future__ import annotations

import numpy as np

from ..hybrid.surface import (SurfaceGrid, correlation_schedule,
                              inception_premium, time_to_expiry_schedule)
from .surface3d import SurfaceAnimation, build_animation

__all__ = ["time_decay_scene", "correlation_scene", "SCENES"]

#: Interactive pages ship every frame's z-grid to the browser, so page weight
#: is roughly ``n_frames * n_grid^2 * 5.3`` bytes (base64 float32). These
#: values land the two-panel page around 5 MB; the video path rasterises
#: server-side and is not constrained.
WEB_GRID, WEB_FRAMES, WEB_RHO_FRAMES = 72, 80, 50
VIDEO_GRID, VIDEO_FRAMES = 140, 300


def time_decay_scene(p, notional: float, n_grid: int = VIDEO_GRID,
                     n_frames: int = VIDEO_FRAMES) -> SurfaceAnimation:
    """Time to expiry collapsing the surface onto the strike/barrier corner."""
    premium = inception_premium(p, notional)
    grid = SurfaceGrid.around(p.S0, p.X0, n=n_grid)
    taus = time_to_expiry_schedule(p.T, n_frames, tau_min_days=0.25)

    return build_animation(
        p, notional, premium, grid,
        taus=taus, rhos=np.full(n_frames, p.rho), sweep="time",
        title="EQ/FX dual digital — mark-to-market PnL",
        subtitle=(f"Short ${notional / 1e6:,.0f}mm  ·  pays if SPX &gt; "
                  f"{p.K:,.0f} <b>and</b> USDJPY &gt; {p.B:.0f} at expiry  "
                  f"·  ρ = {p.rho:+.2f}"),
    )


def correlation_scene(p, notional: float, n_grid: int = VIDEO_GRID,
                      n_frames: int = 150, tau: float | None = None,
                      rho_max: float = 0.9) -> SurfaceAnimation:
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
        subtitle=(f"Short ${notional / 1e6:,.0f}mm  ·  spot, vols and time "
                  f"held fixed at {tau * 12 / 1:.0f}m to expiry  ·  "
                  f"only ρ moves"),
    )


SCENES = {
    "time": time_decay_scene,
    "rho": correlation_scene,
}
