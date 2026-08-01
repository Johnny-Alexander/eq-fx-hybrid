"""Presentation layer for the hybrid pricer.

Kept separate from :mod:`src.hybrid` on purpose: everything in here is about
how a surface *looks*, and none of it is allowed to compute a price. The
numbers come from :mod:`src.hybrid.surface`, which is tested against the
scalar closed form.
"""
from .surface3d import (PNL_DIVERGING, SurfaceAnimation, add_playback_controls,
                        build_animation, diverging_pnl_colorscale,
                        format_time_to_expiry, orbit_camera)

__all__ = ["SurfaceAnimation", "build_animation", "diverging_pnl_colorscale",
           "format_time_to_expiry", "orbit_camera", "add_playback_controls",
           "PNL_DIVERGING"]
