"""The 3-D PnL surface: one figure spec, driven either interactively or to MP4.

Both outputs are built from :func:`build_animation` so the video and the
hosted page cannot drift apart. The only difference is that the MP4 orbits the
camera (the viewer cannot) and the HTML ships play/pause controls (the video
does not need them).

Colour
------
Diverging **blue (profit) to red (loss)** through a neutral grey, not the
trading-floor green/red. Red-green deficiency affects roughly 8% of men, so in
a large audience a green/red surface is illegible to a meaningful slice of the
room; blue/red carries the same polarity and survives every CVD simulation.
The neutral midpoint is pinned to *zero PnL* rather than to the middle of the
data, and ``cmin``/``cmax`` are frozen across every frame -- a colour scale
that rescales per frame would make the animation lie about magnitude.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go

from ..hybrid.surface import (SurfaceGrid, dual_digital_value_grid,
                              hedged_short_pnl_grid, marginal_probabilities)

__all__ = ["SurfaceAnimation", "build_animation", "diverging_pnl_colorscale",
           "format_time_to_expiry", "orbit_camera", "add_playback_controls",
           "PNL_DIVERGING"]

# -- themes --------------------------------------------------------------
#
# Every colour the figure uses lives in a Theme. The rule that makes a theme
# valid is the same in light and dark: **lightness must move monotonically
# away from the neutral, in the direction away from the chart surface.** On a
# dark surface that means the extremes glow; on a light one it means they
# darken. Get this backwards and the largest losses become the least visually
# prominent thing on screen, which is exactly the wrong encoding for a risk
# chart.


@dataclass(frozen=True)
class Theme:
    """Colours and lighting for one look. See module note on lightness."""

    name: str
    surface_bg: str             # page plane
    panel_bg: str               # the three scene walls
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    accent: str                 # the K x B corner marker
    neutral: str                # zero-PnL colour, closest to the surface
    loss: tuple                 # most-negative -> nearest-neutral
    profit: tuple               # nearest-neutral -> most-positive
    contour: str
    ambient: float = 0.62
    diffuse: float = 0.82
    specular: float = 0.18
    show_walls: bool = True
    gridwidth: float = 1.0


THEMES = {
    "dark": Theme(
        name="dark",
        surface_bg="#0d0d0d", panel_bg="#1a1a19",
        ink="#ffffff", ink_secondary="#c3c2b7", ink_muted="#898781",
        grid="#2c2c2a", axis="#383835", accent="#eda100",
        neutral="#383835",
        loss=("#f6b8b8", "#e97070", "#d03b3b", "#8a3230", "#4a3130"),
        profit=("#274f7c", "#2a78d6", "#6da7ec", "#9ec5f4", "#cde2fb"),
        contour="rgba(255,255,255,0.22)",
    ),
    # Near-white editorial page. The safest choice on an unknown projector:
    # light backgrounds survive a washed-out room where dark ones turn grey.
    "paper": Theme(
        name="paper",
        surface_bg="#fcfcfb", panel_bg="#f2f2ef",
        ink="#0b0b0b", ink_secondary="#52514e", ink_muted="#7d7b76",
        grid="#c9c7bc", axis="#9a978c", accent="#a86b00",
        neutral="#f0efec",
        loss=("#6e1717", "#a52626", "#d03b3b", "#ec8a8a", "#f8d4d4"),
        profit=("#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"),
        contour="rgba(0,0,0,0.22)",
        ambient=0.78, diffuse=0.68, specular=0.06, gridwidth=2.0,
    ),
    # Warm newsprint. Reads as financial press rather than software.
    "ivory": Theme(
        name="ivory",
        surface_bg="#fff1e5", panel_bg="#f7e3d2",
        ink="#26211d", ink_secondary="#57504a", ink_muted="#8a7f75",
        grid="#dcc3ab", axis="#bfa287", accent="#96590a",
        neutral="#f4e7db",
        loss=("#6b1414", "#9e2222", "#cc3b39", "#e58b81", "#f6d3c6"),
        profit=("#cfe0ee", "#87b3d8", "#2f6fa8", "#1b4a76", "#10314f"),
        contour="rgba(60,40,25,0.22)",
        ambient=0.78, diffuse=0.68, specular=0.06, gridwidth=2.0,
    ),
    # Cool corporate blue-grey -- closest to a standard bank deck template.
    "institutional": Theme(
        name="institutional",
        surface_bg="#f5f7fa", panel_bg="#e8edf4",
        ink="#0f1c2e", ink_secondary="#44536b", ink_muted="#78859a",
        grid="#c6d0e0", axis="#a6b3c7", accent="#9a5b00",
        neutral="#eef1f6",
        loss=("#6d1526", "#a3202f", "#d0404a", "#ea8b90", "#f7d3d6"),
        profit=("#d3e3f7", "#8ab6e8", "#2e6fc4", "#1a4b8c", "#0e2f5c"),
        contour="rgba(15,28,46,0.20)",
        ambient=0.78, diffuse=0.68, specular=0.06, gridwidth=2.0,
    ),
    # No scene walls: the surface floats on white with only thin axes.
    "minimal": Theme(
        name="minimal",
        surface_bg="#ffffff", panel_bg="#ffffff",
        ink="#111111", ink_secondary="#4a4a4a", ink_muted="#8c8c8c",
        grid="#dcdcdc", axis="#bcbcbc", accent="#a86b00",
        neutral="#f2f2f2",
        loss=("#6e1717", "#a52626", "#d03b3b", "#ec8a8a", "#f8d4d4"),
        profit=("#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"),
        contour="rgba(0,0,0,0.18)",
        ambient=0.82, diffuse=0.62, specular=0.04, show_walls=False,
        gridwidth=2.0,
    ),
}

DEFAULT_THEME = "dark"

# Backwards-compatible aliases for the dark palette.
SURFACE_BG = THEMES["dark"].surface_bg
PANEL_BG = THEMES["dark"].panel_bg
INK = THEMES["dark"].ink
INK_SECONDARY = THEMES["dark"].ink_secondary
INK_MUTED = THEMES["dark"].ink_muted
GRID = THEMES["dark"].grid
AXIS = THEMES["dark"].axis

PNL_DIVERGING = {
    "neutral": THEMES["dark"].neutral,
    "loss": list(THEMES["dark"].loss),
    "profit": list(THEMES["dark"].profit),
}


def resolve_theme(theme) -> Theme:
    """Accept a Theme, or the name of one."""
    if isinstance(theme, Theme):
        return theme
    try:
        return THEMES[theme]
    except KeyError:
        raise ValueError(
            f"unknown theme {theme!r}; choose from {sorted(THEMES)}") from None


def diverging_pnl_colorscale(vmin: float, vmax: float,
                             theme=DEFAULT_THEME) -> list:
    """Colour scale with the neutral pinned to zero PnL, not to mid-range.

    PnL on a short digital is wildly asymmetric -- here roughly +20 against
    -80 -- so an automatically centred scale would paint the entire profit
    plateau in a colour that means "slightly negative". Placing the neutral at
    the true zero costs one function and buys an honest picture.
    """
    if not vmin < 0.0 < vmax:
        raise ValueError(f"range must straddle zero, got [{vmin}, {vmax}]")

    th = resolve_theme(theme)
    zero = (0.0 - vmin) / (vmax - vmin)
    loss, profit = th.loss, th.profit

    scale = []
    # Loss arm: position 0 is the most negative PnL, running up to the neutral.
    for i, colour in enumerate(loss):
        scale.append([zero * i / len(loss), colour])
    scale.append([zero, th.neutral])
    # Profit arm: neutral out to the most positive PnL at position 1.
    for i, colour in enumerate(profit):
        scale.append([zero + (1.0 - zero) * (i + 1) / len(profit), colour])

    scale[-1][0] = 1.0          # guard against float drift at the endpoint
    return scale


def format_time_to_expiry(tau: float) -> str:
    """Human units that keep up with a geometric time schedule."""
    if tau <= 0:
        return "EXPIRY"
    days = tau * 365.0
    if days >= 60:
        return f"{days / 30.44:.1f} months to expiry"
    if days >= 2:
        return f"{days:.0f} days to expiry"
    hours = days * 24.0
    if hours >= 1:
        return f"{hours:.0f} hours to expiry"
    return f"{hours * 60:.0f} minutes to expiry"


def orbit_camera(azimuth_deg: float, elevation_deg: float, radius: float = 2.5):
    """Orbit position in spherical coords.

    The radius has to clear the *terminal* surface, not the opening one: at
    expiry the payout quadrant is a shaft ~100mm deep, and a camera framed on
    the smooth 6-month surface crops its floor off the bottom of the frame.
    """
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    return dict(
        eye=dict(x=radius * np.cos(a) * np.cos(e),
                 y=radius * np.sin(a) * np.cos(e),
                 z=radius * np.sin(e)),
        center=dict(x=0, y=0, z=-0.02),
        up=dict(x=0, y=0, z=1),
    )


@dataclass
class SurfaceAnimation:
    """A built animation: the base figure, its frames, and their captions."""

    figure: go.Figure
    frames: list = field(default_factory=list)
    captions: list = field(default_factory=list)
    theme: "Theme | None" = None
    _still: go.Figure | None = field(default=None, repr=False)

    def frame_figure(self, i: int, camera: dict | None = None) -> go.Figure:
        """A standalone figure for frame ``i`` -- what the MP4 renderer draws.

        Frames carry only the traces that change, so this replays them onto a
        copy of the base figure rather than rebuilding the scene each time.

        The copy is taken from a *frameless* template, cached on first use.
        Copying ``self.figure`` directly would clone its entire frame list on
        every call -- quadratic in frame count, and minutes of pure overhead
        on a 300-frame render.
        """
        if self._still is None:
            self._still = go.Figure(data=self.figure.data,
                                    layout=self.figure.layout)
        fig = go.Figure(self._still)
        frame = self.frames[i]
        for trace_index, trace in zip(frame.traces, frame.data):
            fig.data[trace_index].update(trace)
        if frame.layout is not None:
            fig.update_layout(frame.layout)
        if camera is not None:
            fig.update_scenes(camera=camera)
        # Direct assignment, not update_layout(updatemenus=[]): an empty list
        # is a no-op merge there, which silently left Play/Pause and the
        # scrub slider baked into the exported poster PNG.
        fig.layout.updatemenus = []
        fig.layout.sliders = []
        return fig


#: Plotly hands any text containing a *pair* of dollar signs to MathJax, which
#: silently renders the readout as italic maths and swallows the <br> tags.
#: The HTML entity is invisible to that parser.
USD = "&#36;"


def _readout(p, tau: float, rho: float, premium: float, notional: float,
             pnl_at_spot: float, sweep: str, hedged: bool = False) -> str:
    """The corner block: what the surface is worth where we actually are."""
    P_eq, P_fx = marginal_probabilities(p, tau, p.S0, p.X0)
    sign = "+" if pnl_at_spot >= 0 else "−"

    # At spot the hedge contributes exactly zero -- it is struck there -- so
    # the number is the same either way. The label still has to change, or the
    # hedged clip appears to be showing an unhedged mark.
    label = "Hedged PnL at spot" if hedged else "MTM PnL at spot"

    # The rho sweep already carries its value in the headline caption, so the
    # readout spends that line on the joint probability instead -- the thing
    # correlation is actually moving.
    lines = [
        f"<b>{label}</b>   {sign}{USD}{abs(pnl_at_spot):,.1f}mm",
        f"Premium received   {USD}{premium / 1e6:,.1f}mm",
        f"P(SPX &gt; K)  {P_eq:.0%}      P(USDJPY &gt; B)  {P_fx:.0%}",
    ]
    if sweep == "rho":
        # Back out the joint probability from the mark: the PnL is
        # premium - N * df * P_joint, so P_joint = value / (N * df).
        value = premium - pnl_at_spot * 1e6
        joint = value / (notional * float(np.exp(-p.r_d * tau)))
        lines.append(f"<b>P(both legs land)   {joint:.1%}</b>")
    return "<br>".join(lines)


def build_animation(p, notional: float, premium: float, grid: SurfaceGrid,
                    taus: np.ndarray, rhos: np.ndarray, sweep: str,
                    title: str, subtitle: str, theme=DEFAULT_THEME,
                    deltas: tuple | None = None) -> SurfaceAnimation:
    """Assemble the surface, its frames and all the static scene furniture.

    ``taus`` and ``rhos`` are per-frame and must be the same length: a time
    sweep holds rho flat, a correlation sweep holds tau flat, and the caller
    decides which. ``sweep`` only selects the caption wording.
    """
    if len(taus) != len(rhos):
        raise ValueError("taus and rhos must be per-frame and equal length")

    th = resolve_theme(theme)
    S, X = grid.S, grid.X
    mm = 1e6

    def pnl_grid(tau, rho):
        if deltas is None:
            z = premium - dual_digital_value_grid(grid, p, tau, notional,
                                                  rho=rho)
        else:
            z = hedged_short_pnl_grid(grid, p, tau, notional, premium,
                                      deltas, rho=rho)
        # float32 halves the page weight: plotly base64-encodes numpy arrays,
        # so every frame's grid costs 8 bytes/point at float64. On a $100mm
        # trade float32 still resolves to about $10, which is far below the
        # width of a screen pixel on this axis.
        return (z / mm).astype(np.float32)

    def pnl_at_spot(tau, rho):
        point = SurfaceGrid(S=np.array([p.S0]), X=np.array([p.X0]))
        if deltas is None:
            z = premium - dual_digital_value_grid(point, p, tau, notional,
                                                  rho=rho)[0, 0]
        else:
            z = hedged_short_pnl_grid(point, p, tau, notional, premium,
                                      deltas, rho=rho)[0, 0]
        return z / mm

    # Frozen colour/height range. Unhedged, the bounds are known in closed
    # form: the short can keep at most the premium and pay at most the
    # notional. Hedged, the linear hedge is unbounded over the grid, so the
    # extent has to be measured -- once, over every frame, never per frame.
    if deltas is None:
        vmax = premium / mm
        vmin = (premium - notional) / mm
    else:
        extremes = [pnl_grid(t, r) for t, r in zip(taus, rhos)]
        vmax = float(max(z.max() for z in extremes))
        vmin = float(min(z.min() for z in extremes))
    colorscale = diverging_pnl_colorscale(vmin, vmax, theme=th)
    zpad = 0.06 * (vmax - vmin)
    zfloor = vmin - zpad

    Z0 = pnl_grid(taus[0], rhos[0])
    z_spot0 = pnl_at_spot(taus[0], rhos[0])

    surface = go.Surface(
        x=S, y=X, z=Z0,
        colorscale=colorscale, cmin=vmin, cmax=vmax,
        # Contour lines drawn *on* the surface rather than projected to the
        # floor. The floor projection is the textbook choice, but the camera
        # here looks down at an opaque surface that covers its own footprint
        # entirely, so a floor projection is never visible. On-surface
        # contours survive the collapse and give the walls a readable
        # gradient once the transition goes vertical.
        contours=dict(
            z=dict(show=True, usecolormap=False, color=th.contour,
                   project=dict(z=False),
                   width=1.5, start=vmin, end=vmax, size=(vmax - vmin) / 20),
        ),
        lighting=dict(ambient=th.ambient, diffuse=th.diffuse,
                      specular=th.specular, roughness=0.55, fresnel=0.15),
        lightposition=dict(x=1.0e5, y=-6.0e4, z=1.0e5),
        opacity=1.0,
        hovertemplate=("SPX %{x:,.0f}<br>USDJPY %{y:.1f}"
                       "<br><b>PnL $%{z:,.1f}mm</b><extra></extra>"),
        colorbar=dict(
            title=dict(text="PnL ($mm)", side="right",
                       font=dict(color=th.ink_secondary, size=12)),
            tickfont=dict(color=th.ink_muted, size=11),
            outlinewidth=0, thickness=14, len=0.55, x=0.9,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    # Annotations that must never be occluded are drawn as pins rising from
    # the surface into empty space above it, not as floor furniture: the
    # camera looks down, so anything at floor level is hidden under the
    # surface for the whole animation.
    ztop = vmax + zpad * 0.9

    # Where we actually are: a stem from the surface up to a labelled pin.
    spot_marker = go.Scatter3d(
        x=[p.S0, p.S0], y=[p.X0, p.X0], z=[z_spot0, ztop],
        mode="lines+markers+text",
        line=dict(color=th.ink, width=4),
        marker=dict(size=[5, 8], color=th.ink,
                    line=dict(color=th.surface_bg, width=2)),
        text=["", "spot"], textposition="top center",
        textfont=dict(color=th.ink, size=12),
        hovertemplate=f"spot: SPX {p.S0:,.0f} / USDJPY {p.X0:.1f}<extra></extra>",
        showlegend=False,
    )

    # The strike/barrier corner -- the vertex the surface collapses onto, and
    # the thing that makes this a hybrid rather than two separate digitals.
    corner_line = go.Scatter3d(
        x=[p.K, p.K], y=[p.B, p.B], z=[zfloor, ztop],
        mode="lines", line=dict(color=th.accent, width=4, dash="dot"),
        hoverinfo="skip", showlegend=False)
    # Label kept short and pushed right: the levels are already in the
    # subtitle, and the long form collided with the spot pin whenever the two
    # pins projected close together.
    corner_label = go.Scatter3d(
        x=[p.K], y=[p.B], z=[ztop], mode="markers+text",
        marker=dict(size=6, color=th.accent, symbol="diamond"),
        text=["  K × B"], textposition="middle right",
        textfont=dict(color=th.accent, size=13),
        hoverinfo="skip", showlegend=False)

    fig = go.Figure(data=[surface, spot_marker, corner_line, corner_label])

    caption0 = (format_time_to_expiry(taus[0]) if sweep == "time"
                else f"ρ = {rhos[0]:+.2f}")

    fig.update_layout(
        template="plotly_dark" if th.name == "dark" else "plotly_white",
        paper_bgcolor=th.surface_bg,
        plot_bgcolor=th.surface_bg,
        font=dict(family="Inter, Helvetica Neue, Helvetica, Arial, sans-serif",
                  color=th.ink_secondary),
        margin=dict(l=0, r=0, t=96, b=28),
        title=dict(
            text=(f"<b>{title}</b><br>"
                  f"<span style='font-size:14px;color:{th.ink_muted}'>{subtitle}</span>"),
            x=0.035, xanchor="left", y=0.955, yanchor="top",
            font=dict(size=25, color=th.ink)),
        scene=dict(
            xaxis=dict(title=dict(text="SPX spot",
                                  font=dict(color=th.ink_secondary, size=13)),
                       backgroundcolor=th.panel_bg, gridcolor=th.grid,
                       gridwidth=th.gridwidth, zerolinecolor=th.axis,
                       showbackground=th.show_walls,
                       tickfont=dict(color=th.ink_muted, size=11), tickformat=",.0f"),
            yaxis=dict(title=dict(text="USDJPY spot",
                                  font=dict(color=th.ink_secondary, size=13)),
                       backgroundcolor=th.panel_bg, gridcolor=th.grid,
                       gridwidth=th.gridwidth, zerolinecolor=th.axis,
                       showbackground=th.show_walls,
                       tickfont=dict(color=th.ink_muted, size=11)),
            zaxis=dict(title=dict(text="PnL ($mm)",
                                  font=dict(color=th.ink_secondary, size=13)),
                       backgroundcolor=th.panel_bg, gridcolor=th.grid,
                       gridwidth=th.gridwidth, zerolinecolor=th.axis,
                       showbackground=th.show_walls,
                       tickfont=dict(color=th.ink_muted, size=11),
                       range=[zfloor, ztop + zpad * 0.35]),
            aspectratio=dict(x=1.3, y=1.3, z=0.85),
            camera=orbit_camera(48.0, 22.0),
            domain=dict(x=[0.15, 1.0], y=[0.0, 1.0]),
        ),
        annotations=[
            dict(x=0.035, y=0.845, xref="paper", yref="paper", xanchor="left",
                 showarrow=False, name="caption", text=caption0,
                 font=dict(size=30, color=th.ink)),
            dict(x=0.035, y=0.775, xref="paper", yref="paper", xanchor="left",
                 yanchor="top", align="left", showarrow=False, name="readout",
                 text=_readout(p, taus[0], rhos[0], premium, notional,
                               pnl_at_spot(taus[0], rhos[0]), sweep,
                               hedged=deltas is not None),
                 font=dict(size=13, color=th.ink_secondary)),
            dict(x=0.035, y=0.045, xref="paper", yref="paper", xanchor="left",
                 showarrow=False, align="left",
                 text=("Illustrative. Flat vols, constant correlation, no smile "
                       "or term structure — not a production risk view."),
                 font=dict(size=11, color=th.ink_muted)),
        ],
    )

    frames, captions = [], []
    for i, (tau, rho) in enumerate(zip(taus, rhos)):
        Z = pnl_grid(tau, rho)
        z_spot = pnl_at_spot(tau, rho)
        caption = (format_time_to_expiry(tau) if sweep == "time"
                   else f"ρ = {rho:+.2f}")
        captions.append(caption)

        layout = go.Layout(annotations=[
            dict(x=0.035, y=0.845, xref="paper", yref="paper", xanchor="left",
                 showarrow=False, text=caption, font=dict(size=30, color=th.ink)),
            dict(x=0.035, y=0.775, xref="paper", yref="paper", xanchor="left",
                 yanchor="top", align="left", showarrow=False,
                 text=_readout(p, tau, rho, premium, notional, z_spot, sweep,
                               hedged=deltas is not None),
                 font=dict(size=13, color=th.ink_secondary)),
            dict(x=0.035, y=0.045, xref="paper", yref="paper", xanchor="left",
                 showarrow=False, align="left",
                 text=("Illustrative. Flat vols, constant correlation, no smile "
                       "or term structure — not a production risk view."),
                 font=dict(size=11, color=th.ink_muted)),
        ])

        frames.append(go.Frame(
            name=str(i),
            data=[go.Surface(z=Z),
                  go.Scatter3d(x=[p.S0, p.S0], y=[p.X0, p.X0],
                               z=[z_spot, ztop])],
            traces=[0, 1],
            layout=layout,
        ))

    fig.frames = frames
    return SurfaceAnimation(figure=fig, frames=frames, captions=captions,
                            theme=th)


def add_playback_controls(anim: SurfaceAnimation, frame_ms: int = 45,
                          slider_label: str = "frame") -> go.Figure:
    """Play / pause / scrub, for the interactive build only."""
    fig = anim.figure
    th = anim.theme or resolve_theme(DEFAULT_THEME)
    transition = dict(duration=0)

    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            x=0.035, y=0.16, xanchor="left", yanchor="top",
            pad=dict(t=0, r=8),
            bgcolor=th.panel_bg, bordercolor=th.axis, borderwidth=1,
            font=dict(color=th.ink, size=13),
            buttons=[
                dict(label="▶  Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                      transition=transition,
                                      fromcurrent=True, mode="immediate")]),
                dict(label="‖  Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        transition=transition,
                                        mode="immediate")]),
            ])],
        sliders=[dict(
            active=0, x=0.035, y=0.1, len=0.55, xanchor="left",
            pad=dict(t=8, b=8),
            currentvalue=dict(visible=False),
            transition=transition,
            tickcolor=th.axis, bgcolor=th.axis, activebgcolor=th.ink,
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=th.ink_muted, size=10),
            steps=[dict(method="animate", label="",
                        args=[[str(i)],
                              dict(frame=dict(duration=0, redraw=True),
                                   transition=transition, mode="immediate")])
                   for i in range(len(anim.frames))])],
    )
    return fig
