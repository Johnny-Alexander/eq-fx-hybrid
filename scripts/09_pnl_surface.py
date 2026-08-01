"""Interactive 3-D PnL surface for the EQ/FX dual digital -> docs/surface.html.

Writes a self-contained page with play/pause, a scrub slider and free camera
rotation, plus a poster still for slide decks that cannot play video.

    python3 scripts/09_pnl_surface.py

The page is written into ``docs/`` so GitHub Pages serves it alongside the
existing pricer app.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotly.io as pio

from src.trade_config import EXAMPLE_TRADE as P, NOTIONAL
from src.viz.scenes import (WEB_FRAMES, WEB_GRID, WEB_RHO_FRAMES,
                            correlation_scene, hedged_time_decay_scene,
                            time_decay_scene)
from src.viz.surface3d import THEMES, add_playback_controls

#: Keep in step with scripts/10 -- the page and the video should not disagree
#: about what the surfaces look like.
RENDER_THEME = "paper"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
FIGURES = os.path.join(ROOT, "figures")

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "width": 1920, "height": 1080,
                             "scale": 2, "filename": "eqfx_pnl_surface"},
}

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>EQ/FX Dual Digital — PnL Surface</title>
<style>
  html, body {{ margin: 0; padding: 0; background: {bg}; color: {ink};
                font-family: Inter, -apple-system, "Helvetica Neue", Arial, sans-serif; }}
  .wrap {{ max-width: 1600px; margin: 0 auto; padding: 8px 16px 48px; }}
  .chart {{ width: 100%; height: min(82vh, 900px); }}
  nav {{ display: flex; gap: 20px; padding: 14px 4px 4px; font-size: 14px; }}
  nav a {{ color: #75726b; text-decoration: none; padding-bottom: 4px;
           border-bottom: 2px solid transparent; }}
  nav a.on {{ color: #0b0b0b; border-bottom-color: #2a78d6; }}
  nav a:hover {{ color: #0b0b0b; }}
  .note {{ font-size: 13px; line-height: 1.6; color: #75726b;
           max-width: 70ch; padding: 8px 4px; }}
  .note b {{ color: #1a1917; font-weight: 600; }}
</style>
</head>
<body>
<div class="wrap">
  <nav>
    <a href="index.html">Pricer</a>
    <a href="surface.html" class="on">PnL surface</a>
    <a href="https://github.com/Johnny-Alexander/hybrid-pricer">Source</a>
  </nav>
  {time_div}
  {hedged_div}
  {rho_div}
  <p class="note">
    Drag to rotate, scroll to zoom, and press <b>Play</b> on any panel.
    Height is PnL on a short {notional:,.0f}mm position: the bank keeps the
    premium if the pair of conditions misses, and pays the notional if both
    land. The first panel runs time down to expiry with the market held still.
    The second subtracts a delta hedge struck once at inception and held —
    note that no desk hedges that way, so the wings are what a <i>static</i>
    hedge costs, not what hedging costs. The third holds time still and moves
    correlation alone.
    Priced with the closed form in <b>src/hybrid/</b>, cross-checked against
    Monte Carlo in the repo's test suite.
  </p>
</div>
</body>
</html>
"""


def _write_html(figs, notional):
    divs = {}
    for i, (key, fig) in enumerate(figs.items()):
        divs[f"{key}_div"] = '<div class="chart">%s</div>' % pio.to_html(
            fig, include_plotlyjs="cdn" if i == 0 else False, full_html=False,
            div_id=f"surface-{key}", config=PLOTLY_CONFIG,
            auto_play=False, default_height="100%")

    page = PAGE_TEMPLATE.format(
        bg=THEMES[RENDER_THEME].surface_bg,
        ink=THEMES[RENDER_THEME].ink_secondary,
        notional=notional / 1e6, **divs)
    path = os.path.join(DOCS, "surface.html")
    with open(path, "w") as fh:
        fh.write(page)
    return path


def main():
    os.makedirs(DOCS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    print(f"Building interactive surfaces ({WEB_GRID}x{WEB_GRID} grid, "
          f"theme={RENDER_THEME})...")
    anims = {
        "time": time_decay_scene(P, NOTIONAL, n_grid=WEB_GRID,
                                 n_frames=WEB_FRAMES, theme=RENDER_THEME),
        "hedged": hedged_time_decay_scene(P, NOTIONAL, n_grid=WEB_GRID,
                                          n_frames=WEB_FRAMES,
                                          theme=RENDER_THEME),
        "rho": correlation_scene(P, NOTIONAL, n_grid=WEB_GRID,
                                 n_frames=WEB_RHO_FRAMES, theme=RENDER_THEME),
    }
    figs = {k: add_playback_controls(a, frame_ms=45) for k, a in anims.items()}

    path = _write_html(figs, NOTIONAL)
    time_anim = anims["time"]
    size_mb = os.path.getsize(path) / 1e6
    print(f"  wrote {os.path.relpath(path, ROOT)}  ({size_mb:.1f} MB)")

    # Poster still: the most striking frame of the time sweep, for decks that
    # cannot play video.
    poster_index = int(len(time_anim.frames) * 0.88)
    poster = time_anim.frame_figure(poster_index)
    poster_path = os.path.join(FIGURES, "09_pnl_surface_poster.png")
    poster.write_image(poster_path, width=1920, height=1080, scale=2)
    print(f"  wrote {os.path.relpath(poster_path, ROOT)}  "
          f"(frame {poster_index}: {time_anim.captions[poster_index]})")


if __name__ == "__main__":
    main()
