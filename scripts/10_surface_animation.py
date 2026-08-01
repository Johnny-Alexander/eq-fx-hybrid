"""Render the PnL surfaces to MP4 for embedding in a slide deck.

    python3 scripts/10_surface_animation.py              # both clips, 1080p
    python3 scripts/10_surface_animation.py --scene time # just the time decay
    python3 scripts/10_surface_animation.py --scale 2    # 4K

Frames come from the same figure spec as the interactive page, so the video
and the hosted version cannot drift apart. Output is H.264 / yuv420p, which is
what PowerPoint and Keynote will both accept without transcoding.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio_ffmpeg
import numpy as np
import plotly.io as pio

from src.trade_config import EXAMPLE_TRADE as P, NOTIONAL
from src.viz.scenes import VIDEO_FRAMES, VIDEO_GRID, correlation_scene, time_decay_scene
from src.viz.surface3d import orbit_camera

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA = os.path.join(ROOT, "media")

FPS = 30
#: Seconds of frozen first/last frame. An animation that starts and stops dead
#: is unreadable on a projector -- the audience needs a beat to find the axes
#: before it moves, and a beat on the final shape before it cuts.
HOLD_START, HOLD_END = 1.2, 2.0

SCENES = {
    "time": dict(
        builder=lambda n_grid, n_frames: time_decay_scene(
            P, NOTIONAL, n_grid=n_grid, n_frames=n_frames),
        n_frames=VIDEO_FRAMES,
        filename="eqfx_dual_digital_time_decay.mp4",
        # A slow orbit through the clip: enough parallax to read the surface
        # as a solid, not enough to disorient anyone reading the axes.
        azimuth=(40.0, 74.0),
        elevation=(24.0, 17.0),
    ),
    "rho": dict(
        builder=lambda n_grid, n_frames: correlation_scene(
            P, NOTIONAL, n_grid=n_grid, n_frames=n_frames),
        n_frames=150,
        filename="eqfx_dual_digital_correlation.mp4",
        azimuth=(44.0, 64.0),
        elevation=(22.0, 22.0),
    ),
}


def _orbit(spec, n_frames):
    """Per-frame camera. The rho clip ping-pongs, so its orbit does too."""
    az0, az1 = spec["azimuth"]
    el0, el1 = spec["elevation"]
    t = np.linspace(0.0, 1.0, n_frames)
    # Smoothstep, so the camera eases in and out instead of starting at speed.
    t = t * t * (3.0 - 2.0 * t)
    return [orbit_camera(az0 + (az1 - az0) * ti, el0 + (el1 - el0) * ti)
            for ti in t]


def _encode(frame_dir, out_path, n_frames, fps, width, height):
    """PNG sequence -> H.264 MP4, with the first and last frames held."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    hold_start, hold_end = int(HOLD_START * fps), int(HOLD_END * fps)

    # A concat list is the reliable way to hold frames: repeating entries
    # avoids re-encoding the same PNG through a filter graph.
    listing = os.path.join(frame_dir, "frames.txt")
    with open(listing, "w") as fh:
        order = ([0] * hold_start + list(range(n_frames))
                 + [n_frames - 1] * hold_end)
        for i in order:
            fh.write(f"file '{os.path.join(frame_dir, f'f{i:05d}.png')}'\n")
            fh.write(f"duration {1.0 / fps:.6f}\n")
        fh.write(f"file '{os.path.join(frame_dir, f'f{n_frames - 1:05d}.png')}'\n")

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", listing,
        "-vf", f"scale={width}:{height}:flags=lanczos,fps={fps}",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return len(order) / fps


def render(scene_key, scale, n_grid, fps):
    spec = SCENES[scene_key]
    width, height = 1920 * scale, 1080 * scale

    print(f"\n[{scene_key}] building {spec['n_frames']} frames "
          f"on a {n_grid}x{n_grid} grid...")
    anim = spec["builder"](n_grid, spec["n_frames"])
    n_frames = len(anim.frames)
    cameras = _orbit(spec, n_frames)

    frame_dir = tempfile.mkdtemp(prefix=f"surface_{scene_key}_")
    try:
        figs = [anim.frame_figure(i, camera=cameras[i]) for i in range(n_frames)]
        paths = [os.path.join(frame_dir, f"f{i:05d}.png") for i in range(n_frames)]

        print(f"[{scene_key}] rasterising at {width}x{height}...")
        t0 = time.time()
        # Batch export keeps one browser alive for the whole sequence; going
        # frame by frame pays the ~5s startup every time.
        pio.write_images(figs, paths, width=1920, height=1080, scale=scale)
        print(f"[{scene_key}] {n_frames} frames in {time.time() - t0:.0f}s "
              f"({(time.time() - t0) / n_frames:.2f}s/frame)")

        os.makedirs(MEDIA, exist_ok=True)
        out_path = os.path.join(MEDIA, spec["filename"])
        duration = _encode(frame_dir, out_path, n_frames, fps, width, height)

        size_mb = os.path.getsize(out_path) / 1e6
        print(f"[{scene_key}] wrote {os.path.relpath(out_path, ROOT)}  "
              f"({duration:.1f}s, {size_mb:.1f} MB, {width}x{height})")
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", choices=[*SCENES, "all"], default="all")
    ap.add_argument("--scale", type=int, default=1,
                    help="1 = 1920x1080, 2 = 3840x2160")
    ap.add_argument("--grid", type=int, default=VIDEO_GRID)
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    keys = list(SCENES) if args.scene == "all" else [args.scene]
    for key in keys:
        render(key, args.scale, args.grid, args.fps)


if __name__ == "__main__":
    main()
