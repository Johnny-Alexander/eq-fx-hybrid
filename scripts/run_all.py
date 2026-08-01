"""Run all analysis scripts in order; regenerates every figure."""
import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))

# Excluded from the sweep: rasterising ~600 frames through a headless browser
# takes minutes and writes video, not figures. Run it directly when you need
# the clips: python scripts/10_surface_animation.py
SKIP = {"10_surface_animation.py"}

scripts = sorted(f for f in os.listdir(HERE)
                 if f.startswith(("0", "1")) and f.endswith(".py")
                 and f != os.path.basename(__file__) and f not in SKIP)

for s in scripts:
    print(f"\n{'='*72}\n  Running {s}\n{'='*72}")
    subprocess.run([sys.executable, os.path.join(HERE, s)], check=True)

print("\nAll scripts complete. Figures in ../figures/")
print("Video: python scripts/10_surface_animation.py")
