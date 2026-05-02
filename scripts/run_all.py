"""Run all analysis scripts in order; regenerates every figure."""
import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
scripts = sorted(f for f in os.listdir(HERE)
                 if f.startswith(("0", "1")) and f.endswith(".py")
                 and f != os.path.basename(__file__))

for s in scripts:
    print(f"\n{'='*72}\n  Running {s}\n{'='*72}")
    subprocess.run([sys.executable, os.path.join(HERE, s)], check=True)

print("\nAll scripts complete. Figures in ../figures/")
