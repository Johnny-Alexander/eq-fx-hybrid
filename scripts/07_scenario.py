"""Walk through PnL of a delta-hedged hybrid through a sharp adverse FX move.

Starting state: deep ITM SPX, FX leg ITM, perfectly delta-hedged on SPX.
Shock:          USDJPY drops sharply, flipping the FX leg to OTM.
Result:         large MTM loss because the SPX hedge does nothing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.parse_math"] = False
from dataclasses import replace

from src.hybrid_pricer import price_hybrid_call
from src.trade_config import EXAMPLE_TRADE as p, NOTIONAL


def price_at(S0, X0, **ov):
    return price_hybrid_call(replace(p, S0=S0, X0=X0, **ov))["price"]


def delta_at(S0, X0, dS=35.0):
    return (price_at(S0 + dS, X0) - price_at(S0 - dS, X0)) / (2 * dS)


def main():
    # State 0: deep ITM SPX, FX leg comfortably above barrier
    S0 = 7600.0
    X0 = 162.0
    P0 = price_at(S0, X0)
    D0 = delta_at(S0, X0)
    units0 = NOTIONAL / S0
    hedge_units = D0 * units0           # SPX units short to delta-hedge
    opt_value0 = P0 * units0

    print("=" * 72)
    print("STATE 0: deep ITM SPX, FX leg ITM, delta-hedged")
    print("=" * 72)
    print(f"  SPX={S0:.0f}, USDJPY={X0:.0f}, MTM=${opt_value0/1e6:.2f}mm, "
          f"delta={D0:.3f}")
    print(f"  Hedge: short {hedge_units:,.0f} SPX = ${hedge_units*S0/1e6:.2f}mm short")
    print()

    # Three flavors of shock
    scenarios = [
        ("(a) FX -6, SPX flat", S0,        156.0),
        ("(b) FX -6, SPX -2%",  S0 * 0.98, 156.0),
        ("(c) FX -6, SPX -4%",  S0 * 0.96, 156.0),
    ]
    print("=" * 72)
    print("SHOCK: USDJPY 162 -> 156 (sharp JPY rally)")
    print("=" * 72)
    print(f"{'Scenario':<22}{'OptPnL':>10}{'HedgePnL':>11}{'NetPnL':>10}")
    print("-" * 72)
    for name, S1, X1 in scenarios:
        opt_pnl = (price_at(S1, X1) - P0) * units0
        hedge_pnl = hedge_units * (S0 - S1)
        net = opt_pnl + hedge_pnl
        print(f"{name:<22}{opt_pnl/1e6:>+10.3f}{hedge_pnl/1e6:>+11.3f}{net/1e6:>+10.3f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: PnL vs FX shock at fixed SPX, with delta+gamma decomposition
    ax = axes[0]
    fx_path = np.linspace(-8, 2, 60)
    fx_d0 = (price_at(S0, X0 + 0.5) - price_at(S0, X0 - 0.5)) / 1.0
    fx_g0 = (price_at(S0, X0 + 0.5) - 2 * P0 + price_at(S0, X0 - 0.5)) / (0.5 ** 2)

    full = [(price_at(S0, X0 + dx) - P0) * units0 / 1e6 for dx in fx_path]
    lin = [fx_d0 * dx * units0 / 1e6 for dx in fx_path]
    quad = [(fx_d0 * dx + 0.5 * fx_g0 * dx ** 2) * units0 / 1e6 for dx in fx_path]

    ax.plot(fx_path, full, "b-", linewidth=2.5, label="Option MTM PnL (full)")
    ax.plot(fx_path, lin, "g--", linewidth=1.5, alpha=0.7, label="FX delta approx")
    ax.plot(fx_path, quad, "orange", linestyle="--", linewidth=1.5, alpha=0.8,
            label="FX delta + gamma")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(-6, color="red", linestyle=":", alpha=0.6, label="-6 yen shock")
    ax.set_xlabel("USDJPY shock (yen)")
    ax.set_ylabel("PnL ($mm)")
    ax.set_title(f"PnL vs FX shock (SPX={S0:.0f} held flat, hedge unchanged)\n"
                 f"Starting state: USDJPY={X0:.0f}, FX leg ITM")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    # Panel 2: 2D PnL surface with hedge held fixed
    ax = axes[1]
    spx_shocks = np.linspace(-0.06, 0.04, 40)
    fx_shocks = np.linspace(-8, 2, 40)
    PNL = np.zeros((len(fx_shocks), len(spx_shocks)))
    for i, dx in enumerate(fx_shocks):
        for j, ds in enumerate(spx_shocks):
            S1 = S0 * (1 + ds); X1 = X0 + dx
            opt_pnl = (price_at(S1, X1) - P0) * units0
            hedge_pnl = hedge_units * (S0 - S1)
            PNL[i, j] = (opt_pnl + hedge_pnl) / 1e6
    vmax = np.abs(PNL).max()
    cf = ax.contourf(spx_shocks * 100, fx_shocks, PNL,
                     levels=np.linspace(-vmax, vmax, 21), cmap="RdYlGn")
    ax.contour(spx_shocks * 100, fx_shocks, PNL, levels=[0],
               colors="black", linewidths=1.2)
    plt.colorbar(cf, ax=ax, label="Net PnL ($mm)")
    ax.scatter([0], [0], color="white", edgecolor="black", s=80, zorder=5, label="Start")
    ax.scatter([0], [-6], color="black", s=80, marker="X", zorder=5, label="(a)")
    ax.scatter([-2], [-6], color="black", s=80, marker="^", zorder=5, label="(b)")
    ax.set_xlabel("SPX shock (%)"); ax.set_ylabel("USDJPY shock (yen)")
    ax.set_title("Net PnL surface (delta-hedged at start)\n"
                 "Hedge captures SPX, leaves FX naked")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.4); ax.axvline(0, color="black", linewidth=0.4)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "07_scenario.png")
    plt.savefig(out, dpi=125, bbox_inches="tight")
    print(f"\nsaved: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
