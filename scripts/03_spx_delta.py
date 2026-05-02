"""SPX delta and notional-equivalent hedge size vs SPX spot."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.parse_math"] = False
from dataclasses import replace
from matplotlib.ticker import FuncFormatter

from src.hybrid_pricer import price_hybrid_call
from src.trade_config import EXAMPLE_TRADE as p, NOTIONAL


def price_at(S0, X0, **ov):
    return price_hybrid_call(replace(p, S0=S0, X0=X0, **ov))["price"]


def main():
    spx_grid = np.linspace(6200, 8000, 80)
    fx_levels = [150.0, 153.0, 156.0, 159.0, 162.0]
    dS = 0.005 * p.K

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: unitless delta
    ax = axes[0]
    for X0 in fx_levels:
        deltas = [(price_at(s + dS, X0) - price_at(s - dS, X0)) / (2 * dS) for s in spx_grid]
        ax.plot(spx_grid, deltas, label=f"USDJPY = {X0:.0f}", linewidth=2)
    vanilla = [(price_at(s + dS, 1000.0) - price_at(s - dS, 1000.0)) / (2 * dS) for s in spx_grid]
    ax.plot(spx_grid, vanilla, "k--", label="Vanilla (no barrier)", linewidth=1.5, alpha=0.7)
    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.6, label=f"Strike {p.K:.0f}")
    ax.axvline(p.S0, color="red", linestyle=":", alpha=0.5, label=f"Spot {p.S0:.0f}")
    ax.set_xlabel("SPX spot")
    ax.set_ylabel(r"SPX delta $\partial V / \partial S$")
    ax.set_title(f"SPX Delta vs Spot, by USDJPY Level\n"
                 f"(K={p.K:.0f}, B={p.B:.0f}, T={p.T}y, rho={p.rho})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # Right: notional-equivalent ($mm of SPX to delta-hedge)
    ax = axes[1]
    for X0 in fx_levels:
        ne = [(price_at(s + dS, X0) - price_at(s - dS, X0)) / (2 * dS) * NOTIONAL / 1e6
              for s in spx_grid]
        ax.plot(spx_grid, ne, label=f"USDJPY = {X0:.0f}", linewidth=2)
    vanilla_ne = [(price_at(s + dS, 1000.0) - price_at(s - dS, 1000.0)) / (2 * dS) * NOTIONAL / 1e6
                  for s in spx_grid]
    ax.plot(spx_grid, vanilla_ne, "k--", label="Vanilla (no barrier)", linewidth=1.5, alpha=0.7)
    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.6)
    ax.axvline(p.S0, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel("SPX spot")
    ax.set_ylabel("SPX notional-equivalent ($mm)")
    ax.set_title(f"SPX Hedge Size on ${NOTIONAL/1e6:.0f}mm trade")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}mm"))

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "03_spx_delta.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
