"""FX vega vs USDJPY spot at different SPX levels."""
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


def main():
    fx_grid = np.linspace(140, 175, 80)
    spx_levels = [6600, 6900, 7200, 7500, 7800]
    dv = 0.01
    units = NOTIONAL / p.S0

    fig, ax = plt.subplots(figsize=(10, 6))
    for S0 in spx_levels:
        vegas = []
        for X0 in fx_grid:
            up = price_at(S0, X0, sig_X=p.sig_X + dv)
            dn = price_at(S0, X0, sig_X=p.sig_X - dv)
            vegas.append((up - dn) / 2 * units / 1e3)
        ax.plot(fx_grid, vegas, label=f"SPX = {S0}", linewidth=2)

    ax.axvline(p.B, color="k", linestyle="--", alpha=0.6, label=f"Barrier {p.B:.0f}")
    ax.axvline(p.X0, color="red", linestyle=":", alpha=0.5, label=f"USDJPY spot {p.X0:.0f}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("USDJPY spot")
    ax.set_ylabel("FX vega ($k per vol point, on $100mm notional)")
    ax.set_title(f"FX Vega vs USDJPY Spot, by SPX Level\n"
                 f"(K={p.K:.0f}, B={p.B:.0f}, T={p.T}y, rho={p.rho})\n"
                 f"FX vega flips sign at the barrier")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "..", "figures", "04_fx_vega.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
