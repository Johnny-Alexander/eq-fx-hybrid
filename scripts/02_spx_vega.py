"""SPX vega vs SPX spot at different USDJPY levels."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.parse_math"] = False
from dataclasses import replace

from src.hybrid_pricer import price_hybrid_call
from src.trade_config import EXAMPLE_TRADE as p, NOTIONAL


def price_at(S0, X0, **overrides):
    return price_hybrid_call(replace(p, S0=S0, X0=X0, **overrides))["price"]


def main():
    spx_grid = np.linspace(6200, 8000, 80)
    fx_levels = [150.0, 153.0, 156.0, 159.0, 162.0]
    dv = 0.01
    units = NOTIONAL / p.S0

    fig, ax = plt.subplots(figsize=(10, 6))
    for X0 in fx_levels:
        vegas = []
        for S0 in spx_grid:
            up = price_at(S0, X0, sig_S=p.sig_S + dv)
            dn = price_at(S0, X0, sig_S=p.sig_S - dv)
            vegas.append((up - dn) / 2 * units / 1e3)  # $k per vol point
        ax.plot(spx_grid, vegas, label=f"USDJPY = {X0:.0f}", linewidth=2)

    # Vanilla reference: barrier essentially always breached
    vanilla = []
    for S0 in spx_grid:
        up = price_at(S0, 1000.0, sig_S=p.sig_S + dv)
        dn = price_at(S0, 1000.0, sig_S=p.sig_S - dv)
        vanilla.append((up - dn) / 2 * units / 1e3)
    ax.plot(spx_grid, vanilla, "k--", label="Vanilla (no barrier)",
            linewidth=1.5, alpha=0.7)

    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.6, label=f"Strike {p.K:.0f}")
    ax.axvline(p.S0, color="red", linestyle=":", alpha=0.5, label=f"Spot {p.S0:.0f}")
    ax.set_xlabel("SPX spot")
    ax.set_ylabel("SPX vega ($k per vol point, on $100mm notional)")
    ax.set_title(f"SPX Vega vs Spot, by USDJPY Level\n"
                 f"(K={p.K:.0f}, B={p.B:.0f}, T={p.T}y, rho={p.rho})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "..", "figures", "02_spx_vega.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
