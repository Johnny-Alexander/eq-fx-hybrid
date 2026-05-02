"""Six-panel dashboard of non-trivial Greeks dynamics for the EQ/FX hybrid:

  1. Cross-gamma:   d(SPX_delta)/d(USDJPY) - the hedge breaks when FX moves
  2. FX delta:      vs SPX spot - peaks near strike (digital-shaped)
  3. Cega term:     correlation risk vs time to expiry
  4. Cega vs rho:   nonlinear and sign-changing in FX level
  5. SPX vanna:     skew sensitivity flips by FX level
  6. Cross-vega:    SPX vega sensitivity to FX
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

from src.hybrid_pricer import price_hybrid_call
from src.trade_config import EXAMPLE_TRADE as p


def price_at(S0, X0, **ov):
    return price_hybrid_call(replace(p, S0=S0, X0=X0, **ov))["price"]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("EQ/FX Hybrid: SPX call $\\cdot$ 1{USDJPY > B}  —  Non-trivial Greeks",
                 fontsize=13, y=1.00)

    dS, dX = 0.005 * p.K, 0.5
    dv = 0.01

    # 1. Cross-gamma
    ax = axes[0, 0]
    spx_grid = np.linspace(p.S0 - 1100, p.S0 + 900, 50)
    fx_grid = np.linspace(p.X0 - 11, p.X0 + 14, 50)
    SS, XX = np.meshgrid(spx_grid, fx_grid)
    cg = np.zeros_like(SS)
    for i in range(SS.shape[0]):
        for j in range(SS.shape[1]):
            s, x = SS[i, j], XX[i, j]
            d_up = (price_at(s + dS, x + dX) - price_at(s - dS, x + dX)) / (2 * dS)
            d_dn = (price_at(s + dS, x - dX) - price_at(s - dS, x - dX)) / (2 * dS)
            cg[i, j] = (d_up - d_dn) / (2 * dX)
    cf = ax.contourf(SS, XX, cg, levels=20, cmap="RdBu_r",
                     vmin=-np.abs(cg).max(), vmax=np.abs(cg).max())
    ax.axhline(p.B, color="k", linestyle="--", alpha=0.5, label="Barrier")
    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.5, label="Strike")
    plt.colorbar(cf, ax=ax, label=r"$\partial \Delta_S / \partial X$")
    ax.set_xlabel("SPX spot"); ax.set_ylabel("USDJPY spot")
    ax.set_title("Cross-gamma: SPX delta sensitivity to FX")
    ax.legend(loc="lower right", fontsize=8)

    # 2. FX delta vs SPX
    ax = axes[0, 1]
    spx2 = np.linspace(p.S0 - 1200, p.S0 + 800, 80)
    for X0 in [150, 153, 156, 160, 162]:
        fx_d = [(price_at(s, X0 + dX) - price_at(s, X0 - dX)) / (2 * dX) for s in spx2]
        ax.plot(spx2, fx_d, label=f"USDJPY={X0}", linewidth=1.8)
    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("SPX spot"); ax.set_ylabel(r"FX delta $\partial V/\partial X$")
    ax.set_title("FX delta vs SPX spot\n(rises with moneyness; digital-shape)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3. Cega term structure
    ax = axes[0, 2]
    T_grid = np.linspace(0.05, 3.0, 60)
    drho = 0.01
    for X0 in [150, 156, 160]:
        cegas = []
        for T in T_grid:
            up = price_at(p.S0, X0, T=T, rho=p.rho + drho)
            dn = price_at(p.S0, X0, T=T, rho=p.rho - drho)
            cegas.append((up - dn) / 2)
        ax.plot(T_grid, cegas, label=f"USDJPY={X0}", linewidth=1.8)
    ax.set_xlabel("Time to expiry (years)"); ax.set_ylabel("Cega (per 1% corr)")
    ax.set_title("Cega term structure")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 4. Cega vs rho
    ax = axes[1, 0]
    rho_grid = np.linspace(-0.8, 0.9, 60)
    for X0 in [150, 156, 160]:
        cegas = []
        for rho in rho_grid:
            up = price_at(p.S0, X0, rho=rho + drho)
            dn = price_at(p.S0, X0, rho=rho - drho)
            cegas.append((up - dn) / 2)
        ax.plot(rho_grid, cegas, label=f"USDJPY={X0}", linewidth=1.8)
    ax.set_xlabel(r"Correlation $\rho$"); ax.set_ylabel("Cega (per 1% corr)")
    ax.set_title("Cega vs correlation level\n(non-linear, sign of slope flips with FX)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 5. SPX vanna
    ax = axes[1, 1]
    spx5 = np.linspace(p.S0 - 1200, p.S0 + 800, 60)
    for X0 in [150, 156, 160, 162]:
        vannas = []
        for s in spx5:
            d_up = (price_at(s + dS, X0, sig_S=p.sig_S + dv)
                    - price_at(s - dS, X0, sig_S=p.sig_S + dv)) / (2 * dS)
            d_dn = (price_at(s + dS, X0, sig_S=p.sig_S - dv)
                    - price_at(s - dS, X0, sig_S=p.sig_S - dv)) / (2 * dS)
            vannas.append((d_up - d_dn) / 2)
        ax.plot(spx5, vannas, label=f"USDJPY={X0}", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("SPX spot"); ax.set_ylabel(r"Vanna $\partial \Delta_S / \partial \sigma_S$")
    ax.set_title("SPX vanna by FX level")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 6. Cross-vega
    ax = axes[1, 2]
    fx6 = np.linspace(p.X0 - 11, p.X0 + 14, 80)
    for S0 in [p.S0 - 300, p.S0, p.S0 + 300]:
        cvs = []
        for X0 in fx6:
            v_up = (price_at(S0, X0 + dX, sig_S=p.sig_S + dv)
                    - price_at(S0, X0 + dX, sig_S=p.sig_S - dv)) / 2
            v_dn = (price_at(S0, X0 - dX, sig_S=p.sig_S + dv)
                    - price_at(S0, X0 - dX, sig_S=p.sig_S - dv)) / 2
            cvs.append((v_up - v_dn) / (2 * dX))
        ax.plot(fx6, cvs, label=f"SPX={S0}", linewidth=1.8)
    ax.axvline(p.B, color="k", linestyle="--", alpha=0.5, label="Barrier")
    ax.set_xlabel("USDJPY spot"); ax.set_ylabel(r"$\partial \nu_S / \partial X$")
    ax.set_title("Cross-vega: SPX vega sensitivity to FX")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "06_greeks_dashboard.png")
    plt.savefig(out, dpi=125, bbox_inches="tight")
    print(f"saved: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
