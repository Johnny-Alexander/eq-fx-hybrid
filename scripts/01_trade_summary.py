"""Print a summary of the example trade with closed-form vs MC and Greeks."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid_pricer import price_hybrid_call, price_hybrid_call_mc, greeks
from src.trade_config import EXAMPLE_TRADE as p, NOTIONAL


def main():
    res = price_hybrid_call(p)
    mc = price_hybrid_call_mc(p)
    g = greeks(p)
    units = NOTIONAL / p.S0

    print("=" * 72)
    print(f"  EQ/FX Hybrid: SPX call x 1{{USDJPY > {p.B:.0f}}}   |   "
          f"Notional ${NOTIONAL/1e6:.0f}mm")
    print("=" * 72)
    print(f"  SPX:    spot={p.S0:.0f}  strike={p.K:.0f}  "
          f"({(p.S0/p.K-1)*100:+.1f}% ITM)")
    print(f"  USDJPY: spot={p.X0:.0f}  barrier={p.B:.0f}  "
          f"({(p.B/p.X0-1)*100:+.1f}% above spot)")
    print(f"  T={p.T}y, sig_S={p.sig_S:.0%}, sig_X={p.sig_X:.0%}, rho={p.rho}")
    print()
    print(f"  Premium per SPX unit:        ${res['price']:.2f}")
    print(f"  Premium % of notional:       {res['price']/p.S0:.2%}")
    print(f"  Premium $:                   ${res['price']*units/1e6:.2f}mm")
    print(f"  Vanilla equivalent:          ${res['vanilla_SPX_call']*units/1e6:.2f}mm")
    print(f"  Hybrid / Vanilla:            {res['price']/res['vanilla_SPX_call']:.1%}")
    print(f"  P(joint exercise):           {res['P_joint_exercise']:.1%}")
    print(f"  P(USDJPY > B):               {res['P_FX_above_barrier']:.1%}")
    print()
    print(f"  Closed-form price:           ${res['price']:.4f}")
    print(f"  Monte Carlo (2M paths):      ${mc['price']:.4f}  "
          f"(95% CI [{mc['ci95'][0]:.4f}, {mc['ci95'][1]:.4f}])")
    print()
    print("  Greeks scaled to $100mm notional:")
    print(f"    Delta SPX:   {g['delta_SPX']:.4f}     "
          f"-> ${g['delta_SPX']*NOTIONAL/1e6:.1f}mm SPX equivalent")
    # FX delta now in percentage convention: $ impact per 1% USDJPY move,
    # scaled by the position's notional-equivalent (units * V_per_unit).
    print(f"    Delta FX:    {g['delta_FX']:.2f}     "
          f"-> ${g['delta_FX']*units/1e3:.1f}k per +1% USDJPY move "
          f"(${g['delta_FX']*units/1e6:.3f}mm)")
    print(f"    Vega SPX:    {g['vega_SPX']:.2f}      "
          f"-> ${g['vega_SPX']*units/1e3:.1f}k per vol pt")
    print(f"    Vega FX:     {g['vega_FX']:.2f}      "
          f"-> ${g['vega_FX']*units/1e3:.1f}k per vol pt")
    print(f"    Cega:        {g['cega_corr']:.3f}     "
          f"-> ${g['cega_corr']*units/1e3:.1f}k per 1% corr")


if __name__ == "__main__":
    main()
