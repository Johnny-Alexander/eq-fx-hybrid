"""Summary of the EQ/IR example trade: SPX call x 1{CMS10 > 4%}.

Mirrors 01_trade_summary.py, but reports risk in rates conventions -- DV01
per basis point rather than delta per 1% spot move -- and shows what the
convexity adjustment is worth.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace

from src.hybrid import RateCondition, conditional_european, conditional_european_mc
from src.hybrid.eqir import greeks
from src.trade_config import EQIR_TRADE as t, NOTIONAL


def main():
    res = t.result()
    eq, cms = t.legs()
    mc = conditional_european_mc(eq, cms, t.T, t.rho, t.eq_type, t.cond_dir)
    g = greeks(t)
    units = NOTIONAL / t.F

    print("=" * 72)
    print(f"  EQ/IR Hybrid: SPX call x 1{{CMS10 > {t.B:.0%}}}   |   "
          f"Notional ${NOTIONAL/1e6:.0f}mm")
    print("=" * 72)
    print(f"  SPX:    forward={t.F:.0f}  strike={t.K:.0f}  "
          f"({(t.F/t.K-1)*100:+.1f}% ITM forward)")
    print(f"  CMS10:  forward swap={t.R_0:.3%}  barrier={t.B:.2%}")
    print(f"          convexity-adjusted={t.R_adj:.4%}  "
          f"(+{(t.R_adj-t.R_0)*1e4:.2f}bp)")
    print(f"  T={t.T}y, sig_S={t.sig_S:.0%}, sig_R={t.sig_R*1e4:.0f}bp/yr, "
          f"rho={t.rho}")
    print()
    print(f"  Premium per SPX unit:        ${res['price']:.2f}")
    print(f"  Premium % of forward:        {res['price']/t.F:.2%}")
    print(f"  Premium $:                   ${res['price']*units/1e6:.2f}mm")
    print(f"  Vanilla equivalent:          "
          f"${res['vanilla_equivalent']*units/1e6:.2f}mm")
    print(f"  Hybrid / Vanilla:            "
          f"{res['price']/res['vanilla_equivalent']:.1%}")
    print(f"  P(joint exercise):           {res['P_joint_exercise']:.1%}")
    print(f"  P(CMS10 > {t.B:.0%}):              {res['P_condition']:.1%}")
    print()
    print(f"  Closed-form price:           ${res['price']:.4f}")
    print(f"  Monte Carlo (2M paths):      ${mc['price']:.4f}  "
          f"(95% CI [{mc['ci95'][0]:.4f}, {mc['ci95'][1]:.4f}])")
    print()

    # What ignoring convexity would cost you.
    naive_cms = RateCondition(R_adj=t.R_0, B=t.B, sig_R=t.sig_R)
    naive = conditional_european(eq, naive_cms, t.T, t.rho, t.eq_type,
                                 t.cond_dir)
    gap = res["price"] - naive["price"]
    print("  Convexity adjustment impact:")
    print(f"    Unadjusted (wrong):        ${naive['price']:.4f}")
    print(f"    Adjusted:                  ${res['price']:.4f}")
    print(f"    Understated by:            ${gap:.4f}  "
          f"({gap/res['price']:.1%} of premium, "
          f"${gap*units/1e6:.2f}mm on notional)")
    print()

    print("  Risk, in each desk's convention, scaled to $100mm notional:")
    print(f"    Delta EQ:    {g['delta_eq']:.4f}      "
          f"-> ${g['delta_eq']*NOTIONAL/1e6:.1f}mm SPX equivalent")
    print(f"    Gamma EQ:    {g['gamma_eq']:.3e}")
    print(f"    Vega EQ:     {g['vega_eq']:.2f}       "
          f"-> ${g['vega_eq']*units/1e3:.1f}k per vol pt")
    print(f"    DV01:        {g['dv01']:.4f}      "
          f"-> ${g['dv01']*units/1e3:.1f}k per +1bp on CMS10")
    print(f"    Rate vega:   {g['rate_vega']:.4f}     "
          f"-> ${g['rate_vega']*units/1e3:.1f}k per +1bp of normal vol")
    print(f"    Cega:        {g['cega']:.4f}      "
          f"-> ${g['cega']*units/1e3:.1f}k per 1% corr")
    print(f"    Discount DV01: {g['discount_dv01']:.4f}    "
          f"-> ${g['discount_dv01']*units/1e3:.1f}k per +1bp on the curve")
    print()
    print("    (DV01 and discount DV01 are partial derivatives -- a real")
    print("     parallel shift would move both, plus the equity forward.)")
    print()

    print("  Sensitivity of premium to correlation:")
    for rho in (-0.6, -0.3, 0.0, 0.3, 0.6):
        p = replace(t, rho=rho).price()
        print(f"    rho={rho:+.1f}   ${p:8.2f}   "
              f"({p/res['price']-1:+.1%} vs base)")


if __name__ == "__main__":
    main()
