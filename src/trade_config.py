"""Default example trades used across the analysis scripts and notebooks."""
import numpy as np

from src.hybrid import EquityLeg, RateCondition  # noqa: F401
from src.hybrid_pricer import HybridInputs

NOTIONAL = 100e6  # $100mm

# -- EQ/FX: SPX call x 1{USDJPY > 160} -----------------------------------

EXAMPLE_TRADE = HybridInputs(
    S0=7200.0,    # SPX spot
    X0=156.0,     # USDJPY spot
    K=7000.0,     # SPX strike (~2.9% ITM)
    B=160.0,      # USDJPY barrier (~2.6% above spot)
    T=0.5,        # 6 months
    r_d=0.045,    # USD rate
    r_f=0.005,    # JPY rate
    q=0.015,      # SPX dividend yield
    sig_S=0.16,   # SPX vol
    sig_X=0.10,   # USDJPY vol
    rho=0.30,     # SPX/USDJPY correlation
)

# -- EQ/IR: SPX call x 1{CMS10 > 4%} -------------------------------------
#
# Longer dated than the FX trade, because a rate condition needs time to be
# interesting. Note the correlation sign: SPX and rates are usually taken
# negatively correlated over this horizon, which makes the joint event
# {SPX up, rates up} rarer and so cheapens the structure relative to rho=0.

EQIR_T = 2.0
EQIR_EQUITY = EquityLeg.from_market(
    F=7200.0 * np.exp((0.045 - 0.015) * EQIR_T),  # SPX forward to 2y
    K=7000.0,
    P0T=np.exp(-0.045 * EQIR_T),                  # USD discount factor
    sig_S=0.16,                                   # vol of the FORWARD
    T=EQIR_T,
)
EQIR_FORWARD_SWAP = 0.0410  # unadjusted 10y forward swap rate, 2y forward
EQIR_CONDITION = RateCondition.from_forward_swap(
    R_0=EQIR_FORWARD_SWAP,
    B=0.04,         # 4% barrier
    sig_R=0.0080,   # 80bp/yr normal vol
    T=EQIR_T,
    tenor=10,       # CMS10
    freq=2,         # semiannual swap
)  # -> R_adj ~= 4.1445%, i.e. +4.45bp of convexity
EQIR_RHO = -0.30    # SPX / CMS10 correlation
