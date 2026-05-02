"""Default example trade used across all analysis scripts."""
from src.hybrid_pricer import HybridInputs

NOTIONAL = 100e6  # $100mm

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
