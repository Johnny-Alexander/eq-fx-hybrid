# Cross-Asset Hybrid Option Pricer

Closed-form bivariate Black–Scholes pricing and risk analysis for two cross-asset structures — an equity payoff gated by a condition on **another asset class**. The conditioning observable is pluggable: **FX** (`USDJPY > 160`) and **rates** (`CMS10 > 4%`) share one closed form.

Taking SPX and USDJPY as the worked example:

1. **Conditional European** — a vanilla SPX call/put gated by a USDJPY barrier:
   $$V_T = \max\!\big(\eta_S(S_T-K),0\big) \cdot \mathbf{1}\{\eta_X(X_T-B)>0\}$$
2. **Joint cash-or-nothing double digital** — a fixed cash payout if both an SPX and an FX condition are met:
   $$V_T = N \cdot \mathbf{1}\{\eta_S(S_T-K)>0\} \cdot \mathbf{1}\{\eta_X(X_T-B)>0\}$$

The signs $\eta_S, \eta_X \in \{+1,-1\}$ select between calls/puts and above/below the barrier, giving four sign variants for each structure. All eight closed forms reduce to one bivariate normal CDF call.

---

## Start here — the two notebooks

The repo is built around two self-contained walkthroughs:

| Notebook | Topic |
|---|---|
| [`notebooks/conditional_european.ipynb`](notebooks/conditional_european.ipynb) | Vanilla call/put × FX above/below barrier. Closed-form pricing for all 4 variants, FX forwards (with the quanto correction), Greeks across variants, deep dive on SPX delta / FX delta / SPX vega / FX vega / cega for the canonical `call × X > B` case, and a scenario walkthrough showing why static delta-hedging fails on a 6-yen FX shock. |
| [`notebooks/digital.ipynb`](notebooks/digital.ipynb) | Joint cash-or-nothing double digital. All 4 sign partitions priced; identity check that the four sum to PV(notional); cega sign-flip across variants; gamma concentration near the strike. |

Trade attributes are defined inline at the top of each notebook — no external config to track. The pricer code stays in `src/hybrid_pricer.py`.

```bash
pip install -r requirements.txt
jupyter notebook notebooks/conditional_european.ipynb
```

---

## The model

Both products price under the **USD risk-neutral measure** $\mathbb{Q}^{USD}$:

$$\frac{dS}{S} = (r_{USD} - q)\,dt + \sigma_S\,dW^S$$
$$\frac{dX}{X} = (r_{JPY} - r_{USD} + \sigma_X^2)\,dt + \sigma_X\,dW^X$$
$$d\langle W^S, W^X\rangle = \rho\,dt$$

The $+\sigma_X^2$ on the FX drift is the **quanto convexity correction** from changing numeraire from JPY-cash (where USDJPY's textbook Garman–Kohlhagen drift is $r_{JPY} - r_{USD}$) to USD-cash, applying Itô to $1/X$. It matters whenever the payoff settles in a currency other than the FX pair's "domestic" leg.

In the code, the variable `r_d` = USD rate (settlement currency, used for discounting) and `r_f` = JPY rate. This naming is "domestic/foreign from the trade's perspective" rather than from USDJPY's quote convention. Confusing, but stable.

### FX forwards

Two FX forwards are worth distinguishing:
- **Market FX forward** (CIP, model-free): $F_X^{\text{mkt}} = X_0\, e^{(r_{JPY}-r_{USD})T}$ — what a USDJPY forward actually trades at.
- **USD-measure expectation** (model-dependent): $\mathbb{E}^{Q^{USD}}[X_T] = X_0\, e^{(r_{JPY}-r_{USD}+\sigma_X^2)T}$ — what the pricer uses internally.

The notebooks show both, and the size of the convexity gap.

### Closed form (conditional european)

$$V_0 = \eta_S\, S_0 e^{-qT}\, M_2\!\big(\eta_S d_1^S,\; \eta_X(d_2^X+\rho\sigma_S\sqrt{T});\; \eta_S\eta_X\rho\big) \;-\; \eta_S\, K e^{-r_{USD}T}\, M_2\!\big(\eta_S d_2^S,\; \eta_X d_2^X;\; \eta_S\eta_X\rho\big)$$

with the standard $d_1, d_2$ for SPX and FX (FX uses the USD-measure drift). The first term reflects a numeraire change to the equity asset; under that measure the FX log-mean shifts by $\rho\sigma_S\sqrt{T}$. The pair of $M_2$ calls reduces to the unconditional vanilla Black–Scholes when both signs are $+1$ and $B \to 0$.

### Closed form (double digital)

$$V_0 = N\, e^{-r_{USD}T}\, M_2\!\big(\eta_S d_2^S,\; \eta_X d_2^X;\; \eta_S\eta_X\rho\big)$$

No numeraire change is needed: the payoff is a fixed cash amount, evaluated under $\mathbb{Q}^{USD}$ directly. The four sign variants partition the $(S, X)$ plane and sum to $N\,e^{-r_{USD}T}$.

### Identity checks

The notebooks verify, to floating-point precision, the structural identities that fall out of $\mathbf{1}\{X>B\} + \mathbf{1}\{X<B\} = 1$:

- $V_{\text{call,above}} + V_{\text{call,below}} = V_{\text{vanilla call}}$
- $V_{\text{put,above}}  + V_{\text{put,below}}  = V_{\text{vanilla put}}$
- $\sum_{\text{4 digital variants}} = N\,e^{-r_{USD}T}$

All eight prices are also cross-checked against Monte Carlo.

---

## Example trade

Both notebooks ship with the same scenario (parameters set inline at the top of each):

| | |
|---|---|
| SPX | spot 7200, strike 7000 |
| USDJPY | spot 156, barrier 160 |
| Tenor | 6 months |
| Vols | $\sigma_S$ = 16%, $\sigma_X$ = 10% |
| Rates | USD 4.5%, JPY 0.5%, SPX div 1.5% |
| Correlation | $\rho$ = 0.30 |

Notional: $100mm for the conditional european, $20mm for the double digital.

---

## Why these trades are interesting

A vanilla SPX call costs you the full premium regardless of FX. Gating it on a USDJPY barrier reduces premium but introduces three intertwined exposures:

1. **SPX risk** — delta, gamma, vega, vanna. Familiar shapes, but each is *fanned by FX level*.
2. **FX risk** — delta, gamma, vega. The FX vega **flips sign at the barrier**: above the barrier you're short FX vol, below you're long.
3. **Correlation risk (cega)** — the unhedgeable one. No vanilla product carries pure $\rho$ risk.

A "perfectly SPX-delta-hedged" position can lose 25%+ of MTM on a single liquid day in USDJPY. The conditional-european notebook quantifies this in §8.

---

## Repository layout

```
hybrid-pricer/
├── notebooks/
│   ├── conditional_european.ipynb   # vanilla x FX-indicator, 4 variants
│   └── digital.ipynb                # joint cash-or-nothing, 4 variants
├── src/
│   ├── hybrid/                      # the pricer, split by responsibility
│   │   ├── bivariate.py             #   shared normal machinery
│   │   ├── equity.py                #   equity leg: forward, d1/d2, vanilla
│   │   ├── conditions.py            #   pluggable conditioning legs (FX, rates)
│   │   ├── cms.py                   #   CMS convexity adjustment
│   │   ├── products.py              #   conditional european, double digital
│   │   └── greeks.py                #   generic bump-and-revalue
│   ├── hybrid_pricer.py             # compatibility shim over src/hybrid
│   └── trade_config.py              # example trade for scripts/tests
├── scripts/                         # 7 standalone analysis scripts -> figures/
├── figures/                         # pre-generated PNGs from scripts/
├── app/
│   ├── app.py                       # Streamlit pricer (mobile-friendly)
│   └── MOBILE_DEPLOY.md
├── docs/                            # GitHub Pages (stlite/Pyodide) deploy
├── tests/
│   ├── test_hybrid.py               # legacy API: closed form vs MC, identities
│   ├── test_hybrid_package.py       # layered API + abstraction invariants
│   ├── test_rates.py                # EQ/IR: closed form vs MC, unit guards
│   └── test_cms.py                  # convexity: derivatives, scaling, sign
├── requirements.txt
└── LICENSE
```

The Streamlit app and the analysis scripts continue to work — they're a different surface on the same pricer.

`src/hybrid_pricer.py` keeps the original flat `HybridInputs` API and delegates to
`src/hybrid/`, so nothing downstream had to change. New code should prefer
`from src.hybrid import ...`.

### Measure

Pricing is under the **T-forward measure** — numeraire P(0,T), so
`V0 = P(0,T) · E^T[payoff]`. A constant short rate is contradictory once the
payoff is *conditioned on a rate*, so the discount factor and equity forward
are observables rather than things derived from a flat `r`:

```python
eq = EquityLeg.from_market(F=7280.0, K=7000.0, P0T=0.978, sig_S=0.16, T=0.5)
```

`EquityLeg.from_spot(...)` keeps the original flat-rate convention and is what
the legacy `HybridInputs` API uses, so existing prices are unchanged. Note that
under Q^T, `sig_S` is the vol of the *forward*, which equals the spot vol only
when rates are deterministic.

### Adding an asset class

The conditioning leg enters the closed form through just two scalars — a
standardised threshold `h`, and the shift `rho * sig_S * sqrt(T)` that `h`
picks up when the numeraire changes from cash to the equity. That shift is
*dynamics-independent*, so a new asset class means implementing `h` and
nothing else:

```python
# FX, lognormal
h = (log(X0/B) + (mu_X - 0.5*sig_X**2)*T) / (sig_X*sqrt(T))

# CMS rate, normal/Bachelier — no log, no -0.5*sig^2*T
h = (R_adj - B) / (sig_R*sqrt(T))
```

So an SPX call contingent on `CMS10 > 4%` is the same call with a different leg:

```python
from src.hybrid import EquityLeg, RateCondition, conditional_european

eq  = EquityLeg.from_market(F=7647.0, K=7000.0, P0T=0.9139, sig_S=0.16, T=2.0)
cms = RateCondition.from_forward_swap(
    R_0=0.0410, B=0.04, sig_R=0.0080,   # 80bp/yr normal vol
    T=2.0, tenor=10, freq=2,            # CMS10, semiannual
)

conditional_european(eq, cms, T=2.0, rho=-0.30)["price"]   # 387.44
```

Rates default to **normal/Bachelier** (post-2015 market convention, handles
negative rates); `ShiftedLognormalRateCondition` covers shifted-lognormal
quoting. Units are guarded — passing `4` for 4%, or `80` for 80bp, raises
rather than returning a plausible wrong number.

### CMS convexity

A swap rate is a martingale under the annuity measure, not under the
$T$-forward measure the hybrid prices in, so $\mathbb{E}^T[R_T] \neq R_0$.
`from_forward_swap` applies the adjustment for you; it is the path that can't
silently omit it. Passing `R_adj` directly is still supported, but it trusts
you to have adjusted the rate already.

The approximation is second-order in the annuity's curvature,

$$\mathbb{E}^T[R_T] \approx R_0 - \tfrac{1}{2}\,\sigma^2 T\,\frac{G''(R_0)}{G'(R_0)}, \qquad G(y)=\sum_{i=1}^{n}\frac{\delta}{(1+\delta y)^i}$$

which for CMS10 fixing in 2y at 80bp vol is **+4.45bp** — worth 3.8% of this
structure's premium. It ignores the **smile** (one vol, no skew), so expect a
few tenths of a bp to a few bp of difference against full static replication,
growing with maturity, tenor and skew steepness. Only the mean is adjusted;
the leg keeps its vol and its shape. There is no payment-delay term because
these products condition on the CMS *fixing* and settle at the same date.

---

## Running

```bash
pip install -r requirements.txt

# The two notebooks (recommended starting point)
jupyter notebook notebooks/conditional_european.ipynb
jupyter notebook notebooks/digital.ipynb

# Standalone scripts (regenerate figures/)
python scripts/run_all.py
python tests/test_hybrid.py

# Streamlit app
./run_app.sh
```

For the Streamlit Cloud deploy, see `app/MOBILE_DEPLOY.md`.

---

## Caveats

The bivariate BS gives clean Greeks and a good benchmark, but a real desk would layer on:

1. **SPX skew.** Calls struck OTM should be priced off the SPX surface, not flat ATM vol.
2. **FX skew.** Barrier digitals are extremely sensitive to wing vol; in production, replicate the digital as a tight call spread on the FX surface.
3. **Correlation skew.** Empirically, SPX/USDJPY correlation rises in risk-off. Constant $\rho$ likely understates the price.
4. **Stochastic vol.** Long-dated FX barriers especially benefit from a 2-factor SLV.

Treat this code as a teaching benchmark and risk-attribution tool, not a production mark.

---

## References

- Heynen & Kat (1994), "Crossing Barriers" — dual-digital and outside-barrier formulas
- Haug, *Complete Guide to Option Pricing Formulas* — multi-asset closed forms
- Lipton, *Mathematical Methods for Foreign Exchange* — quanto and FX hybrid mechanics

## License

MIT — see [LICENSE](LICENSE).
