# EQ/FX Hybrid Option Pricer

Closed-form bivariate Black–Scholes pricing and risk analysis for two cross-asset structures linking SPX and USDJPY:

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
│   ├── hybrid_pricer.py             # generalized closed-form + MC + Greeks
│   └── trade_config.py              # example trade for scripts/tests
├── scripts/                         # 7 standalone analysis scripts -> figures/
├── figures/                         # pre-generated PNGs from scripts/
├── app/
│   ├── app.py                       # Streamlit pricer (mobile-friendly)
│   └── MOBILE_DEPLOY.md
├── docs/                            # GitHub Pages (stlite/Pyodide) deploy
├── tests/test_hybrid.py             # closed-form vs MC, identity, monotonicity
├── requirements.txt
└── LICENSE
```

The Streamlit app and the analysis scripts continue to work — they're a different surface on the same pricer.

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
