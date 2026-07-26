# Deploy to GitHub Pages

The `docs/` folder is set up to deploy as a static site that runs the pricer **entirely in the browser** via [stlite](https://github.com/whitphx/stlite) (Streamlit compiled to WebAssembly via Pyodide).

## How it works

- `docs/index.html` — the landing page; loads stlite from a CDN
- `docs/app.py` — the Streamlit app, fetched and run by stlite client-side
- `docs/hybrid_pricer.py` — the pricer module, also fetched

When a user visits the GitHub Pages URL, their browser:
1. Loads `index.html`
2. Loads stlite (~few MB)
3. Loads Pyodide (CPython for WASM, ~10MB)
4. Loads numpy + scipy + matplotlib (~30-40MB)
5. Loads `app.py` and `hybrid_pricer.py`
6. Runs the Streamlit app in a sandboxed iframe

**First load: 30-60 seconds.** Cached after that.

## Enabling GitHub Pages

After pushing the repo:

1. Go to your repo on github.com
2. **Settings → Pages**
3. Under "Source": select **Deploy from a branch**
4. Branch: `main`, folder: `/docs`
5. Click Save

Wait ~1 minute, then your URL is:

```
https://<your-username>.github.io/hybrid-pricer/
```

That's it. No build step, no CI, no deploy keys. Pushing to `main` auto-redeploys.

## Trade-offs vs Streamlit Cloud

| | GitHub Pages (stlite) | Streamlit Cloud |
|---|---|---|
| First load | 30-60 sec | Instant |
| Re-render speed | 2-5 sec (browser) | 0.5-2 sec (server) |
| Cost | Free | Free tier |
| Privacy | Public URL | Public or auth-gated |
| Compute | User's device | Streamlit's server |
| Battery on phone | Higher (browser-side compute) | Lower |
| Offline | Works once cached | No |
| Deploy effort | Settings click | Settings click |

**Pick GitHub Pages if:** you want zero infrastructure ever, you're fine with 30-60s first load, and you like the "no backend" purity.

**Pick Streamlit Cloud if:** you want fast iteration on the desk, expect to use it from your phone often, or share with colleagues who'll get frustrated by the load wait.

## Caveats

- **Pyodide doesn't have everything.** numpy, scipy, and matplotlib all work. But scipy.stats.multivariate_normal is heavy — it adds notable load time. You could speed things up by replacing the `bivariate_normal_cdf` call with a custom implementation (Genz's algorithm in pure Python is ~20 lines).
- **Caching is per-browser.** A new device or a hard refresh = full 30-60s reload again.
- **Mobile Safari memory limits.** On iPhones with <4GB RAM, you might hit memory pressure with very large grids. The grid resolution in `docs/app.py` is already reduced (n=30 instead of n=40) to compensate.
- **Updates to `src/hybrid_pricer.py` don't auto-propagate to `docs/hybrid_pricer.py`.** They're separate files — you'll need to copy the changes manually, or add a pre-commit hook / CI step that syncs them. There's a sync script you can run from the repo root: `cp src/hybrid_pricer.py docs/hybrid_pricer.py`.

## Hybrid approach

Nothing stops you from using **both** — Streamlit Cloud at `hybrid-pricer.streamlit.app` for desk use, and a GitHub Pages mirror at `username.github.io/hybrid-pricer` as a portfolio link. They use the same code.
