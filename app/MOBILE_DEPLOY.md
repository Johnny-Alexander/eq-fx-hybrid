# Open the App on Your iPhone

Two options, depending on whether you want it accessible just on home/office WiFi or anywhere on cellular.

---

## Option 1: Same WiFi as your Mac (simplest)

The app runs on your Mac. You connect from your iPhone over the local WiFi network.

**Pros:** Free, no public deploy, takes 30 seconds.
**Cons:** Only works when iPhone and Mac are on the same WiFi. Mac must be running.

### Steps

```bash
cd hybrid-pricer
pip install -r requirements.txt streamlit
./run_app.sh
```

The script prints something like:

```
============================================================
  Open this URL on your iPhone (must be on same WiFi):

    http://192.168.1.42:8501

  On your Mac:  http://localhost:8501
============================================================
```

On your iPhone, open Safari and type the LAN URL. Add to Home Screen for an app-like icon (Share button → Add to Home Screen).

### Troubleshooting

- **"Site can't be reached" on iPhone:** Check your Mac's firewall. System Settings → Network → Firewall → Options → make sure Python is allowed. Or temporarily disable the firewall to test.
- **Mac IP not detected:** Find it manually under System Settings → Wi-Fi → Details → IP Address. Then on your Mac run: `streamlit run app/app.py --server.address 0.0.0.0` and use `http://<that-ip>:8501` from the phone.
- **Sleep mode:** If the Mac sleeps, the app stops. Disable sleep, or use Option 2.

---

## Option 2: Streamlit Community Cloud (works anywhere)

Deploy publicly so the app has a URL you can hit from anywhere — cellular, hotel WiFi, anywhere. Free, no card needed.

**Pros:** Always available, works on cellular, you can share the URL.
**Cons:** Public by default (anyone with the URL can use it). Requires GitHub.

### Steps

1. **Push the repo to GitHub** (Claude Code can handle this — see `HANDOFF.md`).

2. **Go to https://share.streamlit.io** and sign in with GitHub.

3. **Click "Create app"** and point it at:
   - Repository: `<your-github-username>/hybrid-pricer`
   - Branch: `main`
   - Main file path: `app/app.py`

4. **Click Deploy.** First build takes ~2 minutes. You'll get a URL like:

   ```
   https://hybrid-pricer.streamlit.app
   ```

5. **Open on iPhone** and add to Home Screen.

### Privacy notes

- Streamlit Cloud gives the app a public URL. Anyone with the link can use it.
- For real desk use with sensitive params, **make the GitHub repo private** — Streamlit Cloud supports private repos and adds a Google sign-in gate. From the app's settings page on share.streamlit.io: Settings → Sharing → "Private (only viewers I invite)".
- The pricer code itself contains no proprietary information, just the closed-form bivariate BS formula.

---

## Tip: Add to Home Screen on iPhone

For a native-app feel:

1. Open the URL in Safari (must be Safari, not Chrome)
2. Tap the Share button (square with arrow)
3. Scroll down → "Add to Home Screen"
4. Name it something short like "EQ/FX Pricer"
5. Tap Add

You now have an icon that opens the app full-screen, no browser chrome.

---

## Mobile UX notes

- **Sidebar starts collapsed** on phone. Tap the `>` arrow at top-left to open inputs.
- **Plots stack vertically** (one above the other) instead of side-by-side, so they're readable without zooming.
- **Sliders are touch-friendly** but for precise values, tap the number directly to bring up the keyboard.
- **Re-render is ~1-2 seconds per slider change** on a typical phone. If it feels sluggish, the easy speed-up is reducing the grid resolution in `app.py` (search for `n=40` and lower it).
