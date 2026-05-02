#!/usr/bin/env bash
# Sync the pricer module into docs/ so GitHub Pages picks up the latest version.
# Run this whenever src/hybrid_pricer.py changes.
set -e
cp src/hybrid_pricer.py docs/hybrid_pricer.py
echo "Synced src/hybrid_pricer.py -> docs/hybrid_pricer.py"
