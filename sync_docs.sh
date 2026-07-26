#!/usr/bin/env bash
# Sync the pricer package into docs/ so GitHub Pages picks up the latest
# version. Run this whenever anything under src/ changes.
#
# The pricer used to be a single module, copied in flat. It is now a package,
# so the whole tree is mirrored under docs/src/ and stlite mounts it at the
# same import paths the repo uses -- that way docs/app.py and app/app.py can
# share their imports verbatim.
set -e

cd "$(dirname "$0")"

rm -rf docs/src
mkdir -p docs/src/hybrid

cp src/__init__.py docs/src/__init__.py
cp src/hybrid_pricer.py docs/src/hybrid_pricer.py
cp src/hybrid/*.py docs/src/hybrid/

echo "Synced src/ -> docs/src/:"
find docs/src -name '*.py' | sort | sed 's/^/  /'

# stlite mounts an explicit file list, so a new module silently 404s unless
# index.html is updated too. Fail loudly rather than shipping a broken page.
missing=0
while read -r f; do
  rel="${f#docs/}"
  if ! grep -q "\"$rel\"" docs/index.html; then
    echo "WARNING: $rel is not listed in docs/index.html" >&2
    missing=1
  fi
done < <(find docs/src -name '*.py' | sort)

if [ "$missing" -ne 0 ]; then
  echo "docs/index.html is out of step with src/ -- update its module list." >&2
  exit 1
fi
echo "docs/index.html module list is in step."
