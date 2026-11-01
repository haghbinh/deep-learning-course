#!/usr/bin/env bash
# Build the GitHub Pages site into site/.
# Copies the notes into docs/ (the Obsidian folders stay untouched), then runs MkDocs.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf docs site
mkdir -p docs/assets
cp -r "Chapter 10" "Chapter 11" "Chapter 12" img docs/
cp README.md docs/index.md
cp web/extra.css web/mathjax.js docs/assets/
mkdocs build --strict
