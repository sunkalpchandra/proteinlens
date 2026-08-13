#!/usr/bin/env bash
# macOS arm64: faiss-cpu and torch each vendor their own libomp.dylib. Loading
# both in one process aborts with "OMP: Error #15" or segfaults. This replaces
# faiss's copy with a symlink to torch's, so a single OpenMP runtime is loaded.
#
# Usage:  bash scripts/fix_macos_libomp.sh [path-to-venv]   (default: .venv)
set -euo pipefail

VENV="${1:-.venv}"
SITE=$("$VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")
FAISS_OMP="$SITE/faiss/.dylibs/libomp.dylib"
TORCH_OMP="$SITE/torch/lib/libomp.dylib"

if [[ ! -f "$TORCH_OMP" ]]; then
    echo "No torch libomp at $TORCH_OMP — nothing to do."; exit 0
fi
if [[ -L "$FAISS_OMP" ]]; then
    echo "Already symlinked."; exit 0
fi
if [[ ! -f "$FAISS_OMP" ]]; then
    echo "No faiss libomp at $FAISS_OMP — nothing to do."; exit 0
fi

mv "$FAISS_OMP" "$FAISS_OMP.bak"
ln -s "$TORCH_OMP" "$FAISS_OMP"
echo "Linked faiss libomp → torch libomp."
"$VENV/bin/python" -c "import torch, faiss; print('coexistence OK')"
