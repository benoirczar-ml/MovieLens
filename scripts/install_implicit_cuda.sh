#!/usr/bin/env bash
set -euo pipefail

CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Install CUDA toolkit first." >&2
  exit 1
fi

echo "[1/3] Rebuilding implicit with CUDA from source..."
CUDA_HOME="$CUDA_ROOT" \
CUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" \
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
python -m pip install --force-reinstall --no-binary implicit --no-cache-dir implicit

echo "[2/3] Verifying implicit GPU extension..."
python - <<'PY'
import implicit.gpu as g
print('HAS_CUDA=', g.HAS_CUDA)
if not g.HAS_CUDA:
    raise SystemExit('implicit GPU extension is not available')
PY

echo "[3/3] Running quick ALS GPU smoke test..."
python - <<'PY'
import numpy as np
from scipy import sparse
from implicit.als import AlternatingLeastSquares

rng=np.random.default_rng(42)
rows=rng.integers(0,128,size=4000)
cols=rng.integers(0,256,size=4000)
vals=np.ones(4000,dtype=np.float32)
mat=sparse.coo_matrix((vals,(rows,cols)),shape=(128,256)).tocsr()

m=AlternatingLeastSquares(factors=32,iterations=2,use_gpu=True,random_state=42)
m.fit((mat*40).astype(np.float32))
print('ALS_GPU_OK')
PY

echo "Done. implicit is CUDA-ready."
