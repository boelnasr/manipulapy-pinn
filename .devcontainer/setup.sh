#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Codespaces post-create setup.
#
# Installs a CPU-only PyTorch before the package itself. This matters: nothing
# in this repo uses CUDA (the networks are three 128-wide float64 layers, and
# a full training cycle takes well under two minutes on CPU), but PyPI's
# default linux torch wheel drags in ~2.5 GB of nvidia-* packages that will
# never be touched. On the default 2-core Codespace that is a third of the
# disk and several minutes of build time spent on nothing.
set -euo pipefail

echo "==> Installing CPU-only PyTorch"
if ! pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; then
    echo "!!  download.pytorch.org unreachable (restricted network?); falling back to PyPI."
    echo "!!  This pulls the CUDA build — larger, but functionally identical here."
    pip install --no-cache-dir torch
fi

echo "==> Installing manipulapy-pinn with dev extras"
pip install --no-cache-dir -e ".[dev]"

echo "==> Verifying"
python - <<'PY'
import importlib.metadata as md
import torch
print(f"  torch       {torch.__version__} (cuda={torch.cuda.is_available()})")
print(f"  ManipulaPy  {md.version('ManipulaPy')}")
from manipulapy_pinn import load_robot
r = load_robot("panda")
print(f"  panda loads: {r.n_joints} DOF")
PY

cat <<'EOF'

Ready. A full training cycle:

    python scripts/train_forward_dynamics.py --robot panda    # ~10s
    python scripts/train_inverse_kinematics.py --robot panda  # ~45s
    python scripts/train_trajectory.py --robot panda          # ~30s
    python scripts/benchmark.py --robot panda                 # vs classical solvers

Checkpoints and figures land in runs/ (gitignored). Run the tests with:

    python -m pytest tests/ -q

EOF
