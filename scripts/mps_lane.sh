#!/usr/bin/env bash
# The MPS lane: Apple-silicon device placement and the float32 transform paths.
#
# Every `device.type == "mps"` branch in torchfits/transforms (base, helpers,
# mask, rgb) is reachable only on an Apple-silicon Mac, and no CANFAR image has
# an MPS device. This lane runs those paths on real hardware and compares the
# result with the same pipeline on the CPU, where torchfits uses float64. It is
# the companion to scripts/launch_canfar_all_lanes.sh, not a replacement for it.
#
# Usage:
#   bash scripts/mps_lane.sh             # parity + MPS-touching tests + DataLoader bench
#   bash scripts/mps_lane.sh --quick     # parity + tests, no benchmark
#
# Env:
#   TORCHFITS_PY            interpreter holding the torchfits wheel
#                           (default: the pixi environment's python)
#   TORCHFITS_MPS_RUN_ID    run id (default: mps_<UTC timestamp>)
#   TORCHFITS_MPS_TOL       mps-vs-cpu relative tolerance [2e-3]
#   TORCHFITS_MPS_OUT       results directory
#                           [benchmarks_results/mps_<run id>]
#
# Prints one "[ OK ]" line per stage, or "[FAIL] ..." with the tail of the log.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QUICK=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    -h | --help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

: "${TORCHFITS_MPS_RUN_ID:=mps_$(date -u +%Y%m%d_%H%M%S)}"
: "${TORCHFITS_MPS_TOL:=2e-3}"
RUN_DIR="${TORCHFITS_MPS_OUT:-${ROOT_DIR}/benchmarks_results/${TORCHFITS_MPS_RUN_ID}}"
mkdir -p "$RUN_DIR"

# The lane must never run against a CUDA build on a Linux box that happens to
# have this script checked out: a silent fallback to CPU would report "green"
# for a device path it never touched.
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[FAIL] mps: this lane needs macOS (got $(uname -s))"
  exit 1
fi

PY="${TORCHFITS_PY:-}"
if [[ -z "$PY" ]]; then
  PY="$(pixi run python -c 'import sys; print(sys.executable)' | tail -1)"
fi
echo "=== torchfits MPS lane (${TORCHFITS_MPS_RUN_ID}) ==="
echo "host=$(hostname) arch=$(uname -m) python=${PY}"
echo "results=${RUN_DIR} tolerance=${TORCHFITS_MPS_TOL}"

if ! "$PY" -c '
import sys, torch
ok = torch.backends.mps.is_available() and torch.backends.mps.is_built()
print("torch", torch.__version__, "mps_available", ok)
sys.exit(0 if ok else 1)
'; then
  echo "[FAIL] mps: torch reports no usable MPS device"
  exit 1
fi
"$PY" -c 'import torchfits; print("torchfits", torchfits.__version__)'

STATUS=0

# --- stage 1: device parity ---------------------------------------------------
# One pipeline, run twice: once where torchfits uses float64 (CPU) and once
# where it must use float32 (MPS). A branch that "works" only because it never
# ran is the failure mode this stage exists to catch.
cat > "${RUN_DIR}/parity.py" <<'PY'
"""MPS vs CPU parity for read() and the transforms with MPS dtype branches."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

import torchfits
from torchfits.transforms import (
    ArcsinhStretch,
    LogStretch,
    SigmaClip,
    ZScaleNormalize,
    estimate_background,
    mask_from_nan,
)

TOL = float(os.environ.get("TORCHFITS_MPS_TOL", "2e-3"))
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def close(a: torch.Tensor, b: torch.Tensor, *, tol: float = TOL) -> tuple[bool, str]:
    a64 = a.detach().to("cpu").double()
    b64 = b.detach().to("cpu").double()
    if a64.shape != b64.shape:
        return False, f"shape {tuple(a64.shape)} vs {tuple(b64.shape)}"
    if a64.numel() == 0:
        return True, "empty"
    # Compare only where both sides are finite; the planted NaNs are checked
    # separately, and a NaN on one side alone must still fail the comparison.
    both = torch.isfinite(a64) & torch.isfinite(b64)
    nan_ok = torch.equal(torch.isnan(a64), torch.isnan(b64))
    if both.any():
        a_f, b_f = a64[both], b64[both]
        rel = ((a_f - b_f).abs() / b_f.abs().clamp_min(1e-12)).max().item()
    else:
        rel = 0.0
    ok = bool(rel <= tol and nan_ok)
    detail = f"max rel {rel:.3e} (tol {tol:g})"
    if not nan_ok:
        detail += ", NaN placement differs"
    return ok, detail


rng = np.random.default_rng(7)
plane = rng.normal(100.0, 5.0, (96, 128)).astype(np.float32)
plane[5:9, 7:11] = np.nan
tmp = Path(tempfile.mkdtemp(prefix="tf-mps-"))
path = tmp / "plane.fits"
fits.PrimaryHDU(data=plane).writeto(path)

cpu = torchfits.read(path, device="cpu")
mps = torchfits.read(path, device="mps")
check("read lands on mps", mps.device.type == "mps", str(mps.device))
check("read shape preserved", tuple(mps.shape) == tuple(cpu.shape),
      f"{tuple(mps.shape)}")
ok, detail = close(mps, cpu)
check("read values match cpu", ok, detail)

finite = torch.nan_to_num(cpu, nan=0.0)
x_cpu, x_mps = finite.to("cpu"), finite.to("mps")

for name, transform in (
    ("ArcsinhStretch", ArcsinhStretch()),
    ("LogStretch", LogStretch()),
    ("ZScaleNormalize", ZScaleNormalize()),
    ("SigmaClip", SigmaClip()),
):
    out_cpu = transform(x_cpu.clone())
    out_mps = transform(x_mps.clone())
    out_cpu = out_cpu[0] if isinstance(out_cpu, tuple) else out_cpu
    out_mps = out_mps[0] if isinstance(out_mps, tuple) else out_mps
    ok, detail = close(out_mps, out_cpu)
    check(f"{name} mps == cpu", ok, detail)
    if name in {"ArcsinhStretch", "LogStretch"}:
        check(f"{name} keeps mps placement", out_mps.device.type == "mps",
              str(out_mps.device))

mask_cpu = mask_from_nan(cpu.float())
mask_mps = mask_from_nan(mps.float())
check("mask_from_nan mps == cpu", torch.equal(mask_cpu, mask_mps.to("cpu")),
      f"{int(mask_mps.sum())} flagged")
check("mask_from_nan stays on mps", mask_mps.device.type == "mps",
      str(mask_mps.device))

bg_cpu = estimate_background(x_cpu)
bg_mps = estimate_background(x_mps)
bg_cpu_t = bg_cpu if isinstance(bg_cpu, torch.Tensor) else torch.as_tensor(bg_cpu)
bg_mps_t = bg_mps if isinstance(bg_mps, torch.Tensor) else torch.as_tensor(bg_mps)
ok, detail = close(bg_mps_t, bg_cpu_t, tol=5e-3)
check("estimate_background mps == cpu", ok, detail)

# The transforms are documented to use float32 on MPS and float64 elsewhere;
# an accidental widening here is what makes a later comparison meaningless.
stretched = ArcsinhStretch()(x_mps.clone())
stretched = stretched[0] if isinstance(stretched, tuple) else stretched
check("mps transform dtype is float32", stretched.dtype == torch.float32,
      str(stretched.dtype))

print(f"parity: {len(FAILURES)} failure(s)")
sys.exit(1 if FAILURES else 0)
PY

echo "--- stage: device parity"
if ! TORCHFITS_MPS_TOL="${TORCHFITS_MPS_TOL}" "$PY" "${RUN_DIR}/parity.py" \
    > "${RUN_DIR}/parity.log" 2>&1; then
  STATUS=1
  tail -30 "${RUN_DIR}/parity.log"
  echo "[FAIL] mps: device parity"
else
  grep -E '^(PASS|FAIL)' "${RUN_DIR}/parity.log" | sed -n '1,40p'
  echo "[ OK ] mps: device parity"
fi

# --- stage 2: the tests that carry MPS branches -------------------------------
# Those files are torchfits' own, so this lane needs a torchfits checkout next
# to the harness (or TORCHFITS_SOURCE). They are run inside that checkout, with
# its own environment: torchfits' conftest puts the checkout's `src/` on
# sys.path, and the compiled `_C` extension only resolves from an interpreter
# that has the built package, which the harness env does not.
TF_DIR="${TORCHFITS_SOURCE:-}"
if [[ -z "$TF_DIR" && -f "${ROOT_DIR}/../torchfits/pyproject.toml" ]]; then
  TF_DIR="${ROOT_DIR}/../torchfits"
fi
echo "--- stage: pytest (mps-branch files)"
if [[ -z "$TF_DIR" || ! -d "${TF_DIR}/tests" ]]; then
  STATUS=1
  echo "[FAIL] mps: no torchfits checkout (set TORCHFITS_SOURCE to one)"
else
  echo "torchfits source: ${TF_DIR}"
  if ! ( cd "$TF_DIR" && pixi run python -m pytest \
      tests/test_transforms.py \
      tests/test_maskrgb_stretch.py \
      tests/test_io.py \
      tests/test_core_library.py \
      -q -p no:cacheprovider ) > "${RUN_DIR}/pytest.log" 2>&1; then
    STATUS=1
    tail -30 "${RUN_DIR}/pytest.log"
    echo "[FAIL] mps: pytest"
  else
    tail -2 "${RUN_DIR}/pytest.log"
    echo "[ OK ] mps: pytest"
  fi
fi

# --- stage 3: DataLoader on the device ----------------------------------------
if [[ "$QUICK" -eq 0 ]]; then
  echo "--- stage: bench_ml_loader --device mps (bounded)"
  if ! "$PY" benchmarks/bench_ml_loader.py \
      --device mps --n-files 8 --shape 512,512 --batch-size 2 \
      --num-workers 2 --epochs 1 --repeats 1 \
      --run-id "${TORCHFITS_MPS_RUN_ID}" \
      --output "${RUN_DIR}/ml_results.csv" \
      > "${RUN_DIR}/bench_ml.log" 2>&1; then
    STATUS=1
    tail -20 "${RUN_DIR}/bench_ml.log"
    echo "[FAIL] mps: bench_ml_loader"
  elif ! grep -q "device: mps" "${RUN_DIR}/bench_ml.log"; then
    # The benchmark falls back to CPU silently when MPS is missing, and a green
    # row from the CPU is exactly the false positive this lane must not report.
    STATUS=1
    grep -E "device:" "${RUN_DIR}/bench_ml.log" || true
    echo "[FAIL] mps: bench_ml_loader fell back off mps"
  else
    grep -E "device:" "${RUN_DIR}/bench_ml.log"
    echo "[ OK ] mps: bench_ml_loader"
  fi
fi

verdict=PASS
[[ "$STATUS" -ne 0 ]] && verdict=FAIL
{
  echo "run_id=${TORCHFITS_MPS_RUN_ID}"
  echo "host=$(hostname) arch=$(uname -m)"
  echo "torch=$("$PY" -c 'import torch; print(torch.__version__)')"
  echo "torchfits=$("$PY" -c 'import torchfits; print(torchfits.__version__)')"
  echo "tolerance=${TORCHFITS_MPS_TOL}"
  echo "RESULT: ${verdict}"
} > "${RUN_DIR}/summary.txt"

echo "=== summary ==="
cat "${RUN_DIR}/summary.txt"
echo "RESULT: ${verdict}"
exit "$STATUS"
