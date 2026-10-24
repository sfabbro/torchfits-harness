#!/usr/bin/env bash
# Run the whole torchfits-harness suite inside an astroai/base CANFAR session.
#
# Companion to canfar_gpu_bench_incontainer.sh: that one produces benchmark
# scorecards, this one proves the suite itself is green against a real wheel.
# The subject is an installed torchfits wheel - this repository never builds
# torchfits and never vendors CFITSIO.
#
#   canfar create headless images.canfar.net/astroai/base:latest --name \
#     torchfits-harness-verify --cpu 8 --memory 32 \
#     -- bash -c "<tab-encoded: git clone ...; cd ...; bash scripts/canfar_verify_incontainer.sh>"
#
# Env:
#   TORCHFITS_HARNESS_RUN_ID   run id (default: verify_<UTC timestamp>)
#   TORCHFITS_WHEEL            pip spec or path for the subject
#   TORCHFITS_VERSION          exact release to install (wins over the default
#                              floor; use this from `canfar --env`, which
#                              drops values containing `>` or `==`)

#   TORCHFITS_HARNESS_PYTEST_ARGS  extra pytest args (default: -q -m "not performance")
set -euo pipefail

: "${TORCHFITS_HARNESS_RUN_ID:=verify_$(date -u +%Y%m%d_%H%M%S)}"
SCRATCH="${TMP_SCRATCH_DIR:-/scratch}"
RUN_DIR="${SCRATCH}/torchfits-harness/${TORCHFITS_HARNESS_RUN_ID}"
mkdir -p "$RUN_DIR"

# notebook/base images ship PIXI_HOME=/usr/local/share/pixi, which the job user
# cannot write, so only an explicit TORCHFITS_PIXI_HOME may override scratch.
export PIXI_HOME="${TORCHFITS_PIXI_HOME:-${SCRATCH}/torchfits-pixi-home}"
export PIXI_CACHE_DIR="${TORCHFITS_PIXI_CACHE:-${SCRATCH}/torchfits-pixi-cache}"
mkdir -p "$PIXI_HOME" "$PIXI_CACHE_DIR"

# Keep user-site off the shared /arc home; stale ~/.local packages break
# setuptools uninstalls across concurrent sessions.
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${SCRATCH}/pip-cache"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The image ships some pixi, but pixi.toml pins one. Fetch the pinned release
# into scratch instead of self-updating: the image's pixi lives in a read-only
# prefix, and `requires-pixi` otherwise refuses to run at all.
. "${REPO_ROOT}/scripts/canfar_pinned_pixi.sh"

# No process substitution: the platform already captures stdout, and a `tee`
# child is killed as the session tears down, which swallowed the last lines of
# the run - including the verdict. Stage results are also written to
# ${RUN_DIR}/summary.txt for anyone reading the scratch tree before it ends.

NPROC="$(nproc 2>/dev/null || echo 4)"
echo "=== torchfits-harness headless verify ==="
echo "run_id=${TORCHFITS_HARNESS_RUN_ID} run_dir=${RUN_DIR}"
echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown) nproc=${NPROC}"
echo "PIXI_HOME=${PIXI_HOME} PIXI_CACHE_DIR=${PIXI_CACHE_DIR}"
python3 --version
command -v pixi >/dev/null || { echo "FAIL: pixi not on PATH in this image"; exit 1; }
use_pinned_pixi

pixi install
pixi run dev

# From here on, drive the environment's interpreter directly. Every `pixi run`
# re-syncs the environment from pixi.lock, and the lock carries its own
# torchfits: invoking pixi again after installing the subject silently put the
# locked wheel back, so a job that pinned 1.2.2 verified 1.2.1.
ENV_PY="$(pixi run python -c 'import sys; print(sys.executable)' | tail -1)"
echo "env python: ${ENV_PY}"

# The subject: a released wheel, never this checkout. TORCHFITS_WHEEL takes any
# pip spec or path; TORCHFITS_VERSION pins an exact release by number, because
# Skaha drops env values that carry `>` or `==`.
if [[ -n "${TORCHFITS_WHEEL:-}" ]]; then
  WHEEL_SPEC="${TORCHFITS_WHEEL}"
elif [[ -n "${TORCHFITS_VERSION:-}" ]]; then
  WHEEL_SPEC="torchfits==${TORCHFITS_VERSION}"
else
  WHEEL_SPEC="torchfits>=1.3.0"
fi
echo "subject spec: ${WHEEL_SPEC}"
"$ENV_PY" -m pip install --no-cache-dir --no-deps "${WHEEL_SPEC}"
"$ENV_PY" -c "import torchfits_harness; print('subject:', torchfits_harness.subject())"
"$ENV_PY" -m pip show torchfits 2>/dev/null | sed -n '1,2p'

if [[ -n "${TORCHFITS_VERSION:-}" ]]; then
  installed_version="$("$ENV_PY" -c 'import torchfits; print(torchfits.__version__)' | tail -1)"
  if [[ "${installed_version}" != "${TORCHFITS_VERSION}" ]]; then
    echo "FAIL: subject is torchfits ${installed_version}, expected ${TORCHFITS_VERSION}"
    exit 1
  fi
  echo "subject pin confirmed: ${installed_version}"
fi

rc_gate=0
rc_suite=0

echo "=== gate: upstream parity smokes ==="
# Same three files as the `test` task in pixi.toml; run directly so the subject
# installed above survives.
"$ENV_PY" -m pytest tests/test_fitsio_upstream_smoke.py \
  tests/test_astropy_upstream_smoke.py tests/test_layout.py -q || rc_gate=$?

echo "=== suite: pytest tests/ ==="
# shellcheck disable=SC2086
"$ENV_PY" -m pytest tests/ ${TORCHFITS_HARNESS_PYTEST_ARGS:--q -m "not performance"} \
  --junitxml="${RUN_DIR}/junit.xml" || rc_suite=$?

echo "=== bounded bench: bench_cpp_backend.py ==="
rc_bench=0
"$ENV_PY" benchmarks/bench_cpp_backend.py || rc_bench=$?

verdict=PASS
if [[ "${rc_gate}" -ne 0 || "${rc_suite}" -ne 0 || "${rc_bench}" -ne 0 ]]; then
  verdict=FAIL
fi
summary="gate_rc=${rc_gate} suite_rc=${rc_suite} bench_rc=${rc_bench} run_dir=${RUN_DIR}"
echo "=== summary ==="
echo "${summary}"
echo "RESULT: ${verdict}"
echo "${summary}" > "${RUN_DIR}/summary.txt"
echo "RESULT: ${verdict}" >> "${RUN_DIR}/summary.txt"
# One failing stage must not hide the other's result.
if [[ "${rc_gate}" -ne 0 || "${rc_suite}" -ne 0 || "${rc_bench}" -ne 0 ]]; then
  exit 1
fi
