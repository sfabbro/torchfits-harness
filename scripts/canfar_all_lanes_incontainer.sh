#!/usr/bin/env bash
# Every harness lane that can run inside one CANFAR session.
#
# The matrix lanes answer "does each interpreter compile and pass torchfits?".
# This script answers the other half: the corpus parity gates, the upstream
# replays, the contract/surface/tour gates, the benchmark modules that had no
# lane at all, and the real-archive data paths. Lanes are independent: one red
# lane never hides another's result.
#
# Everything durable goes on the scratch volume (${TORCHFITS_DATA_ROOT}), which
# is mounted across sessions: the corpus and the CFHT samples are fetched once
# and reused by every later lane and every later run.
#
#   canfar create headless images.canfar.net/astroai/base:latest --name \
#     torchfits-lanes --cpu 8 --memory 32 \
#     --env TORCHFITS_LANES=all -- bash -c "<tab-encoded clone + script>"
#
# Env:
#   TORCHFITS_LANES        comma/space list, or "all"                     [all]
#   TORCHFITS_DATA_ROOT    persistent data (corpus, samples, checkout)    [/scratch/torchfits-data]
#   TORCHFITS_HARNESS_RUN_ID                                              [lanes_<UTC>]
#   TORCHFITS_WHEEL        pip spec or path for the subject               [torchfits>=1.3.0]
#   TORCHFITS_VERSION      exact release to install (wins over the floor)
#   TORCHFITS_REF          torchfits checkout ref, for examples/oracle    [main]
#   TORCHFITS_CORPUS_TIER  ci | full                                      [ci]
#   TORCHFITS_LANE_SECS    per-lane wall-clock cap                        [1800]
#   TORCHFITS_GPU          set to 1 by the launcher for GPU lanes
set -uo pipefail

: "${TORCHFITS_HARNESS_RUN_ID:=lanes_$(date -u +%Y%m%d_%H%M%S)}"
SCRATCH="${TMP_SCRATCH_DIR:-/scratch}"
DATA_ROOT="${TORCHFITS_DATA_ROOT:-${SCRATCH}/torchfits-data}"
RUN_DIR="${TORCHFITS_RUN_ROOT:-${SCRATCH}/torchfits-harness}/${TORCHFITS_HARNESS_RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR" "$DATA_ROOT"

# The image sets PIXI_HOME=/usr/local/share/pixi, which the job user cannot
# write, so only an explicit TORCHFITS_PIXI_HOME may override scratch. Falling
# back with ${PIXI_HOME:-...} silently kept the read-only image value.
export PIXI_HOME="${TORCHFITS_PIXI_HOME:-${SCRATCH}/torchfits-pixi-home}"
export PIXI_CACHE_DIR="${TORCHFITS_PIXI_CACHE:-${SCRATCH}/torchfits-pixi-cache}"
export PIP_CACHE_DIR="${SCRATCH}/pip-cache"
export TMPDIR="${SCRATCH}/tmp"
export PYTHONNOUSERSITE=1
mkdir -p "$PIXI_HOME" "$PIXI_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

CORPUS_TIER="${TORCHFITS_CORPUS_TIER:-ci}"
export TORCHFITS_CORPUS_DIR="${DATA_ROOT}/corpus"
export TORCHFITS_CORPUS_TIER="$CORPUS_TIER"
LANE_SECS="${TORCHFITS_LANE_SECS:-1800}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== torchfits-harness lanes ==="
echo "run_id=${TORCHFITS_HARNESS_RUN_ID} run_dir=${RUN_DIR}"
echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "data_root=${DATA_ROOT} corpus_tier=${CORPUS_TIER} lane_cap=${LANE_SECS}s"
echo "host=$(hostname) nproc=$(nproc 2>/dev/null || echo '?')"
command -v nvidia-smi >/dev/null && { nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true; }

# --- environment --------------------------------------------------------------
command -v pixi >/dev/null || { echo "FAIL: pixi not on PATH in this image"; exit 1; }
# shellcheck disable=SC1091
. "${REPO_ROOT}/scripts/canfar_pinned_pixi.sh"
use_pinned_pixi || exit 1
python3 --version

# GPU lanes need a CUDA torch before the subject is installed; torchfits is
# built against whatever torch is present, so this cannot be swapped afterwards.
if [[ "${TORCHFITS_GPU:-0}" == "1" ]]; then
  export TORCHFITS_TORCH_INDEX="${TORCHFITS_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
  echo "--- GPU lanes: torch from ${TORCHFITS_TORCH_INDEX}"
fi

echo "--- pixi install"
pixi install > "${LOG_DIR}/pixi-install.log" 2>&1 || {
  tail -20 "${LOG_DIR}/pixi-install.log"
  echo "FAIL: pixi install"
  exit 1
}
pixi run dev > "${LOG_DIR}/pixi-dev.log" 2>&1 || {
  tail -20 "${LOG_DIR}/pixi-dev.log"
  echo "FAIL: pixi run dev"
  exit 1
}

# From here on the environment's interpreter is driven directly: every further
# `pixi run` re-syncs from pixi.lock and would put the locked torchfits back,
# which is how a lane silently tests a different version than it reports.
ENV_PY="$(pixi run python -c 'import sys; print(sys.executable)' | tail -1)"
echo "env python: ${ENV_PY}"
# The wheel's console script lands in the environment's bin, which is not on
# PATH when the interpreter is invoked by absolute path. The docs tour runs
# `torchfits`, so without this it errors on a missing console script.
export PATH="$(dirname "$ENV_PY"):${PATH}"

if [[ -n "${TORCHFITS_TORCH_INDEX:-}" ]]; then
  echo "--- installing torch from ${TORCHFITS_TORCH_INDEX}"
  # --force-reinstall: the environment already carries torch at the same
  # version, so a plain install prints "already satisfied" and leaves the CPU
  # build in place, which then reports cuda False on an H100 session.
  "$ENV_PY" -m pip install -q --no-cache-dir --force-reinstall \
    "torch==${TORCHFITS_TORCH:-2.14.1}" \
    --index-url "${TORCHFITS_TORCH_INDEX}" > "${LOG_DIR}/torch.log" 2>&1 || {
    tail -20 "${LOG_DIR}/torch.log"
    echo "FAIL: torch install"
    exit 1
  }
fi

if [[ -n "${TORCHFITS_WHEEL:-}" ]]; then
  WHEEL_SPEC="$TORCHFITS_WHEEL"
elif [[ -n "${TORCHFITS_VERSION:-}" ]]; then
  WHEEL_SPEC="torchfits==${TORCHFITS_VERSION}"
else
  WHEEL_SPEC="torchfits>=1.3.0"
fi
echo "subject spec: ${WHEEL_SPEC}"
"$ENV_PY" -m pip install -q --no-cache-dir --no-deps "$WHEEL_SPEC" \
  > "${LOG_DIR}/subject.log" 2>&1 || {
  tail -20 "${LOG_DIR}/subject.log"
  echo "FAIL: subject install ${WHEEL_SPEC}"
  exit 1
}
"$ENV_PY" - <<'PY'
import torch, torchfits
print("subject:", torchfits.__version__, "torch", torch.__version__,
      "cuda", torch.cuda.is_available())
PY

# A torchfits checkout is needed by three things that deliberately live outside
# the wheel: the vendored CFITSIO the oracle compiles against, the example
# gallery, and the real-data tests. It is not the subject - the wheel is.
TF_DIR="${TORCHFITS_SOURCE:-${DATA_ROOT}/src/torchfits}"
if [[ ! -f "${TF_DIR}/pyproject.toml" ]]; then
  mkdir -p "$(dirname "$TF_DIR")"
  echo "--- cloning torchfits ${TORCHFITS_REF:-main} into ${TF_DIR}"
  git clone --depth 1 --branch "${TORCHFITS_REF:-main}" \
    https://github.com/astroai/torchfits.git "$TF_DIR" \
    > "${LOG_DIR}/clone-torchfits.log" 2>&1 || {
    tail -10 "${LOG_DIR}/clone-torchfits.log"
    echo "FAIL: clone torchfits"
    exit 1
  }
else
  echo "--- reusing torchfits checkout ${TF_DIR}"
fi
export TORCHFITS_SOURCE="$TF_DIR"

# Keep the data on the scratch volume. Both repositories look for
# benchmarks_data/ under their own root, so point theirs at the shared tree.
# Anything already there is left alone: on a developer machine that path is the
# 15 GB corpus (often itself a symlink), and a lane run must never repoint it.
# A fresh CANFAR clone has nothing at that path, which is the case this covers.
SHARED_BENCH_DATA="${DATA_ROOT}/benchmarks_data"
mkdir -p "$SHARED_BENCH_DATA"
for root in "$REPO_ROOT" "$TF_DIR"; do
  target="${root}/benchmarks_data"
  if [[ -e "$target" || -L "$target" ]]; then
    echo "--- keeping existing ${target}"
  else
    ln -sfn "$SHARED_BENCH_DATA" "$target"
    echo "--- data: ${target} -> ${SHARED_BENCH_DATA}"
  fi
done

# --- lane helpers -------------------------------------------------------------
fail=0
declare -a RESULTS=()

pytest_lane() {
  "$ENV_PY" -m pytest "$@" -q -p no:cacheprovider
}

# A lane that hangs must not take the session's other lanes with it, so each one
# runs in the background under a watchdog. Lanes are pytest runs and benchmark
# modules: anything still alive at the cap is hung, not slow.
run_with_cap() {
  local secs="$1" fn="$2" waited=0 pid
  "$fn" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if (( waited >= secs )); then
      kill -TERM "$pid" 2>/dev/null || true
      sleep 5
      kill -KILL "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      return 124
    fi
    sleep 5
    waited=$(( waited + 5 ))
  done
  wait "$pid"
}

fetch_corpus() {
  "$ENV_PY" scripts/fetch_corpus.py --tier "$CORPUS_TIER" || return 1
  "$ENV_PY" scripts/fetch_corpus.py --tier "$CORPUS_TIER" --verify || return 1
}

fetch_real_data() {
  local sample
  # The calibration frames are not optional for the denoise lane: it needs two
  # dark frames and otherwise writes zero rows while still exiting 0.
  for sample in fetch_cfht_megacam_sample fetch_cfht_megapipe_sample fetch_cfht_calib_frames; do
    if ! bash "${TF_DIR}/scripts/${sample}.sh"; then
      echo "FAIL: ${sample}"
      return 1
    fi
  done
  find "$SHARED_BENCH_DATA" -name '*.fits*' | wc -l
}

# --- lanes --------------------------------------------------------------------
lane_corpus() {
  fetch_corpus || return 1
  pytest_lane tests/test_corpus_manifest.py tests/test_corpus_parity.py \
    tests/test_operation_matrix.py tests/test_write_parity.py \
    tests/test_cfitsio_oracle.py tests/test_astropy_corpus_diff.py
}

lane_replay() {
  pytest_lane tests/test_replay_fitsio.py tests/test_replay_astropy.py \
    tests/test_replay_cfitsio.py tests/test_replay_ccfits.py \
    tests/test_replay_elefits.py tests/test_replay_qfits.py \
    tests/test_replay_sfitsio.py tests/test_upstream_parity_inventory.py
}

lane_contract() {
  pytest_lane tests/test_bench_corpus.py tests/test_cli_suite.py \
    tests/test_public_surface.py tests/test_tour.py
}

lane_surface() {
  "$ENV_PY" scripts/surface_report.py
}

lane_coverage() {
  pytest_lane tests --cov=torchfits --cov-report=term:skip-covered \
    --cov-report="html:${RUN_DIR}/python-coverage"
}

lane_perf() {
  pytest_lane tests -m performance
}

lane_examples() {
  local rc=0 f
  if [[ -f "${TF_DIR}/examples/test_examples.py" ]]; then
    TORCHFITS_EXAMPLE_FAST=1 "$ENV_PY" "${TF_DIR}/examples/test_examples.py" || rc=1
  else
    echo "FAIL: no example gallery in ${TF_DIR}"
    rc=1
  fi
  for f in examples/example_*.py; do
    [[ -f "$f" ]] || continue
    echo "--- example ${f}"
    "$ENV_PY" "$f" || rc=1
  done
  return $rc
}

lane_real_data() {
  fetch_real_data || return 1
  # Those two files are torchfits' own and need its environment: its conftest
  # puts the checkout's src/ first, and the compiled _C extension only resolves
  # from an interpreter where torchfits was built in place. The env persists on
  # the scratch volume, so only the first run pays for the build.
  if ! ( cd "$TF_DIR" && pixi run python -c 'import torchfits._C' ) >/dev/null 2>&1; then
    echo "--- building the torchfits dev environment in ${TF_DIR}"
    ( cd "$TF_DIR" && pixi install && pixi run dev ) || return 1
  fi
  ( cd "$TF_DIR" && pixi run python -m pytest \
      tests/test_real_data_reads.py tests/test_core_library_real_data.py \
      -q -p no:cacheprovider ) || return 1
  pytest_lane tests/test_reads_real_data.py
}

lane_bench_cpu() {
  local rc=0
  "$ENV_PY" -m benchmarks.bench_corpus --tier "$CORPUS_TIER" \
    --output-dir "${RUN_DIR}/bench-corpus" || rc=1
  "$ENV_PY" benchmarks/bench_metadata.py --out "${RUN_DIR}/metadata_results.csv" || rc=1
  "$ENV_PY" benchmarks/bench_arrow_tables.py \
    --output "${RUN_DIR}/table_arrow_results.csv" || rc=1
  "$ENV_PY" scripts/api_overhead.py --json "${RUN_DIR}/api-overhead.json" || rc=1
  return $rc
}

lane_bench_ml() {
  "$ENV_PY" benchmarks/bench_ml_loader.py --n-files 20 --epochs 1 --repeats 1 \
    --run-id "${TORCHFITS_HARNESS_RUN_ID}" \
    --output "${RUN_DIR}/ml_results.csv"
}

lane_bench_megacam() {
  fetch_real_data || return 1
  local data="${SHARED_BENCH_DATA}/cfht_megacam" rc=0
  "$ENV_PY" benchmarks/bench_megacam_cutouts.py --data-dir "$data" \
    --run-id "${TORCHFITS_HARNESS_RUN_ID}" || rc=1
  "$ENV_PY" benchmarks/bench_median_stack.py --data-dir "$data" --max-files 4 \
    --run-id "${TORCHFITS_HARNESS_RUN_ID}" || rc=1
  "$ENV_PY" benchmarks/bench_http_stream.py --data-dir "$data" --max-files 3 \
    --run-id "${TORCHFITS_HARNESS_RUN_ID}" || rc=1
  "$ENV_PY" benchmarks/bench_science_pipeline.py --data-dir "$data" \
    --max-files 3 --cutouts-per-file 2 --threads 4 \
    --run-id "${TORCHFITS_HARNESS_RUN_ID}" || rc=1
  return $rc
}

lane_bench_gpu() {
  if ! command -v nvidia-smi >/dev/null; then
    echo "FAIL: no GPU in this session"
    return 1
  fi
  local rc=0
  "$ENV_PY" benchmarks/bench_gpu_memory.py || rc=1
  fetch_real_data || return 1
  "$ENV_PY" benchmarks/bench_denoise.py \
    --data-dir "${SHARED_BENCH_DATA}/cfht_megacam" --max-files 4 --max-hdus 1 \
    --epochs 1 --run-id "${TORCHFITS_HARNESS_RUN_ID}" || rc=1
  return $rc
}

lane_cfitsio_direct() {
  if [[ ! -f "${TF_DIR}/extern/cfitsio/CMakeLists.txt" ]]; then
    echo "FAIL: ${TF_DIR} has no vendored CFITSIO"
    return 1
  fi
  ln -sfn "${TF_DIR}/extern" "${REPO_ROOT}/extern"
  "$ENV_PY" benchmarks/run_cfitsio_direct_bench.py
}

LANES="${TORCHFITS_LANES:-all}"
if [[ "$LANES" == "all" ]]; then
  LANES="corpus replay contract surface coverage perf examples real-data bench-cpu bench-ml bench-megacam cfitsio-direct"
  if [[ "${TORCHFITS_GPU:-0}" == "1" ]]; then
    LANES="${LANES} bench-gpu"
  fi
fi
LANES="${LANES//,/ }"

echo "=== lanes: ${LANES}"
for lane in $LANES; do
  fn="lane_${lane//-/_}"
  if ! declare -F "$fn" >/dev/null; then
    echo "[FAIL] lane=${lane}: unknown lane"
    RESULTS+=("lane=${lane} status=2 secs=0")
    fail=1
    continue
  fi
  echo "--- lane=${lane} start $(date -u +%H:%M:%S)"
  start=$(date +%s)
  log="${LOG_DIR}/${lane}.log"
  run_with_cap "$LANE_SECS" "$fn" > "$log" 2>&1
  rc=$?
  secs=$(( $(date +%s) - start ))
  if [[ "$rc" -eq 124 ]]; then
    echo "[FAIL] lane=${lane}: exceeded ${LANE_SECS}s"
    fail=1
  elif [[ "$rc" -ne 0 ]]; then
    echo "[FAIL] lane=${lane} (rc=${rc}) after ${secs}s; tail of ${log}"
    tail -25 "$log"
    fail=1
  else
    echo "[ OK ] lane=${lane} ${secs}s"
  fi
  RESULTS+=("lane=${lane} status=${rc} secs=${secs}")
done

echo "=== summary ==="
printf '%s\n' "${RESULTS[@]}" | tee "${RUN_DIR}/summary.txt"
if [[ "$fail" -ne 0 ]]; then
  echo "RESULT: FAIL"
  echo "RESULT: FAIL" >> "${RUN_DIR}/summary.txt"
  exit 1
fi
echo "RESULT: PASS"
echo "RESULT: PASS" >> "${RUN_DIR}/summary.txt"
