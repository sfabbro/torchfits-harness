#!/usr/bin/env bash
# Launch the harness lane suite on CANFAR headless: one CPU session for the
# gates and benchmarks, optionally one GPU session for the device lanes.
#
# Each session clones this repository into /scratch, runs
# scripts/canfar_all_lanes_incontainer.sh, and leaves its per-lane logs and CSVs
# under /scratch/torchfits-harness/<run id>/ on the scratch volume.
#
# Usage:
#   bash scripts/launch_canfar_all_lanes.sh                      # CPU lanes
#   bash scripts/launch_canfar_all_lanes.sh --gpu                # CPU + GPU sessions
#   bash scripts/launch_canfar_all_lanes.sh --lanes corpus,replay
#   bash scripts/launch_canfar_all_lanes.sh --collect-only <run dir>
#
# Env:
#   TORCHFITS_LANES          lane list for the CPU session        [all]
#   TORCHFITS_GPU_LANES      lanes for the GPU session            [bench-gpu]
#   TORCHFITS_DATA_ROOT      persistent data on the scratch mount [/scratch/torchfits-data]
#   TORCHFITS_CORPUS_TIER    ci | full                            [ci]
#   TORCHFITS_VERSION        exact torchfits release to test      [unset: >=1.3.0]
#   TORCHFITS_REF            torchfits checkout ref               [main]
#   TORCHFITS_ALL_CPU/_MEMORY/_MAX_WAIT_SECS/_IMAGE
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COLLECT_DIR=""
GPU=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lanes) TORCHFITS_LANES="$2"; shift 2 ;;
    --gpu) GPU=1; shift ;;
    --collect-only) COLLECT_DIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

LANES="${TORCHFITS_LANES:-all}"
GPU_LANES="${TORCHFITS_GPU_LANES:-bench-gpu}"
DATA_ROOT="${TORCHFITS_DATA_ROOT:-/scratch/torchfits-data}"
TIER="${TORCHFITS_CORPUS_TIER:-ci}"
IMAGE="${TORCHFITS_ALL_IMAGE:-images.canfar.net/astroai/base:latest}"
CPU="${TORCHFITS_ALL_CPU:-8}"
MEMORY="${TORCHFITS_ALL_MEMORY:-32}"
GPU_CPU="${TORCHFITS_ALL_GPU_CPU:-8}"
GPU_MEMORY="${TORCHFITS_ALL_GPU_MEMORY:-48}"
MAX_WAIT="${TORCHFITS_ALL_MAX_WAIT_SECS:-10800}"
POLL="${TORCHFITS_ALL_POLL_SECS:-30}"
RUN_ID="${TORCHFITS_ALL_RUN_ID:-lanes_$(date -u +%Y%m%d_%H%M%S)}"

if [[ -n "$COLLECT_DIR" ]]; then
  OUT="$(cd "$COLLECT_DIR" && pwd)"
  RUN_ID="$(basename "$OUT")"
else
  OUT="${ROOT_DIR}/benchmarks_results/canfar_${RUN_ID}"
  mkdir -p "$OUT"
fi

command -v canfar >/dev/null || { echo "canfar CLI not found" >&2; exit 1; }

session_status() {
  canfar info "$1" 2>/dev/null | awk '$1 == "Status" { print $2; exit }'
}

launch_session() {
  local tag="$1" lanes="$2" cpus="$3" memory="$4" gpu="$5" extra_env="$6"
  local dir="${OUT}/${tag}"
  mkdir -p "$dir"
  local name="tf-${tag}-${RUN_ID}"
  name="$(printf '%s' "$name" | tr '_.' '--' | tr -cd '[:alnum:]-')"
  local gpu_args=""
  [[ "$gpu" == "1" ]] && gpu_args="--gpu 1"

  # The platform splits its command on whitespace and rejects `$`, so the
  # remote is tab-encoded and contains no shell expansion.
  local remote
  remote="git clone --depth 1 https://github.com/sfabbro/torchfits-harness.git /scratch/src/harness-${RUN_ID}-${tag}; cd /scratch/src/harness-${RUN_ID}-${tag}; bash scripts/canfar_all_lanes_incontainer.sh"
  local tabs="${remote// /$'\t'}"

  echo "== launching ${tag} (${name}) lanes=${lanes}"
  # shellcheck disable=SC2086
  if ! canfar create headless "$IMAGE" \
      --name "$name" \
      --cpu "$cpus" \
      --memory "$memory" \
      ${gpu_args} \
      --env "TORCHFITS_LANES=${lanes}" \
      --env "TORCHFITS_HARNESS_RUN_ID=${RUN_ID}-${tag}" \
      --env "TORCHFITS_DATA_ROOT=${DATA_ROOT}" \
      --env "TORCHFITS_CORPUS_TIER=${TIER}" \
      --env "TORCHFITS_REF=${TORCHFITS_REF:-main}" \
      ${extra_env} \
      -- bash -c "$tabs" > "${dir}/create.log" 2>&1; then
    echo "[FAIL] ${tag}: session create failed (see ${dir}/create.log)"
    return 1
  fi

  local session_id
  session_id="$(python3 - "${dir}/create.log" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"ID:\s*([^)]+)\)", text)
print(m.group(1).strip() if m else "")
PY
)"
  if [[ -z "$session_id" ]]; then
    echo "[FAIL] ${tag}: could not parse session id (see ${dir}/create.log)"
    return 1
  fi
  echo "$session_id" > "${dir}/session_id.txt"
  echo "   session ${session_id}"
  return 0
}

echo "=== CANFAR harness lanes (run ${RUN_ID}) ==="
echo "cpu lanes=${LANES} gpu=${GPU} (${GPU_LANES}) tier=${TIER}"
echo "data_root=${DATA_ROOT} out=${OUT}"

FAILED=0
if [[ -z "$COLLECT_DIR" ]]; then
  launch_session cpu "$LANES" "$CPU" "$MEMORY" 0 "" || FAILED=$((FAILED + 1))
  if [[ "$GPU" -eq 1 ]]; then
    sleep 3
    launch_session gpu "$GPU_LANES" "$GPU_CPU" "$GPU_MEMORY" 1 \
      "--env TORCHFITS_GPU=1" || FAILED=$((FAILED + 1))
  fi
fi

echo "=== launched; collecting (max wait ${MAX_WAIT}s) ==="
START="$(date +%s)"
for dir in "$OUT"/*/; do
  [[ -f "${dir}/session_id.txt" ]] || continue
  session_id="$(cat "${dir}/session_id.txt")"
  tag="$(basename "$dir")"
  while true; do
    # awk, not sed: the CLI pads its labels with spaces and BSD sed has no \s.
    status="$(session_status "$session_id")"
    case "$status" in Succeeded | Completed | Failed | Error | Terminating) break ;; esac
    if (( $(date +%s) - START > MAX_WAIT )); then
      echo "[FAIL] ${tag}: still ${status:-unknown} after ${MAX_WAIT}s"
      break
    fi
    sleep "$POLL"
  done
  canfar logs "$session_id" > "${dir}/canfar_logs.txt" 2>&1 || true
  if grep -q '^RESULT: PASS' "${dir}/canfar_logs.txt"; then
    grep -E '^\[ (OK|FAIL) \]' "${dir}/canfar_logs.txt" || true
  else
    echo "[FAIL] ${tag}: session ${status:-unknown}"
    grep -E '^\[ (OK|FAIL) \]|^RESULT:' "${dir}/canfar_logs.txt" | tail -20 || true
    FAILED=$((FAILED + 1))
  fi
done

echo "=== summary: ${FAILED} failing session(s), results in ${OUT} ==="
[[ "$FAILED" -eq 0 ]]
