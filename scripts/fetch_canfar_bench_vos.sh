#!/usr/bin/env bash
# Download torchfits CANFAR bench artifacts from VOSpace to benchmarks_results/<run-id>/.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?usage: $0 <run-id>}"
VOS_BASE="${TORCHFITS_VOS_BASE:-vos:sfabbro/torchfits-gpu-bench}"
VOS_URI="${VOS_BASE}/${RUN_ID}"
LOCAL_DIR="${ROOT_DIR}/benchmarks_results/${RUN_ID}"

if command -v vcp >/dev/null; then
  VCP=(vcp)
  VLS=(vls)
else
  # vos is a pixi pypi-dependency; the scripts are not on PATH outside the env.
  VCP=(pixi run vcp)
  VLS=(pixi run vls)
fi

mkdir -p "${LOCAL_DIR}"

# Copy the artifacts one by one rather than recursing: `--keep-temp` leaves a
# `_raw` fixture tree that vcp cannot resume into (it fails creating the .part
# file), and older uploads nest <run-id>/ inside the destination.
SRC="${VOS_URI}"
entries="$("${VLS[@]}" "${SRC}/")"
if grep -qx "${RUN_ID}" <<<"${entries}"; then
  SRC="${VOS_URI}/${RUN_ID}"
  entries="$("${VLS[@]}" "${SRC}/")"
fi

copied=0
while IFS= read -r name; do
  case "${name}" in
    "" | "_raw" | "${RUN_ID}") continue ;;
  esac
  if "${VCP[@]}" "${SRC}/${name}" "${LOCAL_DIR}/${name}"; then
    copied=$((copied + 1))
  else
    echo "skip ${name}: not a file vcp can fetch" >&2
  fi
done <<<"${entries}"

if [[ "${copied}" -eq 0 ]]; then
  echo "no artifacts under ${SRC}" >&2
  exit 1
fi
echo "fetched ${copied} artifacts from ${SRC} -> ${LOCAL_DIR}"
