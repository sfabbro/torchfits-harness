#!/usr/bin/env bash
# Line coverage of torchfits' C/C++ core (extension + vendored CFITSIO).
#
# `pixi run coverage-python` measures the Python API of the installed wheel but
# cannot see inside `_C`, `torchfits_core` or CFITSIO: the wheel is compiled
# without instrumentation and its sources are not in this repo. This lane
# builds torchfits *from a checkout* with `-DTORCHFITS_COVERAGE=ON`, runs a test
# selection against that build, and summarises the gcov data with gcovr.
#
# It is opt-in and slow (a full -O0 build, several minutes). It never touches
# the harness Pixi environment: the instrumented wheel is installed into a
# throwaway venv that borrows torch/astropy from the active environment, so the
# wheel you normally benchmark against stays the one under test everywhere else.
#
# Usage:
#   pixi run coverage-cpp                       # sibling ../torchfits checkout
#   TORCHFITS_SOURCE=/path/to/torchfits pixi run coverage-cpp
#   scripts/cpp_coverage.sh --tests tests/test_replay_cfitsio.py --keep
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${TORCHFITS_SOURCE:-$ROOT/../torchfits}"
TESTS=("tests/test_replay_cfitsio.py" "tests/test_cfitsio_oracle.py")
KEEP=0
SKIP_BUILD=0
VENV="${CPP_COVERAGE_VENV:-$ROOT/.cov-cpp/venv}"
REPORT_DIR="$ROOT/reports"
GCOV_EXECUTABLE="${GCOV_EXECUTABLE:-gcov}"

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) SOURCE="$2"; shift 2 ;;
        --tests) TESTS=("$2"); shift 2 ;;
        --keep) KEEP=1; shift ;;
        --no-build) SKIP_BUILD=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

SOURCE="$(cd "$SOURCE" && pwd)"
if [[ ! -f "$SOURCE/src/torchfits/cpp_src/CMakeLists.txt" ]]; then
    echo "not a torchfits checkout: $SOURCE (set TORCHFITS_SOURCE)" >&2
    exit 2
fi
if ! grep -q "TORCHFITS_COVERAGE" "$SOURCE/src/torchfits/cpp_src/CMakeLists.txt"; then
    echo "$SOURCE predates -DTORCHFITS_COVERAGE; update the checkout" >&2
    exit 2
fi
mkdir -p "$REPORT_DIR"

echo "== torchfits source: $SOURCE"
echo "== venv:            $VENV"

WHEELS="$ROOT/.cov-cpp/wheels"
if [[ $SKIP_BUILD -eq 0 ]]; then
    rm -rf "$VENV" "$WHEELS"
    mkdir -p "$WHEELS"
    echo "== building instrumented wheel (-O0, --coverage)"
    # Build with the checkout's own Pixi environment: the harness env carries no
    # cmake/nanobind, and `pixi run dev` is how this source tree is built anyway.
    # scikit-build-core keeps its work tree in $SOURCE/build/<tag>, which is
    # where the .gcda files land once the tests run.
    (cd "$SOURCE" && CMAKE_ARGS="-DTORCHFITS_COVERAGE=ON" \
        pixi run python -m pip wheel "$SOURCE" \
        --no-build-isolation --no-deps --quiet -w "$WHEELS")
    # The test venv borrows torch/astropy/fitsio/pytest from the harness env and
    # only adds the instrumented torchfits, so nothing else is rebuilt or
    # shadowed.
    python -m venv --system-site-packages "$VENV"
    # --force-reinstall: the borrowed system site-packages already contain a
    # (non-instrumented) torchfits, so plain pip would answer "already
    # satisfied" and every test would import that wheel instead of this one.
    "$VENV/bin/python" -m pip install --no-deps --force-reinstall --quiet \
        "$WHEELS"/*.whl
fi

echo "== running ${TESTS[*]}"
set +e
(cd "$ROOT" && "$VENV/bin/python" -m pytest "${TESTS[@]}" -q)
STATUS=$?
set -e
if [[ $STATUS -ne 0 ]]; then
    echo "tests failed (exit $STATUS); still collecting coverage" >&2
fi

# find, not globstar: macOS ships bash 3.2, where `shopt -s globstar` fails.
GCDA=()
while IFS= read -r -d '' gcd_file; do
    GCDA+=("$gcd_file")
done < <(find "$SOURCE/build" -name '*.gcda' -print0 2>/dev/null)
if [[ ${#GCDA[@]} -eq 0 ]]; then
    echo "no .gcda files under $SOURCE/build — was the build instrumented?" >&2
    exit 3
fi
echo "== ${#GCDA[@]} gcov data files under $SOURCE/build"

# Scope the report to torchfits' own code plus vendored CFITSIO; the build tree
# also holds generated sources, nanobind glue and test programs.
# --gcov-ignore-errors=all: CFITSIO's flex/bison-generated lexer and parser
# record the intermediate .l/.y path in their notes, which no longer exists
# after generation; gcovr otherwise aborts the whole report on those three
# files. Every other unreadable note would be reported by gcov as well, so
# this does not hide a real build problem — the .gcda count is checked above.
gcovr \
    --root "$SOURCE" \
    --gcov-executable "$GCOV_EXECUTABLE" \
    --gcov-ignore-errors=all \
    --filter "$SOURCE/src/torchfits/cpp_src/" \
    --filter "$SOURCE/extern/cfitsio/" \
    --txt "$REPORT_DIR/cpp-coverage.txt" \
    --html-details "$REPORT_DIR/cpp-coverage.html" \
    --print-summary >"$REPORT_DIR/cpp-coverage-summary.txt"

cat "$REPORT_DIR/cpp-coverage-summary.txt"
echo "== wrote $REPORT_DIR/cpp-coverage.txt and $REPORT_DIR/cpp-coverage.html"

if [[ $KEEP -eq 0 ]]; then
    rm -rf "$VENV"
fi
exit "$STATUS"
