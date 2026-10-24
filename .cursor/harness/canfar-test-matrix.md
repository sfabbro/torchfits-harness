# CANFAR test matrix (per-CPython, per-device)

The wheel is ABI-locked to one PyTorch minor and one CPython build, and the
suite ships with the source tree rather than in the wheel. So the only way to
answer "does this lane pass on 3.15 on a GPU" is to build the checkout on the
interpreter in question and run `tests/` there. That is what these two scripts
do. This is a test matrix, not a bench: it compiles torchfits, which the bench
launchers deliberately do not.

```bash
bash scripts/launch_canfar_matrix_tests.sh                      # 10 legs: 3.11-3.15 x {cpu, cu130}
bash scripts/launch_canfar_matrix_tests.sh --pythons 3.13 --devices cpu
TORCHFITS_MT_REAL_DATA=1 bash scripts/launch_canfar_matrix_tests.sh \
    --pythons 3.11 --devices cpu                                # + 7.5 GB CFHT corpus
bash scripts/launch_canfar_matrix_tests.sh \
    --collect-only benchmarks_results/canfar_matrix_tests_<run>  # read an earlier run
```

One session per (CPython, device), on `images.canfar.net/astroai/base`.

## What a leg provisions, and why

| Need | How | Why not the obvious thing |
|---|---|---|
| CPython | `uv venv --python 3.x` | The image ships 3.11-3.14 only; uv fetches 3.15 without a conda solve. |
| torch | `torch==2.14.1` from `whl/cpu` or `whl/cu130` | The lane is exactly what the wheel is built against. |
| zlib | image prefix if usable, else build 1.3.1 | `find_package(ZLIB REQUIRED)` fails on the image: the runtime is there, the headers are not. |
| bzip2 | image prefix if usable, else build 1.0.8 **with `-fPIC`** | Without it CFITSIO compiles BZIP2_1 out and six tests skip; without `-fPIC` the static archive cannot link into the extension. |
| CFHT samples | both fetch scripts, into `$SCRATCH` | The tests skip without them, which silently shrinks the run. The checkout is symlinked at `benchmarks_data`, so the tests' own resolution is untouched. |

The suite runs from the harness checkout, not from the repository root: a test
that only passes from one CWD is a bug, and this is the cheapest place to see
it. `-rs` records why anything skipped.

## Result, 2026-10-22 (torch 2.14.1, main `73b3be9`)

| Legs | Suite |
|---|---|
| 3.11-3.15 x {cpu, cu130}, no samples | 912 passed, 32 skipped, ~2.5 min per leg |
| 3.11 cpu and 3.15 cu130, samples + bzip2 | 939 passed, 5 skipped, ~2.6 min per leg |

The five remaining skips are platform- or network-conditional (`nm -m` and the
Homebrew `libomp` cases are macOS-only, KMP is set on Darwin, the spawn hazard
is macOS, one index probe needs the network). `cuda_available` was `True` with
CUDA 13.0 on every GPU leg.

## What it caught

- `scripts/check_duplicate_cpp.py` resolved its tree relative to the CWD, so the
  guard scanned nothing when run from anywhere but the repo root (fixed: the
  console and its test landed in torchfits).
- `tests/test_core_library_real_data.py` skipped only when *both* samples were
  absent, so a MegaCam-only corpus died on `MEGAPIPE[0]` with `IndexError`
  (fixed: per-dataset skip marks).
- `tests/test_docs_integrity.py` shelled out to `git ls-files` without `cwd`,
  so two checks skipped with "not a git checkout" whenever the suite started
  outside the repo (fixed: `cwd=ROOT`).
- The launcher parsed `canfar info` with `sed 's/^\s*Status…'`; BSD sed has no
  `\s`, so completed legs were polled until `MAX_WAIT` (fixed: `awk`).

`canfar logs` returns only the tail of a session, so a leg keeps its own
`build.log`, `pytest-*.log` and fetch logs under `/scratch/torchfits-matrix-tests/<tag>/`.

## The verify lane

`scripts/canfar_verify_incontainer.sh` answers a different question: it installs
the **released wheel** (`TORCHFITS_WHEEL`, or `torchfits>=1.3.0`) and runs all of
`tests/` plus a bounded bench. It is the only lane that runs every test, so it
is where contract tests that still poke a previous release's internals surface.

Result, 2026-10-24 (torchfits 1.3.0, harness `be24d72`): 2663 passed, 189
skipped, 17 deselected, 5 xfailed in 5:06; `RESULT: PASS`.

The first run of the lane was red by 20 tests that pinned the pre-1.3 contract:
`read(handle_cache_capacity=)`, the deprecated `configure_cache` /
`configure_cpp_cache` no-ops, positional `TableHDU(..., header)` (keyword-only
since `4811f12`), the `_C.open_and_read_headers` batch helper, and writable
raw-transport buffers. None was a torchfits regression: every one fails the same
way against the dev tree, because 1.3.0 removed or changed the thing tested.
The suite now pins the 1.3.0 contract, including that the removed knob is
rejected by name.
