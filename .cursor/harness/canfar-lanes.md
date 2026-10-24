# CANFAR lane suite

The matrix (`scripts/launch_canfar_matrix_tests.sh`) answers "does each
interpreter compile and pass torchfits?". This suite answers the rest: the
corpus parity gates, the upstream replays, the contract/surface/tour gates, the
benchmarks that had no lane, and the real-archive data paths.

```bash
pixi run lanes-canfar                 # CPU session, every lane
pixi run lanes-canfar-gpu             # CPU + one H100 session for bench-gpu
bash scripts/launch_canfar_all_lanes.sh --lanes corpus,replay
```

One session runs the lanes in order, each under its own watchdog; a red lane
never stops the rest. Verdicts come from `[ OK ] lane=…` / `[FAIL] lane=…`
lines collected into `benchmarks_results/canfar_<run id>/<tag>/canfar_logs.txt`.

| Lane | What it proves |
|---|---|
| `corpus` | CI-tier corpus fetched and hash-verified, then manifest, parity, operation matrix, write parity, pure-CFITSIO oracle and astropy diff |
| `replay` | fitsio, astropy, CFITSIO, CCfits, EleFits, qfits, sfitsio behaviour replays |
| `contract` | bench-corpus, CLI suite, public surface, executed docs tour |
| `surface` | every exported symbol is mentioned by a test (exits 1 otherwise) |
| `coverage` | whole suite with line coverage of the installed wheel |
| `perf` | the `performance`-marked tests the matrix legs exclude |
| `examples` | torchfits example gallery plus the harness cooks |
| `real-data` | CFHT MegaCam/MegaPipe samples, torchfits real-data tests, harness real-archive reads |
| `bench-cpu` | `bench_corpus`, `bench_metadata`, `bench_arrow_tables`, `api_overhead` |
| `bench-ml` | DataLoader throughput |
| `bench-megacam` | cutouts, median stack, HTTP stream, science pipeline (bounded; `TORCHFITS_MEGACAM_FILES/HDUS`) |
| `cfitsio-direct` | pure-C CFITSIO microbench built against the vendored tree |
| `bench-gpu` | GPU memory suite and the dark→blank denoiser (GPU session only) |

## Data on the scratch volume

Every artifact and every downloaded byte lives under `TORCHFITS_DATA_ROOT`
(default `/scratch/torchfits-data`): corpus, CFHT samples, calibration frames,
the torchfits checkout, the zlib prefix and the pixi/pip caches. Nothing is
written under `$HOME`, because `/arc/home` is shared and a stale `~/.local`
poisons later sessions.

On the runs below `/scratch` behaved as **per-session** scratch: each session
re-cloned the checkout and re-fetched the corpus and the 3.3 GB MegaCam samples.
Point `TORCHFITS_DATA_ROOT` at a mounted project volume if one is available and
the fetch should be paid once.

## Results, 2026-10-24 (torchfits 1.3.0)

| Lane | Session | Seconds |
|---|---|---|
| corpus (with the oracle built) | `oyaoje9y` | 35 |
| replay, contract, surface, perf, examples, bench-cpu, bench-ml | `eow4e6y7` | 5–70 |
| coverage — 2690 passed, 89 skipped, 5 xfailed, 75% of 14376 statements | `z1950272` | 295 |
| real-data | `bghihf8v` | 450 |
| bench-megacam | `piy5yr7y` | 606 |
| cfitsio-direct | `oyaoje9y` | 35 |
| bench-gpu (H100 NVL, `cuda True`) | `xou7sx5m` | 326 |

## What the lanes exposed

1. **The pure-CFITSIO oracle was skipping on every CANFAR run.** The image has
   the zlib runtime but no headers, and `extern/cfitsio` is gitignored, so
   `build_cfitsio_oracle.py` either failed to configure or found no tree and the
   tests turned into skips — which pytest reports as a pass. The lane now builds
   zlib into scratch and vendors CFITSIO in the checkout, and fails outright if
   the oracle cannot be built.
2. **The docs tour needs the console script on PATH.** Invoking the environment
   interpreter by absolute path leaves its `bin/` off PATH, so `test_tour.py`
   errored five times. `contract` and `coverage` were red for that reason alone.
3. **A GPU session keeps the CPU torch without `--force-reinstall`.** pip prints
   "already satisfied" for the same version and the H100 leg then reports
   `cuda False`; `bench_gpu_memory` fails while the log looks healthy.
4. **Two benchmarks still passed `handle_cache_capacity=`**, removed in 1.3.0:
   `bench_ml_loader.py` and `bench_fitstable_io.py`. Found by the MPS lane,
   fixed here.
5. **The denoiser needs the calibration frames.** Without
   `fetch_cfht_calib_frames.sh` it writes zero rows and exits 0.

## MPS lane (this Mac only)

`pixi run mps-lane` / `bash scripts/mps_lane.sh`. No CANFAR image has an MPS
device, and every `device.type == "mps"` branch in `torchfits/transforms`
(base, helpers, mask, rgb) is reachable only on Apple silicon.

The lane compares `read(..., device="mps")` and the transforms with MPS dtype
branches against the CPU float64 result (tolerance `TORCHFITS_MPS_TOL`, default
2e-3), checks the result stays on `mps` and stays float32, runs the torchfits
files that carry those branches, and benchmarks the DataLoader on the device.
It fails if that benchmark silently falls back to CPU.

2026-10-24 on this Mac (torch 2.14.1, torchfits 1.3.0): 13 parity checks pass,
`max rel` 1.6e-07 for the stretches, 274 tests pass, DataLoader runs on MPS.
`tests/test_mps.py` referenced by a comment in `tests/test_io.py` does not
exist, so device placement has no dedicated test on either platform.
