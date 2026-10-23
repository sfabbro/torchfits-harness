# Coverage

Three complementary measurements, from cheapest to most expensive. None of
them runs in the default `pixi run test` lane.

| Question | Lane | Output |
|---|---|---|
| Is every public torchfits symbol mentioned by a test? | `pixi run surface-report` | `docs/public-surface.md` |
| Which Python lines does the harness execute? | `pixi run coverage-python` | `reports/python-coverage/` |
| Which C/C++ lines does it execute? | `pixi run coverage-cpp` | `reports/cpp-coverage.txt` |

## Public surface

`benchmarks/surface/public_surface.json` is a tracked snapshot of every public
function, class, method and property the installed wheel exports. Regenerate it
after a torchfits upgrade:

```bash
pixi run surface-update        # rewrite the JSON + docs/public-surface.md
pixi run surface-report        # summary; exit 1 when a symbol has no test
```

`tests/test_public_surface.py` fails when the registry drifts from the
installed wheel or when a symbol appears in no test file. Matching is by name
against `tests/test_*.py`, deliberately cheap and stable: it answers "does any
suite mention this?", not "does it run?". Symbols the harness cannot sensibly
exercise belong in `benchmarks/surface/exemptions.json` with a written reason;
stale exemptions fail the same test.

## Python line coverage

```bash
pixi run coverage-python
```

Runs the full suite with `pytest-cov` against the installed wheel and writes an
HTML report to `reports/python-coverage/index.html` (gitignored). This measures
the Python modules; the compiled extension is a black box to it, which is what
the next lane is for.

## C/C++ line coverage

The wheel on PyPI is compiled without instrumentation, so the extension, the
torch-free core and vendored CFITSIO are invisible to `coverage.py`. Build a
checkout with `-DTORCHFITS_COVERAGE=ON` and run a selection against that build:

```bash
pixi run coverage-cpp
# or, against another checkout / narrower selection:
TORCHFITS_SOURCE=/path/to/torchfits scripts/cpp_coverage.sh \
    --tests tests/test_replay_cfitsio.py
```

The CMake option instruments `_C`, `_core`, `torchfits_core` and vendored
CFITSIO with `--coverage`, pins optimisation to `-O0` (inlining merges lines
and skews gcov counts) and disables IPO for the build. The lane then:

1. builds a wheel in the checkout's Pixi environment,
2. installs it into a throwaway venv that borrows torch, astropy, fitsio and
   pytest from the harness environment — your normal benchmark wheel is
   untouched,
3. runs the selected tests,
4. summarises `*.gcda` under the checkout's `build/` directory with gcovr.

Expect several minutes: it is a full `-O0` build of the extension and CFITSIO.
`--keep` preserves the venv, `--no-build` reuses it for a faster re-run. The
gcov executable defaults to `gcov`; override with `GCOV_EXECUTABLE` (for
example `GCOV_EXECUTABLE="xcrun llvm-cov gcov"`).

This lane is the only one that compiles torchfits. It never vendors CFITSIO
into this repository: the sources come from the checkout named by
`TORCHFITS_SOURCE`, which defaults to the sibling `../torchfits`.
