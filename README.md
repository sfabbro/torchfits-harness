# torchfits-harness

Benchmarks, upstream parity, migration notes, and CANFAR evaluation for
[torchfits](https://github.com/astroai/torchfits).

The library, wheels, and tests that do not import astropy or fitsio stay in
torchfits. This repo compares an installed torchfits wheel with astropy and
fitsio. Apart from the opt-in `coverage-cpp` lane — which builds a checkout you
point it at — it does not compile torchfits or vendor CFITSIO.

## Run

```bash
pixi install
pixi run subject /path/to/torchfits-*.whl
pixi run dev
pixi run test          # fitsio and astropy upstream smokes
pixi run bench-cpp     # bounded C++ backend comparison
pixi run corpus-fetch  # real archive files, gitignored (~8 MB for the CI tier)
pixi run corpus-test   # value parity against astropy/fitsio and pure CFITSIO
```

`pixi run bench-fits`, `pixi run bench-fitstable`, and `pixi run bench-all`
run the larger suites in this same environment. Full suites are opt-in.
`pixi run bench-corpus` times the fetched archive files against astropy and
fitsio, and `pixi run replay-test` replays the upstream suites' documented
behaviour.

## Docs

| | |
|---|---|
| [Benchmarks](docs/benchmarks.md) | Methodology and published results |
| [Parity](docs/parity.md) | Behavior compared with astropy and fitsio |
| [Tour](docs/tour.md) | A tested walkthrough of the CLI and API on a real archive file |
| [Coverage](docs/coverage.md) | Public-surface, Python and C/C++ coverage lanes |
| [API overhead](docs/api-overhead.md) | Measured per-call costs and a proposal |
| [Public surface](docs/public-surface.md) | Generated: symbols no test exercises |
| [From astropy](docs/migration_astropy.md) | Migration notes |
| [From fitsio](docs/migration_fitsio.md) | Migration notes |
