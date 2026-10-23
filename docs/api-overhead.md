# Proposal: cut the Python overhead in row-window table reads

`torchfits.table.read(..., row_slice=...)` is the call every catalogue workflow
starts with, and on a wide table it is roughly **39× slower than the C++ binding
underneath it** — 15.4 ms against 0.4 ms for the same 1000 rows. That gap is not
I/O: the binding has already copied the bytes. It is Python turning those bytes
into Arrow.

This note records the measurements, identifies where the time goes, and proposes
four changes in descending order of payoff. It is a proposal, not an
implementation: nothing in the library changes here.

## How the numbers were produced

```bash
pixi run api-overhead                       # the table below
pixi run python scripts/api_overhead.py --number 2000 --json reports/api-overhead.json
```

Host: Apple arm64, macOS, `torchfits 1.3.0` (installed wheel), PyTorch 2.14.
Payload: `plates-dr18.fits` from the CI corpus — 7.1 MB, 6885 rows × 131 columns
(100 numeric, 31 fixed-width strings). Every figure is a median of five samples
after a warm-up call, so the CFITSIO handle and the process caches are warm;
`clear_file_cache + read_header` is the one deliberately cold row.

| Layer | Call | µs / call |
|---|---|---:|
| header | `_C.read_header_dict` (binding only) | 31.7 |
| header | `torchfits.read_header` (cached) | 8.8 |
| header | `torchfits.clear_file_cache` + `read_header` | 48.9 |
| image | `_C.read_full` (binding only) | 40.4 |
| image | `torchfits.read_tensor` | 45.9 |
| image | `_C.read_num_hdus` (binding only) | 0.15 |
| image | `torchfits.read_num_hdus` | 1.86 |
| table | `_C.read_fits_table_rows_raw`, 1000 rows | 396 |
| table | `torchfits.table.read`, 1000 rows, all 131 columns | **15 442** |
| table | `torchfits.table.read`, 1000 rows, 5 numeric columns | 1 203 |
| table | `torchfits.table.read`, 1000 rows, 5 string columns | 3 057 |
| handle | `torchfits.open` + `len` | 79 |

Two rows are already in good shape and are worth stating so the proposal does
not overreach:

* **Headers.** The public call is *faster* than the binding (0.3×) because it
  memoises parsed headers. Uncached, the full open-and-parse is ~49 µs, which is
  the floor for a cold call and not a target.
* **Images.** `read_tensor` adds 5.6 µs to `_C.read_full` (1.1×): one Python
  dispatch over a call that is dominated by the copy. Nothing to win.

`read_num_hdus` is 12× its binding, but the absolute cost is 1.7 µs — worth a
one-line fix at most, listed last.

## Where the 15 ms goes

A `cProfile` run of one 1000-row, all-column read attributes ~14 ms of the
~19 ms cold call to `_table/arrow_convert.py`, split as:

| Function | Share |
|---|---:|
| `_decode_uint8_matrix_to_arrow` (31 string columns, via `numpy.char.decode`) | ~12 ms |
| `_numpy_to_arrow_array` (100 numeric columns) | ~2 ms |
| `fits_schema.parse_tform` / `iter_table_columns` (schema re-parsed per call) | ~2 ms |

String columns are 24 % of the fields but ~80 % of the conversion time: each one
is a `uint8` matrix that Python decodes with `numpy`'s `_vec_string`, then hands
to Arrow as an object array. Numeric columns are cheap per column but there are
100 of them, each entering and leaving Python.

The C++ layer already knows how to build Arrow for a whole table
(`_C.read_fits_table`); the row-window path deliberately returns a numpy dict
and rebuilds Arrow in Python, which is why the same bytes cost 39× more.

## Proposed changes

### 1. Decode fixed-width strings in C++ (expected: −10 ms, ~65 %)

Expose the string columns from `read_fits_table_rows_raw` as a byte buffer plus
`StringArray` offsets — or return Arrow `StringArray` directly — instead of a
`uint8` matrix. Arrow's `StringArray::from_buffers` validates offsets without
copying, so the Python side becomes one call per column.

*Acceptance:* the 5-string-column case (3.06 ms) drops below the 5-numeric case
(1.20 ms) adjusted for its size; the all-column read falls under 6 ms.
*Risk:* the existing `strip=True` behaviour (right-trimming padded FITS strings)
must move into the same layer, or be applied as an Arrow compute kernel over
offsets.

### 2. Build the row-window record batch in C++ (expected: −3 ms on wide tables)

The full-table path already returns Arrow from `_C`; teach the row-window path
the same trick (`start_row`/`num_rows`) so `_table/arrow_convert.py` handles only
the fallback cases (scaled columns, variable-length arrays). This removes 131
Python→Arrow transitions per call, independent of change 1.

*Acceptance:* `torchfits.table.read(row_slice=...)` within 2× of
`_C.read_fits_table_rows_raw` on the same window; both give identical Arrow
schemas.

### 3. Cache the parsed schema per `(path, hdu)` (expected: −1.5 ms)

`parse_tform` runs 586 times for a 131-column read — 4.5 parses per column,
because each conversion helper asks the schema again. A small LRU keyed on the
file's identity (path, mtime, size, hdu) removes the repeat work and is the same
cache the header path already keeps.

*Acceptance:* one `parse_tform` per column per cold read; warmed reads show zero
schema calls in the profile.

### 4. Make the `read_num_hdus` cache hit cheap (expected: −1.5 µs)

The cached call does 1.86 µs of path validation against a 0.15 µs binding. Fold
the validation into the cache key (already computed for the header cache) and
return the stored count.

*Acceptance:* ≤ 0.5 µs warm; no change to the uncached path.

## What this proposal does not ask for

* No change to the C++ read path, chunking, or CFITSIO usage — that layer is not
  the bottleneck on these files.
* No new dependency: Arrow is already a hard requirement.
* No bypass of validation, scaling or `NULL` handling to win the microbenchmark;
  the tests in `tests/test_corpus_parity.py` and `tests/test_write_parity.py`
  compare values against astropy and must keep passing unchanged.

## Verification

Any implementation lands with a benchmark delta, not a claim: rerun
`pixi run api-overhead` and `pixi run bench-corpus` on the same host, and quote
both the µs table above and the corpus `read_rows` row for
`sdss_plates_dr18`. A change that improves only a synthetic fixture is not
evidence.
