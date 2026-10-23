# A tour on a real archive file

One SDSS spectrum file and no hand-written FITS: read the archive, query a
table, verify checksums, convert to Parquet, write and cut out an image. Every
block below is executed by `tests/test_tour.py`, so it cannot drift from the CLI
it documents.

Fetch the files once (about 8 MB for the CI tier):

```bash
pixi run corpus-fetch --tier ci
```

Then run the blocks in order from the repository root. They share shell state,
so keep them in one session; outputs go to a scratch directory.

```bash
export SPEC="benchmarks_data/corpus/sdss_spec_0266_51630_0001/spec-0266-51630-0001.fits"
export OUT="${TMPDIR:-/tmp}/torchfits-tour"
rm -rf "$OUT" && mkdir -p "$OUT"
```

## 1. What is in the file?

`info` lists every HDU without reading data pages. The COADD extension is the
flux vector; the eleven `B1`/`R1` extensions are per-exposure spectra.

```bash
pixi run torchfits info "$SPEC"
```

## 2. One header, three keywords

`--hdu` selects an extension (`-e` is the short form), `--keyword` accepts
wildcards. Values print as FITS cards, so they can be pasted into a header
editor.

```bash
pixi run torchfits header --hdu 1 --keyword 'NAXIS*' "$SPEC"
```

## 3. The table schema, without loading the flux

`table` reports column names and types plus a preview. It reads the header and a
bounded number of rows, so it is safe on the 1.6-million-row plate catalogue.

```bash
pixi run torchfits table --hdu 1 "$SPEC"
```

## 4. Are the checksums intact?

`verify` reads `DATASUM`/`CHECKSUM`; files without them report
`no_checksums` and still exit 0. A corrupt data block exits 4.

```bash
pixi run torchfits verify "$SPEC"
```

## 5. Extract the table to Parquet

`convert` infers the format from the output extension. The Parquet file has the
same column names and row count as the FITS extension.

```bash
pixi run torchfits convert "$SPEC" "$OUT/coadd.parquet" --hdu 1
pixi run python - <<'PY'
import os
import pyarrow.parquet as pq

table = pq.read_table(f"{os.environ['OUT']}/coadd.parquet")
print(table.num_rows, "rows;", table.column_names[:4], "...")
PY
```

## 6. The same data as Arrow, with a row window and a predicate

`torchfits.table.read` returns an Arrow table. `row_slice` bounds the rows before
conversion and `where` filters inside the reader, so a wide catalogue is never
materialised whole just to look at a window.

```bash
pixi run python - <<'PY'
import os

from torchfits import table as tf

spec = os.environ["SPEC"]
head = tf.read(spec, hdu=1, row_slice=(0, 5))
print(head.num_rows, "rows;", head.column_names[:3], "...")

positive = tf.read(spec, hdu=1, where="flux > 0")
print("positive flux rows:", positive.num_rows)
PY
```

## 7. Write an image, then cut a box out of it

`write_tensor` takes a `torch.Tensor` (wrap a NumPy array with
`torch.from_numpy`; it does not convert for you) and writes a standard FITS
primary HDU. `cutout` extracts a half-open pixel box.

```bash
pixi run python - <<'PY'
import os

import numpy as np
import torch
import torchfits

cube = torch.from_numpy(np.arange(4 * 64 * 64, dtype=np.float32).reshape(4, 64, 64))
path = f"{os.environ['OUT']}/cube.fits"
torchfits.write_tensor(path, cube, overwrite=True)
print("wrote", path, "shape", torchfits.read_shape(path, 0))
PY
pixi run torchfits cutout "$OUT/cube.fits" "$OUT/cutout.fits" --box 8,8,24,24
pixi run torchfits stats "$OUT/cutout.fits"
```

## 8. Edit a copy, leave the archive untouched

Editing in place is `setkey`'s default; work on a copy when the input is
read-only, as archive files usually are. The last command prints the card that
was just written.

```bash
pixi run torchfits copy "$SPEC" "$OUT/edited.fits"
pixi run torchfits setkey "$OUT/edited.fits" --hdu 1 -k OBJECT --value "SDSS J0266"
pixi run torchfits header "$OUT/edited.fits" --hdu 1 -k OBJECT
```

## Where next

- `pixi run bench-corpus` times the same kind of file against astropy and
  fitsio on this host.
- `docs/api-overhead.md` explains why a row-window table read spends most of its
  time in Python, with the measurements that back the proposal.
- `pixi run surface-report` shows which public torchfits symbols this repository
  exercises and which it does not.
