#!/usr/bin/env python3
"""Benchmark on the real-archive corpus: same file, three readers, one timer.

The synthetic suites in ``bench_all.py`` control every byte, which is what makes
them good for tracking a regression but a poor predictor of archive work: real
files are tiled, gzipped, heterogeneous MEFs whose HDUs disagree about
compression and scaling. This module times the corpus that
``scripts/fetch_corpus.py`` downloads, with astropy and fitsio reading the same
HDU so the comparison stays on one host and one file.

Operations, chosen per HDU:

* ``read_header`` — metadata only, no data pages.
* ``read_full`` — whole image, when its payload is under ``--max-mb``.
* ``read_window`` — a central 256×256 window of a bigger image.
* ``read_rows`` — the first ``--rows`` rows of a table.

Rows use the shared result schema, so ``results.csv`` can be read next to a
``bench_all.py`` run. Nothing here downloads; a file that is not present is
reported as missing and skipped.

    pixi run bench-corpus
    pixi run bench-corpus -- --tier ci --limit 1 --list
    python -m benchmarks.bench_corpus --families image table
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from astropy.io import fits as astropy_fits

import torchfits
from torchfits import table as tf_table

from benchmarks import corpus
from benchmarks.bench_contract import (
    RESULT_COLUMNS,
    annotate_rankings,
    make_run_id,
    write_csv,
    write_summary,
)
from benchmarks.bench_timing import time_medians_interleaved
from benchmarks.config import DEFAULT_OUTPUT_DIR

Tier = str

#: Families this benchmark knows how to pick a case from; the corpus also
#: declares "mef", whose files simply contain several of these.
BENCH_FAMILIES = ("image", "cube", "spectrum", "table")

#: Central window, in pixels, for images above the payload cap.
WINDOW = 256

_HOST = socket.gethostname()


@dataclass(frozen=True)
class Case:
    """One timed operation on one HDU of one file."""

    case_id: str
    label: str
    operation: str
    family: str
    path: Path
    hdu: int
    kind: str  # "image" | "table"
    size_mb: float
    n_points: int
    window: tuple[int, int, int, int] | None = None
    rows: int = 0

    def metadata(self) -> dict[str, Any]:
        payload = {
            "file": self.path.name,
            "source": self.path.parent.name,
            "hdu": self.hdu,
        }
        if self.window is not None:
            payload["window"] = list(self.window)
        if self.rows:
            payload["rows"] = self.rows
        return payload


def _payload_bytes(header: Any) -> tuple[int, tuple[int, ...], int]:
    """(bytes, shape, itemsize) of an image HDU, from header keys only.

    Reading the header is enough: tile-compressed HDUs keep the logical shape in
    ``ZNAXISn``/``ZBITPIX`` and the compressed payload in ``NAXISn``, and probing
    the data here would decompress every file before the timing loop.
    """
    compressed = "ZNAXIS" in header
    naxis = int(header.get("ZNAXIS", header.get("NAXIS", 0)) or 0)
    bitpix = int(header.get("ZBITPIX", header.get("BITPIX", 8)) or 8)
    shape = tuple(
        int(header.get(f"{'Z' if compressed else ''}NAXIS{i}", 0) or 0)
        for i in range(1, naxis + 1)
    )
    itemsize = abs(bitpix) // 8
    return int(np.prod(shape)) * itemsize, shape, itemsize


def _first_hdu(path: Path, predicate: Callable[[Any], bool]) -> tuple[int, Any] | None:
    try:
        with astropy_fits.open(path, memmap=True, disable_image_compression=True) as hdul:
            for index, hdu in enumerate(hdul):
                if predicate(hdu):
                    return index, hdu.header
    except Exception:  # noqa: BLE001 - a file no reader opens is a skip, not a crash
        return None
    return None


def _is_table(hdu: Any) -> bool:
    return isinstance(hdu, (astropy_fits.BinTableHDU, astropy_fits.TableHDU))


def _is_image(hdu: Any) -> bool:
    return not _is_table(hdu) and int(hdu.header.get("NAXIS", 0) or 0) > 0


def _image_cases(
    source: dict[str, Any],
    path: Path,
    *,
    family: str,
    max_mb: float,
) -> list[Case]:
    found = _first_hdu(path, _is_image)
    if found is None:
        return []
    index, header = found
    nbytes, shape, itemsize = _payload_bytes(header)
    size_mb = nbytes / 1e6
    ident = f"corpus_{source['id']}_image"
    cases = [
        Case(
            case_id=f"{ident}_header",
            label=f"corpus [{source['id']}] image header",
            operation="read_header",
            family=family,
            path=path,
            hdu=index,
            kind="image",
            size_mb=0.0,
            n_points=len(header.cards) if hasattr(header, "cards") else 0,
        )
    ]
    if nbytes / (1024.0**2) <= max_mb:
        cases.append(
            Case(
                case_id=f"{ident}_full",
                label=f"corpus [{source['id']}] image full {shape}",
                operation="read_full",
                family=family,
                path=path,
                hdu=index,
                kind="image",
                size_mb=size_mb,
                n_points=int(np.prod(shape)),
            )
        )
    elif len(shape) == 2:
        # A central window. ``shape`` is in FITS axis order (NAXIS1 first), so
        # shape[0] is x and shape[1] is y; using the last two axes would
        # transpose non-square images and ask read_subset for rows past the
        # image. Cubes above the cap are header-only: the readers disagree on
        # what a two-index slice means for a third axis, so a window there
        # would compare different pixels.
        x0 = max(0, shape[0] // 2 - WINDOW // 2)
        y0 = max(0, shape[1] // 2 - WINDOW // 2)
        x1 = min(shape[0], x0 + WINDOW)
        y1 = min(shape[1], y0 + WINDOW)
        cases.append(
            Case(
                case_id=f"{ident}_window",
                label=f"corpus [{source['id']}] image window {x1 - x0}×{y1 - y0}",
                operation="read_window",
                family=family,
                path=path,
                hdu=index,
                kind="image",
                size_mb=(x1 - x0) * (y1 - y0) * itemsize / 1e6,
                n_points=(x1 - x0) * (y1 - y0),
                window=(x0, y0, x1, y1),
            )
        )
    return cases


def _table_cases(
    source: dict[str, Any],
    path: Path,
    *,
    family: str,
    rows: int,
) -> list[Case]:
    found = _first_hdu(path, _is_table)
    if found is None:
        return []
    index, header = found
    nrows = int(header.get("NAXIS2", 0) or 0)
    row_bytes = int(header.get("NAXIS1", 0) or 0)
    selected = min(rows, nrows)
    ident = f"corpus_{source['id']}_table"
    return [
        Case(
            case_id=f"{ident}_header",
            label=f"corpus [{source['id']}] table header",
            operation="read_header",
            family=family,
            path=path,
            hdu=index,
            kind="table",
            size_mb=0.0,
            n_points=len(header.cards) if hasattr(header, "cards") else 0,
        ),
        Case(
            case_id=f"{ident}_rows",
            label=f"corpus [{source['id']}] table first {selected} rows",
            operation="read_rows",
            family=family,
            path=path,
            hdu=index,
            kind="table",
            size_mb=selected * row_bytes / 1e6,
            n_points=selected,
            rows=selected,
        ),
    ]


def select_cases(
    *,
    tier: str = "full",
    families: tuple[str, ...] = BENCH_FAMILIES,
    limit: int = 3,
    max_mb: float = 128.0,
    rows: int = 200_000,
) -> tuple[list[Case], list[dict[str, Any]]]:
    """Pick cases from the corpus; also return the sources that were skipped."""
    cases: list[Case] = []
    skipped: list[dict[str, Any]] = []
    per_family: dict[str, int] = {}
    for source in corpus.sources(tier=tier):
        declared = corpus.families(source)
        family = next((f for f in declared if f in families), None)
        if family is None:
            continue
        if per_family.get(family, 0) >= limit:
            continue
        files = corpus.source_files(source)
        present = [path for path in files if path.is_file()]
        if not present:
            skipped.append({"source": source["id"], "reason": "not fetched"})
            continue
        path = present[0]
        # A table family may still ship an image extension and vice versa
        # (SDSS spectra are binary tables); probe the declared family first and
        # fall back to whatever the file actually contains.
        try:
            if family == "table":
                found = _table_cases(source, path, family=family, rows=rows)
                found = found or _image_cases(
                    source, path, family=family, max_mb=max_mb
                )
            else:
                found = _image_cases(source, path, family=family, max_mb=max_mb)
                found = found or _table_cases(
                    source, path, family=family, rows=rows
                )
        except Exception as exc:  # noqa: BLE001 - report, never abort the run
            skipped.append({"source": source["id"], "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if not found:
            skipped.append({"source": source["id"], "reason": "no image or table HDU"})
            continue
        cases.extend(found)
        per_family[family] = per_family.get(family, 0) + 1
    return cases, skipped


def _methods(case: Case) -> dict[str, Callable[[], Any]]:
    """Reader closures for one case. Missing peers are simply absent."""
    path = str(case.path)
    methods: dict[str, Callable[[], Any]] = {}

    def torchfits_read() -> Any:
        if case.operation == "read_header":
            return torchfits.read_header(path, case.hdu)
        if case.operation == "read_full":
            return torchfits.read_tensor(path, case.hdu)
        if case.operation == "read_window":
            x0, y0, x1, y1 = case.window or (0, 0, WINDOW, WINDOW)
            return torchfits.read_subset(path, case.hdu, x0, y0, x1, y1)
        return tf_table.read(path, hdu=case.hdu, row_slice=(0, case.rows))

    def astropy_read() -> Any:
        if case.operation == "read_header":
            return astropy_fits.getheader(path, case.hdu)
        if case.operation == "read_full":
            return astropy_fits.getdata(path, case.hdu, memmap=False)
        if case.operation == "read_rows":
            # memmap: slicing touches the requested rows, not the 6.7 GB file.
            with astropy_fits.open(path, memmap=True) as hdul:
                return hdul[case.hdu].data[: case.rows]
        x0, y0, x1, y1 = case.window or (0, 0, WINDOW, WINDOW)
        with astropy_fits.open(path, memmap=True) as hdul:
            return np.array(hdul[case.hdu].data[y0:y1, x0:x1])

    methods["torchfits"] = torchfits_read
    methods["astropy"] = astropy_read
    try:
        import fitsio
    except ImportError:  # pragma: no cover - fitsio is a harness test dependency
        fitsio = None  # type: ignore[assignment]

    if fitsio is not None:

        def fitsio_read() -> Any:
            # fitsio's hdu numbers are 0-based like astropy's. Slicing the HDU
            # object avoids its `rows=`/`cols=` argument forms, whose tuple
            # semantics differ from the contiguous windows we need.
            with fitsio.FITS(path) as handle:
                hdu = handle[case.hdu]
                if case.operation == "read_header":
                    return hdu.read_header()
                if case.operation == "read_full":
                    return hdu.read()
                if case.operation == "read_window":
                    x0, y0, x1, y1 = case.window or (0, 0, WINDOW, WINDOW)
                    return hdu[y0:y1, x0:x1]
                return hdu[0 : case.rows]

        methods["fitsio"] = fitsio_read
    return methods


def _row(
    *,
    run_id: str,
    case: Case,
    method: str,
    time_s: float | None,
    peak_rss_mb: float | None,
    peak_cuda_mb: float | None,
    err: str | None,
) -> dict[str, Any]:
    status = "OK" if time_s is not None else "FAILED"
    library = method
    return {
        "run_id": run_id,
        "domain": "corpus",
        "suite": "corpus",
        "case_id": case.case_id,
        "case_label": case.label,
        # The orchestrator's deficit pass groups writes separately; keep the
        # operation distinct so a header read never ranks against pixel reads.
        "operation": case.operation,
        "family": case.family,
        "library": library,
        "method": method,
        "mode": "corpus",
        "status": status,
        "skip_reason": "" if err is None else err,
        "comparable": status == "OK",
        "mmap_target": "n/a",
        "host": _HOST,
        "time_s": time_s,
        "peak_rss_mb": peak_rss_mb,
        "peak_cuda_alloc_mb": peak_cuda_mb,
        "throughput": (case.size_mb / time_s) if time_s and case.size_mb else None,
        "unit": "MB/s",
        "size_mb": case.size_mb,
        "n_points": case.n_points,
        "metadata": json.dumps(case.metadata(), sort_keys=True),
    }


def run(
    *,
    tier: str = "full",
    families: tuple[str, ...] = BENCH_FAMILIES,
    limit: int = 3,
    max_mb: float = 128.0,
    rows: int = 200_000,
    runs: int = 3,
    warmup: int = 1,
    run_id: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Time every selected case; returns (rows, skipped sources)."""
    run_id = run_id or make_run_id()
    cases, skipped = select_cases(
        tier=tier, families=families, limit=limit, max_mb=max_mb, rows=rows
    )
    torchfits.clear_file_cache()
    out: list[dict[str, Any]] = []
    for case in cases:
        methods = _methods(case)
        print(
            f"[corpus] {case.case_id} peers={','.join(methods)} "
            f"runs={runs} size={case.size_mb:.1f}MB",
            flush=True,
        )
        timed = time_medians_interleaved(methods, runs=runs, warmup=warmup)
        for method, (time_s, peak_rss, peak_cuda, err) in timed.items():
            if err is not None:
                print(f"[corpus]   {method}: failed — {err}", flush=True)
            out.append(
                _row(
                    run_id=run_id,
                    case=case,
                    method=method,
                    time_s=time_s,
                    peak_rss_mb=peak_rss,
                    peak_cuda_mb=peak_cuda,
                    err=err,
                )
            )
    annotate_rankings(out)
    return out, skipped


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tier", choices=["ci", "full"], default="ci")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=BENCH_FAMILIES,
        default=list(BENCH_FAMILIES),
        help="corpus families to draw cases from",
    )
    parser.add_argument("--limit", type=int, default=2, help="files per family")
    parser.add_argument("--max-mb", type=float, default=128.0, help="full-image cap")
    parser.add_argument("--rows", type=int, default=200_000, help="table rows per case")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--list", action="store_true", help="print cases and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    families = tuple(args.families)
    cases, skipped = select_cases(
        tier=args.tier,
        families=families,
        limit=args.limit,
        max_mb=args.max_mb,
        rows=args.rows,
    )
    if skipped:
        print(f"[corpus] {len(skipped)} source(s) skipped:", flush=True)
        for entry in skipped:
            print(f"[corpus]   {entry['source']}: {entry['reason']}", flush=True)
    if args.list:
        for case in cases:
            print(
                f"{case.case_id:55s} {case.operation:12s} "
                f"hdu={case.hdu:<3d} {case.size_mb:8.2f} MB",
                flush=True,
            )
        return 0 if cases else 1
    if not cases:
        print(
            "[corpus] no cases to run; fetch the corpus with "
            "`pixi run corpus-fetch --tier ci`",
            flush=True,
        )
        return 1

    rows, skipped = run(
        tier=args.tier,
        families=families,
        limit=args.limit,
        max_mb=args.max_mb,
        rows=args.rows,
        runs=args.runs,
        warmup=args.warmup,
        run_id=args.run_id,
        output_dir=args.output_dir,
    )
    run_id = rows[0]["run_id"] if rows else (args.run_id or make_run_id())
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "results.csv", rows, RESULT_COLUMNS)
    write_summary(
        run_dir / "summary.md",
        run_id=run_id,
        scopes=[f"corpus:{args.tier}"],
        rows=rows,
        deficits=[],
    )
    failures = [row for row in rows if row["status"] != "OK"]
    print(
        f"Wrote {len(rows)} corpus rows to {run_dir / 'results.csv'}"
        f" ({len(failures)} failed)",
        flush=True,
    )
    if failures:
        for row in failures:
            print(
                f"[corpus] {row['case_id']} {row['method']}: {row['skip_reason']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:  # pragma: no cover - `| head`
        traceback.print_exc()
        os._exit(1)
