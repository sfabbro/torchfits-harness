#!/usr/bin/env python
"""Per-call cost of the public torchfits API, next to the binding underneath it.

A slow benchmark can mean slow I/O or slow Python. This script separates the
two by timing each public entry point beside the nanobind function it calls, on
the same warm file, in the same process:

    public call          binding call
    torchfits.read_header  _C.read_header_dict
    torchfits.read_tensor  _C.read_full
    torchfits.table.read   _C.read_fits_table

The difference is the Python layer: argument normalisation, cache lookups and
object construction. It is the part a pure-I/O optimisation cannot touch, which
is why it is measured on its own.

    pixi run api-overhead
    pixi run api-overhead -- --json reports/api-overhead.json --number 2000

Files default to the CI corpus (an image and a table); point --path/--table at
anything else to measure a different payload. Nothing is written unless --json
is given.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import timeit
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torchfits  # noqa: E402
from torchfits import _C  # noqa: E402
from torchfits import table as tf_table  # noqa: E402


def _first_corpus_file(family: str) -> Path | None:
    from benchmarks import corpus

    for source in corpus.sources(tier="ci"):
        if family not in corpus.families(source):
            continue
        files = corpus.source_files(source)
        if files and all(path.is_file() for path in files):
            return files[0]
    return None


def _us(fn: Callable[[], Any], *, number: int, repeat: int = 5) -> float:
    """Median µs per call, after one untimed warm-up call."""
    fn()
    timer = timeit.Timer(fn)
    samples = [elapsed / number for elapsed in timer.repeat(repeat=repeat, number=number)]
    return statistics.median(samples) * 1e6


def _measure(
    rows: list[dict[str, Any]],
    *,
    layer: str,
    call: str,
    fn: Callable[[], Any],
    number: int,
    note: str = "",
) -> float:
    us = _us(fn, number=number)
    rows.append({"layer": layer, "call": call, "us_per_call": us, "note": note})
    return us


def measure(
    *,
    image: Path,
    table: Path,
    number: int = 1000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    image_path = str(image)
    table_path = str(table)

    # -- header: the public call builds a Header object every time ----------
    _measure(
        rows,
        layer="header",
        call="torchfits.read_header",
        fn=lambda: torchfits.read_header(image_path, 0),
        number=number,
    )
    _measure(
        rows,
        layer="header",
        call="_C.read_header_dict",
        fn=lambda: _C.read_header_dict(image_path, 0),
        number=number,
        note="binding only; no Header object",
    )

    header = torchfits.read_header(image_path, 0)
    key = "NAXIS1" if "NAXIS1" in header else next(iter(header.keys()))
    plain = {card.key: card.value for card in header.cards}
    _measure(
        rows,
        layer="header",
        call="Header.get",
        fn=lambda: header.get(key),
        number=number * 10,
        note="pure Python, warm object",
    )
    _measure(
        rows,
        layer="header",
        call="dict.get",
        fn=lambda: plain.get(key),
        number=number * 10,
        note="baseline for the same lookup",
    )
    _measure(
        rows,
        layer="header",
        call="Header.__getitem__",
        fn=lambda: header[key],
        number=number * 10,
    )

    # -- image: public read vs the raw tensor/array builder -----------------
    _measure(
        rows,
        layer="header",
        call="clear_file_cache + read_header",
        fn=lambda: (torchfits.clear_file_cache(), torchfits.read_header(image_path, 0)),
        number=max(5, number // 20),
        note="uncached public path: open + parse + Header",
    )

    _measure(
        rows,
        layer="image",
        call="torchfits.read_tensor",
        fn=lambda: torchfits.read_tensor(image_path, 0),
        number=max(10, number // 20),
    )
    _measure(
        rows,
        layer="image",
        call="_C.read_full",
        fn=lambda: _C.read_full(image_path, 0),
        number=max(10, number // 20),
        note="binding only; no wrapper dispatch",
    )
    _measure(
        rows,
        layer="image",
        call="torchfits.read_num_hdus",
        fn=lambda: torchfits.read_num_hdus(image_path),
        number=number,
    )
    _measure(
        rows,
        layer="image",
        call="_C.read_num_hdus",
        fn=lambda: _C.read_num_hdus(image_path),
        number=number,
        note="binding only",
    )

    # -- table: raw extraction vs the public Arrow build -------------------
    # Same 1000-row window through both layers, so the difference is the
    # Python-side schema handling and Arrow conversion, not I/O.
    _measure(
        rows,
        layer="table",
        call="_C.read_fits_table_rows_raw",
        fn=lambda: _C.read_fits_table_rows_raw(table_path, 1, [], 1, 1000),
        number=max(5, number // 20),
        note="binding only; numpy dict, 1000 rows",
    )
    _measure(
        rows,
        layer="table",
        call="torchfits.table.read (all cols)",
        fn=lambda: tf_table.read(table_path, hdu=1, row_slice=(0, 1000)),
        number=max(3, number // 100),
        note="public Arrow build, 1000 rows",
    )
    sample = tf_table.read(table_path, hdu=1, row_slice=(0, 1000))
    numeric = [f.name for f in sample.schema if "string" not in str(f.type)][:5]
    strings = [f.name for f in sample.schema if "string" in str(f.type)][:5]
    if numeric:
        _measure(
            rows,
            layer="table",
            call="torchfits.table.read (5 numeric)",
            fn=lambda: tf_table.read(
                table_path, hdu=1, row_slice=(0, 1000), columns=numeric
            ),
            number=max(3, number // 50),
        )
    if strings:
        _measure(
            rows,
            layer="table",
            call="torchfits.table.read (5 string)",
            fn=lambda: tf_table.read(
                table_path, hdu=1, row_slice=(0, 1000), columns=strings
            ),
            number=max(3, number // 50),
            note="fixed-width strings are decoded in Python",
        )

    # -- handle layer -------------------------------------------------------
    _measure(
        rows,
        layer="handle",
        call="torchfits.open + len",
        fn=lambda: len(torchfits.open(image_path)),
        number=max(5, number // 20),
    )
    return rows


def _ratio(rows: list[dict[str, Any]], left: str, right: str) -> str:
    by_call = {row["call"]: row for row in rows}
    if left not in by_call or right not in by_call:
        return ""
    ratio = by_call[left]["us_per_call"] / max(by_call[right]["us_per_call"], 1e-9)
    return f"{ratio:5.1f}×"


def _print(rows: list[dict[str, Any]]) -> None:
    width = max(len(row["call"]) for row in rows)
    layer = ""
    for row in rows:
        if row["layer"] != layer:
            layer = row["layer"]
            print(f"\n{layer}")
        note = f"  # {row['note']}" if row["note"] else ""
        print(f"  {row['call']:<{width}}  {row['us_per_call']:9.2f} µs{note}")

    print("\nratios (public / binding):")
    for public, raw in (
        ("torchfits.read_header", "_C.read_header_dict"),
        ("torchfits.read_tensor", "_C.read_full"),
        ("torchfits.read_num_hdus", "_C.read_num_hdus"),
        ("torchfits.table.read (all cols)", "_C.read_fits_table_rows_raw"),
    ):
        ratio = _ratio(rows, public, raw)
        if ratio:
            print(f"  {public:<34} {ratio}")
    print(
        "  (read_header is < 1× because the public path caches parsed "
        "headers; the binding is uncached)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, help="image file (default: CI corpus)")
    parser.add_argument("--table", type=Path, help="table file (default: CI corpus)")
    parser.add_argument("--number", type=int, default=1000, help="calls per sample")
    parser.add_argument("--json", type=Path, help="write the measurements here")
    args = parser.parse_args(argv)

    image = args.path or _first_corpus_file("image")
    table = args.table or _first_corpus_file("table")
    if image is None or table is None:
        print(
            "no corpus file to measure; pass --path and --table, or fetch the "
            "corpus with `pixi run corpus-fetch --tier ci`",
            file=sys.stderr,
        )
        return 1

    rows = measure(image=image, table=table, number=args.number)
    print(f"image: {image}\ntable: {table}\ncalls per sample: {args.number}")
    _print(rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"image": str(image), "table": str(table), "rows": rows}, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
