"""Value parity on real archive files, with astropy and fitsio as oracles.

The corpus is fetched by ``scripts/fetch_corpus.py`` and is gitignored, so every
test skips cleanly until the files are present. Set
``TORCHFITS_CORPUS_TIER=ci`` to keep a run to the ~8 MB tier.

What is compared, per HDU:

* HDU count, type string and EXTNAME;
* image shape and pixels, in full when small enough and otherwise as a central
  window read through ``read_subset``;
* table row count, column names/order and column values over the first and last
  row windows (never the whole 6.7 GB catalog);
* header keyword sequence and values.

Column differences that are legitimate FITS semantics are normalized rather
than skipped: fixed-width strings are right-stripped (``strip=True`` is the
public default) and scaled/null sentinels are compared as NaN.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torchfits  # noqa: E402
import torchfits.table as tf_table  # noqa: E402
from astropy.io import fits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import corpus  # noqa: E402

TIER = os.environ.get("TORCHFITS_CORPUS_TIER", "full")
SOURCES = [
    source
    for source in corpus.sources(tier=TIER)
    if source["kind"] != "astropy"  # covered by test_astropy_corpus_diff.py
]

#: Above this payload we compare a central window instead of the whole image.
FULL_IMAGE_LIMIT = 64 * 1024**2
#: Above this file size only a sample of HDUs is compared.
HUGE_FILE = 512 * 1024**2
#: Rows compared at each end of a table.
ROWS = 2000


def _sample_indices(hdul: Any, path: Path) -> list[int]:
    """HDUs worth opening: every one, unless each open costs a decompression."""
    expensive = str(path).endswith(".gz") or Path(path).stat().st_size > HUGE_FILE
    if not expensive:
        return list(range(len(hdul)))
    indices = [0]
    tables = [
        i for i, h in enumerate(hdul) if isinstance(h, (fits.BinTableHDU, fits.TableHDU))
    ]
    images = [
        i for i, h in enumerate(hdul) if h.data is not None and i not in tables
    ]
    if tables:
        indices.append(tables[0])
    if images:
        indices.append(images[len(images) // 2])
    if len(hdul) - 1 not in indices:
        indices.append(len(hdul) - 1)
    return indices


@pytest.mark.parametrize("source", SOURCES, ids=[s["id"] for s in SOURCES])
def test_hdu_inventory_matches_astropy(source: dict[str, Any]) -> None:
    path = corpus.require(source)
    if isinstance(path, list):
        pytest.skip("multi-file source")
    with fits.open(path) as hdul:
        assert torchfits.read_num_hdus(str(path)) == len(hdul)
        for index in _sample_indices(hdul, Path(path)):
            hdu = hdul[index]
            expected = {
                fits.PrimaryHDU: "IMAGE",
                fits.ImageHDU: "IMAGE",
                fits.CompImageHDU: "IMAGE",
                fits.GroupsHDU: "IMAGE",
                fits.BinTableHDU: "BINARY_TABLE",
                fits.TableHDU: "ASCII_TABLE",
            }.get(type(hdu))
            if expected is None:
                continue
            assert torchfits.read_hdu_type(str(path), index) == expected, index
            extname = (hdu.header.get("EXTNAME") or "").strip() or None
            assert torchfits.read_extname(str(path), index) == extname, index


def _header_cards(header: Any) -> list[tuple[str, Any]]:
    """Non-blank cards in file order (COMMENT/HISTORY included)."""
    cards: list[tuple[str, Any]] = []
    for card in header.cards:
        key = getattr(card, "keyword", None) or getattr(card, "key", "")
        key = (key or "").strip()
        if not key or key == "CONTINUE":
            continue
        cards.append((key, card.value))
    return cards


@pytest.mark.parametrize("source", SOURCES, ids=[s["id"] for s in SOURCES])
def test_header_cards_match_astropy(source: dict[str, Any]) -> None:
    path = corpus.require(source)
    if isinstance(path, list):
        pytest.skip("multi-file source")
    # Raw view: torchfits' Header is the on-disk one, and for tile-compressed
    # HDUs astropy's default view rewrites it into the decompressed image's.
    with fits.open(path, disable_image_compression=True, memmap=True) as hdul:
        for index in _sample_indices(hdul, Path(path)):
            hdu = hdul[index]
            got = _header_cards(torchfits.read_header(str(path), index))
            expected = _header_cards(hdu.header)
            assert [key for key, _ in got] == [key for key, _ in expected], index
            for (key, a), (_, b) in zip(got, expected, strict=True):
                if isinstance(a, str) and isinstance(b, str):
                    # COMMENT/HISTORY values keep the card's post-key padding in
                    # astropy and drop it in torchfits; compare the text itself.
                    assert a.strip() == b.strip(), (index, key, a, b)
                elif isinstance(b, float) and math.isnan(b):
                    assert isinstance(a, float) and math.isnan(a), (index, key)
                else:
                    assert a == b, (index, key, a, b)


@pytest.mark.parametrize("source", SOURCES, ids=[s["id"] for s in SOURCES])
def test_image_pixels_match_astropy(source: dict[str, Any]) -> None:
    path = corpus.require(source)
    if isinstance(path, list):
        pytest.skip("multi-file source")
    size = Path(path).stat().st_size
    with fits.open(path) as hdul:
        image_indices = [
            index
            for index, hdu in enumerate(hdul)
            if hdu.data is not None and not isinstance(hdu, fits.BinTableHDU | fits.TableHDU)
        ]
        if not image_indices:
            pytest.skip("no image HDU")
        if len(image_indices) > 12 or size > HUGE_FILE or str(path).endswith(".gz"):
            sampled = _sample_indices(hdul, Path(path))
            image_indices = [i for i in sampled if i in set(image_indices)] or [
                image_indices[0]
            ]
        for index in image_indices:
            expected = np.asarray(hdul[index].data)
            nbytes = expected.size * expected.dtype.itemsize
            if nbytes > FULL_IMAGE_LIMIT:
                y0 = max(0, expected.shape[0] // 2 - 128)
                x0 = max(0, expected.shape[-1] // 2 - 128)
                y1 = min(expected.shape[0], y0 + 256)
                x1 = min(expected.shape[-1], x0 + 256)
                got = torchfits.read_subset(str(path), index, x0, y0, x1, y1).numpy()
                expected = expected[y0:y1, x0:x1]
            else:
                got = torchfits.read_tensor(str(path), index).numpy()
            assert got.shape == expected.shape, (index, got.shape, expected.shape)
            if expected.dtype.kind in "iu" and got.dtype.kind in "iu":
                np.testing.assert_array_equal(got, expected, err_msg=f"hdu {index}")
            else:
                np.testing.assert_allclose(
                    got.astype(np.float64),
                    expected.astype(np.float64),
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                    err_msg=f"hdu {index}",
                )


def _normalize_expected_column(column: Any) -> Any:
    if isinstance(column, np.ma.MaskedArray):
        if column.dtype.kind in "fc":
            return np.ma.filled(column.astype(np.float64), np.nan)
        if column.dtype.kind in "iu":
            return np.ma.filled(column.astype(np.float64), np.nan)
        return np.array(
            [None if m else str(v).rstrip() for v, m in zip(column.data, column.mask, strict=True)],
            dtype=object,
        )
    return column


def _assert_column(
    name: str, got: Any, expected: Any, *, hdu: int, where: str
) -> None:
    if isinstance(expected, (list, tuple)) or got.dtype == object:
        got_values = got.tolist() if hasattr(got, "tolist") else list(got)
        expected_values = list(expected)
        assert len(got_values) == len(expected_values), (hdu, name, where)
        for i, (a, b) in enumerate(zip(got_values, expected_values, strict=True)):
            if isinstance(b, (list, tuple, np.ndarray)):
                np.testing.assert_allclose(
                    np.asarray(a, dtype=np.float64),
                    np.asarray(b, dtype=np.float64),
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                    err_msg=f"hdu {hdu} col {name} row {i}",
                )
            elif isinstance(b, str):
                assert (a or "").rstrip() == b.rstrip(), (hdu, name, i, a, b)
            elif b is None or (isinstance(b, float) and math.isnan(b)):
                assert a is None or (isinstance(a, float) and math.isnan(a)), (
                    hdu,
                    name,
                    i,
                    a,
                    b,
                )
            else:
                assert a == b, (hdu, name, i, a, b)
        return

    if expected.dtype.kind in "SU":
        assert got.dtype.kind in "SUO", (hdu, name, got.dtype)
        got_s = np.asarray([str(v).rstrip() for v in got], dtype=object)
        exp_s = np.asarray([str(v).rstrip() for v in expected], dtype=object)
        np.testing.assert_array_equal(got_s, exp_s, err_msg=f"hdu {hdu} col {name}")
        return

    got_f = np.asarray(got, dtype=np.float64)
    exp_f = np.asarray(expected, dtype=np.float64)
    assert got_f.shape == exp_f.shape, (hdu, name, got_f.shape, exp_f.shape)
    np.testing.assert_allclose(
        got_f,
        exp_f,
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
        err_msg=f"hdu {hdu} col {name} ({where})",
    )


@pytest.mark.parametrize("source", SOURCES, ids=[s["id"] for s in SOURCES])
def test_table_values_match_astropy(source: dict[str, Any]) -> None:
    path = corpus.require(source)
    if isinstance(path, list):
        pytest.skip("multi-file source")
    with fits.open(path, memmap=True) as hdul:
        for index, hdu in enumerate(hdul):
            if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                continue
            data = hdu.data
            nrows = len(data)
            assert torchfits.read_nrows(str(path), index) == nrows
            assert torchfits.read_colnames(str(path), index) == list(data.columns.names)

            windows = [(0, min(ROWS, nrows))]
            if nrows > ROWS:
                windows.append((max(0, nrows - ROWS), nrows))
            for start, stop in windows:
                arrow = tf_table.read(str(path), hdu=index, row_slice=(start, stop))
                assert arrow.num_rows == stop - start
                for name in data.columns.names:
                    expected = _normalize_expected_column(data[name][start:stop])
                    got = arrow[name].to_numpy(zero_copy_only=False)
                    _assert_column(
                        name, got, expected, hdu=index, where=f"{start}:{stop}"
                    )


@pytest.mark.parametrize("source", SOURCES, ids=[s["id"] for s in SOURCES])
def test_numeric_columns_agree_with_fitsio(source: dict[str, Any]) -> None:
    """fitsio is a second, independent CFITSIO binding; compare plain columns."""
    path = corpus.require(source)
    if isinstance(path, list):
        pytest.skip("multi-file source")
    if str(path).endswith(".gz"):
        pytest.skip("fitsio does not open gzip containers")
    with fits.open(path, memmap=True) as hdul:
        for index, hdu in enumerate(hdul):
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            data = hdu.data
            nrows = len(data)
            if nrows == 0:
                continue
            stop = min(500, nrows)
            try:
                import fitsio

                table = fitsio.read(str(path), ext=index + 1, rows=(0, stop))
            except Exception as exc:  # noqa: BLE001 - comparator availability
                pytest.skip(f"fitsio cannot read hdu {index}: {exc}")
            torch_cols = tf_table.read_torch(str(path), hdu=index, start_row=1, num_rows=stop)
            for name in data.columns.names:
                if name not in torch_cols or name not in table.dtype.names:
                    continue
                expected = np.asarray(table[name])
                if expected.dtype.kind not in "iuf":
                    continue
                got = torch_cols[name].numpy()
                np.testing.assert_allclose(
                    np.asarray(got, dtype=np.float64),
                    np.asarray(expected, dtype=np.float64),
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                    err_msg=f"hdu {index} col {name}",
                )
