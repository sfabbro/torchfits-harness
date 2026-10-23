"""Compare torchfits values with pure CFITSIO, with no Python FITS layer between.

The oracle is ``benchmarks/cfitsio_direct/dump_cfitsio_values.c``: a small C
program linked against the same vendored CFITSIO that torchfits bundles. It
prints raw values (``fits_read_subset`` / ``fits_read_col``) and the test diffs
them against torchfits. astropy never participates, so a torchfits/astropy
agreement cannot hide a shared Python-level assumption.

The vendored tree lives in a torchfits checkout, which this repo deliberately
does not vendor: without one (``TORCHFITS_SOURCE`` or a sibling ``torchfits``)
these tests skip.
"""

from __future__ import annotations

import subprocess
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

BUILD_SCRIPT = ROOT / "scripts" / "build_cfitsio_oracle.py"


@pytest.fixture(scope="session")
def oracle() -> Path:
    proc = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if proc.returncode == 3:
        pytest.skip("no vendored CFITSIO in a torchfits checkout")
    if proc.returncode != 0:
        pytest.skip(f"CFITSIO oracle build failed: {proc.stderr.strip()[:400]}")
    return Path(proc.stdout.strip())


def _run(oracle: Path, *args: str) -> tuple[list[str], np.ndarray]:
    proc = subprocess.run(
        [str(oracle), *args], capture_output=True, text=True, cwd=ROOT
    )
    if proc.returncode != 0:
        pytest.fail(f"oracle {args} failed: {proc.stderr.strip()}")
    header: list[str] = []
    values: list[float] = []
    for line in proc.stdout.splitlines():
        if line.startswith("#"):
            header.append(line[1:].strip())
        elif line.strip():
            values.append(float(line) if line.strip() != "nan" else np.nan)
    return header, np.asarray(values, dtype=np.float64)


def _write_synthetic(path: Path) -> None:
    rng = np.random.default_rng(7)
    data = (rng.normal(size=(64, 80)) * 100).astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header["BSCALE"] = 0.5
    hdu.header["BZERO"] = 10.0
    ints = (rng.normal(size=(32, 40)) * 50).astype(np.int16)
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="I32", format="J", array=np.arange(100, dtype=np.int32)),
            fits.Column(name="F32", format="E", array=rng.normal(size=100).astype(np.float32)),
            fits.Column(
                name="F64", format="D", array=rng.normal(size=100).astype(np.float64)
            ),
        ],
        name="CAT",
    )
    fits.HDUList([hdu, fits.ImageHDU(ints, name="RAW"), table]).writeto(path, overwrite=True)


def test_scaled_image_matches_pure_cfitsio(oracle: Path, tmp_path: Path) -> None:
    path = tmp_path / "oracle_scaled.fits"
    _write_synthetic(path)
    header, expected = _run(oracle, "image", str(path), "0", "0", "0", "80", "64")
    assert header[0].startswith("shape 64 80")
    got = torchfits.read(str(path), hdu=0).numpy()
    assert got.shape == (64, 80)
    # torchfits scales in float32 for float payloads; CFITSIO converts to
    # double first, so the agreement is single-precision, not bit-exact.
    np.testing.assert_allclose(
        got.astype(np.float64), expected.reshape(64, 80), rtol=1e-6, atol=1e-5
    )


def test_unsigned_scaled_image_matches_pure_cfitsio(oracle: Path, tmp_path: Path) -> None:
    """The BZERO=32768 MegaCam convention, checked without astropy."""
    path = tmp_path / "oracle_unsigned.fits"
    raw = np.arange(48, dtype=np.int16).reshape(6, 8)
    fits.PrimaryHDU(raw).writeto(path, overwrite=True)
    with fits.open(path, mode="update") as hdul:
        hdul[0].header["BZERO"] = 32768.0
        hdul[0].header["BSCALE"] = 1.0
    header, expected = _run(oracle, "image", str(path), "0", "0", "0", "8", "6")
    assert header[0].startswith("shape 6 8")
    got = np.asarray(torchfits.read(str(path), hdu=0))
    assert got.dtype.kind in "iu", got.dtype
    np.testing.assert_array_equal(got, expected.reshape(6, 8).astype(got.dtype))


def test_cutout_matches_pure_cfitsio(oracle: Path, tmp_path: Path) -> None:
    path = tmp_path / "oracle_cutout.fits"
    _write_synthetic(path)
    header, expected = _run(oracle, "image", str(path), "1", "23", "17", "11", "9")
    assert header[0].startswith("shape 9 11")
    got = torchfits.read_subset(str(path), 1, 23, 17, 34, 26).numpy()
    np.testing.assert_array_equal(
        got, expected.reshape(9, 11).astype(got.dtype)
    )


def test_table_columns_match_pure_cfitsio(oracle: Path, tmp_path: Path) -> None:
    path = tmp_path / "oracle_table.fits"
    _write_synthetic(path)
    with fits.open(path) as hdul:
        columns = list(hdul[2].data.columns.names)
    cols = tf_table.read_torch(str(path), hdu=2, columns=columns, start_row=1, num_rows=25)
    for name in columns:
        _, expected = _run(oracle, "table", str(path), "2", name, "0", "25")
        got = cols[name].numpy().astype(np.float64)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6, err_msg=name)


def _present_image_sources() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in corpus.sources(tier="full"):
        if "image" not in corpus.families(source):
            continue
        if not corpus.source_available(source):
            continue
        out.append(source)
    return out[:3]


@pytest.mark.parametrize("source", _present_image_sources(), ids=lambda s: s["id"])
def test_real_corpus_image_matches_pure_cfitsio(
    oracle: Path, source: dict[str, Any]
) -> None:
    path = corpus.require(source)
    if isinstance(path, list):
        pytest.skip("multi-file source")
    with fits.open(path) as hdul:
        image_indices = [
            i
            for i, hdu in enumerate(hdul)
            if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
            and hdu.data is not None
            and hdu.header.get("NAXIS", 0) >= 2
        ]
        if not image_indices:
            pytest.skip("no image HDU")
        index = image_indices[0]
        nx_full = int(hdul[index].header["NAXIS1"])
        ny_full = int(hdul[index].header["NAXIS2"])
        x0 = max(0, nx_full // 2 - 32)
        y0 = max(0, ny_full // 2 - 32)
        nx = min(64, nx_full - x0)
        ny = min(64, ny_full - y0)
        header, expected = _run(
            oracle, "image", str(path), str(index), str(x0), str(y0), str(nx), str(ny)
        )
        got = torchfits.read_subset(str(path), index, x0, y0, x0 + nx, y0 + ny).numpy()
        assert got.shape == (ny, nx), (header, got.shape)
        np.testing.assert_allclose(
            got.astype(np.float64),
            expected.reshape(ny, nx),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )
