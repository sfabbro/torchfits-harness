"""Write-path parity: every output format read back by astropy and fitsio.

The read side is covered by ``test_corpus_parity.py`` and the upstream corpus.
Here the direction is reversed: torchfits writes, independent libraries read.
That is where a wrong ``BITPIX``, a leaked ``BSCALE``/``TZERO`` or a stale
``Z*`` card shows up, because the other library has no reason to agree.
"""

from __future__ import annotations

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

DTYPES = {
    torch.uint8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
    torch.float32: -32,
    torch.float64: -64,
    torch.uint16: 16,
    torch.uint32: 32,
}


@pytest.mark.parametrize("dtype,bitpix", DTYPES.items(), ids=[str(d) for d in DTYPES])
def test_bitpix_matrix_matches_astropy(
    dtype: torch.dtype, bitpix: int, tmp_path: Path
) -> None:
    path = tmp_path / f"write_{bitpix}_{dtype}.fits"
    data = (torch.arange(48, dtype=torch.float32).reshape(6, 8) * 3).to(dtype)
    torchfits.write(str(path), data, overwrite=True)

    with fits.open(path) as hdul:
        assert hdul[0].header["BITPIX"] == bitpix
        expected = np.asarray(hdul[0].data)
        assert expected.shape == (6, 8)
        np.testing.assert_array_equal(expected, data.numpy().astype(expected.dtype))

    assert torchfits.read_shape(str(path))[0] == bitpix
    np.testing.assert_array_equal(torchfits.read_tensor(str(path)).numpy(), data.numpy())

    import fitsio

    fitsio_data = fitsio.read(str(path))
    np.testing.assert_array_equal(fitsio_data, data.numpy().astype(fitsio_data.dtype))


def test_unsigned_convention_uses_bzero(tmp_path: Path) -> None:
    path = tmp_path / "unsigned.fits"
    data = torch.tensor([[0, 1, 65535], [32768, 40000, 12345]], dtype=torch.uint16)
    torchfits.write(str(path), data, overwrite=True)
    with fits.open(path) as hdul:
        header = hdul[0].header
        assert header["BITPIX"] == 16
        assert header["BZERO"] == 32768.0
        np.testing.assert_array_equal(np.asarray(hdul[0].data), data.numpy())
    # Re-reading physical values must not double-apply BZERO.
    np.testing.assert_array_equal(torchfits.read_tensor(str(path)).numpy(), data.numpy())


def test_mef_surgery_and_checksums(tmp_path: Path) -> None:
    path = tmp_path / "surgery.fits"
    primary = torch.arange(16, dtype=torch.int16).reshape(4, 4)
    first = torch.ones((5, 6), dtype=torch.float32)
    second = torch.full((3, 3), 7, dtype=torch.int32)

    torchfits.write(str(path), primary, header={"OBJECT": "M31"}, overwrite=True)
    torchfits.insert_hdu(str(path), first, index=1, header={"EXTNAME": "SCI"})
    torchfits.insert_hdu(str(path), second, index=2, header={"EXTNAME": "ERR"})
    torchfits.replace_hdu(
        str(path), "SCI", first * 2, header={"EXTNAME": "SCI", "BUNIT": "counts"}
    )
    torchfits.delete_hdu(str(path), "ERR")
    torchfits.write_checksums(str(path))

    report = torchfits.verify_checksums(str(path))
    assert report["ok"] is True

    with fits.open(path, checksum=True) as hdul:
        assert len(hdul) == 2
        assert hdul[0].header["OBJECT"] == "M31"
        assert hdul[1].header["EXTNAME"] == "SCI"
        assert hdul[1].header["BUNIT"] == "counts"
        np.testing.assert_array_equal(np.asarray(hdul[1].data), (first * 2).numpy())
        hdul.verify("exception")
        assert "CHECKSUM" in hdul[0].header


@pytest.mark.parametrize("algorithm", ["RICE_1", "GZIP_1", "HCOMPRESS_1"])
def test_image_compression_roundtrips_through_astropy(
    algorithm: str, tmp_path: Path
) -> None:
    path = tmp_path / f"compressed_{algorithm}.fits.fz"
    # Smooth data keeps HCOMPRESS quantization error small.
    y, x = np.mgrid[0:64, 0:64]
    data = torch.from_numpy((100 + x + 3 * y).astype(np.int16))
    torchfits.write(str(path), data, compress=algorithm, overwrite=True)

    with fits.open(path) as hdul:
        # A compressed image is stored in a BINTABLE extension, so the empty
        # primary stays at 0 and the payload lives at 1.
        assert len(hdul) == 2
        assert isinstance(hdul[1], fits.CompImageHDU)
        got = np.asarray(hdul[1].data)
    assert torchfits.read_header(str(path), 1)["ZCMPTYPE"] == algorithm
    expected = data.numpy()
    if algorithm in {"RICE_1", "GZIP_1"}:
        np.testing.assert_array_equal(got, expected)
    else:
        assert np.max(np.abs(got.astype(np.int64) - expected.astype(np.int64))) <= 8

    np.testing.assert_array_equal(
        torchfits.read_tensor(str(path), hdu=1).numpy(), got
    )


def test_plio_compression_roundtrips_through_astropy(tmp_path: Path) -> None:
    path = tmp_path / "plio.fits.fz"
    mask = torch.zeros((32, 32), dtype=torch.uint8)
    mask[4:12, 5:9] = 1
    mask[20:28, 20:26] = 1
    torchfits.write(str(path), mask, compress="PLIO_1", overwrite=True)
    with fits.open(path) as hdul:
        assert isinstance(hdul[1], fits.CompImageHDU)
        np.testing.assert_array_equal(np.asarray(hdul[1].data), mask.numpy())


def test_quantize_robust_roundtrips_through_astropy(tmp_path: Path) -> None:
    path = tmp_path / "quantized.fits"
    rng = np.random.default_rng(11)
    data = torch.from_numpy((rng.normal(size=(64, 64)) * 1000).astype(np.float32))
    torchfits.write(str(path), data, quantize="robust", overwrite=True)
    with fits.open(path) as hdul:
        assert hdul[0].header["BITPIX"] == 16
        assert "BSCALE" in hdul[0].header
        got = np.asarray(hdul[0].data)
    # Robust quantization clips a handful of extreme samples by design; the
    # bulk of the distribution must land within the scale step.
    diff = np.abs(got.astype(np.float64) - data.numpy().astype(np.float64))
    assert float(np.mean(diff <= 5.0)) >= 0.99
    np.testing.assert_allclose(
        torchfits.read_tensor(str(path)).numpy(), got, rtol=1e-6, atol=1e-3
    )


def _real_source(source_id: str) -> dict[str, Any]:
    for source in corpus.sources(tier="full"):
        if source["id"] == source_id:
            return source
    raise AssertionError(source_id)


def test_real_table_rewrite_matches_astropy(tmp_path: Path) -> None:
    source = _real_source("sdss_plates_dr18")
    path = corpus.require(source)
    arrow = tf_table.read(str(path), hdu=1, row_slice=(0, 128))
    columns = {
        name: np.asarray(arrow[name].to_numpy(zero_copy_only=False))
        for name in arrow.column_names
    }
    out = tmp_path / "plates_copy.fits"
    tf_table.write(str(out), columns, overwrite=True, extname="PLATE")

    with fits.open(str(path), memmap=True) as hdul:
        expected = hdul[1].data[:128]
    with fits.open(out) as hdul:
        got = hdul[1].data
        assert hdul[1].header["EXTNAME"] == "PLATE"
        for name in expected.columns.names:
            a = np.asarray(got[name])
            b = np.asarray(expected[name])
            if b.dtype.kind in "SU":
                assert [str(v).rstrip() for v in a] == [str(v).rstrip() for v in b]
            else:
                np.testing.assert_allclose(
                    a.astype(np.float64),
                    b.astype(np.float64),
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                    err_msg=name,
                )


def test_real_image_cutout_rewrite_matches_astropy(tmp_path: Path) -> None:
    source = _real_source("sdss_manga_logrss_8485_12701")
    path = corpus.require(source)
    with fits.open(str(path)) as hdul:
        header = hdul["FLUX"].header
        cube = np.asarray(hdul["FLUX"].data)
        plane = cube[:, 100:164]
        expected = plane[300:364, :]
    out = tmp_path / "manga_plane.fits"
    torchfits.write(
        str(out),
        torch.from_numpy(expected.astype(np.float32)),
        header={"EXTNAME": "FLUX", "BUNIT": header.get("BUNIT", "")},
        overwrite=True,
    )
    with fits.open(out) as hdul:
        got = np.asarray(hdul[0].data)
        assert hdul[0].header["EXTNAME"] == "FLUX"
    np.testing.assert_array_equal(got, expected.astype(np.float32))
    np.testing.assert_array_equal(torchfits.read_tensor(str(out)).numpy(), got)
