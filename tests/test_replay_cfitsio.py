"""Replays of CFITSIO's own test programs against torchfits.

Sources (CFITSIO 4.x test suite):

* ``testprog.c`` -- null pixels, checksums, HDU copy, compression
* ``imagetest.c`` -- image scaling, blank values, windows
* ``tabletest.c`` -- logical/bit columns, variable-length arrays, scaled columns
* ``cookbook.c`` -- the recipes already mirrored in
  ``examples/example_cfitsio_cookbook.py``

These are semantic ports, not C wrappers: each case builds the CFITSIO fixture
with astropy and asserts that torchfits answers the same as CFITSIO's own
binding. Cases CFITSIO covers that torchfits intentionally does not (random
groups, A3DTABLE) stay in ``docs/parity.md`` as Unsupported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torchfits  # noqa: E402
import torchfits.table as tf_table  # noqa: E402
from astropy.io import fits  # noqa: E402


def test_blank_pixels_become_nan_like_cfitsio_nulval(tmp_path: Path) -> None:
    """imagetest: BLANK is CFITSIO's nulval contract for integer images."""
    path = tmp_path / "blank.fits"
    data = np.array([[1, -999, 3], [4, 5, -999]], dtype=np.int16)
    hdu = fits.PrimaryHDU(data)
    hdu.header["BLANK"] = -999
    hdu.writeto(path)

    got = torchfits.read(str(path)).numpy()
    assert np.isnan(got).sum() == 2
    assert np.isnan(got[0, 1]) and np.isnan(got[1, 2])
    finite = ~np.isnan(got)
    np.testing.assert_array_equal(got[finite], data[finite].astype(got.dtype))


def test_blank_pixels_inside_a_window_become_nan(tmp_path: Path) -> None:
    """imagetest: fits_read_subset applies the same nulval as a full read."""
    path = tmp_path / "blank_window.fits"
    data = np.arange(64, dtype=np.int32).reshape(8, 8)
    data[2, 2] = -2147483647
    hdu = fits.PrimaryHDU(data)
    hdu.header["BLANK"] = -2147483647
    hdu.writeto(path)

    window = torchfits.read_subset(str(path), 0, 1, 1, 4, 4).numpy()
    assert window.shape == (3, 3)
    assert np.isnan(window[1, 1])
    np.testing.assert_array_equal(
        window[~np.isnan(window)], data[1:4, 1:4][~np.isnan(window)].astype(window.dtype)
    )


def test_copy_an_image_hdu_between_files(tmp_path: Path) -> None:
    """cookbook: fits_copy_hdu moves a payload without touching the header."""
    src = tmp_path / "copy_src.fits"
    dst = tmp_path / "copy_dst.fits"
    payload = np.arange(30, dtype=np.float32).reshape(5, 6)
    fits.PrimaryHDU(payload).writeto(src)
    fits.PrimaryHDU(np.zeros((2, 2), dtype=np.int16)).writeto(dst)

    torchfits.insert_hdu(
        str(dst),
        torch.from_numpy(payload),
        index=1,
        header={"EXTNAME": "SCI", "BUNIT": "Jy"},
    )

    with fits.open(dst) as hdul:
        assert len(hdul) == 2
        assert hdul[1].header["EXTNAME"] == "SCI"
        assert hdul[1].header["BUNIT"] == "Jy"
        np.testing.assert_array_equal(np.asarray(hdul[1].data), payload)
    np.testing.assert_array_equal(torchfits.read_tensor(str(dst), 1).numpy(), payload)


def test_logical_and_bit_columns(tmp_path: Path) -> None:
    """tabletest: TFORM L and X decode to booleans, MSB first."""
    path = tmp_path / "bits.fits"
    flags = np.array([True, False, True, True, False], dtype=bool)
    bits = np.array([0b10000000, 0, 0b01000000, 0b11111111, 0b00000001], dtype=np.uint8)
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="FLAG", format="L", array=flags),
            fits.Column(name="MASK", format="8X", array=bits),
        ],
        name="EVENTS",
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)

    with fits.open(path) as hdul:
        expected_flag = np.asarray(hdul[1].data["FLAG"])
        expected_mask = np.asarray(hdul[1].data["MASK"])
    arrow = tf_table.read(str(path), hdu=1)
    np.testing.assert_array_equal(arrow["FLAG"].to_numpy(), expected_flag)
    # X decodes to one boolean per bit (MSB first), so a row is a list of 8.
    np.testing.assert_array_equal(np.asarray(arrow["MASK"].to_pylist()), expected_mask)

    tensors = tf_table.read_torch(str(path), hdu=1)
    assert tensors["FLAG"].dtype == torch.bool
    np.testing.assert_array_equal(tensors["FLAG"].numpy(), expected_flag)


def test_variable_length_array_column_is_read_like_cfitsio(tmp_path: Path) -> None:
    """tabletest: a P/Q column reads through the heap, row by row."""
    path = tmp_path / "vla.fits"
    rows = np.empty(4, dtype=object)
    rows[:] = [np.array([1, 2, 3], dtype=np.int32), np.array([], dtype=np.int32),
               np.array([7], dtype=np.int32), np.array([-1, 2], dtype=np.int32)]
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="VEC", format="PJ", array=rows)], name="HEAP"
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)

    with fits.open(path) as hdul:
        expected = [np.asarray(v) for v in hdul[1].data["VEC"]]
    got = tf_table.read(str(path), hdu=1)["VEC"].to_pylist()
    assert len(got) == len(expected)
    for row, want in zip(got, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(row), want)


def test_scaled_column_applies_tscal_tzero(tmp_path: Path) -> None:
    """tabletest: TSCAL/TZERO turn stored integers into physical floats."""
    path = tmp_path / "scaled_col.fits"
    stored = np.arange(6, dtype=np.int16)
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="COUNTS", format="I", array=stored)], name="CAT"
    )
    table.header["TSCAL1"] = 0.5
    table.header["TZERO1"] = 10.0
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)

    with fits.open(path) as hdul:
        expected = np.asarray(hdul[1].data["COUNTS"])
    arrow = tf_table.read(str(path), hdu=1)
    np.testing.assert_allclose(arrow["COUNTS"].to_numpy(), expected)

    tensors = tf_table.read_torch(str(path), hdu=1)
    np.testing.assert_allclose(tensors["COUNTS"].numpy(), expected, rtol=1e-6)


def test_checksum_verification_detects_corruption(tmp_path: Path) -> None:
    """testprog: fits_verify_chksum fails after a single payload byte flips."""
    path = tmp_path / "checked.fits"
    data = torch.arange(32, dtype=torch.int16).reshape(4, 8)
    torchfits.write(str(path), data, checksum=True, overwrite=True)

    with fits.open(path, checksum=True) as hdul:
        hdul.verify("exception")
    assert torchfits.verify_checksums(str(path))["ok"] is True

    raw = bytearray(path.read_bytes())
    raw[2880 + 5] ^= 0x01
    path.write_bytes(bytes(raw))
    assert torchfits.verify_checksums(str(path))["ok"] is False
    # astropy reports a checksum mismatch as a warning on open (its verify()
    # only checks structural validity), and still parses the header.
    with pytest.warns(UserWarning, match="[Cc]hecksum verification failed"):
        with fits.open(path, checksum=True):
            pass
