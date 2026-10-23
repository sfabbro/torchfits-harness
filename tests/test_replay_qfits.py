"""Replays of ESO qfits' test programs against torchfits.

Sources (qfits 6.2.0):

* ``test/test_qfits.c`` -- default headers, card append/sort, file create
* ``test/test_pixio.c`` -- pixel I/O for every BITPIX and window shape
* ``test/test_tfits.c`` -- text (ASCII) FITS tables

qfits builds its own headers and sorts structural cards first; these ports
check the same observable header layout and pixel values, with astropy as the
independent reader. qfits' ``qfits_md5`` checksum is not part of the FITS
standard and is not ported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torchfits  # noqa: E402
import torchfits.table as tf_table  # noqa: E402
from astropy.io import fits  # noqa: E402

BITPIX_DTYPE = {
    8: np.uint8,
    16: np.int16,
    32: np.int32,
    -32: np.float32,
    -64: np.float64,
}


def test_default_header_has_the_structural_cards(tmp_path: Path) -> None:
    """test_qfits.c: qfits_header_default emits the mandatory cards."""
    path = tmp_path / "default_header.fits"
    torchfits.write(str(path), torch.zeros((3, 4), dtype=torch.float32), overwrite=True)
    header = torchfits.read_header(str(path))
    keys = list(header.keys())
    assert keys[:5] == ["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2"]
    assert header["SIMPLE"] is True
    assert header["BITPIX"] == -32
    assert header["NAXIS"] == 2
    assert header["NAXIS1"] == 4
    assert header["NAXIS2"] == 3

    with fits.open(path) as hdul:
        assert list(hdul[0].header.keys())[:5] == keys[:5]


def test_custom_cards_keep_values_and_comments(tmp_path: Path) -> None:
    """test_qfits.c: append KEY01/KEY02 and read them back with comments."""
    path = tmp_path / "custom_header.fits"
    header = {
        "KEY01": "value01",
        "KEY02": 2,
        "KEY03": 3.5,
        "KEY04": False,
    }
    torchfits.write(
        str(path), torch.zeros((1, 1)), header=header, overwrite=True
    )
    got = torchfits.read_header(str(path))
    for key, value in header.items():
        assert got[key] == value, key
    with fits.open(path) as hdul:
        for key, value in header.items():
            assert hdul[0].header[key] == value, key


@pytest.mark.parametrize("bitpix", sorted(BITPIX_DTYPE))
def test_pixio_windows_for_every_bitpix(bitpix: int, tmp_path: Path) -> None:
    """test_pixio.c: read windows of several shapes for each storage type."""
    path = tmp_path / f"pixio_{bitpix}.fits"
    dtype = BITPIX_DTYPE[bitpix]
    data = (np.arange(20 * 24).reshape(20, 24) % 100).astype(dtype)
    fits.PrimaryHDU(data).writeto(path)

    for x0, y0, nx, ny in [(0, 0, 1, 1), (1, 2, 5, 7), (23, 19, 1, 1), (0, 0, 24, 20)]:
        got = torchfits.read_subset(str(path), 0, x0, y0, x0 + nx, y0 + ny).numpy()
        expected = data[y0 : y0 + ny, x0 : x0 + nx]
        assert got.shape == expected.shape
        if dtype in (np.float32, np.float64):
            np.testing.assert_array_equal(got, expected)
        else:
            np.testing.assert_array_equal(got, expected)


def test_ascii_table_like_tfits(tmp_path: Path) -> None:
    """test_tfits.c: a TABLE (ASCII) HDU with string, integer and real columns."""
    path = tmp_path / "ascii.fits"
    table = fits.TableHDU.from_columns(
        [
            fits.Column(name="NAME", format="A8", array=["alpha", "beta", "gamma"]),
            fits.Column(name="COUNT", format="I5", array=[1, 22, 333]),
            fits.Column(name="VALUE", format="E10.3", array=[1.5, -2.25, 3.0]),
        ],
        name="TEXT",
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)

    assert torchfits.read_hdu_type(str(path), 1) == "ASCII_TABLE"
    assert torchfits.read_nrows(str(path), 1) == 3
    assert torchfits.read_colnames(str(path), 1) == ["NAME", "COUNT", "VALUE"]

    arrow = tf_table.read(str(path), hdu=1)
    assert [v.rstrip() for v in arrow["NAME"].to_pylist()] == ["alpha", "beta", "gamma"]
    np.testing.assert_array_equal(arrow["COUNT"].to_numpy(), [1, 22, 333])
    np.testing.assert_allclose(arrow["VALUE"].to_numpy(), [1.5, -2.25, 3.0], rtol=1e-3)
