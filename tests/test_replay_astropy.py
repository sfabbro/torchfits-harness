"""Replays of astropy's FITS test cases that the corpus diff does not pin.

Source: ``astropy/io/fits/tests/test_uint.py``, ``test_checksum.py`` and
``test_table.py`` (TDIM). astropy's installed data corpus is already compared
by ``tests/test_astropy_corpus_diff.py``; these are the generated behaviours
that corpus does not contain as files.

TDIM is the one place where torchfits deliberately answers a different
question: it preserves ``TDIM`` as schema metadata and returns the stored
row-major values, where astropy reshapes the column in place. The test pins
that contract instead of pretending the APIs agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torchfits  # noqa: E402
import torchfits.table as tf_table  # noqa: E402
from astropy.io import fits  # noqa: E402


def test_unsigned_scaled_image_is_physical_uint16(tmp_path: Path) -> None:
    """test_uint.py: int16 + BZERO=32768 reads as unsigned, full and windowed."""
    path = tmp_path / "uint16.fits"
    stored = np.array([[0, 1, 32767], [-32768, -1, 1234]], dtype=np.int16)
    hdu = fits.PrimaryHDU(stored)
    hdu.header["BZERO"] = 32768.0
    hdu.header["BSCALE"] = 1.0
    hdu.writeto(path)

    with fits.open(path) as hdul:
        expected = np.asarray(hdul[0].data)
    assert expected.dtype == np.uint16
    got = torchfits.read_tensor(str(path)).numpy()
    assert got.dtype == np.uint16
    np.testing.assert_array_equal(got, expected)

    window = torchfits.read_subset(str(path), 0, 1, 0, 3, 2).numpy()
    np.testing.assert_array_equal(window, expected[0:2, 1:3])


def test_checksum_and_datasum_cards_verify_in_astropy(tmp_path: Path) -> None:
    """test_checksum.py: the CHECKSUM/DATASUM cards torchfits writes are standard."""
    path = tmp_path / "checksum_cards.fits"
    torchfits.write(
        str(path),
        torch.arange(48, dtype=torch.int32).reshape(6, 8),
        header={"EXTNAME": "SCI"},
        checksum=True,
        overwrite=True,
    )
    header = torchfits.read_header(str(path))
    assert isinstance(header["CHECKSUM"], str) and len(header["CHECKSUM"]) == 16
    assert isinstance(header["DATASUM"], str) and header["DATASUM"].isdigit()

    with fits.open(path, checksum=True) as hdul:
        hdul.verify("exception")
        hdul[0].verify_checksum()
        assert hdul[0].header["CHECKSUM"] == header["CHECKSUM"]
        assert hdul[0].header["DATASUM"] == header["DATASUM"]


def test_tdim_roundtrips_through_the_write_schema(tmp_path: Path) -> None:
    """test_table.py TDIM: the card round-trips; values stay row-major.

    torchfits does not reshape a TDIM column on read (the tensor is ``(nrows,
    repeat)``); the FITS metadata is what it preserves. astropy reshapes, which
    is a different API choice, so the flat values are compared to the stored
    buffer rather than to astropy's reshaped view.
    """
    path = tmp_path / "tdim_written.fits"
    stored = np.arange(24, dtype=np.int32).reshape(3, 8)
    tf_table.write(
        str(path),
        {"C": stored},
        schema={"C": {"format": "8J", "dim": "(4,2)", "unit": "m"}},
        extname="T",
    )
    header = torchfits.read_header(str(path), 1)
    assert header["TDIM1"] == "(4,2)"
    assert header["TUNIT1"] == "m"
    assert header["TFORM1"] == "8J"

    with fits.open(path) as hdul:
        assert np.asarray(hdul[1].data["C"]).shape == (3, 2, 4)

    arrow = tf_table.read(str(path), hdu=1)
    np.testing.assert_array_equal(np.asarray(arrow["C"].to_pylist()), stored)
    tensors = tf_table.read_torch(str(path), hdu=1)
    assert tuple(tensors["C"].shape) == (3, 8)
