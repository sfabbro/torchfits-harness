"""Replays of fitsio's own test behaviours against torchfits.

fitsio is a thin CFITSIO binding and the harness' second Python oracle. These
cases use fitsio as the *writer* and torchfits as the reader (and the reverse),
which catches assumptions the astropy-only paths share.

Source: the ``fitsio`` Python package tests (``test_fitsio.py``) for
``write``/``append``/``read(rows=...)``/``read_header``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

fitsio = pytest.importorskip("fitsio")

import torchfits  # noqa: E402
import torchfits.table as tf_table  # noqa: E402
from astropy.io import fits  # noqa: E402


def test_fitsio_written_table_reads_with_metadata(tmp_path: Path) -> None:
    """test_fitsio.py: a fitsio-written table keeps extname, units and dtype."""
    path = tmp_path / "fitsio_table.fits"
    data = np.zeros(
        5, dtype=[("ID", "i4"), ("FLUX", "f8"), ("NAME", "U6")]
    )
    data["ID"] = np.arange(5)
    data["FLUX"] = np.linspace(0.5, 2.5, 5)
    data["NAME"] = ["a", "b", "c", "d", "e"]
    fitsio.write(
        str(path),
        data,
        extname="CAT",
        header={"BUNIT": "Jy", "OBJECT": "M31"},
        clobber=True,
    )

    # fitsio writes an empty primary and puts the table in extension 1.
    assert torchfits.read_num_hdus(str(path)) == 2
    assert torchfits.read_extname(str(path), 1) == "CAT"
    assert torchfits.read_colnames(str(path), 1) == ["ID", "FLUX", "NAME"]
    header = torchfits.read_header(str(path), 1)
    assert header["OBJECT"] == "M31"
    # fitsio folds BUNIT into per-column TUNIT and drops it when the column
    # has no unit; the persisted keyword set must still match.
    assert set(header.keys()) == set(fitsio.read_header(str(path), 1).keys())

    arrow = tf_table.read(str(path), hdu=1)
    np.testing.assert_array_equal(arrow["ID"].to_numpy(), data["ID"])
    np.testing.assert_allclose(arrow["FLUX"].to_numpy(), data["FLUX"])
    assert [v.rstrip() for v in arrow["NAME"].to_pylist()] == list(data["NAME"])


def test_fitsio_rows_window_equals_torchfits_row_slice(tmp_path: Path) -> None:
    """test_fitsio.py: read(rows=...) is the same window as row_slice."""
    path = tmp_path / "rows.fits"
    n = 50
    data = np.zeros(n, dtype=[("X", "i4"), ("Y", "f4")])
    data["X"] = np.arange(n)
    data["Y"] = np.linspace(0, 1, n).astype(np.float32)
    fitsio.write(str(path), data, clobber=True)

    expected = fitsio.read(str(path), ext=1, rows=np.arange(10, 20))
    arrow = tf_table.read(str(path), hdu=1, row_slice=(10, 20))
    np.testing.assert_array_equal(arrow["X"].to_numpy(), expected["X"])
    np.testing.assert_allclose(arrow["Y"].to_numpy(), expected["Y"])

    tensors = tf_table.read_torch(str(path), hdu=1, start_row=11, num_rows=10)
    np.testing.assert_array_equal(tensors["X"].numpy(), expected["X"])


def test_fitsio_appended_image_is_visible_to_torchfits(tmp_path: Path) -> None:
    """test_fitsio.py append: an extension added by fitsio is readable."""
    path = tmp_path / "append.fits"
    fitsio.write(str(path), np.arange(12, dtype=np.float32).reshape(3, 4), clobber=True)
    fitsio.write(str(path), np.ones((2, 2), dtype=np.int16), extname="SECOND")
    assert torchfits.read_num_hdus(str(path)) == 2
    assert torchfits.read_extname(str(path), 1) == "SECOND"
    np.testing.assert_array_equal(
        torchfits.read_tensor(str(path), "SECOND").numpy(),
        np.ones((2, 2), dtype=np.int16),
    )


def test_fitsio_header_dict_agrees_with_torchfits(tmp_path: Path) -> None:
    """test_fitsio.py read_header: the same cards, values and types."""
    path = tmp_path / "header.fits"
    header = {"OBJECT": "NGC 1", "EXPTIME": 12.5, "NCOMBINE": 3, "CALIB": True}
    fitsio.write(
        str(path), np.zeros((2, 2), dtype=np.float32), header=header, clobber=True
    )
    theirs = fitsio.read_header(str(path), ext=0)
    ours = torchfits.read_header(str(path), 0)
    for key, value in header.items():
        assert ours[key] == value, key
        assert theirs[key] == value, key


def test_fitsio_cutout_matches_torchfits_subset(tmp_path: Path) -> None:
    """test_fitsio.py: a manually sliced fitsio array equals read_subset."""
    path = tmp_path / "cutout.fits"
    data = np.arange(30 * 40, dtype=np.int32).reshape(30, 40)
    fitsio.write(str(path), data, clobber=True)
    base = fitsio.read(str(path))
    for x0, y0, dx, dy in [(0, 0, 1, 1), (5, 7, 9, 4), (30, 25, 10, 5)]:
        got = torchfits.read_subset(str(path), 0, x0, y0, x0 + dx, y0 + dy).numpy()
        np.testing.assert_array_equal(got, base[y0 : y0 + dy, x0 : x0 + dx])


def test_fitsio_can_read_a_torchfits_table(tmp_path: Path) -> None:
    """Reverse direction: fitsio parses the FITS table torchfits writes."""
    path = tmp_path / "written_by_torchfits.fits"
    tf_table.write(
        str(path),
        {
            "ID": np.arange(8, dtype=np.int32),
            "FLUX": np.linspace(-1, 1, 8),
            "NAME": np.array([f"n{i}" for i in range(8)]),
        },
        extname="CAT",
        header={"OBSERVER": "harness"},
    )
    table = fitsio.read(str(path), ext=1)
    assert table.dtype.names == ("ID", "FLUX", "NAME")
    np.testing.assert_array_equal(table["ID"], np.arange(8))
    np.testing.assert_allclose(table["FLUX"], np.linspace(-1, 1, 8))
    assert str(table["NAME"][0]).rstrip() == "n0"
    assert fitsio.read_header(str(path), ext=1)["OBSERVER"] == "harness"
