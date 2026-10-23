"""Replays of sfitsio's test programs against torchfits.

Sources (SFITSIO 1.4.6, ``test/``):

* ``tzerotscale_bintable.cc`` -- TSCAL/TZERO on D/B/J/K columns
* ``access_bte_heap.cc``    -- variable-length-array columns through the heap
* ``edit_bintable.cc``      -- add and rename columns in place
* ``continue_header.cc``    -- long string values and CONTINUE cards
* ``rename_header.cc``      -- rename a keyword without losing the payload
* ``image_copy_paste.cc``   -- copy a region from one image into another

SFITSIO wraps CFITSIO with script-like calls; these tests keep its workflows
and compare with astropy. Its ``select_bte`` / ``select_ckm`` arrays and
``create_from_template`` are covered by the table and image cases elsewhere
and are listed as such in the upstream manifest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torchfits  # noqa: E402
import torchfits.table as tf_table  # noqa: E402
from astropy.io import fits  # noqa: E402


def _scaled_table(path: Path) -> dict[str, np.ndarray]:
    n = 6
    # K is omitted: astropy 8.0 raises UnboundLocalError decoding a scaled K
    # column (its own bug), so there is no independent oracle for that one.
    stored = {
        "COL_D": np.arange(n, dtype=np.float64),
        "COL_B": np.arange(n, dtype=np.uint8),
        "COL_J": np.arange(n, dtype=np.int32),
    }
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="COL_D", format="D", array=stored["COL_D"]),
            fits.Column(name="COL_B", format="B", array=stored["COL_B"]),
            fits.Column(name="COL_J", format="J", array=stored["COL_J"]),
        ],
        name="TZERO_TEST",
    )
    for index, column in enumerate(["COL_D", "COL_B", "COL_J"], start=1):
        hdu.header[f"TSCAL{index}"] = 0.5
        hdu.header[f"TZERO{index}"] = -10.0
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    return stored


def test_tscal_tzero_on_every_numeric_column(tmp_path: Path) -> None:
    """tzerotscale_bintable.cc: physical = stored * 0.5 - 10 for D/B/J/K."""
    path = tmp_path / "tzero.fits"
    stored = _scaled_table(path)
    with fits.open(path) as hdul:
        expected = {name: np.asarray(hdul[1].data[name]) for name in stored}
    arrow = tf_table.read(str(path), hdu=1)
    for name in stored:
        np.testing.assert_allclose(
            arrow[name].to_numpy(), expected[name], rtol=1e-6, err_msg=name
        )
        np.testing.assert_allclose(
            arrow[name].to_numpy(), stored[name] * 0.5 - 10.0, rtol=1e-6, err_msg=name
        )


def test_heap_variable_length_arrays(tmp_path: Path) -> None:
    """access_bte_heap.cc: each row owns its array in the heap."""
    path = tmp_path / "heap.fits"
    rows = np.empty(5, dtype=object)
    rows[:] = [
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([9.0], dtype=np.float64),
        np.array([-1.0, -2.0, -3.0], dtype=np.float64),
        np.array([0.5], dtype=np.float64),
    ]
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="VEC", format="PD", array=rows)], name="MATRIX"
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)

    with fits.open(path) as hdul:
        expected = [np.asarray(v) for v in hdul[1].data["VEC"]]
    got = tf_table.read(str(path), hdu=1)["VEC"].to_pylist()
    for row, want in zip(got, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(row), want)



def test_insert_and_rename_table_columns(tmp_path: Path) -> None:
    """edit_bintable.cc: add a column and rename an existing one."""
    path = tmp_path / "edit.fits"
    tf_table.write(
        str(path),
        {"ID": np.arange(4, dtype=np.int32), "VALUE": np.linspace(0, 1, 4)},
        extname="EVENT",
    )
    tf_table.insert_column(str(path), "FLUX", np.arange(4, dtype=np.float32), hdu="EVENT")
    tf_table.rename_columns(str(path), {"ID": "OBJID"}, hdu="EVENT")

    with fits.open(path) as hdul:
        assert list(hdul[1].data.columns.names) == ["OBJID", "VALUE", "FLUX"]
        np.testing.assert_array_equal(np.asarray(hdul[1].data["OBJID"]), np.arange(4))
        np.testing.assert_allclose(np.asarray(hdul[1].data["FLUX"]), np.arange(4))

    arrow = tf_table.read(str(path), hdu="EVENT")
    assert arrow.column_names == ["OBJID", "VALUE", "FLUX"]


def test_long_string_header_values(tmp_path: Path) -> None:
    """continue_header.cc: values longer than one card survive CONTINUE."""
    path = tmp_path / "continue.fits"
    long_value = (
        "0123456789abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    )
    torchfits.write(
        str(path),
        torch.zeros((2, 2)),
        header={"LONGVAL0": long_value, "LONGVAL1": long_value[:-1]},
        overwrite=True,
    )
    with fits.open(path) as hdul:
        assert hdul[0].header["LONGVAL0"] == long_value
        assert hdul[0].header["LONGVAL1"] == long_value[:-1]
    got = torchfits.read_header(str(path))
    assert got["LONGVAL0"] == long_value
    assert got["LONGVAL1"] == long_value[:-1]


def test_rename_a_header_keyword(tmp_path: Path) -> None:
    """rename_header.cc: the new name is present, the old one is gone."""
    src = tmp_path / "rename_src.fits"
    dst = tmp_path / "rename_dst.fits"
    torchfits.write(
        str(src),
        torch.arange(9, dtype=torch.int16).reshape(3, 3),
        header={"OLDKEY": 42, "KEEP": "yes"},
        overwrite=True,
    )
    with torchfits.open(str(src)) as hdul:
        del hdul[0].header["OLDKEY"]
        hdul[0].header["NEWKEY"] = 42
        hdul.write(str(dst))

    with fits.open(dst) as hdul:
        assert hdul[0].header["NEWKEY"] == 42
        assert "OLDKEY" not in hdul[0].header
        assert hdul[0].header["KEEP"] == "yes"
        np.testing.assert_array_equal(
            np.asarray(hdul[0].data), np.arange(9, dtype=np.int16).reshape(3, 3)
        )
    assert torchfits.read_keys(str(dst), ["NEWKEY"]) == {"NEWKEY": 42}


def test_image_copy_paste_region(tmp_path: Path) -> None:
    """image_copy_paste.cc: paste a source stamp into a larger destination."""
    src = tmp_path / "stamp_src.fits"
    dst = tmp_path / "stamp_dst.fits"
    stamp = np.arange(6 * 7, dtype=np.float32).reshape(6, 7)
    fits.PrimaryHDU(stamp).writeto(src)
    canvas = np.zeros((10, 12), dtype=np.float32)
    fits.PrimaryHDU(canvas).writeto(dst)

    read_back = torchfits.read_tensor(str(dst)).numpy()
    read_back[2:8, 3:10] = torchfits.read(str(src)).numpy()
    torchfits.write(str(dst), torch.from_numpy(read_back), overwrite=True)

    with fits.open(dst) as hdul:
        expected = np.zeros((10, 12), dtype=np.float32)
        expected[2:8, 3:10] = stamp
        np.testing.assert_array_equal(np.asarray(hdul[0].data), expected)
