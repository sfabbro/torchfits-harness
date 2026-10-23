"""Replays of the CCfits cookbook against torchfits.

Source: https://heasarc.gsfc.nasa.gov/fitsio/ccfits/html/cookbook.html

The cookbook is organised by CCfits call (create file, write keywords, write a
binary table, read columns, read an image section, walk the HDU chain). The
same steps are performed through torchfits and checked against astropy, which
shares neither CCfits' C++ layer nor torchfits' reader.

``examples/example_ccfits_cookbook.py`` already mirrors the *writing* recipes
for the docs; these tests are the reading/round-trip counterparts, including
the typed-column matrix the example only partly covers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torchfits  # noqa: E402
import torchfits.table as tf_table  # noqa: E402
from astropy.io import fits  # noqa: E402


def _typed_table(path: Path, nrows: int = 12) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    columns = {
        "J": np.arange(nrows, dtype=np.int32),
        "K": (np.arange(nrows, dtype=np.int64) * 10**9),
        "E": rng.normal(size=nrows).astype(np.float32),
        "D": rng.normal(size=nrows).astype(np.float64),
        "L": (np.arange(nrows) % 2 == 0),
        "A": np.array([f"obj{i:03d}" for i in range(nrows)], dtype="S7"),
        "X": np.array([0b10000000 if i % 2 else 0 for i in range(nrows)], dtype=np.uint8),
    }
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="J", format="J", array=columns["J"]),
            fits.Column(name="K", format="K", array=columns["K"]),
            fits.Column(name="E", format="E", array=columns["E"]),
            fits.Column(name="D", format="D", array=columns["D"]),
            fits.Column(name="L", format="L", array=columns["L"]),
            fits.Column(name="A", format="7A", array=columns["A"]),
            fits.Column(name="X", format="8X", array=columns["X"]),
        ],
        name="CATALOG",
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    return columns


def test_typed_columns_roundtrip_values(tmp_path: Path) -> None:
    """Cookbook section 2: every column type, values not just names."""
    path = tmp_path / "typed.fits"
    expected = _typed_table(path)
    arrow = tf_table.read(str(path), hdu=1, columns=list(expected))
    for name in ["J", "K", "E", "D", "L"]:
        np.testing.assert_array_equal(
            arrow[name].to_numpy(), expected[name], err_msg=name
        )
    assert [v.rstrip() for v in arrow["A"].to_pylist()] == [
        v.decode().rstrip() for v in expected["A"]
    ]
    np.testing.assert_array_equal(
        np.asarray(arrow["X"].to_pylist()), np.unpackbits(expected["X"][:, None], axis=1).astype(bool)
    )


def test_column_projection_and_row_window(tmp_path: Path) -> None:
    """Cookbook section 3: Column::read over a range, selected columns only."""
    path = tmp_path / "project.fits"
    expected = _typed_table(path)
    got = tf_table.read(str(path), hdu=1, columns=["D", "A"], row_slice=(2, 8))
    np.testing.assert_array_equal(got["D"].to_numpy(), expected["D"][2:8])
    assert [v.rstrip() for v in got["A"].to_pylist()] == [
        v.decode().rstrip() for v in expected["A"][2:8]
    ]

    tensors = tf_table.read_torch(str(path), hdu=1, columns=["J"], start_row=3, num_rows=4)
    np.testing.assert_array_equal(tensors["J"].numpy(), expected["J"][2:6])


def test_image_section_read(tmp_path: Path) -> None:
    """Cookbook section 4: a section of an image, not the whole array."""
    path = tmp_path / "section.fits"
    cube = np.arange(4 * 10 * 12, dtype=np.float32).reshape(4, 10, 12)
    fits.PrimaryHDU(cube).writeto(path)
    got = torchfits.read_subset(str(path), 0, 2, 3, 7, 9)
    assert tuple(got.shape) == (4, 6, 5)
    np.testing.assert_array_equal(got.numpy(), cube[:, 3:9, 2:7])

    with fits.open(path) as hdul:
        assert torchfits.read_shape(str(path)) == (-32, tuple(hdul[0].data.shape))


def test_hdu_chain_and_extname_lookup(tmp_path: Path) -> None:
    """Cookbook section 6: primary, image extension, binary table, by name."""
    path = tmp_path / "chain.fits"
    primary = np.zeros((4, 4), dtype=np.int16)
    science = np.ones((8, 8), dtype=np.float32) * 5
    table = np.array([(1, "a"), (2, "b")], dtype=[("ID", "i4"), ("NAME", "U1")])
    fits.HDUList(
        [
            fits.PrimaryHDU(primary),
            fits.ImageHDU(science, name="SCI"),
            fits.BinTableHDU(table, name="CAT"),
        ]
    ).writeto(path)

    assert torchfits.read_num_hdus(str(path)) == 3
    assert torchfits.read_extname(str(path), 1) == "SCI"
    assert torchfits.read_extname(str(path), 2) == "CAT"
    assert torchfits.read_hdu_type(str(path), "SCI") == "IMAGE"
    assert torchfits.read_hdu_type(str(path), "CAT") == "BINARY_TABLE"
    np.testing.assert_array_equal(torchfits.read_tensor(str(path), "SCI").numpy(), science)
    assert torchfits.read_colnames(str(path), "CAT") == ["ID", "NAME"]


def test_keyword_types_and_comments(tmp_path: Path) -> None:
    """Cookbook section 1: string, int, float, bool and comments survive."""
    path = tmp_path / "keywords.fits"
    header = {
        "OBJECT": "M31",
        "EXPTIME": 300.0,
        "NCOMBINE": 7,
        "CALIBRATED": True,
        "FILTER": "R",
    }
    torchfits.write(
        str(path),
        torch.zeros((2, 2), dtype=torch.float32),
        header=header,
        overwrite=True,
    )
    got = torchfits.read_header(str(path))
    for key, value in header.items():
        assert got[key] == value, key

    with fits.open(path) as hdul:
        written = {key: hdul[0].header[key] for key in header}
    assert written == header
    assert torchfits.read_keys(str(path), ["OBJECT", "EXPTIME"]) == {
        "OBJECT": "M31",
        "EXPTIME": 300.0,
    }
