"""Replays of EleFits' test suite against torchfits.

Sources (EleFits 4.x):

* ``EleFits/tests/src/MefFile_test.cpp`` -- append/count/remove/access by name
* ``EleFits/tests/src/ImageRaster_test.cpp`` -- raster, 3D slice, 2D region
* ``EleFitsData/tests/src/Segment_test.cpp`` -- segment write without touching
  the rest of the array
* ``EleFitsData/tests/src/Compression_test.cpp`` -- scaling and quantization

EleFits is a header-only C++ layer over CFITSIO. The port keeps its semantics
(multi-extension access, region reads/writes, name lookup) and uses astropy as
the independent answer. ``FileMemSegments`` (several FITS files concatenated
into one stream) is out of scope and recorded as such in the upstream manifest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torchfits  # noqa: E402
import torchfits.table as tf_table  # noqa: E402
from astropy.io import fits  # noqa: E402


def test_mef_append_count_and_access_by_name(tmp_path: Path) -> None:
    """MefFile_test: append_test / count_test / access_single_named_hdu_test."""
    path = tmp_path / "mef.fits"
    # Start from the table (primary + CAT), then insert the image between them.
    tf_table.write(
        str(path),
        {"ID": np.arange(4, dtype=np.int32), "NAME": np.array(["a", "b", "c", "d"])},
        extname="CAT",
    )
    torchfits.insert_hdu(
        str(path), torch.ones((6, 5), dtype=torch.int16), index=1, header={"EXTNAME": "SCI"}
    )

    assert torchfits.read_num_hdus(str(path)) == 3
    assert torchfits.read_extname(str(path), 1) == "SCI"
    assert torchfits.read_extname(str(path), 2) == "CAT"

    with fits.open(path) as hdul:
        assert [h.header.get("EXTNAME") for h in hdul] == [None, "SCI", "CAT"]
        np.testing.assert_array_equal(np.asarray(hdul["SCI"].data), np.ones((6, 5), np.int16))
        assert list(hdul["CAT"].data["NAME"]) == ["a", "b", "c", "d"]

    np.testing.assert_array_equal(
        torchfits.read_tensor(str(path), "SCI").numpy(), np.ones((6, 5), np.int16)
    )
    assert tf_table.read(str(path), hdu="CAT")["ID"].to_pylist() == [0, 1, 2, 3]


def test_mef_remove_primary_leaves_a_valid_file(tmp_path: Path) -> None:
    """MefFile_test: remove_primary_test -- remaining extension is promoted."""
    path = tmp_path / "remove_primary.fits"
    torchfits.write(str(path), torch.zeros((3, 3)), overwrite=True)
    torchfits.insert_hdu(
        str(path), torch.full((2, 2), 9, dtype=torch.int16), index=1, header={"EXTNAME": "ONLY"}
    )
    torchfits.delete_hdu(str(path), 0)
    assert torchfits.read_num_hdus(str(path)) == 1
    with fits.open(path) as hdul:
        assert len(hdul) == 1
        assert hdul[0].header["EXTNAME"] == "ONLY"
        np.testing.assert_array_equal(np.asarray(hdul[0].data), np.full((2, 2), 9))
    np.testing.assert_array_equal(
        torchfits.read_tensor(str(path)).numpy(), np.full((2, 2), 9, dtype=np.int16)
    )


def test_image_raster_slice_and_region(tmp_path: Path) -> None:
    """ImageRaster_test: raster_is_read_back / slice_3d / region_2d."""
    path = tmp_path / "raster.fits"
    cube = np.arange(5 * 9 * 11, dtype=np.float32).reshape(5, 9, 11)
    fits.PrimaryHDU(cube).writeto(path)

    raster = torchfits.read(str(path)).numpy()
    np.testing.assert_array_equal(raster, cube)

    slice_2d = torchfits.read(str(path))[2].numpy()
    np.testing.assert_array_equal(slice_2d, cube[2])

    region = torchfits.read_subset(str(path), 0, 3, 4, 8, 7).numpy()
    assert region.shape == (5, 3, 5)
    np.testing.assert_array_equal(region, cube[:, 4:7, 3:8])


def test_sif_single_image_file(tmp_path: Path) -> None:
    """SifFile: a single-image file has no extension name and one HDU."""
    path = tmp_path / "sif.fits"
    data = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    torchfits.write(str(path), data, header={"OBJECT": "SIF"}, overwrite=True)
    assert torchfits.read_num_hdus(str(path)) == 1
    assert torchfits.read_extname(str(path), 0) is None
    assert torchfits.read_hdu_type(str(path), 0) == "IMAGE"
    np.testing.assert_array_equal(torchfits.read_tensor(str(path)).numpy(), data.numpy())


def test_header_edit_then_write_to_a_new_path(tmp_path: Path) -> None:
    """MefFile_test: append_header_test -- edit a copy, leave the original."""
    src = tmp_path / "original.fits"
    dst = tmp_path / "edited.fits"
    torchfits.write(str(src), torch.zeros((2, 2)), header={"OLD": 1}, overwrite=True)

    with torchfits.open(str(src)) as hdul:
        hdul[0].header["NEW"] = "value"
        hdul[0].header.add_comment("edited by test")
        hdul.write(str(dst))

    assert torchfits.read_keys(str(src), ["OLD"]) == {"OLD": 1}
    with pytest.raises(RuntimeError):
        torchfits.read_keys(str(src), ["NEW"])
    edited = torchfits.read_header(str(dst))
    assert edited["OLD"] == 1
    assert edited["NEW"] == "value"
    assert "edited by test" in edited.get_comment()
    with fits.open(dst) as hdul:
        assert hdul[0].header["NEW"] == "value"


def test_compression_strategy_roundtrip(tmp_path: Path) -> None:
    """Compression_test: the strategy name is recorded and data survives."""
    path = tmp_path / "compressed.fits.fz"
    data = torch.arange(64, dtype=torch.int16).reshape(8, 8)
    torchfits.write(str(path), data, compress="RICE_1", overwrite=True)
    assert torchfits.read_header(str(path), 1)["ZCMPTYPE"] == "RICE_1"
    np.testing.assert_array_equal(
        torchfits.read_tensor(str(path), hdu=1).numpy(), data.numpy()
    )
    with fits.open(path) as hdul:
        np.testing.assert_array_equal(np.asarray(hdul[1].data), data.numpy())
