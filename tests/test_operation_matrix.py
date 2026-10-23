"""Operations a user actually performs, exercised on real archive files.

Each test names the operation, the real product it runs on, and the independent
answer it is checked against. Synthetics already cover the combinatorial grid;
this file is about the shapes and dtypes that only archives produce: 3D IFU
cubes, 6.7 GB catalogs read through row windows, 40-extension compressed
exposures and 435-megapixel mosaics.
"""

from __future__ import annotations

import shutil
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


def _source(source_id: str) -> dict[str, Any]:
    for source in corpus.sources(tier="full"):
        if source["id"] == source_id:
            return source
    raise AssertionError(source_id)


def test_cube_cutout_matches_astropy() -> None:
    source = _source("sdss_manga_logcube_8485_12701")
    path = corpus.require(source)
    with fits.open(str(path)) as hdul:
        cube = np.asarray(hdul["FLUX"].data)
        expected = cube[:, 20:26, 10:16]
        assert expected.shape == (cube.shape[0], 6, 6)
    got = torchfits.read_subset(str(path), "FLUX", 10, 20, 16, 26).numpy()
    np.testing.assert_array_equal(got, expected)


def test_cube_plane_values_match_astropy() -> None:
    source = _source("sdss_manga_logrss_8485_12701")
    path = corpus.require(source)
    with fits.open(str(path)) as hdul:
        data = np.asarray(hdul["FLUX"].data)
        expected = data[256, :]
    got = torchfits.read(str(path), "FLUX")[256, :].numpy()
    np.testing.assert_array_equal(got, expected)


def test_table_window_projection_and_predicate_match_astropy() -> None:
    source = _source("sdss_plates_dr18")
    path = corpus.require(source)
    with fits.open(str(path), memmap=True) as hdul:
        data = hdul[1].data
        expected = np.asarray(data["PLATE"][10:30])
        mask = np.asarray(data["PLATE"]) > 3000
        expected_filtered = np.asarray(data["PLATE"])[mask]

    window = tf_table.read(str(path), hdu=1, columns=["PLATE"], row_slice=(10, 30))
    np.testing.assert_array_equal(window["PLATE"].to_numpy(), expected)

    filtered = tf_table.read(str(path), hdu=1, columns=["PLATE"], where="PLATE > 3000")
    np.testing.assert_array_equal(filtered["PLATE"].to_numpy(), expected_filtered)

    ranged = tf_table.read(
        str(path), hdu=1, columns=["PLATE"], where="PLATE BETWEEN 3000 AND 4000"
    )
    expected_range = np.asarray(data["PLATE"])[
        (np.asarray(data["PLATE"]) >= 3000) & (np.asarray(data["PLATE"]) <= 4000)
    ]
    np.testing.assert_array_equal(ranged["PLATE"].to_numpy(), expected_range)


def test_tensor_where_and_arrow_where_select_the_same_rows() -> None:
    source = _source("sdss_plates_dr18")
    path = corpus.require(source)
    where = "PLATE > 3000 AND PLATE < 4000"
    arrow = tf_table.read(str(path), hdu=1, columns=["PLATE"], where=where)
    tensors = tf_table.read_torch(
        str(path), hdu=1, columns=["PLATE"], where=where, num_rows=-1
    )
    np.testing.assert_array_equal(
        tensors["PLATE"].numpy(), arrow["PLATE"].to_numpy()
    )


def test_header_key_extraction_and_missing_keys() -> None:
    source = _source("sdss_spec_0266_51630_0001")
    path = corpus.require(source)
    with fits.open(str(path)) as hdul:
        expected = hdul[1].header
    keys = ["BITPIX", "NAXIS1", "NAXIS2"]
    got = torchfits.read_keys(str(path), keys, hdu=1)
    for key in keys:
        assert got[key] == expected[key], key
    # read_keys is strict by design: it raises instead of inventing a value.
    with pytest.raises(RuntimeError, match="keyword not found"):
        torchfits.read_keys(str(path), ["NOPE_NOT_A_CARD"], hdu=1)
    header = torchfits.read_header(str(path), 1)
    assert header["EXTNAME"] == "COADD"
    assert header.contains("NAXIS2") if hasattr(header, "contains") else "NAXIS2" in header


def test_mef_surgery_on_a_real_product(tmp_path: Path) -> None:
    source = _source("sdss_spec_0266_51630_0001")
    src = corpus.require(source)
    path = tmp_path / "spec_edited.fits"
    shutil.copy2(src, path)

    n_before = torchfits.read_num_hdus(str(path))
    torchfits.insert_hdu(
        str(path),
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
        index=1,
        header={"EXTNAME": "CUTOUT"},
    )
    assert torchfits.read_num_hdus(str(path)) == n_before + 1
    assert torchfits.read_extname(str(path), 1) == "CUTOUT"

    with fits.open(str(path)) as hdul:
        assert hdul[1].header["EXTNAME"] == "CUTOUT"
        assert np.asarray(hdul[1].data).shape == (3, 4)
        # The untouched science extension still carries the SDSS header.
        coadd = next(h for h in hdul if h.header.get("EXTNAME") == "COADD")
        assert coadd.header["NAXIS2"] > 0

    torchfits.delete_hdu(str(path), "CUTOUT")
    assert torchfits.read_num_hdus(str(path)) == n_before
    with fits.open(str(path)) as hdul:
        assert [h.header.get("EXTNAME") for h in hdul] == [
            None,
            "COADD",
            "SPECOBJ",
            "SPZLINE",
            "B1-00003238-00003246-00003247",
            "B1-00003239-00003246-00003247",
            "B1-00003240-00003246-00003247",
            "B1-00003241-00003246-00003247",
            "R1-00003238-00003246-00003247",
            "R1-00003239-00003246-00003247",
            "R1-00003240-00003246-00003247",
            "R1-00003241-00003246-00003247",
        ]


def test_stack_of_real_frames_matches_numpy() -> None:
    # Odd count: torch.median returns the lower middle value for even stacks,
    # while numpy averages the two, so an odd stack keeps the comparison exact.
    ids = [
        "cfht_megacam_2366188o",
        "cfht_megacam_2366188p",
        "cfht_megacam_2366432o",
    ]
    paths = [corpus.require(_source(source_id)) for source_id in ids]
    frames = [
        torchfits.read_subset(str(path), 1, 1000, 1000, 1128, 1128)
        for path in paths
    ]
    stacked = torch.stack(frames)
    mean = stacked.float().mean(dim=0).numpy()
    median = stacked.float().median(dim=0).values.numpy()

    with fits.open(str(paths[0]), disable_image_compression=False) as hdul:
        expected_first = np.asarray(hdul[1].data)[1000:1128, 1000:1128]
    np.testing.assert_array_equal(frames[0].numpy(), expected_first)
    np.testing.assert_allclose(mean, np.mean([f.numpy() for f in frames], axis=0))
    np.testing.assert_array_equal(median, np.median([f.numpy() for f in frames], axis=0))


def test_bitpix_conversion_of_a_real_frame(tmp_path: Path) -> None:
    # The "o" frames are stored as int16 with BZERO=32768 (uint16 in memory);
    # the "p" frames are already float32 after calibration.
    source = _source("cfht_megacam_2366188o")
    src = corpus.require(source)
    raw = torchfits.read_subset(str(src), 1, 500, 500, 532, 532)
    assert raw.dtype == torch.uint16

    as_float = tmp_path / "frame_float32.fits"
    torchfits.write(str(as_float), raw.float(), header={"ORIGIN": "torchfits"}, overwrite=True)
    with fits.open(as_float) as hdul:
        assert hdul[0].header["BITPIX"] == -32
        np.testing.assert_array_equal(np.asarray(hdul[0].data), raw.float().numpy())

    as_int = tmp_path / "frame_int32.fits"
    torchfits.write(str(as_int), raw.to(torch.int32), overwrite=True)
    with fits.open(as_int) as hdul:
        assert hdul[0].header["BITPIX"] == 32
        np.testing.assert_array_equal(np.asarray(hdul[0].data), raw.to(torch.int32).numpy())


def test_mosaic_cutout_edges_and_refusals() -> None:
    source = _source("cfht_megapipe_R")
    path = corpus.require(source)
    one = torchfits.read_subset(str(path), 0, 0, 0, 1, 1)
    assert tuple(one.shape) == (1, 1)
    with fits.open(str(path)) as hdul:
        assert int(one.item()) == int(np.asarray(hdul[0].data)[0, 0])
    # NumPy-style slicing: a window entirely past the array is empty, not an
    # exception, and a partially overlapping one is clipped to the array.
    empty = torchfits.read_subset(str(path), 0, 10**9, 10**9, 10**9 + 4, 10**9 + 4)
    assert tuple(empty.shape) == (0, 0)
    with fits.open(str(path)) as hdul:
        nx = int(hdul[0].header["NAXIS1"])
        ny = int(hdul[0].header["NAXIS2"])
    clipped = torchfits.read_subset(str(path), 0, nx - 2, ny - 2, nx + 2, ny + 2)
    assert tuple(clipped.shape) == (2, 2)


def test_checksum_verification_detects_a_flipped_pixel(tmp_path: Path) -> None:
    path = tmp_path / "checksummed.fits"
    data = torch.arange(64, dtype=torch.int16).reshape(8, 8)
    torchfits.write(str(path), data, checksum=True, overwrite=True)
    assert torchfits.verify_checksums(str(path))["ok"] is True

    # Flip one byte inside the first data block (2880-byte header, then pixels).
    payload = bytearray(path.read_bytes())
    payload[2880 + 3] ^= 0xFF
    path.write_bytes(bytes(payload))
    assert torchfits.verify_checksums(str(path))["ok"] is False
