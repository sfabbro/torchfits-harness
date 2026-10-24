"""End-to-end CLI contract: the `torchfits` command against astropy and the corpus.

The CLI is a user-facing surface that no unit test in the library exercises end
to end: argument plumbing, exit codes, JSON shapes and file contents all cross
module boundaries. Every test here invokes the installed console entry point in
a subprocess — the same thing a user types — and checks the result against an
oracle (astropy) or against the file it produced.

Exit codes are part of the contract (``torchfits/cli/common.py``): 0 OK,
1 differences, 2 usage, 3 I/O, 4 checksum failure.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import corpus  # noqa: E402

astropy_fits = pytest.importorskip("astropy.io.fits")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

EXIT_OK = 0
EXIT_DIFF = 1
EXIT_IO = 3
EXIT_VERIFY_FAIL = 4

IMAGE = np.arange(32 * 24, dtype=np.float32).reshape(24, 32)
TABLE = np.array(
    [
        (1, 2.5, "alpha"),
        (2, 3.5, "beta"),
        (3, 4.5, "gamma"),
    ],
    dtype=[("id", "i4"), ("flux", "f4"), ("name", "U5")],
)


#: The console script installed next to the interpreter; the installed wheel
#: predates ``python -m torchfits``, so the entry point is the supported path.
_CONSOLE_SCRIPT = shutil.which("torchfits")


def _command(*args: Any) -> list[str]:
    if _CONSOLE_SCRIPT:
        return [_CONSOLE_SCRIPT, *map(str, args)]
    return [
        sys.executable,
        "-c",
        "import sys; from torchfits.cli.main import main; sys.exit(main())",
        *map(str, args),
    ]


def cli(*args: Any, expect: int | None = EXIT_OK) -> subprocess.CompletedProcess[str]:
    """Run the ``torchfits`` command and assert its exit code."""
    proc = subprocess.run(
        _command(*args),
        capture_output=True,
        text=True,
        check=False,
    )
    if expect is not None and proc.returncode != expect:
        raise AssertionError(
            f"torchfits {' '.join(map(str, args))} -> {proc.returncode}, "
            f"expected {expect}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def cli_json(*args: Any, expect: int | None = EXIT_OK) -> Any:
    proc = cli(*args, "--json", expect=expect)
    return json.loads(proc.stdout)


def first_image(path: Path) -> np.ndarray:
    """The first 2-D image HDU; compress rewrites the primary as an empty one."""
    with astropy_fits.open(path) as hdul:
        for hdu in hdul:
            data = hdu.data
            if data is not None and getattr(data, "ndim", 0) == 2 and data.size:
                return np.asarray(data)
    raise AssertionError(f"no 2-D image HDU in {path}")


@pytest.fixture(scope="module")
def mef(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checksummed primary image + binary table, written by astropy."""
    path = tmp_path_factory.mktemp("cli") / "mef.fits"
    astropy_fits.HDUList(
        [
            astropy_fits.PrimaryHDU(IMAGE),
            astropy_fits.BinTableHDU(TABLE, name="CAT"),
        ]
    ).writeto(path, overwrite=True, checksum=True)
    return path


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# inventory: info / header / table / probe
# ---------------------------------------------------------------------------


def test_info_matches_astropy(mef: Path) -> None:
    records = cli_json("info", mef)
    with astropy_fits.open(mef) as hdul:
        assert [r["hdu"] for r in records] == list(range(len(hdul)))
        assert [r["type"] for r in records] == ["IMAGE", "TABLE"]
        assert records[1]["name"] == "CAT"
        assert records[1]["nrows"] == len(hdul[1].data)
        assert records[1]["ncols"] == len(hdul[1].columns)


def test_header_keywords_match_astropy(mef: Path) -> None:
    cards = cli_json("header", mef, "-e", 1)
    with astropy_fits.open(mef, disable_image_compression=True) as hdul:
        expected = [str(card.keyword).strip() for card in hdul[1].header.cards]
    got = [card["keyword"] for card in cards]
    assert got == expected
    by_key = {card["keyword"]: card["value"] for card in cards}
    assert by_key["NAXIS"] == 2
    assert by_key["NAXIS2"] == len(TABLE)


def test_header_keyword_filter(mef: Path) -> None:
    cards = cli_json("header", mef, "-e", 1, "-k", "NAXIS*")
    assert {card["keyword"] for card in cards} == {"NAXIS", "NAXIS1", "NAXIS2"}
    assert all(card["hdu"] == 1 for card in cards)


def test_table_schema_and_preview(mef: Path) -> None:
    (record,) = cli_json("table", mef, "-e", 1)
    assert record["nrows"] == len(TABLE)
    assert [column["name"] for column in record["schema"]] == ["id", "flux", "name"]
    preview = record["preview"]
    assert len(preview) == len(TABLE)
    assert [row["id"] for row in preview] == TABLE["id"].tolist()
    assert [row["name"] for row in preview] == [str(v) for v in TABLE["name"]]


def test_probe_local_matches_info(mef: Path) -> None:
    probe = cli_json("probe", mef)
    info = cli_json("info", mef)
    assert [(r["hdu"], r["name"], r["type"]) for r in probe] == [
        (r["hdu"], r["name"], r["type"]) for r in info
    ]


# ---------------------------------------------------------------------------
# pixel maths: stats / cutout / compress / decompress
# ---------------------------------------------------------------------------


def test_stats_match_numpy(mef: Path) -> None:
    (record,) = cli_json("stats", mef, "-e", 0)
    assert record["shape"] == list(IMAGE.shape)
    assert record["min"] == pytest.approx(float(IMAGE.min()))
    assert record["max"] == pytest.approx(float(IMAGE.max()))
    assert record["mean"] == pytest.approx(float(IMAGE.mean()), rel=1e-6)
    assert record["std"] == pytest.approx(float(IMAGE.std()), rel=1e-5)
    assert record["median"] == pytest.approx(float(np.median(IMAGE)), abs=0.5)


def test_cutout_is_the_requested_window(mef: Path, workdir: Path) -> None:
    out = workdir / "cut.fits"
    cli("cutout", mef, out, "--box", "4,5,12,17")
    got = astropy_fits.getdata(out)
    np.testing.assert_allclose(got, IMAGE[5:17, 4:12], rtol=0, atol=0)


def test_compress_and_decompress_roundtrip(mef: Path, workdir: Path) -> None:
    compressed = workdir / "compressed.fits"
    restored = workdir / "restored.fits"
    cli("compress", mef, compressed, "--algorithm", "GZIP_1")
    # astropy is the oracle: the result must be a standard tile-compressed FITS,
    # which it marks with ZCMPTYPE/ZIMAGE. File size is not a useful assertion
    # here — a 768-pixel image pads to the same number of 2880-byte blocks.
    with astropy_fits.open(compressed, disable_image_compression=True) as hdul:
        assert any("ZCMPTYPE" in hdu.header for hdu in hdul)
        assert any(hdu.name == "CAT" for hdu in hdul), "table extension was dropped"
    np.testing.assert_allclose(first_image(compressed), IMAGE, atol=0)
    cli("decompress", compressed, restored)
    with astropy_fits.open(restored, disable_image_compression=True) as hdul:
        assert not any("ZCMPTYPE" in hdu.header for hdu in hdul)
    np.testing.assert_allclose(first_image(restored), IMAGE, atol=0)


def test_copy_is_byte_identical(mef: Path, workdir: Path) -> None:
    out = workdir / "copy.fits"
    cli("copy", mef, out)
    assert out.read_bytes() == mef.read_bytes()


# ---------------------------------------------------------------------------
# integrity and comparison: verify / diff
# ---------------------------------------------------------------------------


def test_verify_reports_checksums(mef: Path) -> None:
    records = cli_json("verify", mef)
    assert [r["status"] for r in records] == ["ok", "ok"]
    assert all(r["ok"] for r in records)


def test_verify_fails_on_a_corrupt_pixel(mef: Path, workdir: Path) -> None:
    bad = workdir / "corrupt.fits"
    shutil.copyfile(mef, bad)
    # First data byte of the primary image: past the 2880-byte header block.
    payload = 2880 + 100
    with bad.open("r+b") as handle:
        handle.seek(payload)
        byte = handle.read(1)
        handle.seek(payload)
        handle.write(bytes([byte[0] ^ 0xFF]))

    records = cli_json("verify", bad, expect=EXIT_VERIFY_FAIL)
    assert records[0]["ok"] is False
    assert records[0]["status"] == "fail"
    assert records[1]["ok"] is True


def test_verify_without_checksums_is_not_a_failure(workdir: Path) -> None:
    plain = workdir / "plain.fits"
    astropy_fits.PrimaryHDU(IMAGE).writeto(plain)
    (record,) = cli_json("verify", plain)
    assert record["ok"] is True
    assert record["status"] == "no_checksums"


def test_diff_same_file_is_ok(mef: Path) -> None:
    proc = cli("diff", mef, mef, expect=EXIT_OK)
    assert "differ" not in proc.stdout.lower()


def test_diff_reports_a_changed_file(mef: Path, workdir: Path) -> None:
    other = workdir / "other.fits"
    changed = IMAGE.copy()
    changed[0, 0] += 1.0
    astropy_fits.PrimaryHDU(changed).writeto(other)
    proc = cli("diff", mef, other, expect=EXIT_DIFF)
    assert proc.returncode == EXIT_DIFF
    report = proc.stdout + proc.stderr
    assert "NAXIS" in report or "image" in report.lower()


def test_missing_file_is_an_io_error(tmp_path: Path) -> None:
    proc = cli("info", tmp_path / "nope.fits", expect=EXIT_IO)
    assert "nope" in proc.stderr


# ---------------------------------------------------------------------------
# editing: setkey / convert
# ---------------------------------------------------------------------------


def test_setkey_set_rename_delete(mef: Path, workdir: Path) -> None:
    target = workdir / "edited.fits"
    cli("copy", mef, target)
    cli("setkey", target, "-e", 0, "-k", "OBJECT", "--value", "M31")
    cli("setkey", target, "-e", 0, "--rename", "OBJECT=TARGET")
    cli("setkey", target, "-e", 0, "--delete", "TARGET")
    header = astropy_fits.getheader(target)
    assert "OBJECT" not in header
    assert "TARGET" not in header


def test_setkey_writes_the_value(mef: Path, workdir: Path) -> None:
    target = workdir / "keyed.fits"
    cli("copy", mef, target)
    cli("setkey", target, "-e", 0, "-k", "EXPTIME", "--value", "12.5")
    header = astropy_fits.getheader(target)
    assert header["EXPTIME"] == pytest.approx(12.5)


def test_convert_table_to_parquet(mef: Path, workdir: Path) -> None:
    out = workdir / "cat.parquet"
    cli("convert", mef, out, "-e", 1)
    table = pq.read_table(out)
    assert table.num_rows == len(TABLE)
    assert table.column_names == ["id", "flux", "name"]
    assert table.column("id").to_pylist() == TABLE["id"].tolist()


# ---------------------------------------------------------------------------
# the same inventory checks on real archive files
# ---------------------------------------------------------------------------


def _has_table(path: Path) -> bool:
    with astropy_fits.open(path, memmap=True) as hdul:
        return any(
            isinstance(hdu, (astropy_fits.BinTableHDU, astropy_fits.TableHDU))
            for hdu in hdul
        )


def _corpus_file(family: str) -> Path:
    for source in corpus.sources(tier=os.environ.get("TORCHFITS_CORPUS_TIER", "ci")):
        if family not in corpus.families(source):
            continue
        for path in corpus.source_files(source):
            if not path.is_file():
                continue
            # A source declares families collectively (the astropy regression
            # corpus ships images and tables), so the first present file can be
            # the wrong shape; pick one that really carries the family asked for.
            if family == "table" and not _has_table(path):
                continue
            return path
    pytest.skip(f"no {family} corpus file is present; run `pixi run corpus-fetch`")


def test_corpus_info_matches_astropy() -> None:
    path = _corpus_file("image")
    records = cli_json("info", path)
    with astropy_fits.open(path, memmap=True) as hdul:
        assert len(records) == len(hdul)
        for record, hdu in zip(records, hdul, strict=True):
            extname = (hdu.header.get("EXTNAME") or "").strip()
            if extname:
                assert record["name"] == extname


def test_corpus_table_matches_astropy() -> None:
    path = _corpus_file("table")
    with astropy_fits.open(path, memmap=True) as hdul:
        index = next(
            i
            for i, hdu in enumerate(hdul)
            if isinstance(hdu, (astropy_fits.BinTableHDU, astropy_fits.TableHDU))
        )
        expected = hdul[index]
        (record,) = cli_json("table", path, "-e", index)
        assert record["nrows"] == len(expected.data)
        assert [column["name"] for column in record["schema"]] == list(
            expected.data.columns.names
        )


def test_corpus_verify_never_crashes() -> None:
    path = _corpus_file("image")
    # Real archive files may or may not carry CHECKSUM; either way the command
    # must finish with a defined verdict, not an exception.
    records = cli_json("verify", path, expect=None)
    assert records
    assert {r["status"] for r in records} <= {"ok", "fail", "no_checksums"}
