"""The corpus manifest is the contract: validate it without touching the network."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import corpus  # noqa: E402

MANIFEST = json.loads(corpus.MANIFEST_PATH.read_text(encoding="utf-8"))
SOURCES = corpus.sources(tier="full")
BY_ID = {source["id"]: source for source in SOURCES}

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_manifest_ids_are_unique_and_complete() -> None:
    assert MANIFEST["version"] == 1
    assert MANIFEST["root"] == "benchmarks_data/corpus"
    assert len(BY_ID) == len(SOURCES)
    for source in SOURCES:
        assert source["id"]
        assert source["tier"] in {"ci", "full"}
        assert source["archive"]
        assert source["family"]
        assert source["license"]
        if source["kind"] == "https":
            assert source["url"].startswith("https://")
            assert source["filename"]
        elif source["kind"] == "local":
            assert (ROOT / source["path"]).is_file(), source["path"]
        elif source["kind"] == "astropy":
            assert source["glob"]
        else:  # pragma: no cover - guards typos in the data file
            raise AssertionError(f"unknown kind {source['kind']!r}")


def test_full_tier_covers_every_required_family() -> None:
    covered = {family for source in SOURCES for family in corpus.families(source)}
    assert set(corpus.REQUIRED_FAMILIES) <= covered


def test_full_tier_is_broad_enough() -> None:
    declared = sum(int(source.get("bytes") or 0) for source in SOURCES)
    assert declared >= 10 * 1024**3, "full corpus should declare at least 10 GB"
    archives = {source["archive"] for source in SOURCES}
    assert len(archives) >= 3, archives
    assert any("SDSS" in archive for archive in archives)
    assert any("MAST" in archive for archive in archives)
    assert any("CFHT" in archive for archive in archives)


def test_ci_tier_stays_small_and_offline_friendly() -> None:
    ci = corpus.sources(tier="ci")
    assert ci, "the ci tier must not be empty"
    assert all(source["tier"] == "ci" for source in ci)
    remote_bytes = sum(
        int(source.get("bytes") or 0) for source in ci if source["kind"] == "https"
    )
    assert remote_bytes <= 50 * 1024**2, remote_bytes
    assert any(source["kind"] == "astropy" for source in ci)


def test_declared_hashes_are_well_formed_when_present() -> None:
    for source in SOURCES:
        digest = source.get("sha256")
        if digest is not None:
            assert HEX64.match(digest), (source["id"], digest)


def test_astropy_corpus_resolves_from_the_installed_package() -> None:
    source = BY_ID["astropy_fits_corpus"]
    files = corpus.source_files(source)
    assert files, "astropy ships its FITS regression corpus"
    assert all(path.is_file() for path in files)
    assert corpus.source_available(source)


def test_local_cfht_sources_point_at_real_files() -> None:
    local = [source for source in SOURCES if source["kind"] == "local"]
    assert len(local) >= 10
    for source in local:
        path = corpus.source_path(source)
        assert path.is_file()
        assert path.stat().st_size == source["bytes"]


def test_iter_corpus_skips_missing_remote_files(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = list(corpus.iter_corpus(tier="full"))
    for source, path in seen:
        assert source["id"]
        assert path.is_file()


def test_missing_reports_only_absent_sources() -> None:
    absent = corpus.missing(tier="full")
    for source in absent:
        assert not corpus.source_available(source)


def test_fetcher_lists_without_network(capsys: pytest.CaptureFixture[str]) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import fetch_corpus

    assert fetch_corpus.main(["--tier", "ci", "--list"]) == 0
    out = capsys.readouterr().out
    assert "sdss_spec_0266_51630_0001" in out
    assert "astropy_fits_corpus" in out


def test_summary_reports_tiers_and_archives() -> None:
    summary = corpus.summary(tier="full")
    assert summary["sources"] == len(SOURCES)
    assert summary["tier"] == "full"
    assert "image" in summary["families"]
    assert summary["declared_bytes"] >= 10 * 1024**3
