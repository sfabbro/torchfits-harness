"""Real-archive FITS corpus: manifest, paths, availability.

The corpus is described by ``manifest.json`` and fetched by
``scripts/fetch_corpus.py``. Nothing here downloads: tests and benchmarks ask
whether a file is present and skip cleanly when it is not, so a fresh clone
runs without network access.

Layout on disk (gitignored)::

    benchmarks_data/corpus/<source-id>/<filename>

Set ``TORCHFITS_CORPUS_DIR`` to keep the files somewhere else (for example a
scratch volume); the manifest paths stay relative.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
DEFAULT_DATA_ROOT = REPO_ROOT / "benchmarks_data" / "corpus"

#: Families every full-tier run must cover.
REQUIRED_FAMILIES: tuple[str, ...] = (
    "image",
    "cube",
    "spectrum",
    "table",
    "mef",
)


def data_root() -> Path:
    override = os.environ.get("TORCHFITS_CORPUS_DIR")
    return Path(override).expanduser() if override else DEFAULT_DATA_ROOT


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sources(
    *,
    tier: str = "full",
    family: str | None = None,
    archive: str | None = None,
) -> list[dict[str, Any]]:
    """Return manifest entries for a tier (``ci`` is a subset of ``full``)."""
    manifest = load_manifest()
    allowed = {"ci"} if tier == "ci" else {"ci", "full"}
    out = [s for s in manifest["sources"] if s.get("tier", "full") in allowed]
    if family is not None:
        out = [s for s in out if family in s.get("families", [s.get("family")])]
    if archive is not None:
        out = [s for s in out if s.get("archive") == archive]
    return out


def families(source: dict[str, Any]) -> tuple[str, ...]:
    declared = source.get("families")
    if declared:
        return tuple(declared)
    return (source["family"],)


def source_files(source: dict[str, Any]) -> list[Path]:
    """Files this source provides on disk (empty for the astropy source)."""
    kind = source["kind"]
    if kind == "astropy":
        return astropy_files()
    if kind == "local":
        return [REPO_ROOT / source["path"]]
    if kind == "https":
        return [data_root() / source["id"] / source["filename"]]
    raise ValueError(f"unknown corpus kind {kind!r} in {source['id']}")


def source_available(source: dict[str, Any]) -> bool:
    files = source_files(source)
    return bool(files) and all(path.is_file() for path in files)


def source_path(source: dict[str, Any]) -> Path:
    """The single local path for a one-file source."""
    files = source_files(source)
    if len(files) != 1:
        raise ValueError(f"{source['id']} provides {len(files)} files")
    return files[0]


def require(source: dict[str, Any]):
    """Skip the calling test when the corpus file has not been fetched."""
    import pytest

    files = source_files(source)
    if not files or not all(path.is_file() for path in files):
        pytest.skip(
            f"corpus source {source['id']!r} is not present; run "
            "`pixi run corpus-fetch --tier full`"
        )
    return files[0] if len(files) == 1 else files


def astropy_files() -> list[Path]:
    """astropy's own FITS regression corpus (ships with the package)."""
    try:
        from astropy.io import fits
    except ImportError:  # pragma: no cover - harness always has astropy
        return []
    return sorted((Path(fits.__file__).resolve().parent / "tests" / "data").glob("*.fits"))


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_corpus(
    *,
    tier: str = "full",
    family: str | None = None,
    require_present: bool = True,
) -> Iterator[tuple[dict[str, Any], Path]]:
    """Yield ``(source, path)`` pairs; one-file sources only."""
    for source in sources(tier=tier, family=family):
        files = source_files(source)
        if not files:
            continue
        if require_present and not all(path.is_file() for path in files):
            continue
        if len(files) == 1:
            yield source, files[0]
        else:
            for path in files:
                yield {**source, "file": path.name}, path


def missing(*, tier: str = "full") -> list[dict[str, Any]]:
    return [s for s in sources(tier=tier) if not source_available(s)]


def summary(*, tier: str = "full") -> dict[str, Any]:
    entries = sources(tier=tier)
    present = [s for s in entries if source_available(s)]
    declared = sum(int(s.get("bytes") or 0) for s in entries)
    return {
        "tier": tier,
        "sources": len(entries),
        "present": len(present),
        "families": sorted({f for s in entries for f in families(s)}),
        "archives": sorted({s.get("archive", "?") for s in entries}),
        "declared_bytes": declared,
    }
