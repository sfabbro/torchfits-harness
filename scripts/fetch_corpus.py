#!/usr/bin/env python3
"""Fetch and verify the real-archive FITS corpus.

Examples::

    pixi run corpus-fetch                 # CI tier (~8 MB)
    pixi run corpus-fetch --tier full     # the whole corpus (~16 GB)
    pixi run corpus-fetch --only manga    # match source ids
    pixi run corpus-fetch --verify        # hashes only, no network
    pixi run corpus-fetch --update-manifest

Files land in ``benchmarks_data/corpus/<source-id>/`` (gitignored). Downloads
resume from a ``.part`` file, so a dropped connection never restarts 6 GB.

``kind: local`` sources (the CFHT frames that ship outside git) are checked in
place; ``kind: astropy`` is the corpus installed with astropy. Neither touches
the network.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.corpus import (  # noqa: E402
    MANIFEST_PATH,
    data_root,
    families,
    load_manifest,
    sha256_file,
    source_available,
    source_files,
    sources,
)

USER_AGENT = "torchfits-harness-corpus/1.0"
CHUNK = 1 << 20


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def _download(url: str, dest: Path, *, expected: int | None) -> None:
    """Stream ``url`` into ``dest``, resuming a previous ``.part``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and expected is not None and start == expected:
            part.replace(dest)
            return
        raise
    with response:
        partial = response.status == 206
        if start and not partial:
            # The server ignored Range; write the whole body from byte zero.
            start = 0
        mode = "ab" if start and partial else "wb"
        received = start
        last_report = 0.0
        with part.open(mode) as handle:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                received += len(block)
                now = time.monotonic()
                if now - last_report > 2.0:
                    last_report = now
                    if expected:
                        pct = 100.0 * received / expected
                        sys.stderr.write(
                            f"\r  {_human(received)} / {_human(expected)} "
                            f"({pct:5.1f}%)"
                        )
                    else:
                        sys.stderr.write(f"\r  {_human(received)}")
                    sys.stderr.flush()
        if expected and received != expected:
            raise RuntimeError(
                f"size mismatch for {dest.name}: got {received}, expected {expected}"
            )
    sys.stderr.write("\r" + " " * 60 + "\r")
    part.replace(dest)


def _check_hash(source: dict[str, Any], path: Path) -> str | None:
    expected = source.get("sha256")
    if not expected:
        return None
    actual = sha256_file(path)
    if actual != expected:
        return f"sha256 mismatch for {path}: {actual} != {expected}"
    return None


def _fetch(source: dict[str, Any], *, verify_only: bool) -> tuple[bool, str]:
    kind = source["kind"]
    if kind == "astropy":
        files = source_files(source)
        return (bool(files), f"{len(files)} files from astropy")
    if kind == "local":
        path = source_files(source)[0]
        if not path.is_file():
            return False, f"missing local file {path.relative_to(ROOT)}"
        problem = _check_hash(source, path)
        return (problem is None, problem or f"present ({_human(path.stat().st_size)})")

    dest = source_files(source)[0]
    expected = source.get("bytes")
    if verify_only:
        if not dest.is_file():
            return False, "not fetched"
        problem = _check_hash(source, dest)
        return (problem is None, problem or "hash ok")
    if dest.is_file():
        problem = _check_hash(source, dest)
        if problem is None:
            return True, f"present ({_human(dest.stat().st_size)})"
        sys.stderr.write(f"  {problem}; refetching\n")
    sys.stderr.write(f"  GET {source['url']}\n")
    _download(source["url"], dest, expected=expected)
    problem = _check_hash(source, dest)
    return (problem is None, problem or f"downloaded ({_human(dest.stat().st_size)})")


def _update_manifest(chosen: list[dict[str, Any]]) -> None:
    manifest = load_manifest()
    by_id = {entry["id"]: entry for entry in manifest["sources"]}
    updated = 0
    for source in chosen:
        files = source_files(source)
        if not files or not all(path.is_file() for path in files):
            continue
        total = sum(path.stat().st_size for path in files)
        digest = sha256_file(files[0]) if len(files) == 1 else None
        entry = by_id[source["id"]]
        if entry.get("bytes") != total or entry.get("sha256") != digest:
            entry["bytes"] = total
            entry["sha256"] = digest
            updated += 1
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest updated: {updated} source(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["ci", "full"], default="ci")
    parser.add_argument("--only", default="", help="regex on source id / family")
    parser.add_argument("--verify", action="store_true", help="check hashes, no download")
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="write measured bytes and sha256 back into manifest.json",
    )
    parser.add_argument("--list", action="store_true", help="list sources and exit")
    parser.add_argument("--force", action="store_true", help="refetch even if present")
    args = parser.parse_args(argv)

    chosen = sources(tier=args.tier)
    if args.only:
        pattern = re.compile(args.only)
        chosen = [
            s
            for s in chosen
            if pattern.search(s["id"])
            or any(pattern.search(f) for f in families(s))
            or pattern.search(s.get("archive", ""))
        ]
    if not chosen:
        print("no sources selected", file=sys.stderr)
        return 2

    if args.list:
        for source in chosen:
            present = "yes" if source_available(source) else "no"
            size = _human(source.get("bytes") or 0)
            print(
                f"{source['id']:<34} {source['tier']:<4} {present:<3} "
                f"{size:>9}  {','.join(families(source)):<22} {source['archive']}"
            )
        return 0

    if args.force:
        for source in chosen:
            if source["kind"] != "https":
                continue
            for path in source_files(source):
                if path.is_file():
                    path.unlink()

    failures: list[tuple[str, str]] = []
    for source in chosen:
        sys.stderr.write(f"[{source['id']}] ")
        try:
            ok, detail = _fetch(source, verify_only=args.verify)
        except Exception as exc:  # noqa: BLE001 - report and continue
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        print(("ok  " if ok else "FAIL") + f" {detail}")
        if not ok:
            failures.append((source["id"], detail))

    if args.update_manifest:
        _update_manifest(chosen)

    if failures:
        print(f"\n{len(failures)} source(s) failed:", file=sys.stderr)
        for source_id, detail in failures:
            print(f"  {source_id}: {detail}", file=sys.stderr)
        return 1
    total = sum(
        path.stat().st_size
        for source in chosen
        for path in source_files(source)
        if path.is_file()
    )
    print(f"\n{len(chosen)} source(s) ready, {_human(total)} at {data_root()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
