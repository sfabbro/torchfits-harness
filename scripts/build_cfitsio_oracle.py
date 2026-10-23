#!/usr/bin/env python3
"""Build the pure-CFITSIO value oracle against a torchfits checkout.

The harness does not vendor CFITSIO. This script finds the tree that a
torchfits checkout already has (``extern/cfitsio``) and builds
``dump_cfitsio_values`` beside the timing driver.

Search order:

1. ``--cfitsio-root`` / ``TORCHFITS_CFITSIO_ROOT``
2. ``TORCHFITS_SOURCE``/``extern/cfitsio``
3. ``../torchfits/extern/cfitsio`` next to this repo

Exit code 3 means "no vendored CFITSIO here"; the calling test turns that into
a skip, never a failure.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "cfitsio_oracle"


def candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for env in ("TORCHFITS_CFITSIO_ROOT",):
        value = os.environ.get(env)
        if value:
            roots.append(Path(value).expanduser())
    source = os.environ.get("TORCHFITS_SOURCE")
    if source:
        roots.append(Path(source).expanduser() / "extern" / "cfitsio")
    roots.append(ROOT.parent / "torchfits" / "extern" / "cfitsio")
    roots.append(ROOT / "extern" / "cfitsio")
    return roots


def find_root(explicit: str | None) -> Path | None:
    candidates = [Path(explicit).expanduser()] if explicit else candidate_roots()
    for candidate in candidates:
        if (candidate / "CMakeLists.txt").is_file():
            return candidate
    return None


def build(root: Path, *, build_dir: Path = BUILD_DIR) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake = os.environ.get("CMAKE", "cmake")
    subprocess.run(
        [
            cmake,
            "-S",
            str(ROOT / "benchmarks" / "cfitsio_direct"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DTORCHFITS_CFITSIO_ROOT={root}",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [cmake, "--build", str(build_dir), "--target", "dump_cfitsio_values", "-j"],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    exe = build_dir / "dump_cfitsio_values"
    if not exe.is_file():
        candidates = [p for p in build_dir.rglob("dump_cfitsio_values") if p.is_file()]
        if not candidates:
            raise FileNotFoundError("dump_cfitsio_values was not produced")
        exe = candidates[0]
    return exe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfitsio-root", default=None)
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
    args = parser.parse_args(argv)

    root = find_root(args.cfitsio_root)
    if root is None:
        print(
            "no vendored CFITSIO found; set TORCHFITS_CFITSIO_ROOT or "
            "TORCHFITS_SOURCE to a torchfits checkout",
            file=sys.stderr,
        )
        return 3
    try:
        exe = build(root, build_dir=args.build_dir)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr or str(exc))
        return 1
    print(exe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
