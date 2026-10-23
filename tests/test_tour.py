"""Run docs/tour.md: every bash block, in order, then inspect what it produced.

Documentation that is not executed rots. This test concatenates the fenced
``bash`` blocks from the tour into one script — they share shell state on
purpose — runs it from the repository root, and then checks the files it wrote
with astropy. ``pixi run`` is stripped because the test already runs inside the
Pixi environment; the ``torchfits`` command is resolved to the installed
console script so the same executable a user gets is exercised.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import corpus  # noqa: E402

astropy_fits = pytest.importorskip("astropy.io.fits")

TOUR = ROOT / "docs" / "tour.md"
SPEC_SOURCE = "sdss_spec_0266_51630_0001"


def _blocks() -> list[str]:
    text = TOUR.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL)
    # The first block is the prerequisite (`corpus-fetch`), which this test has
    # already asserted: the file is present or the test skips.
    return [block for block in blocks if "corpus-fetch" not in block]


def _script() -> str:
    console = shutil.which("torchfits")
    assert console, "the torchfits console script is not on PATH"
    lines = ["set -eu", f"PATH={Path(console).parent}:$PATH"]
    for block in _blocks():
        for line in block.splitlines():
            if line.startswith("pixi run python "):
                line = sys.executable + line[len("pixi run python") :]
            elif line.startswith("pixi run "):
                stripped = line[len("pixi run ") :]
                if stripped.startswith("torchfits "):
                    line = console + stripped[len("torchfits") :]
                else:
                    line = stripped
            lines.append(line)
        lines.append("")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def tour_output(tmp_path_factory: pytest.TempPathFactory) -> tuple[subprocess.CompletedProcess[str], Path]:
    source = next(
        (s for s in corpus.sources(tier="ci") if s["id"] == SPEC_SOURCE), None
    )
    if source is None or not corpus.source_available(source):
        pytest.skip("run `pixi run corpus-fetch --tier ci` to execute the tour")

    workdir = tmp_path_factory.mktemp("tour")
    env = {**os.environ, "TMPDIR": str(workdir)}
    proc = subprocess.run(
        ["bash", "-c", _script()],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc, workdir / "torchfits-tour"


def test_tour_runs(tour_output: tuple[subprocess.CompletedProcess[str], Path]) -> None:
    proc, _ = tour_output
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_tour_reports_the_predicate_count(
    tour_output: tuple[subprocess.CompletedProcess[str], Path],
) -> None:
    proc, _ = tour_output
    assert re.search(r"positive flux rows: \d+", proc.stdout), proc.stdout


def test_tour_writes_a_parquet_copy(
    tour_output: tuple[subprocess.CompletedProcess[str], Path],
) -> None:
    _, out = tour_output
    pqt = pytest.importorskip("pyarrow.parquet")
    table = pqt.read_table(out / "coadd.parquet")
    assert table.num_rows == 3818
    assert table.column_names[:2] == ["flux", "loglam"]


def test_tour_cutout_is_the_requested_box(
    tour_output: tuple[subprocess.CompletedProcess[str], Path],
) -> None:
    _, out = tour_output
    data = astropy_fits.getdata(out / "cutout.fits")
    # The cube was np.arange(4*64*64).reshape(4, 64, 64); the box is x=8..24,
    # y=8..24 and every plane survives the cut.
    assert data.shape == (4, 16, 16)
    assert data[0, 0, 0] == pytest.approx(8 * 64 + 8)
    assert data[0, -1, -1] == pytest.approx(23 * 64 + 23)
    assert data[-1, 0, 0] == pytest.approx(3 * 64 * 64 + 8 * 64 + 8)


def test_tour_edited_header(
    tour_output: tuple[subprocess.CompletedProcess[str], Path],
) -> None:
    _, out = tour_output
    header = astropy_fits.getheader(out / "edited.fits", ext=1)
    assert header["OBJECT"] == "SDSS J0266"
