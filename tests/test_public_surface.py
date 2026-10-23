"""The public-surface registry, plus behaviour for the symbols it flags.

``benchmarks/surface.py`` records every public torchfits symbol of the
*installed* distribution and matches each one against this test tree by name.
That match is cheap, so it needs a companion: these tests make the flagged
symbols do real work, otherwise a stray mention in a comment would count as
coverage.

When torchfits grows a public symbol, ``pixi run surface-report --update``
regenerates the registry and the first test here reports anything new that no
suite exercises yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import torchfits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import surface  # noqa: E402
from torchfits import cache, transforms, where
from torchfits.data import BandInfo, CutoutSpec
from torchfits.hdu import TableHDU, TableHDURef, TensorHDU
from torchfits.transforms.state import DataState, Payload

pa = pytest.importorskip("pyarrow")


# --------------------------------------------------------------------------
# Registry gates
# --------------------------------------------------------------------------


def test_registry_matches_installed_torchfits() -> None:
    """The tracked registry is what the installed wheel actually exports."""
    entries = surface.collect()
    assert surface.load_registry() == entries, (
        "benchmarks/surface/public_surface.json is stale; run "
        "`pixi run surface-report --update` and commit the diff."
    )


def test_no_public_symbol_is_unexercised() -> None:
    """Every public symbol appears in at least one test file."""
    missing, _ = surface.report(surface.collect())
    assert not missing, (
        "public symbols no test mentions: "
        + ", ".join(entry.path for entry in missing)
    )


def test_exemptions_are_live_and_justified() -> None:
    entries = surface.collect()
    exemptions = surface.load_exemptions()
    assert not surface.stale_exemptions(entries), (
        "exemptions name symbols torchfits no longer exports: "
        + ", ".join(surface.stale_exemptions(entries))
    )
    assert all(reason.strip() for reason in exemptions.values()), (
        "every exemption needs a written reason"
    )


# --------------------------------------------------------------------------
# HDU accessors
# --------------------------------------------------------------------------


def test_table_hdu_feat_types_marks_numeric_columns() -> None:
    """``feat_types`` classifies float/complex columns as numerical."""
    table = TableHDU(
        {
            "FLUX": torch.zeros(3),
            "ID": torch.arange(3, dtype=torch.int32),
            "NAME": np.array(["a", "b", "c"]),
        }
    )
    assert table.feat_types == {
        "FLUX": "numerical",
        "ID": "categorical",
        "NAME": "categorical",
    }


def test_tensor_hdu_dtype_and_shape_strings(tmp_path) -> None:
    """``dtype_str``/``shape_str`` work on in-memory and file-backed HDUs."""
    memory = TensorHDU(data=torch.zeros(2, 3), header=torchfits.Header())
    assert memory.dtype_str == "float32"
    assert memory.shape_str == "(2, 3)"

    path = tmp_path / "img.fits"
    torchfits.write(str(path), np.arange(6, dtype=np.float32).reshape(2, 3))
    on_disk = TensorHDU(header=torchfits.read_header(str(path), 0), source_path=str(path))
    assert on_disk.shape_str == "(2, 3)"
    assert on_disk.dtype_str == "float32"


def test_table_hdu_ref_delete_rows_file_updates_the_file(tmp_path) -> None:
    """``TableHDURef.delete_rows_file`` drops the row and refreshes the view."""
    path = tmp_path / "delete.fits"
    torchfits.table.write(
        str(path), {"A": np.arange(6, dtype=np.int32)}, extname="CAT"
    )
    ref = TableHDURef(
        header=torchfits.read_header(str(path), 1),
        source_path=str(path),
        source_hdu=1,
        columns=["A"],
    )
    refreshed = ref.delete_rows_file(1)
    # The call returns a refreshed view of the same file.
    assert isinstance(refreshed, TableHDURef)
    assert refreshed is not ref
    assert torchfits.read_nrows(str(path), 1) == 5
    np.testing.assert_array_equal(
        torchfits.table.read(str(path), hdu=1)["A"].to_numpy(),
        np.array([0, 2, 3, 4, 5], dtype=np.int32),
    )


# --------------------------------------------------------------------------
# Cache introspection
# --------------------------------------------------------------------------


def test_get_optimal_cache_config_describes_the_environment() -> None:
    config = cache.get_optimal_cache_config()
    assert set(config) == {
        "max_files",
        "max_memory_mb",
        "disk_cache_gb",
        "prefetch_enabled",
        "environment",
    }
    assert config["max_files"] > 0
    assert config["max_memory_mb"] > 0
    assert isinstance(config["prefetch_enabled"], bool)


def test_cache_manager_get_stats_reports_counters() -> None:
    manager = cache.CacheManager()
    stats = manager.get_stats()
    assert {"hits", "misses", "hit_rate", "config", "cpp_cache_size"} <= set(stats)
    assert 0.0 <= stats["hit_rate"] <= 1.0


# --------------------------------------------------------------------------
# data namespace
# --------------------------------------------------------------------------


def test_band_info_serializes_to_a_plain_dict() -> None:
    band = BandInfo(index=1, name="SCI", shape=(4, 5), dtype="float32", zeropoint=25.0)
    payload = band.to_dict()
    assert payload == {
        "index": 1,
        "name": "SCI",
        "shape": (4, 5),
        "dtype": "float32",
        "role": "flux",
        "bitpix": None,
        "zeropoint": 25.0,
        "photflam": None,
        "photplam": None,
        "exptime": None,
    }
    assert band.flux_scale() == pytest.approx(10 ** (-0.4 * 25.0))


def test_cutout_spec_is_the_documented_six_tuple() -> None:
    assert CutoutSpec == tuple[str, int, int, int, int, int]


# --------------------------------------------------------------------------
# Table backends
# --------------------------------------------------------------------------


def test_read_arrow_is_the_arrow_reader(tmp_path) -> None:
    path = tmp_path / "arrow.fits"
    torchfits.table.write(
        str(path),
        {"A": np.arange(4, dtype=np.int32), "B": np.linspace(0, 1, 4)},
    )
    table = torchfits.table.read_arrow(str(path), hdu=1)
    assert isinstance(table, pa.Table)
    assert table.column_names == ["A", "B"]
    np.testing.assert_array_equal(table["A"].to_numpy(), np.arange(4))


def test_table_backends_are_the_accepted_backend_values(tmp_path) -> None:
    path = tmp_path / "backends.fits"
    torchfits.table.write(str(path), {"A": np.arange(4, dtype=np.int32)})
    assert torchfits.table.TABLE_BACKENDS == frozenset({"auto", "torch", "cpp"})
    native = torchfits.table.read(str(path), hdu=1, backend="cpp")
    eager = torchfits.table.read(str(path), hdu=1, backend="torch")
    np.testing.assert_array_equal(native["A"].to_numpy(), eager["A"].to_numpy())
    with pytest.raises(ValueError, match="backend must be one of"):
        torchfits.table.read(str(path), hdu=1, backend="numpy")


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------


def test_as_module_wraps_a_transform_for_nn_sequential() -> None:
    x = torch.arange(4.0)
    module = transforms.AsModule(transforms.AffineTransform(scale=2.0, offset=1.0))
    torch.testing.assert_close(module(x), x * 2.0 + 1.0)
    chained = torch.nn.Sequential(module, module)
    torch.testing.assert_close(chained(x), (x * 2.0 + 1.0) * 2.0 + 1.0)


def test_fitstransform_state_protocol_matches_forward_output() -> None:
    """``inverse_expects`` accepts exactly what ``forward`` produces."""
    scaler = transforms.FITSHeaderScale.from_header({"BSCALE": 2.0})
    assert scaler.inverse_state() is DataState.STORED
    assert scaler.inverse_expects() == frozenset({DataState.PHYSICAL})
    # A multi-state whitelist names no single state to restore.
    assert transforms.InterquantileScale().inverse_state() is None

    stateless = transforms.AffineTransform(scale=2.0)
    assert stateless.inverse_state() is None
    assert stateless.inverse_expects() is None


def test_carries_companions_sees_ivar_only() -> None:
    payload = {"flux": torch.zeros(3), "ivar": torch.ones(3)}
    transform = transforms.AffineTransform(scale=2.0)
    assert transform.carries_companions(payload) is True
    assert transform.carries_companions(torch.zeros(3)) is False


def test_ivar_helpers_propagate_uncertainty() -> None:
    ivar = torch.ones(3)
    torch.testing.assert_close(transforms.FITSTransform.scale_ivar(ivar, 2.0), ivar / 4)
    torch.testing.assert_close(
        transforms.FITSTransform.divide_ivar(ivar, 2.0), ivar * 4
    )
    slope = torch.tensor([2.0, 1.0, 0.0])
    # A locally flat map (slope 0) is no longer invertible: ivar collapses to 0.
    torch.testing.assert_close(
        transforms.FITSTransform.delta_ivar(ivar, slope),
        torch.tensor([0.25, 1.0, 0.0]),
    )
    assert transforms.FITSTransform.scale_ivar(None, 2.0) is None


def test_interquantile_normalize_computes_its_own_scale() -> None:
    x = torch.arange(5.0)
    transform = transforms.InterquantileNormalize(q_low=0.0, q_high=1.0)
    scale = transform.compute_scale(x)
    torch.testing.assert_close(scale.flatten(), torch.tensor([4.0]))
    torch.testing.assert_close(transform(x), x / 4.0)


def test_payload_round_trips_between_container_forms() -> None:
    flux = torch.arange(4.0)
    ivar = torch.ones(4)
    payload = Payload.coerce({"flux": flux, "ivar": ivar, "meta": {"path": "x.fits"}})
    assert isinstance(payload, Payload)
    assert Payload.coerce(payload) is payload
    assert Payload.coerce(flux).ivar is None
    as_dict = payload.to_dict()
    assert as_dict["flux"] is flux
    assert as_dict["ivar"] is ivar
    assert as_dict["meta"] == {"path": "x.fits"}
    with pytest.raises(TypeError):
        Payload.coerce({"ivar": ivar})


# --------------------------------------------------------------------------
# where namespace
# --------------------------------------------------------------------------


def test_normalize_where_syntax_translates_c_operators() -> None:
    assert where.normalize_where_syntax("A == 1 && B > 2") == "A == 1  AND  B > 2"
    assert where.normalize_where_syntax("A == 1 || B != 2") == "A == 1  OR  B != 2"


def test_parse_where_literal_handles_quotes_numbers_and_words() -> None:
    assert where.parse_where_literal("'x y'") == "x y"
    assert where.parse_where_literal("3") == 3
    assert where.parse_where_literal("-2.5") == -2.5
    assert where.parse_where_literal("true") is True
    assert where.parse_where_literal("null") is None
    assert where.parse_where_literal("STAR_A") == "STAR_A"
    with pytest.raises(ValueError):
        where.parse_where_literal("   ")


def test_tokenize_where_expression_splits_words_and_operators() -> None:
    assert where.tokenize_where_expression("A >= 2 AND B < 3") == [
        ("WORD", "A"),
        ("OP", ">="),
        ("WORD", "2"),
        ("WORD", "AND"),
        ("WORD", "B"),
        ("OP", "<"),
        ("WORD", "3"),
    ]


def test_where_columns_from_ast_lists_each_column_once() -> None:
    tree = where.parse_where_expression("A >= 2 AND (B < 3 OR A == 0)")
    assert where.where_columns_from_ast(tree) == ["A", "B"]
    with pytest.raises(ValueError):
        where.where_columns_from_ast(("nonsense",))


def test_where_identifier_re_matches_plain_column_names() -> None:
    assert where.where_identifier_re.match("FLUX_1") is not None
    assert where.where_identifier_re.match("1FLUX") is None
    assert where.where_identifier_re.match("FLUX-BAND") is None
