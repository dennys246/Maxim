"""H2 (2026-09-16, Option A): declared range VALUES enter a GAINED modality's geometry tag.

The hole this closes, as a red gate: `saturation` re-declared `[0,10] → [0,20]` (#726) changed
where every reading lands on its two bases while the tag stayed identical, so persisted world
nodes silently pattern-completed across the change and the committed L11 replay re-printed
different numbers with no mechanism change. Verified RED on the pre-H2 encoder (2026-09-16, with
pre-existing APIs only): the two tags below were EQUAL and no mismatch warning fired. Note the
gate is the tag + the warning — the two readings happened to land in different nodes even
pre-H2 (saturation 10 normalizes to 1.0 vs 0.5, a large enough vector move), so node identity
is asserted only as a sanity check, not as the red condition.

Scope (Option A, owner's call): only where gain applies. Ungained tags are byte-identical to the
H3 golden — pinned literally there and re-asserted here through the live path.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.similarity.encoder import SensorEncoder, encoding_geometry_tag, sensor_geometry_fields

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "encoder_golden_v1.json"

WORLD = {"saturation": 10.0, "light": 9.0, "y": 60.0}
R_OLD = {"saturation": (0, 10), "light": (0, 15), "y": (0, 128)}
R_NEW = {"saturation": (0, 20), "light": (0, 15), "y": (0, 128)}


def _encode(enc: SensorEncoder, ranges, modality="world", agent="a"):
    return enc.encode_sensors(agent_id=agent, sensors=dict(WORLD), modality=modality, ranges=ranges)


def test_saturation_re_declaration_is_a_different_space_and_is_warned_once(caplog: pytest.LogCaptureFixture) -> None:
    """The #726 hole, closed: two range declarations on one gained sensor → two tags; a node stamped
    under the old one is MASKED (not merged) when the new encoder scans, and the mismatch is warned
    exactly once per (modality, stored, live) triple."""
    ec = EntorhinalCortex(ECConfig())
    enc = SensorEncoder(ec=ec)
    old_node = _encode(enc, R_OLD, agent="old")
    assert old_node is not None
    old_tag = ec._substrate_node_geometries[old_node]
    with caplog.at_level(logging.WARNING, logger="maxim.similarity.ec"):
        new_node = _encode(enc, R_NEW, agent="new")
        new_node_again = _encode(enc, R_NEW, agent="new2")
    assert new_node is not None and new_node != old_node, "the old-range node must not absorb the new-range reading"
    new_tag = ec._substrate_node_geometries[new_node]
    assert old_tag != new_tag
    assert ec._substrate_node_geometries[new_node_again] == new_tag
    mismatch = [r for r in caplog.records if "geometry mismatch" in r.getMessage()]
    assert len(mismatch) == 1, [r.getMessage() for r in mismatch]
    assert old_tag in mismatch[0].getMessage() and new_tag in mismatch[0].getMessage()


def test_only_the_range_values_differ_between_the_two_tags() -> None:
    f_old = sensor_geometry_fields(
        modality="world", declared_sensors=sorted(R_OLD), declared_ranges=R_OLD, embedding_dim=384, gain=3.0
    )
    f_new = sensor_geometry_fields(
        modality="world", declared_sensors=sorted(R_NEW), declared_ranges=R_NEW, embedding_dim=384, gain=3.0
    )
    assert {k: v for k, v in f_old.items() if k != "ranges"} == {k: v for k, v in f_new.items() if k != "ranges"}
    assert f_old["ranges"]["saturation"] == [0.0, 10.0] and f_new["ranges"]["saturation"] == [0.0, 20.0]


def test_int_and_float_declarations_are_one_space() -> None:
    a = sensor_geometry_fields(
        modality="world", declared_sensors=["s"], declared_ranges={"s": (0, 20)}, embedding_dim=384, gain=3
    )
    b = sensor_geometry_fields(
        modality="world", declared_sensors=["s"], declared_ranges={"s": (0.0, 20.0)}, embedding_dim=384, gain=3.0
    )
    assert encoding_geometry_tag(**a) == encoding_geometry_tag(**b)


def test_ungained_tags_carry_no_ranges_and_match_the_golden_literally() -> None:
    """Option A's byte-identity half: interoception/audio spaces are untouched by H2."""
    fx = json.loads(GOLDEN.read_text())
    ec = EntorhinalCortex(ECConfig())
    enc = SensorEncoder(ec=ec)
    node = enc.encode_sensors(
        agent_id="i",
        sensors={"core_temperature": 0.4, "hunger": 0.9, "thirst": 0.2},
        modality="interoception",
        ranges={"core_temperature": (-1, 1), "hunger": (0, 1), "thirst": (0, 1)},
    )
    assert node is not None
    assert ec._substrate_node_geometries[node] == fx["tags"]["interoception_infant_operant"]
    f = sensor_geometry_fields(
        modality="interoception",
        declared_sensors=["hunger"],
        declared_ranges={"hunger": (0, 1)},
        embedding_dim=384,
        gain=None,
    )
    assert "ranges" not in f and "gain" not in f


def test_gained_range_blind_encode_has_no_ranges_field() -> None:
    f = sensor_geometry_fields(
        modality="world", declared_sensors=None, declared_ranges=None, embedding_dim=384, gain=3.0
    )
    assert f["normalization"] == "range-blind" and f["gain"] == "p3.0" and "ranges" not in f


# ── the one-helper obligation, as a test ──────────────────────────────────────────────────────


def _unstamped_world_ec_with_provenance(ranges, *, mixed_with=None) -> tuple[EntorhinalCortex, str]:
    ec = EntorhinalCortex(ECConfig())
    nid = "legacy-world-1"
    ec.register_substrate_node(nid, [0.1] * 384, "world", geometry=None)
    ec.record_encoder_provenance(
        "sensor:world",
        {
            "embedding_dim": 384,
            "sensor_names": sorted(ranges),
            "normalization": "range-aware",
            "declared_sensors": sorted(ranges),
            "declared_ranges": {n: [float(lo), float(hi)] for n, (lo, hi) in ranges.items()},
            "gain_exponent": 3.0,
        },
    )
    if mixed_with is not None:
        ec.record_encoder_provenance(
            "sensor:world",
            {"declared_ranges": {n: [float(lo), float(hi)] for n, (lo, hi) in mixed_with.items()}},
        )
    return ec, nid


def test_migrate_derives_the_same_gained_tag_the_live_encode_stamps() -> None:
    ec, nid = _unstamped_world_ec_with_provenance(R_NEW)
    assert ec._migrate_legacy_geometries() == 1
    live_ec = EntorhinalCortex(ECConfig())
    live = SensorEncoder(ec=live_ec)
    live_node = _encode(live, R_NEW)
    assert ec._substrate_node_geometries[nid] == live_ec._substrate_node_geometries[live_node]


def test_migrate_refuses_a_gained_node_without_recorded_ranges_or_with_mixed_ranges(caplog) -> None:
    ec = EntorhinalCortex(ECConfig())
    ec.register_substrate_node("pre-h2", [0.1] * 384, "world", geometry=None)
    ec.record_encoder_provenance(
        "sensor:world",
        {"declared_sensors": ["s"], "normalization": "range-aware", "gain_exponent": 3.0},  # no declared_ranges
    )
    with caplog.at_level(logging.WARNING, logger="maxim.similarity.ec"):
        assert ec._migrate_legacy_geometries() == 0
    assert ec._substrate_node_geometries["pre-h2"] is None
    assert any("declared ranges" in r.getMessage() for r in caplog.records)

    ec2, nid2 = _unstamped_world_ec_with_provenance(R_OLD, mixed_with=R_NEW)
    assert ec2.encoder_provenance["sensor:world"]["declared_ranges_mixed"] is True
    assert ec2._migrate_legacy_geometries() == 0
    assert ec2._substrate_node_geometries[nid2] is None


def test_provenance_records_declared_ranges_and_flags_a_mid_session_change() -> None:
    ec = EntorhinalCortex(ECConfig())
    enc = SensorEncoder(ec=ec)
    _encode(enc, R_OLD, agent="p1")
    prov = ec.encoder_provenance["sensor:world"]
    assert prov["declared_ranges"]["saturation"] == [0.0, 10.0]
    assert "declared_ranges_mixed" not in prov
    _encode(enc, R_NEW, agent="p2")
    prov = ec.encoder_provenance["sensor:world"]
    assert prov["declared_ranges"]["saturation"] == [0.0, 20.0]
    assert prov["declared_ranges_mixed"] is True


def test_ranged_gained_tag_is_stable_across_processes() -> None:
    fields = sensor_geometry_fields(
        modality="world", declared_sensors=sorted(R_NEW), declared_ranges=R_NEW, embedding_dim=384, gain=3.0
    )
    expected = encoding_geometry_tag(**fields)
    code = (
        "import json\n"
        "from maxim.similarity.encoder import encoding_geometry_tag, sensor_geometry_fields\n"
        f"r = {dict(R_NEW)!r}\n"
        "print(encoding_geometry_tag(**sensor_geometry_fields(modality='world', declared_sensors=sorted(r), "
        "declared_ranges=r, embedding_dim=384, gain=3.0)))\n"
    )
    for seed in ("0", "4242"):
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "", "PYTHONPATH": str(REPO_ROOT / "src")},
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == expected


# ── review-fold arms (2026-09-16 two-lens round) ──────────────────────────────────────────────


def test_helper_validates_declarations_loudly() -> None:
    """A DECLARATION error is loud (unlike a weird reading): malformed pairs, non-finite bounds,
    and a `declared_sensors` set that disagrees with the ranges' key set all raise."""
    for bad in ({"s": [0, 1, 2]}, {"s": 5}, {"s": None}, {"s": (0, float("nan"))}):
        with pytest.raises(ValueError):
            sensor_geometry_fields(
                modality="world", declared_sensors=None, declared_ranges=bad, embedding_dim=384, gain=3.0
            )
    with pytest.raises(ValueError):
        sensor_geometry_fields(
            modality="world", declared_sensors=["s", "t"], declared_ranges={"s": (0, 1)}, embedding_dim=384, gain=3.0
        )


def test_signed_zero_is_one_space() -> None:
    a = sensor_geometry_fields(
        modality="world", declared_sensors=None, declared_ranges={"s": (-0.0, 1)}, embedding_dim=384, gain=3.0
    )
    b = sensor_geometry_fields(
        modality="world", declared_sensors=None, declared_ranges={"s": (0.0, 1)}, embedding_dim=384, gain=3.0
    )
    assert encoding_geometry_tag(**a) == encoding_geometry_tag(**b)


def test_corrupt_persisted_ranges_skip_the_node_instead_of_refusing_the_load(tmp_path: Path, caplog) -> None:
    """The migrate half runs inside `EC.load()`; a hand-edited or malformed `declared_ranges` must
    cost one unstamped node, never the whole session (the `sensor_names` hardening, one layer down)."""
    for corrupt in ({"s": [0, 1, 2]}, {"s": 5}, {"other": [0.0, 1.0]}):  # malformed pair / scalar / key-set mismatch
        ec = EntorhinalCortex(ECConfig())
        ec.register_substrate_node("bad", [0.1] * 384, "world", geometry=None)
        ec.record_encoder_provenance(
            "sensor:world",
            {
                "declared_sensors": ["s"],
                "normalization": "range-aware",
                "gain_exponent": 3.0,
                "declared_ranges": corrupt,
            },
        )
        path = tmp_path / f"ec_{len(str(corrupt))}.json"
        ec.save(path)
        with caplog.at_level(logging.WARNING, logger="maxim.similarity.ec"):
            loaded = EntorhinalCortex(ECConfig())
            loaded.load(path)  # must not raise
        assert loaded._substrate_node_geometries["bad"] is None, corrupt


def test_range_blind_stamp_keeps_the_last_real_ranges() -> None:
    ec = EntorhinalCortex(ECConfig())
    enc = SensorEncoder(ec=ec)
    _encode(enc, R_NEW, agent="a1")
    enc.encode_sensors(agent_id="a2", sensors=dict(WORLD), modality="world", ranges=None)  # range-blind
    prov = ec.encoder_provenance["sensor:world"]
    assert prov["declared_ranges"]["saturation"] == [0.0, 20.0]
    assert "declared_ranges_mixed" not in prov
    assert sorted(prov["normalization_modes"]) == ["range-aware", "range-blind"]


def test_sensor_tag_fields_are_built_in_exactly_one_place() -> None:
    """The one-helper invariant, mechanically: the sensor tag's `encoder="sensor"` literal appears in
    `src/maxim/` only inside `sensor_geometry_fields`. A third hand-built site is the drift the
    helper exists to prevent (D66 obligation)."""
    hits: list[str] = []
    for f in sorted((REPO_ROOT / "src" / "maxim").rglob("*.py")):
        text = f.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if 'encoder="sensor"' in line or '"encoder": "sensor"' in line:
                hits.append(f"{f.relative_to(REPO_ROOT)}:{i}")
    assert len(hits) == 1 and hits[0].startswith("src/maxim/similarity/encoder.py"), hits
    src = (REPO_ROOT / "src" / "maxim" / "similarity" / "encoder.py").read_text()
    start = src.index("def sensor_geometry_fields(")
    end = src.index("\ndef ", start + 1)
    assert '"encoder": "sensor"' in src[start:end]
