"""Golden pin for the substrate's encode primitive (H3 of the 2026-09-16 set-point-aware-neutral
design review, `docs/experiments/rationale/setpoint-neutral/regression.md` R1).

`similarity/encoder.py::_sensor_embed` underlies every substrate result in the graduation ledger,
and until this file the only "byte-identical" guard mirrored the formula in-process with
`pytest.approx` — a refactor that changed `_stable_basis`, the accumulation order or the tag
canonicalisation TOGETHER with its test expectations stayed green. This pin holds the actual
output: full 384-float vectors (exact equality, not approx) for a fixed set of readings ×
{gain None, 3.0} × {range-blind, range-aware, range-partial}, and the LITERAL geometry-tag strings
for the shipped spaces, generated at the commit named in the fixture.

Rules (same posture as the NAc golden sequence in `test_decision_provenance.py`):

* The fixture is regenerated ONLY from the pre-change commit, with the diff justified in the PR —
  never by pasting the new output. Regenerate with
  ``python tests/unit/test_encoder_golden_v1.py --regen`` on the commit you mean to pin.
* An intentional geometry change (a new normalization, a set-point, a basis change) MUST show up
  here as a failure; that is what the anti-vacuity arms below prove the pin can see.
* The tag half runs two-process with differing ``PYTHONHASHSEED`` (the stable-hash invariant): a
  tag that only matches inside one process is not a tag.

DO-NOT-SHIP condition for any encoder change (the review's wording): this test passes unchanged
on the shipped commit with no body declaring anything new, and its anti-vacuity arms fail.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from maxim.similarity import encoder as enc_mod
from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.similarity.encoder import SensorEncoder, _sensor_embed, encoding_geometry_tag

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "encoder_golden_v1.json"
DIM = 384
GAINS: tuple[float | None, ...] = (None, 3.0)

# The readings are FROZEN here, not read from body YAML: the pin holds the FUNCTION. A later range
# re-declaration in a body must not silently change the golden's inputs (that is a different
# question, answered by `scripts/lint_body_rest_neutral.py`). `world_*` mirror the 17-sensor
# `minecraft_player` roster and ranges as of 2026-09-16.
_WORLD_RANGES: dict[str, tuple[float, float]] = {
    "health": (0, 40),
    "food": (0, 40),
    "light_level": (0, 15),
    "y_altitude": (0, 128),
    "nearest_hostile_dist": (0, 128),
    "time_of_day": (0, 1),
    "saturation": (0, 20),
    "oxygen": (0, 40),
    "hostile_count": (-32, 32),
    "distance_from_spawn": (-128, 128),
    "speed": (-1, 1),
    "on_ground": (-1, 3),
    "is_raining": (-1, 1),
    "is_in_water": (-1, 1),
    "xp_level": (-50, 50),
    "nearest_player_dist": (0, 128),
    "look_pitch": (-1.5708, 1.5708),
}
_WORLD_SHORE = {
    "health": 20, "food": 20, "light_level": 14, "y_altitude": 66, "nearest_hostile_dist": 64,
    "time_of_day": 0.31, "saturation": 10, "oxygen": 20, "hostile_count": 0, "distance_from_spawn": 36.0,
    "speed": 0.0, "on_ground": 1, "is_raining": 0, "is_in_water": 0, "xp_level": 0,
    "nearest_player_dist": 64, "look_pitch": 0.0,
}  # fmt: skip
_WORLD_SUBMERGED = dict(
    _WORLD_SHORE, light_level=9, y_altitude=60, oxygen=12, on_ground=0, is_in_water=1, look_pitch=0.6
)

CASES: tuple[tuple[str, dict[str, float], dict[str, tuple[float, float]] | None], ...] = (
    ("unit_blind", {"a": 0.3, "b": 0.9, "c": 0.5}, None),
    ("unit_aware", {"a": 0.3, "b": 0.9, "c": 0.5}, {"a": (0, 1), "b": (0, 1), "c": (0, 1)}),
    ("signed_blind", {"azimuth": -0.5, "thermal": 0.25}, None),
    ("signed_aware", {"azimuth": -0.5, "thermal": 0.25}, {"azimuth": (-1, 1), "thermal": (-1, 1)}),
    ("signed_partial", {"azimuth": -0.5, "thermal": 0.25}, {"azimuth": (-1, 1)}),
    ("world_shore", _WORLD_SHORE, _WORLD_RANGES),
    ("world_submerged", _WORLD_SUBMERGED, _WORLD_RANGES),
)

# The shipped spaces. `declared_sensors` is the fixture's own declared set (what the body walk hands
# `encode_sensors` as the `ranges` key set), TRANSCRIBED from the registry-resolved bodies on
# 2026-09-16 (`minecraft_player` world; `infant_operant` interoception = hunger/thirst/core_temperature,
# the Exp 42/48/52/53 body; `reachy_mini_infant` interoception = hunger/thirst; audio = azimuth) and
# pinned literally as FUNCTION inputs — a change to how the tag is derived from these fields, not a
# change to a body, is what trips it.
TAG_FIELDS: dict[str, dict[str, object]] = {
    "world_minecraft_player_gained": dict(
        encoder="sensor",
        modality="world",
        declared_sensors=sorted(_WORLD_RANGES),
        normalization="range-aware",
        embedding_dim=DIM,
        gain="p3.0",
    ),
    "interoception_infant_operant": dict(
        encoder="sensor",
        modality="interoception",
        declared_sensors=["core_temperature", "hunger", "thirst"],
        normalization="range-aware",
        embedding_dim=DIM,
    ),
    "interoception_reachy_mini_infant": dict(
        encoder="sensor",
        modality="interoception",
        declared_sensors=["hunger", "thirst"],
        normalization="range-aware",
        embedding_dim=DIM,
    ),
    "audio_azimuth": dict(
        encoder="sensor",
        modality="audio",
        declared_sensors=["azimuth"],
        normalization="range-aware",
        embedding_dim=DIM,
    ),
}


def _gain_key(g: float | None) -> str:
    return "none" if g is None else f"p{g}"


def generate() -> dict:
    vectors: dict[str, list[float]] = {}
    for name, sensors, ranges in CASES:
        for g in GAINS:
            vectors[f"{name}|{_gain_key(g)}"] = _sensor_embed(sensors, ranges=ranges, dim=DIM, gain_exponent=g)
    tags = {k: encoding_geometry_tag(**f) for k, f in TAG_FIELDS.items()}
    commit = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    return {"generated_at_commit": commit, "dim": DIM, "vectors": vectors, "tags": tags}


def _load() -> dict:
    return json.loads(FIXTURE.read_text())


def _vector_mismatches(fx: dict) -> list[str]:
    bad = []
    for name, sensors, ranges in CASES:
        for g in GAINS:
            key = f"{name}|{_gain_key(g)}"
            got = _sensor_embed(sensors, ranges=ranges, dim=DIM, gain_exponent=g)
            if got != fx["vectors"][key]:  # exact float equality on purpose
                bad.append(key)
    return bad


def test_fixture_is_complete() -> None:
    fx = _load()
    assert fx["dim"] == DIM
    assert set(fx["vectors"]) == {f"{n}|{_gain_key(g)}" for n, _, _ in CASES for g in GAINS}
    assert set(fx["tags"]) == set(TAG_FIELDS)
    assert all(len(v) == DIM for v in fx["vectors"].values())


def test_sensor_embed_matches_golden_exactly() -> None:
    assert _vector_mismatches(_load()) == []


def test_geometry_tags_match_golden_literally() -> None:
    fx = _load()
    for k, f in TAG_FIELDS.items():
        assert encoding_geometry_tag(**f) == fx["tags"][k], k


def test_world_tag_through_the_encode_sensors_seam() -> None:
    """The literal world tag must be what `encode_sensors` stamps on a gained node — pins the
    derivation seam (declared set = the `ranges` key set, mode, dim, gain), not just the helper."""
    fx = _load()
    ec = EntorhinalCortex(ECConfig())
    enc = SensorEncoder(ec=ec)
    node = enc.encode_sensors(agent_id="g", sensors=_WORLD_SUBMERGED, modality="world", ranges=_WORLD_RANGES)
    assert node is not None
    assert ec._substrate_node_geometries[node] == fx["tags"]["world_minecraft_player_gained"]


def test_tags_are_stable_across_processes_with_different_hash_seeds() -> None:
    fx = _load()
    code = (
        "import json,sys\n"
        "sys.path.insert(0, %r)\n"
        "from test_encoder_golden_v1 import TAG_FIELDS\n"
        "from maxim.similarity.encoder import encoding_geometry_tag\n"
        "print(json.dumps({k: encoding_geometry_tag(**f) for k, f in TAG_FIELDS.items()}))\n"
    ) % str(Path(__file__).parent)
    for seed in ("0", "4242"):
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "", "PYTHONPATH": str(REPO_ROOT / "src")},
        )
        assert r.returncode == 0, f"PYTHONHASHSEED={seed}: {r.stderr}"
        assert json.loads(r.stdout) == fx["tags"], f"PYTHONHASHSEED={seed}"


# ── anti-vacuity arms: the pin must SEE a geometry change (D44 strict-red-gate lesson) ──────────


def test_pin_sees_a_normalization_change(monkeypatch: pytest.MonkeyPatch) -> None:
    orig = enc_mod._normalize_value
    monkeypatch.setattr(enc_mod, "_normalize_value", lambda v, r=None: min(1.0, orig(v, r) + 1e-9))
    bad = _vector_mismatches(_load())
    assert bad, "a 1e-9 normalization shift must trip the golden"


def test_pin_sees_a_basis_change(monkeypatch: pytest.MonkeyPatch) -> None:
    orig = enc_mod._stable_basis
    monkeypatch.setattr(enc_mod, "_stable_basis", lambda name, dim, salt="": orig(name, dim, salt=salt + "x"))
    assert _vector_mismatches(_load()), "a basis salt change must trip the golden"


def test_pin_sees_a_set_point_style_weight_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shape of the change the review deferred: one sensor's rest re-centred away from the
    midpoint (`light_level` only). Patched into the NORMALIZATION, so all four world keys move
    (value and weight both shift); a set-point placed in the WEIGHT alone would leave the gain-None
    keys byte-identical and surface only in the `|p3.0` keys — either way the pin sees it. The
    unit/signed cases carry no `(0, 15)` range and must stay put."""
    orig = enc_mod._normalize_value

    def shifted(v, r=None):
        return orig(v, r) if r != (0, 15) else max(0.0, min(1.0, orig(v, r) + 0.0333))  # light_level only

    monkeypatch.setattr(enc_mod, "_normalize_value", shifted)
    bad = _vector_mismatches(_load())
    assert {"world_shore|p3.0", "world_submerged|p3.0", "world_shore|none", "world_submerged|none"} <= set(bad)
    assert not any(k.startswith(("unit_", "signed_")) for k in bad)


def test_pin_sees_a_tag_field_change() -> None:
    fx = _load()
    f = dict(TAG_FIELDS["interoception_infant_operant"], gain="p3.0")
    assert encoding_geometry_tag(**f) != fx["tags"]["interoception_infant_operant"]


if __name__ == "__main__":  # regeneration entry point — read the module docstring first
    if "--regen" not in sys.argv:
        raise SystemExit(
            "usage: python tests/unit/test_encoder_golden_v1.py --regen  (from the commit you mean to pin)"
        )
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(generate(), indent=0) + "\n")
    print(f"wrote {FIXTURE} at {json.loads(FIXTURE.read_text())['generated_at_commit']}")
