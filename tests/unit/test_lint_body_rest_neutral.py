"""scripts/lint_body_rest_neutral.py — fixture-tree tests (H1 of the set-point-aware-neutral review).

Verified RED on the pre-declaration bodies: before `rest:` was declared, every gained sensor without a
homeostatic drive failed rule 1 (the `saturation` [0,10] declaration that broke Exp 60 gate (ii) would
have failed rule 2 the moment `rest: 10` was stated against it). The executor-lens fold added the
inputs the first cut missed: a missing/empty tree, a homeostatic drive with no set_point (runtime
defaults it to 0.0), an `extends:` child overriding only `range:`, NaN, and a non-[lo, hi] range.
"""

from __future__ import annotations

from pathlib import Path

from scripts import lint_body_rest_neutral as L


def _body(root: Path, name: str, sensors: str, *, extends: str | None = None, category: str = "bodies") -> Path:
    d = root / category
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.yaml"
    ext = f"  extends: {extends}\n" if extends else ""
    p.write_text(f"component:\n  name: {name}\n{ext}entity:\n  name: {name}\n  sensors:\n{sensors}")
    return p


GOOD = """\
    health:
      range: [0, 40]
      initial: 20
      modality: world
      drive:
        drift_mode: homeostatic
        set_point: 20
    speed:
      range: [-1, 1]
      rest: 0
      initial: 0
      modality: world
    time_of_day:
      range: [0, 1]
      rest: null
      modality: world
    hunger:
      range: [0, 1]
      initial: 0.0
      drive:
        drift_mode: entropic
"""


def _names(tags: list[str]) -> list[str]:
    return sorted(t.split("::")[1] for t in tags)


def test_gained_modalities_mirror_the_encoder_config() -> None:
    from maxim.similarity.encoder import SensorEncoderConfig

    assert L.GAINED_MODALITIES == SensorEncoderConfig().gain_modalities


def test_clean_body_passes_and_reports_the_no_rest_roster(tmp_path: Path) -> None:
    _body(tmp_path, "b", GOOD)
    fails, no_rest, gained = L.lint_components(tmp_path)
    assert fails == []
    assert _names(no_rest) == ["time_of_day"]
    assert gained == 3


def test_gained_sensor_without_any_rest_declaration_fails(tmp_path: Path) -> None:
    _body(tmp_path, "b", "    y:\n      range: [0, 128]\n      initial: 64\n      modality: world\n")
    fails, _, _ = L.lint_components(tmp_path)
    assert len(fails) == 1 and "declares no rest" in fails[0] and "rest: 64" in fails[0]


def test_the_saturation_lesson_fails_on_rest_off_midpoint(tmp_path: Path) -> None:
    # The Exp 60 defect stated honestly: range [0,10] (midpoint 5) while the game rests at 10.
    _body(tmp_path, "b", "    saturation:\n      range: [0, 10]\n      rest: 10\n      modality: world\n")
    fails, _, _ = L.lint_components(tmp_path)
    assert len(fails) == 1 and "rest 10 != range midpoint 5" in fails[0]


def test_homeostatic_set_point_off_midpoint_fails(tmp_path: Path) -> None:
    _body(
        tmp_path,
        "b",
        "    oxygen:\n      range: [0, 20]\n      modality: world\n      drive:\n        drift_mode: homeostatic\n        set_point: 20\n",
    )
    fails, _, _ = L.lint_components(tmp_path)
    assert len(fails) == 1 and "drive.set_point 20 != range midpoint 10" in fails[0]


def test_homeostatic_without_set_point_is_linted_as_rest_at_zero(tmp_path: Path) -> None:
    # `spec.py::_parse_drive_spec` defaults set_point to 0.0 — the lint must see the drive resting at 0.
    _body(
        tmp_path,
        "b",
        "    oxygen:\n      range: [0, 20]\n      modality: world\n      drive:\n        drift_mode: homeostatic\n",
    )
    fails, _, _ = L.lint_components(tmp_path)
    assert any("drive.set_point 0 != range midpoint 10" in f for f in fails)


def test_homeostatic_sensor_must_not_also_declare_rest(tmp_path: Path) -> None:
    # One rest source per sensor (architecture-lens S2 / bio-faithful SF-4): even an AGREEING `rest:` fails.
    for rest in ("20", "18", "null"):
        _body(
            tmp_path,
            "b",
            f"    health:\n      range: [0, 40]\n      rest: {rest}\n      modality: world\n      drive:\n        drift_mode: homeostatic\n        set_point: 20\n",
        )
        fails, no_rest, _ = L.lint_components(tmp_path)
        assert len(fails) == 1 and "one rest source per sensor" in fails[0], rest
        assert no_rest == []


def test_gained_sensor_without_range_fails_and_ungained_are_ignored(tmp_path: Path) -> None:
    _body(tmp_path, "b", "    light:\n      modality: world\n    az:\n      range: [-1, 1]\n      initial: 0.7\n")
    fails, _, gained = L.lint_components(tmp_path)
    assert len(fails) == 1 and "declares no `range:`" in fails[0]
    assert gained == 1


def test_malformed_range_and_nan_rest_fail(tmp_path: Path) -> None:
    _body(
        tmp_path,
        "b",
        "    a:\n      range: [0, 15, 3]\n      rest: 7.5\n      modality: world\n    b:\n      range: [0, 2]\n      rest: .nan\n      modality: world\n",
    )
    fails, _, _ = L.lint_components(tmp_path)
    assert any("must be `[lo, hi]`" in f for f in fails)
    assert any("finite number or null" in f for f in fails)


def test_initial_is_not_read_as_rest(tmp_path: Path) -> None:
    # `initial` is a boot value (it was the wrong field for saturation); it must neither satisfy nor fail the rule.
    _body(tmp_path, "b", "    light:\n      range: [0, 15]\n      initial: 7\n      rest: 7.5\n      modality: world\n")
    fails, _, _ = L.lint_components(tmp_path)
    assert fails == []


def test_extends_child_overriding_only_range_is_linted_against_inherited_rest(tmp_path: Path) -> None:
    # The Exp 60 shape verbatim, one level down: parent declares rest 10 on [0,20]; child re-declares [0,10].
    _body(tmp_path, "parent", "    saturation:\n      range: [0, 20]\n      rest: 10\n      modality: world\n")
    _body(tmp_path, "child", "    saturation:\n      range: [0, 10]\n", extends="bodies/parent")
    fails, _, gained = L.lint_components(tmp_path)
    assert gained == 2
    assert len(fails) == 1 and "child.yaml::saturation" in fails[0] and "rest 10 != range midpoint 5" in fails[0]


def test_extends_to_a_missing_parent_fails_loudly(tmp_path: Path) -> None:
    _body(
        tmp_path, "child", "    x:\n      range: [0, 2]\n      rest: 1\n      modality: world\n", extends="bodies/nope"
    )
    fails, _, _ = L.lint_components(tmp_path)
    assert len(fails) >= 1 and "does not exist" in fails[0]


def test_other_component_categories_are_linted_too(tmp_path: Path) -> None:
    _body(tmp_path, "b", GOOD)
    _body(tmp_path, "npc", "    y:\n      range: [0, 128]\n      modality: world\n", category="npcs")
    fails, _, _ = L.lint_components(tmp_path)
    assert len(fails) == 1 and "npcs/npc.yaml::y" in fails[0]


def test_missing_or_empty_tree_fails_instead_of_passing_vacuously(tmp_path: Path) -> None:
    fails, _, gained = L.lint_components(tmp_path / "nope")
    assert gained == 0 and len(fails) == 1 and "must not read as a pass" in fails[0]
    _body(tmp_path, "b", "    hunger:\n      range: [0, 1]\n")  # a tree with no gained sensor at all
    fails, _, gained = L.lint_components(tmp_path)
    assert gained == 0 and len(fails) == 1 and "no gained sensor found" in fails[0]


def test_shipped_components_are_clean() -> None:
    fails, no_rest, gained = L.lint_components(L.COMPONENTS_DIR)
    assert fails == [], fails
    assert gained == 29  # 17 player + 5 bench + 5 inherited by bench_satiated + 2 bench57
    # The declared no-rest roster today: the place/time sensors of the Minecraft bodies (one inherited).
    assert _names(no_rest) == ["light_level", "time_of_day", "time_of_day", "time_of_day"]


def test_main_exit_codes(tmp_path: Path, capsys) -> None:
    _body(tmp_path, "b", GOOD)
    assert L.main([str(tmp_path)]) == 0
    _body(tmp_path, "b", "    y:\n      range: [0, 128]\n      modality: world\n")
    assert L.main([str(tmp_path)]) == 1
    assert "FAIL" in capsys.readouterr().err
