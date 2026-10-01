"""#856 -- a config.json written by this build stays readable by an older build.

Four sections (`console`, `tools`, `sim`, `memory`) shipped without bumping `CONFIG_FORMAT_VERSION`,
so a file this build wrote still said "1.0". An older build reading "1.0" treats an unknown key as an
error, and every `maxim` command failed until the operator hand-edited the file. The mechanism for this
exists: a FUTURE-minor file ("1.1" to a "1.0" loader) has its unknown keys tolerated with a warning.

Owner decision 2026-09-28: bump once to "1.1", and pin the schema to the version so the next schema
change cannot ship without a bump.
"""

from __future__ import annotations

from maxim.runtime.config_loader import non_default_paths

import dataclasses
import json
from pathlib import Path

import pytest

from maxim.runtime import config_loader as cl

REPO = Path(__file__).resolve().parents[2]
SCHEMA_FILE = REPO / "tests" / "fixtures" / "config_schema_by_version.json"


def _schema_paths(obj, prefix: str = "") -> list[str]:
    """Every field path of the config, sections included (a key added INSIDE a section breaks an older
    build exactly like a new section does).

    Blind spots, stated: it does not descend into tuple-of-dataclass fields (lane placements, whose
    unknown keys go to their own ``extra``), cannot see a TYPE change, and lists an Optional nested
    dataclass defaulting to None as one path. None of those exist in a way that breaks an older build
    today; a new one needs its own check."""
    out: list[str] = []
    for f in dataclasses.fields(obj):
        if f.name.startswith("_"):
            continue
        value = getattr(obj, f.name)
        path = f"{prefix}{f.name}"
        out += _schema_paths(value, path + ".") if dataclasses.is_dataclass(value) else [path]
    return sorted(out)


def _pinned() -> dict[str, list[str]]:
    return {k: v for k, v in json.loads(SCHEMA_FILE.read_text()).items() if not k.startswith("_")}


def _version_key(version: str) -> tuple[int, int]:
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


def test_the_schema_is_pinned_to_the_current_format_version() -> None:
    """A schema change without a version bump fails here, naming what changed."""
    pinned = _pinned()
    current = _schema_paths(cl.MaximConfig())
    assert cl.CONFIG_FORMAT_VERSION in pinned, (
        f"no pinned schema for CONFIG_FORMAT_VERSION {cl.CONFIG_FORMAT_VERSION!r}: add it to {SCHEMA_FILE.name}"
    )
    expected = pinned[cl.CONFIG_FORMAT_VERSION]
    added, removed = sorted(set(current) - set(expected)), sorted(set(expected) - set(current))
    assert not added and not removed, (
        f"config.json schema changed without a CONFIG_FORMAT_VERSION bump (added {added}, removed {removed}). "
        "Bump the minor version for additions (older builds then tolerate the new keys), add the new schema "
        f"under the new version in {SCHEMA_FILE.name}, and never edit an existing entry."
    )


def test_minor_versions_only_add_to_the_schema() -> None:
    """An older build tolerates a future minor's EXTRA keys; a removed or renamed key is a major bump."""
    pinned = _pinned()
    versions = sorted(pinned, key=_version_key)
    for older, newer in zip(versions, versions[1:]):
        if _version_key(older)[0] == _version_key(newer)[0]:
            lost = sorted(set(pinned[older]) - set(pinned[newer]))
            assert not lost, f"{newer} drops {lost} from {older} within one major version"


def test_a_file_this_build_writes_is_tolerated_by_a_1_0_build(tmp_path: Path, monkeypatch) -> None:
    """The downgrade itself: the file this build writes carries a newer minor than a 1.3.1-era loader
    knows, so that loader tolerates the sections it has never heard of instead of refusing."""
    from maxim.runtime.config_writer import write_config

    path = tmp_path / "config.json"
    write_config(cfg_ := cl.MaximConfig(), path=path, explicit=non_default_paths(cfg_))
    data = json.loads(path.read_text())
    assert data["_format_version"] == cl.CONFIG_FORMAT_VERSION

    monkeypatch.setattr(cl, "CONFIG_FORMAT_VERSION", "1.0")  # the loader an older build carries
    _version, is_future_minor = cl._check_format_version(data)
    assert is_future_minor is True  # unknown keys tolerated with a warning, not a ConfigurationError


def test_rewriting_a_config_loaded_from_an_older_file_stamps_this_version(tmp_path: Path) -> None:
    """The writer serializes THIS build's schema, so it stamps this build's version. It used to pass the
    loaded file's version through, which the stale-version check refuses once the version moves."""
    from maxim.runtime.config_writer import write_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": "1.0", "role": "solo"}))
    loaded = cl.load_config(path)
    write_config(loaded, path=path, explicit=non_default_paths(loaded))
    data = json.loads(path.read_text())
    assert data["_format_version"] == cl.CONFIG_FORMAT_VERSION and data["role"] == "solo"


@pytest.mark.parametrize("section", ["console", "tools", "sim", "memory"])
def test_each_section_that_shipped_unversioned_is_in_the_pinned_schema(section: str) -> None:
    assert any(p.startswith(section + ".") for p in _pinned()[cl.CONFIG_FORMAT_VERSION])


def test_a_1_0_loader_parses_a_file_this_build_writes(tmp_path: Path, monkeypatch) -> None:
    """Through the real parse, not only the classifier: a key the older schema lacks is tolerated."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": cl.CONFIG_FORMAT_VERSION, "role": "solo", "from_the_future": {}}))
    monkeypatch.setattr(cl, "CONFIG_FORMAT_VERSION", "1.0")
    assert cl.load_config(path).role == "solo"  # unknown key tolerated with a warning, not refused


def test_rewriting_a_config_loaded_from_a_newer_file_refuses(tmp_path: Path) -> None:
    """The newer file's unknown keys were dropped at parse; writing would lose them and stamp the file
    DOWN. That stays a loud refusal (it was a ValueError before #856), not a silent loss (#974)."""
    from maxim.exceptions import ConfigurationError
    from maxim.runtime.config_writer import write_config

    major, minor = (int(p) for p in cl.CONFIG_FORMAT_VERSION.split(".")[:2])
    newer = f"{major}.{minor + 1}"
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": newer, "role": "solo", "from_the_future": {"x": 1}}))
    loaded = cl.load_config(path)
    with pytest.raises(ConfigurationError, match="newer config.json"):
        write_config(loaded, path=path, explicit=non_default_paths(loaded))
    assert json.loads(path.read_text())["from_the_future"] == {"x": 1}


# ── what each version MEANS is pinned too (config.json 1.2, 2026-10-01) ─────────────────────────────────

RESOLUTION = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "config_resolution_by_version.json").read_text()
)


@pytest.mark.parametrize("version", sorted(k for k in RESOLUTION if not k.startswith("_")))
def test_each_version_resolves_as_pinned(version, tmp_path) -> None:
    """A change in what a key means (not only the schema) must bump the format: the loader must still read every
    pinned version's probe files exactly as pinned (a default-equal value, a null, a lane tier's unknown keys, a
    null section)."""
    from maxim.runtime.config_loader import explicit_paths, load_config

    for case in RESOLUTION[version]:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(case["probe"]))
        explicit = explicit_paths(load_config(path))
        got = {field: ("set" if field in explicit else "unset") for field in case["expect"]}
        assert got == case["expect"], case["probe"]


def test_the_current_version_has_a_pinned_resolution() -> None:
    from maxim.runtime.config_loader import CONFIG_FORMAT_VERSION

    assert CONFIG_FORMAT_VERSION in RESOLUTION


def test_the_append_only_lint_guards_both_fixtures() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lint_cfg_schema", Path(__file__).resolve().parents[2] / "scripts" / "lint_config_schema_append_only.py"
    )
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)
    assert set(lint.FIXTURES) == {
        "tests/fixtures/config_schema_by_version.json",
        "tests/fixtures/config_resolution_by_version.json",
    }
