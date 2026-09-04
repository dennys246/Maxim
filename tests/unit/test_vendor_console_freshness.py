"""scripts/vendor_console_ui.py — the bundle-provenance half of bugs ledger D47.

maxim-pulse 0.1.0 (2026-08-30) closed the PRODUCER half: the Console bundle now
carries `app_version`, `commit_date`, `describe` and `dirty` in its
`maxim-ui.json`, and ships as a checksummed asset on a tagged release. Before
that, `app_version` had been frozen at `0.0.1` since inception and the artifact
existed only on whichever machine built it.

This is the CONSUMER half. `validate()` was shape-only — an `index.html`, an
`assets/` dir, a matching `contract_version` — so a months-stale bundle vendored
silently. It now reads the provenance fields, and the split between what it
REFUSES and what it merely REPORTS is the design decision under test:

* `dirty: true` is refused. The `commit` the bundle names does not describe its
  contents, so the artifact cannot be traced to source. That is mechanically
  wrong, not a matter of taste.
* Staleness is NOT refused by default. A three-week-old bundle is fine if the
  facade contract has not moved and wrong if it has, and this script cannot tell
  which — so it prints the age and refuses only under `--max-age-days`.
* Missing fields are not an error, or vendoring from a pre-0.1.0 checkout would
  break for no safety gain.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from maxim.console.ui_bundle import CONSOLE_CONTRACT_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "vendor_console_ui.py"


def _load():
    spec = importlib.util.spec_from_file_location("vendor_console_ui", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vcu():
    return _load()


def _bundle(tmp_path: Path, **manifest_overrides) -> Path:
    """A structurally valid bundle; overrides tweak the manifest."""
    src = tmp_path / "dist"
    (src / "assets").mkdir(parents=True)
    (src / "index.html").write_text("<html></html>")
    (src / "assets" / "index-abc.js").write_text("//")
    manifest = {
        "target": "console",
        "app_version": "0.1.0",
        "contract_version": CONSOLE_CONTRACT_VERSION,
        "commit": "3592561",
        "commit_date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "describe": "v0.1.0",
        "dirty": False,
    }
    manifest.update(manifest_overrides)
    (src / "maxim-ui.json").write_text(json.dumps(manifest))
    return src


# ── the refusal: a dirty build cannot be traced to source ────────────────────


def test_dirty_bundle_is_refused(vcu, tmp_path, capsys):
    ok, problems = vcu.validate(_bundle(tmp_path, dirty=True))
    assert not ok
    assert any("DIRTY" in p for p in problems)


def test_clean_bundle_passes(vcu, tmp_path, capsys):
    ok, problems = vcu.validate(_bundle(tmp_path))
    assert ok, problems


def test_dirty_absent_is_not_a_refusal(vcu, tmp_path, capsys):
    """Pre-0.1.0 bundles carry no `dirty` key; that must still vendor."""
    src = _bundle(tmp_path)
    manifest = json.loads((src / "maxim-ui.json").read_text())
    del manifest["dirty"]
    (src / "maxim-ui.json").write_text(json.dumps(manifest))
    ok, problems = vcu.validate(src)
    assert ok, problems


# ── staleness: reported always, refused only on request ──────────────────────


def test_stale_bundle_passes_by_default(vcu, tmp_path, capsys):
    """Staleness is a judgment call — the script must not make it unasked."""
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat().replace("+00:00", "Z")
    ok, problems = vcu.validate(_bundle(tmp_path, commit_date=old))
    assert ok, problems
    assert "200" in capsys.readouterr().out.replace(".0 days", " days").replace("200 days", "200")


def test_stale_bundle_is_refused_when_asked(vcu, tmp_path, capsys):
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat().replace("+00:00", "Z")
    ok, problems = vcu.validate(_bundle(tmp_path, commit_date=old), max_age_days=7)
    assert not ok
    assert any("days old" in p for p in problems)


def test_fresh_bundle_passes_the_age_gate(vcu, tmp_path, capsys):
    ok, problems = vcu.validate(_bundle(tmp_path), max_age_days=7)
    assert ok, problems


def test_age_is_always_reported(vcu, tmp_path, capsys):
    vcu.validate(_bundle(tmp_path))
    assert "days ago" in capsys.readouterr().out


# ── absent / malformed provenance is tolerated, and said out loud ────────────


def test_missing_commit_date_is_reported_not_refused(vcu, tmp_path, capsys):
    src = _bundle(tmp_path)
    manifest = json.loads((src / "maxim-ui.json").read_text())
    del manifest["commit_date"]
    (src / "maxim-ui.json").write_text(json.dumps(manifest))
    ok, problems = vcu.validate(src)
    assert ok, problems
    assert "predates maxim-pulse 0.1.0" in capsys.readouterr().out


def test_unparseable_commit_date_is_reported_not_refused(vcu, tmp_path, capsys):
    ok, problems = vcu.validate(_bundle(tmp_path, commit_date="not-a-date"))
    assert ok, problems
    assert "not parseable" in capsys.readouterr().out


def test_untagged_describe_is_surfaced_not_refused(vcu, tmp_path, capsys):
    """`v0.1.0-3-gabc1234` = three commits past the tag. Worth saying; not wrong."""
    ok, problems = vcu.validate(_bundle(tmp_path, describe="v0.1.0-3-gabc1234"))
    assert ok, problems
    assert "not exactly on a tag" in capsys.readouterr().out


# ── the age helper ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("stamp", ["2026-08-30T22:51:09Z", "2026-08-30T22:51:09+00:00"])
def test_age_parses_both_utc_spellings(vcu, stamp):
    now = datetime(2026, 8, 31, 22, 51, 9, tzinfo=timezone.utc)
    assert vcu.bundle_age_days(stamp, now=now) == pytest.approx(1.0, abs=0.01)


def test_age_returns_none_on_garbage(vcu):
    assert vcu.bundle_age_days("nonsense") is None


def test_naive_timestamp_is_treated_as_utc(vcu):
    """A producer that drops the Z must not make the age wildly wrong."""
    now = datetime(2026, 8, 31, 0, 0, 0, tzinfo=timezone.utc)
    assert vcu.bundle_age_days("2026-08-30T00:00:00", now=now) == pytest.approx(1.0, abs=0.01)


# ── the contract-version check still bites (unchanged behaviour) ─────────────


def test_contract_mismatch_still_refused(vcu, tmp_path, capsys):
    ok, problems = vcu.validate(_bundle(tmp_path, contract_version="0.1.0"))
    assert not ok
    assert any("facade contract mismatch" in p for p in problems)
