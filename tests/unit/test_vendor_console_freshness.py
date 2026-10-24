"""scripts/vendor_console_ui.py and scripts/_console_pin.py: R4, the vendored-console gate (#1333).

pymaxim 1.3.2 vendored pulse v0.4.0 two days after v0.4.1 had been tagged. The vendor step took any directory and
refused only a contract mismatch or a dirty build, and this file PINNED that ("staleness is NOT refused"): with no
definition of "current", the age of a bundle was a judgment call the script could not make.

The committed pin (`scripts/console_ui_pin.json`) is that definition now, so the three old pins are REVERSED, and
rewritten rather than flipped (design pass fold 13): a bundle is refused when it is not the pinned one, whatever its
date or describe, and a bundle that IS the pinned one passes however old it is. `--verify-pin` proves the pin is the
newest pulse tag for the backend's contract.

Every network seam is faked; an autouse fixture makes the real ones raise (fold 6), so a test that forgets its fake
fails instead of reaching the network. Design: docs/plans/archive/r4_console_vendor_gate.md.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import types
from pathlib import Path

import pytest

from tests.unit import _console_bundles as cb

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


@pytest.fixture(autouse=True)
def _no_network(vcu, monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("a test reached a real network seam; inject a fake")

    monkeypatch.setattr(vcu, "http_fetch", refuse)
    monkeypatch.setattr(vcu, "git_ls_remote", refuse)
    monkeypatch.setattr(vcu.subprocess, "run", refuse)
    monkeypatch.setattr(vcu.urllib.request, "urlopen", refuse)


@pytest.fixture
def cp(vcu):
    return vcu.cp


def _validate(vcu, tmp_path, files, pin, **kw):
    return vcu.validate(cb.write_dir(tmp_path / "dist", files), pin=pin, **kw)


# ── the pin decides "current" ────────────────────────────────────────────────


def test_the_pinned_bundle_passes(vcu, cp, tmp_path):
    files = cb.bundle(cp)
    ok, problems = _validate(vcu, tmp_path, files, cb.pin_for(cp, files))
    assert ok, problems


def test_a_stale_bundle_is_refused_by_default(vcu, cp, tmp_path):
    """REVERSED (was test_stale_bundle_passes_by_default). The 1.3.2 failure: v0.4.0's build, freshly dated, with
    v0.4.1 pinned. Refused for being the wrong build, not for its age."""
    pinned = cb.bundle(cp)
    stale = cb.bundle(cp, commit="b39ca63", describe="v0.4.0", app_version="0.4.0")
    ok, problems = _validate(vcu, tmp_path, stale, cb.pin_for(cp, pinned))
    assert not ok
    assert any("not the pinned v0.4.1" in p for p in problems), problems
    assert any("differ from the pinned v0.4.1" in p for p in problems), problems


def test_an_old_dated_bundle_that_is_the_pin_passes(vcu, cp, tmp_path):
    """The date never decides: an old bundle that IS the pinned one is current."""
    files = cb.bundle(cp, commit_date="2020-01-01T00:00:00Z")
    ok, problems = _validate(vcu, tmp_path, files, cb.pin_for(cp, files))
    assert ok, problems


def test_an_untagged_describe_that_is_not_the_pin_is_refused(vcu, cp, tmp_path):
    """REVERSED (was test_untagged_describe_is_surfaced_not_refused): three commits past a tag is another build."""
    pinned = cb.bundle(cp)
    untagged = cb.bundle(cp, describe="v0.4.1-3-gabc1234", commit="abc1234")
    ok, problems = _validate(vcu, tmp_path, untagged, cb.pin_for(cp, pinned))
    assert not ok and any("not the pinned" in p for p in problems), problems


def test_missing_provenance_is_refused(vcu, cp, tmp_path):
    """REVERSED (was test_missing_commit_date_is_reported_not_refused): the stamp is inside the digest, so a bundle
    whose stamp lost a field is not the pinned bundle."""
    pinned = cb.bundle(cp)
    ok, problems = _validate(vcu, tmp_path, cb.bundle(cp, drop=("commit_date",)), cb.pin_for(cp, pinned))
    assert not ok and any("differ from the pinned" in p for p in problems), problems


def test_an_edited_stamp_over_the_right_files_is_refused(vcu, cp, tmp_path):
    """A hand-edited `maxim-ui.json` claiming the pinned tag over other contents: the digest is the gate."""
    pinned = cb.bundle(cp)
    forged = {**pinned, "assets/index-abc.js": b"// v0.4.0 code"}
    ok, problems = _validate(vcu, tmp_path, forged, cb.pin_for(cp, pinned))
    assert not ok and any("differ from the pinned" in p for p in problems), problems


def test_a_describe_mismatch_alone_is_a_note(vcu, cp, tmp_path, capsys):
    """Fold 10: two tags on one commit, or pulse's tolerated `vX-N-g<sha>`, must not refuse the right build."""
    files = cb.bundle(cp, describe="v0.4.0-2-gda20a72")
    ok, problems = _validate(vcu, tmp_path, files, cb.pin_for(cp, files))
    assert ok, problems
    assert "the pin names v0.4.1" in capsys.readouterr().out


@pytest.mark.parametrize("dirty", [True, None, "false"])
def test_a_bundle_not_stamped_clean_is_refused(vcu, cp, tmp_path, dirty):
    files = cb.bundle(cp, dirty=dirty) if dirty is not None else cb.bundle(cp, drop=("dirty",))
    ok, problems = _validate(vcu, tmp_path, files, cb.pin_for(cp, files))
    assert not ok and any("`dirty`" in p for p in problems), problems


def test_the_wrong_target_is_refused(vcu, cp, tmp_path):
    files = cb.bundle(cp, target="reachy")
    ok, problems = _validate(vcu, tmp_path, files, cb.pin_for(cp, files))
    assert not ok and any("target" in p for p in problems)


def test_a_pin_naming_another_repo_is_refused(vcu, cp, tmp_path):
    """Fold 2: a fork could carry a self-consistent pin; the repo is a constant, only compared."""
    files = cb.bundle(cp)
    ok, problems = _validate(vcu, tmp_path, files, cb.pin_for(cp, files, repo="someone/Maxim-pulse-fork"))
    assert not ok and any("comes only from dennys246/Maxim-pulse" in p for p in problems)


def test_validate_requires_a_pin(vcu, tmp_path):
    """Fold 6: no caller can validate against no pin by forgetting the argument."""
    with pytest.raises(TypeError):
        vcu.validate(tmp_path)


# ── the contract: enforced where a release can ship (fold 1) ──────────────────


def test_contract_drift_is_refused_when_releasing(vcu, cp, tmp_path, monkeypatch):
    files = cb.bundle(cp)
    pin = cb.pin_for(cp, files)
    monkeypatch.setattr(cp, "backend_contract", lambda *_: "0.7.0")
    ok, problems = _validate(vcu, tmp_path, files, pin, releasing=True)
    assert not ok and any("move the pin to a 0.7.0-contract" in p for p in problems), problems


def test_contract_drift_is_a_loud_note_on_a_tree_that_is_not_releasing(vcu, cp, tmp_path, monkeypatch, capsys):
    """A PR that bumps CONSOLE_CONTRACT_VERSION can merge before pulse tags a 0.7.0 bundle; the release cannot."""
    files = cb.bundle(cp)
    pin = cb.pin_for(cp, files)
    monkeypatch.setattr(cp, "backend_contract", lambda *_: "0.7.0")
    ok, problems = _validate(vcu, tmp_path, files, pin, releasing=False)
    assert ok, problems
    assert "CONTRACT DRIFT" in capsys.readouterr().out


def test_the_backend_contract_is_read_as_text_not_imported(cp):
    """Fold 11: an editable install of another tree cannot answer for this one."""
    assert cp.contract_from_source('x = 1\nCONSOLE_CONTRACT_VERSION = "0.6.0"\n') == "0.6.0"
    assert cp.contract_from_source("# CONSOLE_CONTRACT_VERSION = '9'\n") is None
    from maxim.console.ui_bundle import CONSOLE_CONTRACT_VERSION

    assert cp.backend_contract() == CONSOLE_CONTRACT_VERSION


# ── the canonical digest (fold 5) ────────────────────────────────────────────


def test_one_digest_whether_read_from_tar_wheel_sdist_or_directory(cp, tmp_path):
    files = cb.bundle(cp)
    with tarfile.open(fileobj=io.BytesIO(cb.tar_bytes(files)), mode="r:gz") as tf:
        from_tar = cp.tree_digest(cp.tar_entries(tf))
    wheel = [(f"maxim/console/ui_dist/{k}", v) for k, v in files.items()] + [("maxim/__init__.py", b"")]
    sdist = [(f"pymaxim-1.3.3/src/maxim/console/ui_dist/{k}", v) for k, v in files.items()]
    from_wheel = cp.tree_digest(cp.prefixed_entries(wheel, "maxim/console/ui_dist/"))
    from_sdist = cp.tree_digest(cp.prefixed_entries(sdist, "pymaxim-1.3.3/src/maxim/console/ui_dist/"))
    from_dir = cp.tree_digest(cp.dir_entries(cb.write_dir(tmp_path / "d", files)))
    assert from_tar == from_wheel == from_sdist == from_dir


@pytest.mark.parametrize(
    "name", ["../evil.js", "/abs.js", "a\\b.js", "assets/.vite/manifest.json", ".DS_Store", "a/./b.js", "x\ny"]
)
def test_refused_member_names(cp, name):
    with pytest.raises(cp.DigestError):
        cp.normalize_member(name)


def test_names_equal_up_to_case_are_refused(cp):
    """A macOS extraction would merge them, so the digest of what ships could not be the digest checked."""
    with pytest.raises(cp.DigestError, match="case"):
        cp.tree_digest([("assets/A.js", b"1"), ("assets/a.js", b"2")])


def test_a_symlink_in_a_directory_bundle_is_refused(cp, tmp_path):
    root = cb.write_dir(tmp_path / "d", cb.bundle(cp))
    (root / "assets" / "link.js").symlink_to(root / "index.html")
    with pytest.raises(cp.DigestError, match="symlink"):
        cp.dir_entries(root)


def test_a_link_member_in_the_asset_is_refused(cp):
    link = tarfile.TarInfo("./assets/link.js")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    with tarfile.open(fileobj=io.BytesIO(cb.tar_bytes(cb.bundle(cp), extra=[link])), mode="r:gz") as tf:
        with pytest.raises(cp.DigestError, match="regular files"):
            cp.tar_entries(tf)


# ── the pin file ─────────────────────────────────────────────────────────────


def test_the_committed_pin_loads(cp):
    """Its contract is deliberately NOT asserted equal to the backend's here: a contract-bump PR must stay green
    until a release (fold 1). `--verify-pin` asserts it where a release ships."""
    pin = cp.load_pin()
    assert cp.pin_identity_problems(pin) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("_format_version"),
        lambda d: d.update(_format_version="2.0"),
        lambda d: d.update(tag="v0.5.0-rc1"),
        lambda d: d.update(commit="da20a72"),
        lambda d: d.pop("tree_sha256"),
        lambda d: d.update(asset_sha256="XYZ"),
        lambda d: d.update(tree_sha256="ab" * 31),
        lambda d: d.update(signed_by="someone"),  # an unknown field is refused, not silently carried
    ],
)
def test_a_malformed_pin_is_unreadable_not_a_pass(cp, tmp_path, mutate):
    raw = json.loads(cp.PIN_PATH.read_text())
    mutate(raw)
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(cp.PinError):
        cp.load_pin(path)


# ── vendoring ────────────────────────────────────────────────────────────────


@pytest.fixture
def dest(vcu, tmp_path, monkeypatch):
    target = tmp_path / "ui_dist"
    monkeypatch.setattr(vcu, "DEST", target)
    return target


def test_vendor_copies_the_pinned_bundle(vcu, cp, tmp_path, dest):
    files = cb.bundle(cp)
    assert vcu.vendor(cb.write_dir(tmp_path / "dist", files), pin=cb.pin_for(cp, files)) == 0
    assert cp.tree_digest(cp.dir_entries(dest)) == cb.pin_for(cp, files).tree_sha256


def test_vendor_refuses_a_bundle_that_is_not_the_pin(vcu, cp, tmp_path, dest):
    pin = cb.pin_for(cp, cb.bundle(cp))
    assert vcu.vendor(cb.write_dir(tmp_path / "dist", cb.bundle(cp, commit="b39ca63")), pin=pin) == 1
    assert not dest.exists()


def _asset_fetch(data: bytes, calls: list[str] | None = None):
    def fetch(url):
        if calls is not None:
            calls.append(url)
        assert url == "https://github.com/dennys246/Maxim-pulse/releases/download/v0.4.1/console-dist.tar.gz", url
        return data

    return fetch


def test_from_pin_vendors_the_release_asset(vcu, cp, dest):
    files = cb.bundle(cp)
    assert vcu.from_pin(cb.pin_for(cp, files), fetch=_asset_fetch(cb.tar_bytes(files))) == 0
    assert cp.tree_digest(cp.dir_entries(dest)) == cb.pin_for(cp, files).tree_sha256


def test_from_pin_refuses_an_asset_with_the_wrong_sha256(vcu, cp, dest):
    files = cb.bundle(cp)
    swapped = cb.tar_bytes({**files, "index.html": b"<html>other</html>"})
    assert vcu.from_pin(cb.pin_for(cp, files), fetch=_asset_fetch(swapped)) == 1
    assert not dest.exists()


def test_from_pin_refuses_other_bytes_even_with_the_same_files(vcu, cp, dest):
    """The sha256 check stands on its own: a re-packed asset with identical files is still not the pinned asset."""
    files = cb.bundle(cp)
    assert vcu.from_pin(cb.pin_for(cp, files), fetch=_asset_fetch(cb.tar_bytes(files, mtime=1))) == 1
    assert not dest.exists()


def test_from_pin_cannot_fetch_is_unverified(vcu, cp, dest):
    def down(_url):
        raise vcu.Unverified("offline")

    assert vcu.from_pin(cb.pin_for(cp, cb.bundle(cp)), fetch=down) == 2


def test_from_pin_reuses_a_cached_asset_by_sha256(vcu, cp, tmp_path, dest):
    files = cb.bundle(cp)
    pin = cb.pin_for(cp, files)
    calls: list[str] = []
    assert vcu.from_pin(pin, fetch=_asset_fetch(cb.tar_bytes(files), calls), cache_dir=tmp_path / "c") == 0
    assert vcu.from_pin(pin, fetch=_asset_fetch(cb.tar_bytes(files), calls), cache_dir=tmp_path / "c") == 0
    assert len(calls) == 1


def test_from_pin_refuses_a_link_in_the_asset_before_extracting(vcu, cp, dest):
    files = cb.bundle(cp)
    link = tarfile.TarInfo("./assets/link.js")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../../../etc/passwd"
    data = cb.tar_bytes(files, extra=[link])
    pin = cb.pin_for(cp, files, asset_sha256=hashlib.sha256(data).hexdigest())
    assert vcu.from_pin(pin, fetch=_asset_fetch(data)) == 1
    assert not dest.exists()


# ── --verify-pin: is the pin the newest tag for the contract? ────────────────


class FakePulse:
    """pulse's tags, contracts and releases, as `git ls-remote` and HTTP would answer."""

    def __init__(self, cp, files, *, tags=None, contracts=None, digest=None):
        self.cp = cp
        self.asset = cb.tar_bytes(files)
        self.tags = tags or {"v0.4.0": "b" * 40, "v0.4.1": cb.COMMIT}
        self.contracts = contracts or {}
        self.digest = hashlib.sha256(self.asset).hexdigest() if digest is None else digest

    def ls_remote(self, url):
        assert url == "https://github.com/dennys246/Maxim-pulse.git", url
        lines = []
        for name, commit in self.tags.items():
            lines.append(f"{'f' * 40}\trefs/tags/{name}")  # the annotated tag object
            lines.append(f"{commit}\trefs/tags/{name}^{{}}")
        return "\n".join(lines) + "\n"

    def fetch(self, url):
        if url.startswith("https://raw.githubusercontent.com/dennys246/Maxim-pulse/"):
            commit = url.split("/")[5]
            version = self.contracts.get(commit, self.cp.backend_contract())
            if version is None:
                return b"404: Not Found"
            return json.dumps({"info": {"version": version}}).encode()
        if url.startswith("https://api.github.com/repos/dennys246/Maxim-pulse/releases/tags/"):
            digest = None if self.digest == "" else f"sha256:{self.digest}"
            return json.dumps({"assets": [{"name": "console-dist.tar.gz", "digest": digest}]}).encode()
        if url.startswith("https://github.com/dennys246/Maxim-pulse/releases/download/"):
            return self.asset
        raise AssertionError(url)


def _verify(vcu, cp, pulse, pin):
    return vcu.verify_pin(pin, ls_remote=pulse.ls_remote, fetch=pulse.fetch)


def test_verify_passes_a_current_pin(vcu, cp):
    files = cb.bundle(cp)
    assert _verify(vcu, cp, FakePulse(cp, files), cb.pin_for(cp, files)) == 0


def test_verify_refuses_when_a_newer_same_contract_tag_exists(vcu, cp, capsys):
    """The 1.3.2 failure, caught at the release PR."""
    files = cb.bundle(cp)
    pulse = FakePulse(cp, files, tags={"v0.4.1": cb.COMMIT, "v0.4.2": "c" * 40})
    assert _verify(vcu, cp, pulse, cb.pin_for(cp, files)) == 1
    assert "--update-pin v0.4.2" in capsys.readouterr().err


def test_verify_ignores_newer_tags_for_another_contract_and_pre_releases(vcu, cp):
    files = cb.bundle(cp)
    pulse = FakePulse(
        cp,
        files,
        tags={"v0.4.1": cb.COMMIT, "v0.5.0": "c" * 40, "v0.4.2-rc1": "d" * 40, "v0.4.10.1": "e" * 40},
        contracts={"c" * 40: "0.7.0"},
    )
    assert _verify(vcu, cp, pulse, cb.pin_for(cp, files)) == 0


def test_verify_orders_by_semver_not_by_string(vcu, cp):
    """v0.4.10 is newer than v0.4.9."""
    files = cb.bundle(cp)
    pulse = FakePulse(cp, files, tags={"v0.4.9": cb.COMMIT, "v0.4.10": "c" * 40})
    assert _verify(vcu, cp, pulse, cb.pin_for(cp, files, tag="v0.4.9")) == 1


def test_verify_refuses_a_moved_tag(vcu, cp, capsys):
    files = cb.bundle(cp)
    pulse = FakePulse(cp, files, tags={"v0.4.1": "9" * 40})
    assert _verify(vcu, cp, pulse, cb.pin_for(cp, files)) == 1
    assert "was the tag moved" in capsys.readouterr().err


def test_verify_refuses_a_re_uploaded_asset(vcu, cp):
    files = cb.bundle(cp)
    pulse = FakePulse(cp, files, digest="0" * 64)
    assert _verify(vcu, cp, pulse, cb.pin_for(cp, files)) == 1


def test_verify_refuses_a_pin_for_another_repo(vcu, cp):
    files = cb.bundle(cp)
    assert _verify(vcu, cp, FakePulse(cp, files), cb.pin_for(cp, files, repo="x/y")) == 1


def test_verify_refuses_a_pin_whose_contract_is_not_the_backends(vcu, cp, monkeypatch):
    files = cb.bundle(cp)
    pulse = FakePulse(cp, files)
    pin = cb.pin_for(cp, files)
    monkeypatch.setattr(cp, "backend_contract", lambda *_: "0.7.0")
    pulse.contracts = {cb.COMMIT: "0.6.0", "b" * 40: "0.6.0"}
    assert _verify(vcu, cp, pulse, pin) == 1


def test_verify_offline_is_unverified(vcu, cp):
    def down(_url):
        raise vcu.Unverified("offline")

    files = cb.bundle(cp)
    assert vcu.verify_pin(cb.pin_for(cp, files), ls_remote=down, fetch=down) == 2


def test_verify_a_newer_tag_with_an_unreadable_contract_is_unverified_not_skipped(vcu, cp):
    """Fold 9 and the design table: never silently exclude a newer tag that might be the newest."""
    files = cb.bundle(cp)
    pulse = FakePulse(cp, files, tags={"v0.4.1": cb.COMMIT, "v0.4.2": "c" * 40}, contracts={"c" * 40: None})
    assert _verify(vcu, cp, pulse, cb.pin_for(cp, files)) == 2


def test_verify_a_release_without_a_digest_is_unverified(vcu, cp):
    files = cb.bundle(cp)
    assert _verify(vcu, cp, FakePulse(cp, files, digest=""), cb.pin_for(cp, files)) == 2


def test_lightweight_tags_are_read(vcu):
    assert vcu.parse_ls_remote(f"{'a' * 40}\trefs/tags/v1.0.0\n") == {"v1.0.0": "a" * 40}


# ── --update-pin ─────────────────────────────────────────────────────────────


def test_update_pin_writes_what_verify_then_accepts(vcu, cp, tmp_path):
    files = cb.bundle(cp)
    pulse = FakePulse(cp, files)
    path = tmp_path / "pin.json"
    assert vcu.update_pin("v0.4.1", path=path, ls_remote=pulse.ls_remote, fetch=pulse.fetch) == 0
    pin = cp.load_pin(path)
    assert pin == cb.pin_for(cp, files)
    assert vcu.verify_pin(pin, ls_remote=pulse.ls_remote, fetch=pulse.fetch) == 0


def test_update_pin_refuses_a_tag_built_for_another_contract(vcu, cp, tmp_path):
    files = cb.bundle(cp)
    pulse = FakePulse(cp, files, contracts={cb.COMMIT: "0.5.0"})
    assert vcu.update_pin("v0.4.1", path=tmp_path / "p.json", ls_remote=pulse.ls_remote, fetch=pulse.fetch) == 1
    assert not (tmp_path / "p.json").exists()


# ── "is this tree releasing?" fails closed ───────────────────────────────────


def _fake_nightlies(monkeypatch, *, releasing):
    mod = types.ModuleType("check_nightlies")
    mod.releasing = lambda: (releasing, "faked")
    monkeypatch.setitem(sys.modules, "check_nightlies", mod)


def test_releasing_is_check_nightlies_one_definition(cp, monkeypatch):
    """The fail-closed rules themselves are tested in test_check_nightlies.py::test_releasing_*."""
    _fake_nightlies(monkeypatch, releasing=False)
    assert cp.releasing() == (False, "faked")


def test_verify_only_when_releasing_is_a_no_op_between_releases(vcu, cp, monkeypatch):
    _fake_nightlies(monkeypatch, releasing=False)
    assert vcu.main(["--verify-pin", "--only-when-releasing"]) == 0  # no network: the autouse guard would raise


# ── review folds (2026-10-18) ────────────────────────────────────────────────


def test_a_local_vite_build_vendors_without_its_build_only_manifest(vcu, cp, tmp_path, dest):
    """Wire review SF-2: every local pulse build has `.vite/manifest.json`; pulse's pack-dist drops it from the
    release asset, so the directory path skips that one top-level directory and the result matches the pin."""
    files = cb.bundle(cp)
    src = cb.write_dir(tmp_path / "dist", {**files, ".vite/manifest.json": b"{}"})
    assert vcu.vendor(src, pin=cb.pin_for(cp, files)) == 0
    assert not (dest / ".vite").exists()


def test_a_nested_dot_directory_is_still_refused(cp, tmp_path):
    root = cb.write_dir(tmp_path / "d", {**cb.bundle(cp), "assets/.vite/x.json": b"{}"})
    with pytest.raises(cp.DigestError, match="dotfile"):
        cp.dir_entries(root)


def test_contract_drift_is_a_github_warning_in_ci(cp, monkeypatch, capsys):
    """Architecture review: a print inside a green log is a signal nobody reads."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    files = cb.bundle(cp)
    cp.stamp_problems(json.loads(files[cp.MANIFEST_NAME]), cb.pin_for(cp, files), "0.7.0", releasing=False)
    assert "::warning title=Console contract drift (R4)::" in capsys.readouterr().out


# ── executor-review folds: drive main(), and make every guard fail when deleted ─────────────────────────────────


def _spy(monkeypatch, vcu, name, result=0):
    calls: list[dict] = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(vcu, name, spy)
    return calls


@pytest.mark.parametrize("is_release", [True, False])
def test_main_threads_releasing_into_from_pin_and_check(vcu, monkeypatch, is_release):
    """Fold 1's wiring: a hardcoded False would waive drift on a release; a hardcoded True would bring back the
    deadlock for contract-bump PRs."""
    _fake_nightlies(monkeypatch, releasing=is_release)
    from_pin = _spy(monkeypatch, vcu, "from_pin")
    check = _spy(monkeypatch, vcu, "check")
    assert vcu.main(["--from-pin", "--contract-drift-ok-unless-releasing"]) == 0
    assert vcu.main(["--check", "--contract-drift-ok-unless-releasing"]) == 0
    assert from_pin[0]["releasing"] is is_release and check[0]["releasing"] is is_release


def test_main_is_strict_without_the_ci_flag(vcu, monkeypatch):
    monkeypatch.setitem(sys.modules, "check_nightlies", None)  # must not even ask
    from_pin = _spy(monkeypatch, vcu, "from_pin")
    assert vcu.main(["--from-pin"]) == 0 and from_pin[0]["releasing"] is True


def test_verify_only_when_releasing_runs_the_check_on_a_release(vcu, monkeypatch):
    """The release-PR freshness check, the 1.3.2 guard, must not become a permanent no-op."""
    _fake_nightlies(monkeypatch, releasing=True)
    verify = _spy(monkeypatch, vcu, "verify_pin", result=1)
    assert vcu.main(["--verify-pin", "--only-when-releasing"]) == 1 and len(verify) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["--from-pin", "--only-when-releasing"],
        ["--from-pin", "--force"],
        ["--from-pin", "x"],
        ["--check", "--asset-cache", "c"],
    ],
)
def test_main_refuses_flags_that_would_be_silently_ignored(vcu, argv):
    with pytest.raises(SystemExit):
        vcu.main(argv)


def test_main_an_unreadable_backend_contract_is_unverified(vcu, cp, monkeypatch):
    def boom(*_a):
        raise cp.PinError("no ui_bundle.py")

    monkeypatch.setattr(cp, "backend_contract", boom)
    assert vcu.main(["--check"]) == 2


def test_vendor_refuses_a_copy_that_does_not_re_digest(vcu, cp, tmp_path, dest, monkeypatch):
    files = cb.bundle(cp)
    real = vcu.shutil.copytree

    def lossy(src, dst, *args, **kw):
        out = real(src, dst, *args, **kw)
        if Path(dst) == dest:  # copytree recurses through the module name; drop a file at the top level only
            (Path(dst) / "assets" / "index-abc.css").unlink()
        return out

    monkeypatch.setattr(vcu.shutil, "copytree", lossy)
    assert vcu.vendor(cb.write_dir(tmp_path / "dist", files), pin=cb.pin_for(cp, files)) == 1
    assert not dest.exists()


def test_vendor_refuses_to_vendor_the_destination_into_itself(vcu, cp, tmp_path, dest):
    files = cb.bundle(cp)
    cb.write_dir(dest, files)
    assert vcu.vendor(dest, pin=cb.pin_for(cp, files)) == 1
    assert dest.is_dir()


def test_the_asset_is_scanned_before_a_byte_is_extracted(vcu, cp, tmp_path, monkeypatch):
    link = tarfile.TarInfo("./assets/link.js")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    extracted = []
    monkeypatch.setattr(tarfile.TarFile, "extractall", lambda *a, **k: extracted.append(a))
    with pytest.raises(cp.DigestError):
        vcu.extract_asset(cb.tar_bytes(cb.bundle(cp), extra=[link]), tmp_path)
    assert extracted == []


def test_an_old_python_without_the_data_filter_refuses_to_extract(vcu, cp, tmp_path, monkeypatch):
    monkeypatch.delattr(tarfile, "data_filter")
    with pytest.raises(RuntimeError, match="data"):
        vcu.extract_asset(cb.tar_bytes(cb.bundle(cp)), tmp_path)


def test_a_cached_asset_with_the_wrong_bytes_is_downloaded_again(vcu, cp, tmp_path, dest):
    files = cb.bundle(cp)
    pin = cb.pin_for(cp, files)
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / f"{pin.asset_sha256}.tar.gz").write_bytes(b"tampered")
    calls: list[str] = []
    assert vcu.from_pin(pin, fetch=_asset_fetch(cb.tar_bytes(files), calls), cache_dir=tmp_path / "c") == 0
    assert len(calls) == 1


@pytest.mark.parametrize("body", [b"\xc3\x28", b'{"assets": ["x"]}', b"[]"])
def test_an_unreadable_release_body_is_unverified(vcu, cp, body):
    files = cb.bundle(cp)
    pulse = FakePulse(cp, files)
    real = pulse.fetch
    pulse.fetch = lambda url: body if url.startswith("https://api.github.com/repos/") else real(url)
    assert _verify(vcu, cp, pulse, cb.pin_for(cp, files)) == 2


def test_an_asset_that_is_not_a_tar_is_unverified(vcu, cp):
    files = cb.bundle(cp)
    pulse = FakePulse(cp, files)
    pulse.asset = b"not a tarball"
    pulse.digest = hashlib.sha256(pulse.asset).hexdigest()
    assert _verify(vcu, cp, pulse, cb.pin_for(cp, files, asset_sha256=pulse.digest)) == 2
