"""#1806 PR 0 (owner decision D14): an agent's file tools never write Maxim's own control-plane files, in any mode.

The denied set is the resolved config directory (``$XDG_CONFIG_HOME/maxim`` or ``~/.config/maxim``: mesh.yml, the API
keys, the console token, config.json) and ``robots.yaml``, ``robots.yaml.bak``, ``util/device_state.json``,
``util/internet_access.json`` (#1345), ``util/internet_policy.json`` (#832), ``util/hive_release_sequence.json`` and
``util/cost_state.json``, ``config/llm.json`` under both the process's data home and the default ``~/.maxim``.
Each root is the union of the current env's resolver output, the ``Path.home()`` default and the passwd home's
default; three independent tests match a path (#957's every-ancestor stat-identity walk, a realpath + NFC +
case-fold string compare, and an env-independent name rule).

Everything here runs in ``tmp_path`` with ``HOME``, ``XDG_CONFIG_HOME``, ``MAXIM_DATA_HOME`` and the passwd home
patched: no test touches the operator's real ``~/.maxim`` or ``~/.config/maxim``.
"""

from __future__ import annotations

import os
import types
import unicodedata
from pathlib import Path

import pytest

ORIGINAL = "host: 10.0.0.5\nhardware_id: abc\n"
REFUSAL = "control-plane"


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    """A fake operator: HOME = passwd home = ``tmp/home``, no XDG, no MAXIM_DATA_HOME; the workspace is ``tmp/ws``."""
    import pwd

    from maxim.tools.filesystem import EditFileTool, WriteFileTool
    from maxim.utils import paths

    # tmp_path is under /private/var on macOS, a FORBIDDEN_PATHS root (test_edit_file_tool.py's pattern).
    monkeypatch.setattr(WriteFileTool, "FORBIDDEN_PATHS", ())
    monkeypatch.setattr(EditFileTool, "FORBIDDEN_PATHS", ())
    home = tmp_path / "home"
    (home / ".maxim").mkdir(parents=True)
    (tmp_path / "ws").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("MAXIM_DATA_HOME", raising=False)
    set_pw_home(monkeypatch, pwd, home)
    paths._reset_caches()
    return types.SimpleNamespace(root=tmp_path, home=home, ws=tmp_path / "ws")


def set_pw_home(monkeypatch, pwd, home: Path) -> None:
    real = pwd.getpwuid

    def fake(uid):
        entry = real(uid)
        return types.SimpleNamespace(pw_dir=str(home), pw_name=entry.pw_name, pw_uid=entry.pw_uid)

    monkeypatch.setattr(pwd, "getpwuid", fake)


def _write(root: Path, path: Path | str, content: str = "host: 6.6.6.6\n"):
    from maxim.tools.filesystem import WriteFileTool

    return WriteFileTool(allowed_dirs=[str(root)]).execute(path=str(path), content=content, overwrite=True)


def _edit(root: Path, path: Path | str):
    from maxim.tools.filesystem import EditFileTool

    return EditFileTool(allowed_dirs=[str(root)]).execute(path=str(path), old_text="10.0.0.5", new_text="6.6.6.6")


TOOLS = pytest.mark.parametrize("tool", ["write_file", "edit_file"])


def _assert_refused(env, tool: str, target: Path, *, seed: bool = True) -> None:
    """Through ``tool``, a write to ``target`` is refused and leaves it as it was (absent, or the seeded content)."""
    if seed:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(ORIGINAL)
    existed = os.path.lexists(target)
    result = (_write if tool == "write_file" else _edit)(env.root, target)
    assert result.success is False, f"{tool} wrote a control-plane file: {target}"
    assert REFUSAL in (result.error or ""), result.error
    if existed:
        assert Path(target).read_text() == ORIGINAL
    else:
        assert not os.path.lexists(target)


# ── Red gates ──────────────────────────────────────────────────────────────────────────────────────────────────────


@TOOLS
def test_overwriting_an_existing_robots_yaml_is_refused(env, tool: str) -> None:
    _assert_refused(env, tool, env.home / ".maxim" / "robots.yaml")


@TOOLS
def test_the_registry_backup_is_refused(env, tool: str) -> None:
    _assert_refused(env, tool, env.home / ".maxim" / "robots.yaml.bak")


@TOOLS
def test_a_new_device_state_file_is_refused(env, tool: str) -> None:
    """Planting a file that does not exist yet, under a ``util/`` that does not exist either."""
    _assert_refused(env, tool, env.home / ".maxim" / "util" / "device_state.json", seed=False)


@TOOLS
@pytest.mark.parametrize(
    "name", ["internet_access.json", "internet_policy.json", "hive_release_sequence.json", "cost_state.json"]
)
def test_the_operators_internet_files_are_refused(env, tool: str, name: str) -> None:
    """Review A2 (owner decision): an agent cannot grant itself internet access by editing a file."""
    _assert_refused(env, tool, env.home / ".maxim" / "util" / name)


@TOOLS
@pytest.mark.parametrize(
    "name", ["internet_access.json", "internet_policy.json", "hive_release_sequence.json", "cost_state.json"]
)
def test_a_child_data_homes_internet_files_are_refused(env, tool: str, name: str, monkeypatch) -> None:
    """Under a harness child's ``MAXIM_DATA_HOME`` (no ``.maxim`` in its path, so the roots match, not the name rule)."""
    from maxim.utils import paths

    child = env.root / "child-data"
    child.mkdir()
    monkeypatch.setenv("MAXIM_DATA_HOME", str(child))
    paths._reset_caches()
    _assert_refused(env, tool, child / "util" / name)


@TOOLS
def test_a_new_file_in_an_existing_config_subdir_is_refused(env, tool: str) -> None:
    sub = env.home / ".config" / "maxim" / "keys"
    sub.mkdir(parents=True)
    _assert_refused(env, tool, sub / "new.txt", seed=(tool == "edit_file"))


@TOOLS
def test_an_absent_config_dir_is_refused(env, tool: str) -> None:
    """A fresh machine: ``~/.config/maxim`` does not exist, so the deny cannot rely on it existing."""
    assert not (env.home / ".config").exists()
    _assert_refused(env, tool, env.home / ".config" / "maxim" / "mesh.yml", seed=False)
    assert not (env.home / ".config" / "maxim").exists()


@TOOLS
def test_the_xdg_config_dir_is_refused(env, tool: str, monkeypatch) -> None:
    """``$XDG_CONFIG_HOME/maxim``, the resolver's answer, named nothing like ``.config`` (so only the roots match)."""
    xdg = env.root / "xdg-conf"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    _assert_refused(env, tool, xdg / "maxim" / "config.json")
    _assert_refused(env, tool, xdg / "maxim" / "fresh" / "api_key", seed=False)


@TOOLS
def test_a_harness_child_still_refuses_the_operators_default_files(env, tool: str, monkeypatch) -> None:
    """A child runs with ``MAXIM_DATA_HOME=<tmp>``: its own registry AND the operator's real default are refused."""
    from maxim.utils import paths

    child = env.root / "child-data"
    child.mkdir()
    monkeypatch.setenv("MAXIM_DATA_HOME", str(child))
    paths._reset_caches()
    _assert_refused(env, tool, env.home / ".maxim" / "robots.yaml")
    _assert_refused(env, tool, child / "robots.yaml")


@TOOLS
def test_an_nfd_spelling_of_the_config_dir_is_refused(env, tool: str, monkeypatch) -> None:
    """The config root spelled NFC, the write spelled NFD (one file on APFS; never ``.config``, so not the name rule)."""
    nfc = unicodedata.normalize("NFC", "confé")
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd
    xdg = env.root / nfc
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    (xdg / "maxim").mkdir(parents=True)
    target = env.root / nfd / "maxim" / "mesh.yml"
    if tool == "edit_file":
        (xdg / "maxim" / "mesh.yml").write_text(ORIGINAL)
    result = (_write if tool == "write_file" else _edit)(env.root, target)
    assert result.success is False and REFUSAL in (result.error or ""), result.error
    if tool == "edit_file":
        assert (xdg / "maxim" / "mesh.yml").read_text() == ORIGINAL


@TOOLS
def test_a_symlink_into_the_config_dir_is_refused(env, tool: str) -> None:
    cfg = env.home / ".config" / "maxim"
    cfg.mkdir(parents=True)
    (cfg / "mesh.yml").write_text(ORIGINAL)
    link = env.ws / "innocent.yml"
    os.symlink(cfg / "mesh.yml", link)
    result = (_write if tool == "write_file" else _edit)(env.root, link)
    assert result.success is False and REFUSAL in (result.error or ""), result.error
    assert (cfg / "mesh.yml").read_text() == ORIGINAL


@TOOLS
def test_a_hardlink_to_the_registry_is_refused(env, tool: str) -> None:
    """Caught only by inode: the link's own path names nothing Maxim owns (the identity walk's gate)."""
    registry = env.home / ".maxim" / "robots.yaml"
    registry.write_text(ORIGINAL)
    link = env.ws / "notes.yml"
    os.link(registry, link)
    result = (_write if tool == "write_file" else _edit)(env.root, link)
    assert result.success is False and REFUSAL in (result.error or ""), result.error
    assert registry.read_text() == ORIGINAL


@TOOLS
def test_the_name_rule_refuses_an_operator_home_no_root_points_at(env, tool: str, monkeypatch) -> None:
    """HOME, MAXIM_DATA_HOME, XDG_CONFIG_HOME and the passwd home all re-pointed away from the operator's home: only
    the env-independent name rule still sees ``<operator home>/.maxim/robots.yaml``. Deleting the rule fails this."""
    import pwd

    from maxim.utils import paths

    elsewhere = env.root / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("HOME", str(elsewhere))
    monkeypatch.setenv("MAXIM_DATA_HOME", str(elsewhere / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(elsewhere / "xdg"))
    set_pw_home(monkeypatch, pwd, elsewhere)
    paths._reset_caches()
    _assert_refused(env, tool, env.home / ".maxim" / "robots.yaml")
    _assert_refused(env, tool, env.home / ".config" / "maxim" / "peer.yml")


def test_the_sandbox_tools_refuse_control_plane_files(env, tmp_path: Path, monkeypatch) -> None:
    """``write_sandbox_file`` (a sandbox inside the operator's data home) and ``write_to_shared_outputs`` (an outputs
    dir that IS ``util/``) refuse at their #957 sites too."""
    from maxim.tools.sandbox import WriteSandboxFileTool, WriteToSharedOutputsTool

    sandbox = env.home / ".maxim"
    result = WriteSandboxFileTool(sandbox_dir=str(sandbox)).execute(path="robots.yaml", content="x")
    assert result.success is False and REFUSAL in (result.error or ""), result.error
    assert not (sandbox / "robots.yaml").exists()
    outputs = env.home / ".maxim" / "util"
    result = WriteToSharedOutputsTool(instance_outputs_dir=str(outputs)).execute(
        filename="device_state.json", content="{}"
    )
    assert result.success is False and REFUSAL in (result.error or ""), result.error
    assert not (outputs / "device_state.json").exists()


# ── Negative controls: component-wise, not substring ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "relative",
    [
        "ws/robots.yaml",
        "home/.config/maximal/x",
        "home/.maxim/sim_reports/x.json",
        "home/.maxim/util/other_state.json",
        "ws/util/internet_access.json",
        "home/maxim/robots.yaml",
    ],
)
def test_ordinary_files_stay_writable(env, relative: str) -> None:
    target = env.root / relative
    result = _write(env.root, target, content="ok\n")
    assert result.success is True, result.error
    assert target.read_text() == "ok\n"


def test_control_plane_write_refusal_does_not_expand_tilde(env) -> None:
    """It takes the tools' resolved ``real_path`` string; a literal ``~`` is a relative name, as the write sees it."""
    from maxim.tools.base import control_plane_write_refusal

    assert control_plane_write_refusal(str(env.ws / "~" / "notes.txt")) is None


# ── Supervised approval: edit_file is a CWD write too ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tool", ["write_file", "edit_file"])
def test_a_cwd_write_needs_approval_in_supervised_mode(tmp_path: Path, monkeypatch, tool: str) -> None:
    from maxim.agents.autonomy import VERDICT_ALLOW, VERDICT_CONFIRM, SupervisionPolicy
    from maxim.utils import filesystem_policy

    monkeypatch.setattr(filesystem_policy, "get_effective_cwd", lambda: str(tmp_path))
    policy = SupervisionPolicy()
    cwd_file = {"tool_name": tool, "params": {"path": str(tmp_path / "src.py")}}
    verdict = policy.verdict(cwd_file)
    assert verdict.kind == VERDICT_CONFIRM and "CWD write" in (verdict.reason or ""), verdict
    workspace = filesystem_policy.get_workspace_path(str(tmp_path))
    in_workspace = {"tool_name": tool, "params": {"path": os.path.join(workspace, "scratch.py")}}
    assert policy.verdict(in_workspace).kind == VERDICT_ALLOW


# ── SandboxExecutor.write_script (create_sandbox_script) ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["../../escaped", "/abs/path", "sub/dir", "..", ".hidden", "", "a\x00b"])
def test_write_script_refuses_a_name_that_is_a_path(tmp_path: Path, name: str) -> None:
    from maxim.tools.sandbox import CreateSandboxScriptTool
    from maxim.utils.sandbox_executor import SandboxExecutor

    executor = SandboxExecutor(sandbox_dir=str(tmp_path / "sb"))
    with pytest.raises(ValueError):
        executor.write_script(name, "print(1)\n")
    assert not (tmp_path / "escaped.py").exists()
    if name:
        result = CreateSandboxScriptTool(executor).execute(name=name, content="print(1)\n")
        assert result.success is False
    assert list((tmp_path / "sb" / "scripts").iterdir()) == []


def test_write_script_refuses_a_symlink_out_of_the_sandbox(tmp_path: Path) -> None:
    from maxim.utils.sandbox_executor import SandboxExecutor

    executor = SandboxExecutor(sandbox_dir=str(tmp_path / "sb"))
    outside = tmp_path / "outside.py"
    os.symlink(outside, tmp_path / "sb" / "scripts" / "planted.py")
    with pytest.raises(PermissionError, match="within the sandbox"):
        executor.write_script("planted", "print(1)\n")
    assert not outside.exists()


def test_write_script_applies_both_write_refusals(env) -> None:
    """A sandbox rooted at the operator's config dir (control-plane) and one inside a git dir (#957) both refuse."""
    from maxim.utils.sandbox_executor import SandboxExecutor

    executor = SandboxExecutor(sandbox_dir=str(env.home / ".config" / "maxim"))
    with pytest.raises(PermissionError, match=REFUSAL):
        executor.write_script("ok_name", "print(1)\n")
    executor = SandboxExecutor(sandbox_dir=str(env.ws / ".git"))
    with pytest.raises(PermissionError, match=r"\.git"):
        executor.write_script("ok_name", "print(1)\n")


def test_write_script_still_writes_a_plain_name(tmp_path: Path) -> None:
    from maxim.utils.sandbox_executor import SandboxExecutor

    executor = SandboxExecutor(sandbox_dir=str(tmp_path / "sb"))
    path = executor.write_script("my_script-1.v2", "print(1)\n")
    assert path == os.path.realpath(tmp_path / "sb" / "scripts" / "my_script-1.v2.py")
    assert Path(path).read_text() == "print(1)\n"


# ── Each matching layer on its own (so a deletion probe of any one layer fails a test) ────────────────────────────


def test_the_roots_are_the_union_of_env_home_and_passwd_home(env, monkeypatch) -> None:
    """HOME re-pointed by a child: the passwd home's ``.config/maxim`` and ``.maxim`` files are still denied roots."""
    from maxim.tools.base import _control_plane_roots

    child_home = env.root / "child-home"
    monkeypatch.setenv("HOME", str(child_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(env.root / "child-xdg"))
    monkeypatch.setenv("MAXIM_DATA_HOME", str(env.root / "child-data"))
    config_dirs, files = _control_plane_roots()
    assert str(env.root / "child-xdg" / "maxim") in config_dirs  # the resolver's output
    assert str(child_home / ".config" / "maxim") in config_dirs  # the Path.home() default
    assert str(env.home / ".config" / "maxim") in config_dirs  # the passwd home
    for data in (env.root / "child-data", child_home / ".maxim", env.home / ".maxim"):
        assert str(data / "robots.yaml") in files
        assert str(data / "robots.yaml.bak") in files
        assert str(data / "util" / "device_state.json") in files
        assert str(data / "util" / "internet_access.json") in files
        assert str(data / "util" / "internet_policy.json") in files
        assert str(data / "util" / "hive_release_sequence.json") in files
        assert str(data / "util" / "cost_state.json") in files
        assert str(data / "config" / "llm.json") in files


def test_the_roots_include_the_data_home_this_process_resolved(env, monkeypatch) -> None:
    """Review A4: the data home ``data_home()`` cached stays a denied root after the env var moves on."""
    from maxim.tools.base import _control_plane_roots
    from maxim.utils import paths

    resolved = env.root / "resolved-data"
    monkeypatch.setenv("MAXIM_DATA_HOME", str(resolved))
    paths._reset_caches()
    assert paths.data_home() == resolved
    monkeypatch.setenv("MAXIM_DATA_HOME", str(env.root / "later-data"))
    _config_dirs, files = _control_plane_roots()
    assert str(resolved / "robots.yaml") in files
    paths._reset_caches()


def test_the_string_compare_folds_nfd_and_case(env) -> None:
    from maxim.tools.base import _matches_by_string

    root = str(env.root / unicodedata.normalize("NFC", "Confé") / "maxim")
    spelled = str(env.root / unicodedata.normalize("NFD", "confé") / "MAXIM" / "mesh.yml")
    assert _matches_by_string(spelled, [root], []) is True
    assert _matches_by_string(str(env.root / "confe" / "maxim" / "x"), [root], []) is False


def test_the_identity_walk_anchors_a_root_that_does_not_exist_yet(env) -> None:
    """Neither the config root nor the target exists: matched at the deepest existing ancestor plus the tail."""
    from maxim.tools.base import _matches_by_identity

    root = str(env.root / "fresh" / "maxim")
    assert _matches_by_identity(str(env.root / "fresh" / "maxim" / "a" / "b.yml"), [root], []) is True
    assert _matches_by_identity(str(env.root / "fresh" / "maximal" / "b.yml"), [root], []) is False
    denied = str(env.home / ".maxim" / "util" / "device_state.json")
    assert _matches_by_identity(denied, [], [denied]) is True
    assert _matches_by_identity(denied + ".tmp", [], [denied]) is False


@TOOLS
def test_the_llm_config_is_refused(env, tool: str, monkeypatch) -> None:
    """Round 3 D1: llm.json holds the spend caps, ``cost_state_path``, ``cloud_enabled`` and provider base URLs.
    Refused at the default data home (an existing file) and, as a new file, under a child's ``MAXIM_DATA_HOME``."""
    from maxim.utils import paths

    _assert_refused(env, tool, env.home / ".maxim" / "config" / "llm.json")
    child = env.root / "child-data-llm"
    child.mkdir()
    monkeypatch.setenv("MAXIM_DATA_HOME", str(child))
    paths._reset_caches()
    _assert_refused(env, tool, child / "config" / "llm.json", seed=False)  # a new one, under a new config/


def test_an_llm_json_planted_in_the_launch_directory_is_not_read(tmp_path: Path, monkeypatch, caplog) -> None:
    """Round 3 (owner decision) + round 4 S2/S3: with no data-home llm.json, a ``./llm.json`` or
    ``./data/util/llm.json`` an agent planted in the launch directory is NOT the config: the loader ignores it (with
    one WARNING naming it), and neither CPU fallback reads it either (each still applies the CPU profile pin)."""
    import json
    import logging
    import types

    from maxim import cli_utils
    from maxim.embodied_runtime import agentic_runtime
    from maxim.models.language import config as C
    from maxim.utils import paths

    home = tmp_path / "data-home"
    home.mkdir()
    monkeypatch.setenv("MAXIM_DATA_HOME", str(home))
    for name in ("MAXIM_LLM_CONFIG", "MAXIM_LLM_PROFILE", "MAXIM_LLM_N_GPU_LAYERS", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    paths._reset_caches()
    work = tmp_path / "work"
    (work / "data" / "util").mkdir(parents=True)
    planted = {"cloud_enabled": True, "profile": "p", "profiles": {"p": {"backend": "llama_cpp"}}}
    for rel in ("llm.json", "data/util/llm.json"):
        (work / rel).write_text(json.dumps(planted))
    monkeypatch.chdir(work)
    # as on a wheel install: no source checkout, so no repo-root candidates (those are a stated residual)
    monkeypatch.setattr(C, "_source_checkout_root", lambda package_file=None: None)
    monkeypatch.setattr(C, "_ignored_cwd_warning_emitted", False)
    assert C.llm_config_file() is None
    assert C.configured_profile_uses_llama_cpp() is False
    with caplog.at_level(logging.WARNING, logger=C.__name__):
        assert C.load_llm_config().cloud_enabled is False
    warned = [r.getMessage() for r in caplog.records if "is NOT read" in r.getMessage()]
    assert len(warned) == 1 and str(work / "data" / "util" / "llm.json") in warned[0]

    # the CLI fallback, handed the planted directory as its home_dir, still pins the CPU profile
    cli_utils.configure_cpu_fallback_model(logging.getLogger("test"), home_dir=str(work / "data"))
    assert os.environ.get("MAXIM_LLM_PROFILE") == "smollm-1.7b-instruct"
    monkeypatch.delenv("MAXIM_LLM_PROFILE")
    # ... and so does the robot runtime's twin
    monkeypatch.setattr(agentic_runtime, "is_gpu_available", lambda: False)
    runtime = types.SimpleNamespace(home_dir=str(work / "data"), log=logging.getLogger("test"))
    agentic_runtime.AgenticRuntimeMixin._apply_llm_backend_fallback(runtime)  # type: ignore[arg-type]
    assert os.environ.get("MAXIM_LLM_PROFILE") == "smollm-1.7b-instruct"
    paths._reset_caches()


def test_a_checkout_run_from_its_root_does_not_warn(tmp_path: Path, monkeypatch, caplog) -> None:
    """Round 5 S1: in a source checkout run from its root, ``./data/util/llm.json`` IS a repo-root candidate (shadowed
    by the user's file, as before #1806), not an ignored launch-directory file, so it gets no "NOT read" WARNING."""
    import json
    import logging

    from maxim.models.language import config as C
    from maxim.utils import paths

    home = tmp_path / "data-home"
    (home / "config").mkdir(parents=True)
    (home / "config" / "llm.json").write_text(json.dumps({"cloud_enabled": False}))
    monkeypatch.setenv("MAXIM_DATA_HOME", str(home))
    monkeypatch.delenv("MAXIM_LLM_CONFIG", raising=False)
    paths._reset_caches()
    repo = tmp_path / "repo"
    (repo / "data" / "util").mkdir(parents=True)
    (repo / "data" / "util" / "llm.json").write_text(json.dumps({"cloud_enabled": True}))
    monkeypatch.chdir(repo)
    monkeypatch.setattr(C, "_source_checkout_root", lambda package_file=None: str(repo))
    monkeypatch.setattr(C, "_ignored_cwd_warning_emitted", False)
    with caplog.at_level(logging.WARNING, logger=C.__name__):
        assert C.load_llm_config().cloud_enabled is False
    assert not [r for r in caplog.records if "is NOT read" in r.getMessage()]
    paths._reset_caches()


def test_a_deleted_launch_directory_does_not_break_loading(tmp_path: Path, monkeypatch) -> None:
    """Round 5 S2: the launch-directory check must not turn ``os.getcwd()``'s FileNotFoundError into a load failure."""
    from maxim.models.language import config as C
    from maxim.utils import paths

    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data-home"))
    monkeypatch.delenv("MAXIM_LLM_CONFIG", raising=False)
    paths._reset_caches()
    monkeypatch.setattr(C, "_ignored_cwd_warning_emitted", False)

    def gone() -> str:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(C.os, "getcwd", gone)
    assert C.load_llm_config() is not None
    paths._reset_caches()


def test_repo_root_candidates_only_in_a_source_checkout(tmp_path: Path) -> None:
    """Round 4 S1: on a wheel install the repo-root arithmetic lands in ``<venv>/lib/python3.X/`` (an in-project
    ``.venv`` an agent can write), so those candidates exist only for a real checkout."""
    from maxim.models.language import config as C

    wheel = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages" / "maxim" / "models" / "language"
    wheel.mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "pyproject.toml").write_text("")  # even a stray pyproject there does not count
    assert C._source_checkout_root(str(wheel / "config.py")) is None
    checkout = tmp_path / "repo" / "src" / "maxim" / "models" / "language"
    checkout.mkdir(parents=True)
    assert C._source_checkout_root(str(checkout / "config.py")) is None  # no pyproject.toml
    (tmp_path / "repo" / "pyproject.toml").write_text("")
    assert C._source_checkout_root(str(checkout / "config.py")) == str((tmp_path / "repo").resolve())


def test_the_cpu_fallbacks_read_the_loaders_llm_json(tmp_path: Path, monkeypatch) -> None:
    """Round 3: ``configured_profile_uses_llama_cpp`` reads the loader's own file; only an explicit llama.cpp
    profile entry counts (round 4 N1: a non-string ``profile`` is no entry, not a crash)."""
    import json

    from maxim.models.language import config as C

    cfg = tmp_path / "llm.json"
    monkeypatch.setenv("MAXIM_LLM_CONFIG", str(cfg))
    cfg.write_text(json.dumps({"profile": "p", "profiles": {"p": {"backend": "llama_cpp"}}}))
    assert C.llm_config_file() == str(cfg)
    assert C.configured_profile_uses_llama_cpp() is True
    for raw in (
        {"profile": "p", "profiles": {"p": {"backend": "transformers"}}},
        {"profile": ["p"], "profiles": {"p": {"backend": "llama_cpp"}}},
        {"profile": "p", "profiles": ["p"]},
        ["not", "a", "mapping"],
    ):
        cfg.write_text(json.dumps(raw))
        assert C.configured_profile_uses_llama_cpp() is False, raw
    cfg.write_text("{not json")
    assert C.configured_profile_uses_llama_cpp() is False
