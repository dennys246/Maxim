"""The console handle's filesystem tools are scoped to the agent's OWN workspace.

Before sandbox-launch, ``MaximHandle`` built its registry with
``operational_mode="active"`` and no override, so the agent's ``read_file`` /
``write_file`` / ``edit_file`` (and ``bash``, when armed) covered the SERVER'S
CWD — wherever the operator happened to launch ``maxim serve`` — and the
registry builder scaffolded ``.maxim_workspace/`` into that CWD on the first
Talk request, which is fatal on a read-only root. Both are pinned here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="requires the `console` extra (fastapi/uvicorn)")

from tests.unit.test_console_tool_allowlist import _build_handle  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_effective_cwd(monkeypatch):
    # `filesystem_policy.get_effective_cwd()` caches os.getcwd() process-wide
    # on first use; a chdir after that is invisible to the registry builder.
    from maxim.utils import filesystem_policy

    monkeypatch.setattr(filesystem_policy, "_current_working_directory", None)
    yield
    monkeypatch.setattr(filesystem_policy, "_current_working_directory", None)


class TestRegistryBuilder:
    def test_override_root_owns_the_scaffold_not_the_cwd(self, tmp_path, monkeypatch):
        from maxim.runtime.bootstrap import build_tool_registry

        cwd = tmp_path / "cwd"
        root = tmp_path / "root"
        cwd.mkdir()
        root.mkdir()
        monkeypatch.chdir(cwd)
        build_tool_registry(operational_mode="active", allowed_dirs_override=[str(root)])
        assert not (cwd / ".maxim_workspace").exists(), "scaffold landed in the CWD despite an override"
        assert (root / ".maxim_workspace").is_dir()

    def test_no_override_keeps_the_cwd_scaffold(self, tmp_path, monkeypatch):
        # Negative control: the CLI/agent-loop path is unchanged.
        from maxim.runtime.bootstrap import build_tool_registry

        monkeypatch.chdir(tmp_path)
        build_tool_registry(operational_mode="active")
        assert (tmp_path / ".maxim_workspace").is_dir()


class TestHandleScope:
    def test_tools_are_scoped_to_the_agent_workspace(self, monkeypatch, tmp_path):
        cwd = tmp_path / "server_cwd"
        cwd.mkdir()
        handle, _ = _build_handle(monkeypatch, tmp_path)  # chdirs to tmp_path; home = tmp_path/home
        registry = handle.instance.tool_registry
        expected = os.path.realpath(str(tmp_path / "home" / "workspace"))
        for name in ("read_file", "write_file", "edit_file"):
            tool = registry.get(name)
            assert tool is not None, name
            assert tool._allowed_dirs == [expected], (name, tool._allowed_dirs)
        assert handle._workspace == Path(tmp_path / "home" / "workspace")
        assert handle._workspace.is_dir()
        assert not (Path(os.getcwd()) / ".maxim_workspace").exists(), "scaffold leaked into the server CWD"

    def test_relative_paths_resolve_against_the_workspace(self, monkeypatch, tmp_path):
        # The mode prompts teach `.maxim_workspace/notes/x` style RELATIVE
        # paths; with the tools scoped away from the CWD those must resolve
        # against the workspace, not fail closed against the process CWD.
        handle, _ = _build_handle(monkeypatch, tmp_path)
        note = handle._workspace / ".maxim_workspace" / "notes" / "hello.txt"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text("from the workspace")
        run = handle.instance.executor.execute
        result = run({"tool_name": "read_file", "params": {"path": ".maxim_workspace/notes/hello.txt"}})
        assert result.success, result.error
        assert "from the workspace" in str(result.output)
        # And a relative path can still not escape the root.
        escaped = run({"tool_name": "read_file", "params": {"path": "../../outside.txt"}})
        assert not escaped.success
