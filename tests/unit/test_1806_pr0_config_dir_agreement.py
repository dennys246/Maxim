"""#1806 PR 0 (review A1): every config-dir file resolves through ``utils/paths.py::maxim_config_dir``.

The file tools' control-plane write deny (``tools/base.py::_control_plane_roots``) denies ONE config directory. That is
only sound while every reader and writer of a config-dir file agrees on where it is: the six resolvers below used to be
hand-copied APPDATA/XDG bodies, so one drifting copy (a new override in one of them) would silently move that file out
of the deny. This pins the agreement structurally, under every branch of the resolver, and pins the resolver's own
semantics (so the agreement cannot be met by all six drifting together).
"""

from __future__ import annotations

import ast
import platform
from pathlib import Path

import pytest


def _resolved_files() -> dict[str, Path]:
    from maxim.models.language.profile_loader import profiles_config_path
    from maxim.peer.config import peer_config_path
    from maxim.peer.mesh_config import mesh_config_path
    from maxim.runtime.config_loader import config_path
    from maxim.runtime.role import _mesh_yml_path
    from maxim.tunnel.keys import CONSOLE_KEY_NAME, MESH_KEY_NAME, key_file_path

    return {
        "peer.yml": peer_config_path(),
        "config.json": config_path(),
        "role mesh.yml": Path(_mesh_yml_path()),
        "profiles.yml": profiles_config_path(),
        "api_key": key_file_path(MESH_KEY_NAME),
        "console_token": key_file_path(CONSOLE_KEY_NAME),
        "mesh.yml": mesh_config_path(),
    }


def _assert_all_in(expected: Path) -> None:
    from maxim.tools.base import _control_plane_roots
    from maxim.utils.paths import maxim_config_dir

    assert maxim_config_dir() == expected
    for name, path in _resolved_files().items():
        assert path.parent == expected, f"{name} resolves outside maxim_config_dir(): {path}"
    assert str(expected) in _control_plane_roots()[0]


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    return home


@pytest.mark.parametrize("xdg", [None, ""])
def test_posix_without_xdg_is_home_dot_config(home: Path, monkeypatch, xdg: str | None) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    if xdg is not None:
        monkeypatch.setenv("XDG_CONFIG_HOME", xdg)  # empty counts as unset
    _assert_all_in(home / ".config" / "maxim")


def test_posix_honours_xdg_config_home(home: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _assert_all_in(tmp_path / "xdg" / "maxim")


def test_windows_honours_appdata_and_ignores_xdg(home: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _assert_all_in(tmp_path / "roaming" / "maxim")


@pytest.mark.parametrize("appdata", [None, ""])
def test_windows_without_appdata_is_home_appdata_roaming(home: Path, monkeypatch, appdata: str | None) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    if appdata is not None:
        monkeypatch.setenv("APPDATA", appdata)
    _assert_all_in(home / "AppData" / "Roaming" / "maxim")


def test_maxim_config_dir_never_creates_the_directory(home: Path, monkeypatch) -> None:
    from maxim.utils.paths import maxim_config_dir

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    maxim_config_dir()
    assert not (home / ".config").exists()


def test_data_home_if_resolved_has_no_side_effects(tmp_path: Path, monkeypatch) -> None:
    """None before :func:`data_home` runs (and nothing created); the cached path after."""
    from maxim.utils import paths

    target = tmp_path / "data"
    monkeypatch.setenv("MAXIM_DATA_HOME", str(target))
    paths._reset_caches()
    assert paths.data_home_if_resolved() is None
    assert not target.exists()
    assert paths.data_home() == target
    assert paths.data_home_if_resolved() == target
    paths._reset_caches()


_CONFIG_ENV_NAMES = frozenset({"XDG_CONFIG_HOME", "APPDATA"})
_CONFIG_SCAN_ALLOWED = frozenset(
    {
        # the one resolver
        ("utils/paths.py", "maxim_config_dir"),
        # git's config (not Maxim's), the deny matcher's own pair and roots, and the host-tool env passthrough set;
        # module-level hits are keyed by the assignment they sit in, so a new top-level read is not allowed (round 3 N2)
        ("tools/base.py", "_global_git_config_paths"),
        ("tools/base.py", "_control_plane_roots"),
        ("tools/base.py", "<module>:_CONTROL_PLANE_CONFIG_PAIR"),
        ("tools/base.py", "<module>:HOST_TOOL_ENV_ALLOW"),
    }
)


def _flagged(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if value == ".config" or value in _CONFIG_ENV_NAMES:
        return True
    # a one-constant path piece (round 3 N1); whitespace means prose (a fix hint naming the path), not a path
    if (".config/maxim" in value or ".config\\maxim" in value) and not any(c.isspace() for c in value):
        return True
    return any(value.startswith(f"{p}{n}") for p in ("$", "${", "%") for n in _CONFIG_ENV_NAMES)


def _strip_docstring(body: list) -> list:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        return body[1:]
    return body


class _ConfigDirReads(ast.NodeVisitor):
    """Collects ``(file, Class.function)`` scopes holding a flagged string constant (docstrings skipped)."""

    def __init__(self, rel: str) -> None:
        self.rel, self.stack, self.hits, self.assign = rel, [], set(), ""

    def _scoped(self, node) -> None:
        for child in node.decorator_list:
            self.visit(child)
        for field_name in ("args", "returns", "bases", "keywords"):
            value = getattr(node, field_name, None)
            for child in value if isinstance(value, list) else [value]:
                if isinstance(child, ast.AST):
                    self.visit(child)
        self.stack.append(node.name)
        for child in _strip_docstring(node.body):
            self.visit(child)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def _assigned(self, node) -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        outer, self.assign = self.assign, ",".join(names) or "?"
        self.generic_visit(node)
        self.assign = outer

    visit_Assign = visit_AnnAssign = _assigned

    def visit_Constant(self, node: ast.Constant) -> None:
        if _flagged(node.value):
            scope = ".".join(self.stack) or f"<module>:{self.assign or '?'}"
            self.hits.add((self.rel, scope))


def _config_dir_scan(root: Path) -> set[tuple[str, str]]:
    hits: set[tuple[str, str]] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = _strip_docstring(tree.body)
        visitor = _ConfigDirReads(path.relative_to(root).as_posix())
        visitor.visit(tree)
        hits |= visitor.hits
    return hits


def test_no_other_module_reads_the_config_dir_env_vars() -> None:
    """Review D5 (tightened in PR 0 delta round 2, N1/A4): no new hand-copied config-dir resolver can appear.

    The scan flags EVERY string constant in ``src/maxim`` that names a config-dir env var (``XDG_CONFIG_HOME``,
    ``APPDATA``, or a ``$``/``%`` expansion of one) or is the ``".config"`` path piece, however it is read: a
    literal ``os.environ.get``, an aliased ``environ``, a module constant, ``in os.environ``, ``getenv(key=)``,
    ``setdefault``, ``expandvars``, and a hand-built ``Path.home() / ".config" / "maxim"`` all carry one. Scopes
    are the fully qualified ``Class.function`` chain, so a same-named method elsewhere is not allowed by its name.

    A constant holding the whole path piece (``"~/.config/maxim"``, ``f"{h}/.config/maxim"``) is flagged too.

    Stated limit: a name assembled at runtime (``"XDG_" + "CONFIG_HOME"``, ``"." + "config"``) carries no
    constant. Docstrings are skipped: they describe the directory, they do not resolve it.
    """
    hits = _config_dir_scan(Path(__file__).resolve().parents[2] / "src" / "maxim")
    assert hits <= _CONFIG_SCAN_ALLOWED, sorted(hits - _CONFIG_SCAN_ALLOWED)
    # every allowance is still needed (not vacuous, and none outlives the code it excused)
    assert hits == _CONFIG_SCAN_ALLOWED, sorted(_CONFIG_SCAN_ALLOWED - hits)


@pytest.mark.parametrize(
    "planted",
    [
        'import os\nK = "XDG_CONFIG_HOME"\ndef f():\n    return os.environ.get(K)\n',
        'from os import environ as env\ndef f():\n    return env["APPDATA"]\n',
        'import os\ndef f():\n    return "XDG_CONFIG_HOME" in os.environ\n',
        'import os\ndef f():\n    return os.getenv(key="APPDATA")\n',
        'import os\ndef f():\n    return os.environ.setdefault("APPDATA", "x")\n',
        'from pathlib import Path\ndef f():\n    return Path.home() / ".config" / "maxim"\n',
        'import os\ndef f():\n    return os.path.expandvars("$XDG_CONFIG_HOME/maxim")\n',
        'import os\ndef f():\n    return os.path.expanduser("~/.config/maxim")\n',
        'def f(h):\n    return f"{h}/.config/maxim"\n',
        # a new module-level read is not covered by an allowed assignment name
        'import os\nX = os.environ.get("XDG_CONFIG_HOME")\n',
        # a same-named method inside the allowed FILE is still not the allowed function
        'import os\nclass C:\n    def maxim_config_dir(self):\n        return os.environ.get("APPDATA")\n',
    ],
)
def test_config_dir_scan_catches_the_planted_forms(planted: str, tmp_path: Path) -> None:
    """The round-2 reviewers' planted forms: each is flagged outside the allowed scopes."""
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils" / "paths.py").write_text(planted, encoding="utf-8")
    hits = _config_dir_scan(tmp_path)
    assert hits and not hits <= _CONFIG_SCAN_ALLOWED, hits


def test_config_dir_scan_skips_docstrings(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text('"""Lives in $XDG_CONFIG_HOME."""\ndef f():\n    """APPDATA"""\n', "utf-8")
    assert _config_dir_scan(tmp_path) == set()
