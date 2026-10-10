"""scripts/lint_loop_mode_reads.py -- ONE operational-mode reader in the agent loop (#963).

Every gate test drives ``main()`` on a fixture tree: a tiny ``src/maxim`` with the lint's required seeds, clean
as written, and its own allowlist. Each failing fixture adds ONE evasion shape and must fail (rule A per shape,
then rules B-E and the allowlist checks); the scope tests must exit 2. The last tests run the lint on THIS
checkout (clean), and on a copy of it with the pre-#963 ``state.data.get("mode", "live")`` restored at the
follow-up in ``tool_dispatch`` (the deletion probe), with the root taken from this file's path -- never
``maxim.__file__``, which in a worktree without ``PYTHONPATH`` would scan another tree.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts import lint_loop_mode_reads as L

_LOOP_STATE = """
from maxim.modes.definitions import DEFAULT_RUN_MODE


def run_mode(state):
    mode = state.data.get("mode")
    return mode if isinstance(mode, str) and mode else None


def shutdown_requested(state):
    return run_mode(state) == "shutdown"


def seed_runtime_run_mode(state):
    runtime = state.data.get("maxim_runtime")
    mode = run_mode(state)
    if isinstance(runtime, dict) and mode is not None and "mode" not in runtime:
        runtime["mode"] = mode


def operational_mode(executor, state):
    mode = executor.effective_operational_mode() if executor is not None else None
    return mode if mode is not None else (run_mode(state) or DEFAULT_RUN_MODE)
"""

_LOOP_SETUP = """
from maxim.runtime.loop_state import run_mode


def _prepare_executor(executor, action_sink, state):
    executor.set_mode_source(lambda: run_mode(state))
    return executor
"""

_TOOL_DISPATCH = """
from maxim.modes.definitions import get_tool_followup_type
from maxim.runtime.loop_state import operational_mode
from maxim.runtime.loop_types import ActionFollowup


def execute_and_learn(executor, state, autonomy_controller, tool_name):
    autonomy_controller.log_action(action_type="executed", mode=operational_mode(executor, state))
    current_mode = operational_mode(executor, state)
    followup_type = get_tool_followup_type(tool_name, current_mode)
    return ActionFollowup(tool=tool_name, followup_type=followup_type, mode=current_mode)
"""

_AGENT_LOOP = '''
from maxim.runtime import tool_dispatch as td
from maxim.runtime.loop_gates import pre_tick_gate
from maxim.runtime.loop_setup import _prepare_executor
from maxim.runtime.loop_state import operational_mode, seed_runtime_run_mode
from maxim.modes.definitions import get_mode


def run_agentic_loop(executor, state, ctrl):
    """The loop: the docstring's "mode" is not a read."""
    executor = _prepare_executor(executor, None, state)
    seed_runtime_run_mode(state)
    pre_tick_gate(executor, state, ctrl)
    mode_name = operational_mode(executor, state)
    mode_def = get_mode(mode_name) or get_mode("passive")
    info = {"mode": mode_name, "percept_source": state.data.get("percept_source")}
    return mode_def, info, td
'''

_LOOP_GATES = """
from maxim.runtime.loop_state import operational_mode, shutdown_requested


def pre_tick_gate(executor, state, ctrl):
    if shutdown_requested(state):
        return "break"
    ctrl.configure_dn_for_mode(operational_mode(executor, state))
    return None
"""

_EXECUTOR = """
class Executor:
    def __init__(self):
        self._mode_source = None
        self._operational_override = None

    def set_mode_source(self, source):
        self._mode_source = source

    def set_operational_override(self, mode):
        self._operational_override = mode

    def effective_operational_mode(self):
        if self._operational_override is not None:
            return self._operational_override
        if self._mode_source is None:
            return None
        return self._mode_source() or "observe"

    def _mode_denial(self, tool_name):
        from maxim.modes.definitions import get_mode

        mode_name = self.effective_operational_mode()
        if mode_name is None:
            return None
        return get_mode(mode_name) or get_mode("passive")
"""

_WRAPPER = """
class Wrapper:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)
"""

FILES = {
    "src/maxim/__init__.py": "",
    "src/maxim/runtime/__init__.py": "",
    "src/maxim/runtime/loop_state.py": _LOOP_STATE,
    "src/maxim/runtime/loop_setup.py": _LOOP_SETUP,
    "src/maxim/runtime/loop_gates.py": _LOOP_GATES,
    "src/maxim/runtime/loop_types.py": "class ActionFollowup:\n    pass\n",
    "src/maxim/runtime/tool_dispatch.py": _TOOL_DISPATCH,
    "src/maxim/runtime/agent_loop.py": _AGENT_LOOP,
    "src/maxim/runtime/executor.py": _EXECUTOR,
    "src/maxim/runtime/fear_gate.py": _WRAPPER,
    "src/maxim/runtime/pain_interceptor.py": _WRAPPER,
    "src/maxim/runtime/helper.py": "def helper():\n    return 1\n",
    "src/maxim/simulation/__init__.py": "",
    "src/maxim/simulation/instrumented_executor.py": _WRAPPER,
    "src/maxim/modes/__init__.py": "",
    "src/maxim/modes/definitions.py": 'DEFAULT_RUN_MODE = "observe"\n',
}

_ALLOW = [
    {
        "file": "runtime/loop_state.py",
        "qualname": "run_mode",
        "count": 1,
        "axis": "lifecycle",
        "ref": "#1",
        "reason": "r",
    },
    {
        "file": "runtime/loop_state.py",
        "qualname": "seed_runtime_run_mode",
        "count": 2,
        "axis": "lifecycle",
        "ref": "#1",
        "reason": "r",
    },
]
SCOPE = 11  # the fixture's scan set: seeds + executor + wrappers + the runtime closure (incl. __init__)


@pytest.fixture
def tree(tmp_path: Path, monkeypatch) -> Path:
    for rel, text in FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(L, "MIN_SCOPE", SCOPE)
    monkeypatch.setattr(L, "ALLOWLIST", [dict(e) for e in _ALLOW])
    return tmp_path


def run(capsys) -> tuple[int, str]:
    rc = L.main()
    out = capsys.readouterr()
    return rc, out.out + out.err


def add(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text((p.read_text() if p.exists() else "") + "\n\n" + text.strip() + "\n")


# ── the clean fixture, and what every run prints ─────────────────────────────


def test_the_clean_fixture_passes_and_prints_the_scan_set_and_the_allowlist(tree, capsys):
    rc, out = run(capsys)
    assert rc == 0, out
    assert f"scan set (rules A, E): {SCOPE} files" in out
    assert "src/maxim/runtime/fear_gate.py" in out and "src/maxim/simulation/instrumented_executor.py" in out
    assert "src/maxim/runtime/loop_state.py::run_mode = 1 (pin 1, lifecycle, #1)" in out
    assert "src/maxim/runtime/helper.py" not in out  # not imported by the loop: out of scope


def test_a_module_the_loop_imports_lazily_or_relatively_joins_the_scope(tree, capsys):
    add(tree, "src/maxim/runtime/helper.py", 'def read(state):\n    return state.data.get("mode")')
    rc, out = run(capsys)  # not imported yet: out of scope, so only rule D (repo-wide) sees it
    assert rc == 1 and "[D]" in out and "[A]" not in out
    add(tree, "src/maxim/runtime/agent_loop.py", "def lazy():\n    from . import helper\n    return helper")
    rc, out = run(capsys)
    assert rc == 1 and "helper.py" in out and "[A]" in out and 'the constant "mode"' in out


def test_from_runtime_import_submodule_as_alias_joins_the_scope(tree, monkeypatch, capsys):
    monkeypatch.setattr(L, "MIN_SCOPE", SCOPE + 1)
    add(tree, "src/maxim/runtime/helper.py", 'def read(state):\n    return state.data["mode"]')
    add(tree, "src/maxim/runtime/tool_dispatch.py", "from maxim.runtime import helper as h")
    rc, out = run(capsys)
    assert rc == 1 and "runtime/helper.py" in out


def test_import_module_with_a_constant_joins_the_scope(tree, monkeypatch, capsys):
    monkeypatch.setattr(L, "MIN_SCOPE", SCOPE + 1)
    add(tree, "src/maxim/runtime/helper.py", 'def read(state):\n    return state.data["mode"]')
    add(
        tree,
        "src/maxim/runtime/tool_dispatch.py",
        'import importlib\nh = importlib.import_module("maxim.runtime.helper")',
    )
    assert run(capsys)[0] == 1


# ── rule A: one failing fixture per evasion shape ────────────────────────────

_RULE_A = {
    "constant_get": ('def f(state):\n    return state.data.get("mode", "live")', 'the constant "mode"'),
    "constant_subscript_write": ('def f(state):\n    state.data["mode"] = "live"', 'the constant "mode"'),
    "constant_in_test": ('def f(state):\n    return "mode" in state.data', 'the constant "mode"'),
    "constant_via_snapshot": ('def f(state):\n    return state.snapshot()["data"]["mode"]', 'the constant "mode"'),
    "constant_via_state_get": ('def f(state):\n    return state.get("mode")', 'the constant "mode"'),
    "double_star_dict": ("def f(state):\n    return {**state.data}", "**<x>.data"),
    "double_star_call": ("def f(state, g):\n    return g(**state.data)", "**<x>.data"),
    "dict_copy": ("def f(state):\n    return dict(state.data)", "dict(<x>.data)"),
    "copy": ("def f(state):\n    return state.data.copy()", "<x>.data.copy()"),
    "items": ("def f(state):\n    return list(state.data.items())", "<x>.data.items()"),
    "values": ("def f(state):\n    return list(state.data.values())", "<x>.data.values()"),
    "update_kw": ('def f(state):\n    state.update(mode="live")', '.update() writing "mode"'),
    "update_dict": ('def f(state):\n    state.data.update({"mode": "live"})', '.update() writing "mode"'),
    "ior": ('def f(state):\n    state.data |= {"x": 1}', "|= on <x>.data"),
    "nonconst_subscript": ('def f(state):\n    k = "mo" + "de"\n    return state.data[k]', "a non-constant key"),
    "nonconst_get": ('def f(state):\n    k = "mo" + "de"\n    return state.data.get(k)', "non-constant .get()"),
    "nonconst_pop": ('def f(state):\n    k = "mo" + "de"\n    return state.data.pop(k)', "non-constant .pop()"),
    "nonconst_setdefault": ("def f(state, k):\n    return state.data.setdefault(k, 1)", "non-constant .setdefault()"),
    "alias_subscript": ("def f(state, k):\n    d = state.data\n    return d[k]", "a non-constant key"),
    "alias_get": ("def f(state, k):\n    d = state.data\n    return d.get(k)", "non-constant .get()"),
    "alias_snapshot": (
        'def f(state, k):\n    snap = state.snapshot()\n    return snap["data"][k]',
        "a non-constant key",
    ),
    "alias_walrus": ("def f(state, k):\n    if (d := state.data):\n        return d[k]", "a non-constant key"),
    "vars": ("def f(state):\n    return vars(state)", "vars(...)"),
    "dunder_dict": ("def f(state):\n    return state.__dict__", ".__dict__"),
    "getattr_data": ('def f(state):\n    return getattr(state, "data", {})', 'getattr(<x>, "data")'),
}


@pytest.mark.parametrize("shape", list(_RULE_A))
def test_rule_a_fails_each_evasion_shape(tree, capsys, shape):
    code, what = _RULE_A[shape]
    add(tree, "src/maxim/runtime/tool_dispatch.py", code)
    rc, out = run(capsys)
    assert rc == 1, out
    assert "[A]" in out and what in out, out


@pytest.mark.parametrize(
    "module", ["runtime/executor.py", "runtime/fear_gate.py", "simulation/instrumented_executor.py"]
)
def test_rule_a_covers_the_executor_and_its_wrappers(tree, capsys, module):
    add(tree, f"src/maxim/{module}", 'def f(state):\n    return state.data.get("mode")')
    rc, out = run(capsys)
    assert rc == 1 and module in out


def test_a_dict_display_key_and_a_docstring_are_not_reads(tree, capsys):
    add(
        tree,
        "src/maxim/runtime/tool_dispatch.py",
        'def f(m):\n    """Says "mode".\n\n    mode"""\n    return {"mode": m}',
    )
    add(tree, "src/maxim/runtime/tool_dispatch.py", 'def g():\n    """mode"""\n    return 1')
    assert run(capsys)[0] == 0


# ── rule B: the grant and the mode source are read only by the Executor ──────


@pytest.mark.parametrize(
    "code",
    [
        "def f(ex):\n    return ex._operational_override",
        "def f(ex):\n    return ex.operational_override",
        "def f(ex):\n    return ex._mode_source()",
        'def f(ex):\n    return getattr(ex, "operational_override", None)',
        'def f(ex):\n    return hasattr(ex, "_mode_source")',
        "class W:\n    def effective_operational_mode(self):\n        return None",
        'def f(ctrl):\n    return getattr(ctrl, "configure_dn_for_mode")("singularity")',
        'def f(ctrl):\n    return hasattr(ctrl, "get_tool_followup_type")',
        "class W:\n    @property\n    def operational_override(self):\n        return None",
    ],
)
def test_rule_b_repo_wide(tree, capsys, code):
    add(tree, "src/maxim/agents/other.py", code)  # outside the rule-A scope: rule B is repo-wide
    rc, out = run(capsys)
    assert rc == 1 and "[B]" in out, out


def test_rule_b_holds_inside_the_executor_outside_its_four_members(tree, capsys):
    add(tree, "src/maxim/runtime/executor.py", "def peek(ex):\n    return ex._operational_override")
    rc, out = run(capsys)
    assert rc == 1 and "[B]" in out
    tree.joinpath("src/maxim/runtime/executor.py").write_text(
        _EXECUTOR
        + "\n    @property\n    def operational_override(self):\n        return self.effective_operational_mode()\n"
    )
    rc, out = run(capsys)
    assert rc == 1 and "operational_override is gone" in out


# ── rule C: the run mode is for the lifecycle helpers only ───────────────────


@pytest.mark.parametrize(
    ("rel", "code"),
    [
        (
            "src/maxim/runtime/tool_dispatch.py",
            "from maxim.runtime.loop_state import run_mode\n\ndef f(s):\n    return run_mode(s)",
        ),
        (
            "src/maxim/runtime/tool_dispatch.py",
            "from maxim.runtime import loop_state\n\ndef f(s):\n    return loop_state.run_mode(s)",
        ),
        (
            "src/maxim/runtime/tool_dispatch.py",
            'from maxim.runtime import loop_state\n\nf = getattr(loop_state, "run_mode")',
        ),
        ("src/maxim/agents/other.py", "from maxim.runtime.loop_state import run_mode as rm"),
        ("src/maxim/runtime/loop_setup.py", "def other(state):\n    return run_mode(state)"),
        (
            "src/maxim/runtime/loop_setup.py",
            "def _prepare_executor2(executor, state):\n    executor.set_mode_source(lambda: run_mode(state))",
        ),
        ("src/maxim/runtime/loop_state.py", "def other(state):\n    return run_mode(state)"),
    ],
)
def test_rule_c_repo_wide(tree, capsys, rel, code):
    add(tree, rel, code)
    rc, out = run(capsys)
    assert rc == 1 and "[C]" in out, out


def test_rule_c_an_alias_inside_prepare_executor_outside_its_lambda_fails(tree, capsys):
    tree.joinpath("src/maxim/runtime/loop_setup.py").write_text(
        _LOOP_SETUP.replace(
            "    executor.set_mode_source(lambda: run_mode(state))",
            "    rm = run_mode\n    executor.set_mode_source(lambda: rm(state))",
        )
    )
    rc, out = run(capsys)
    assert rc == 1 and "[C]" in out


# ── rule D: mode-keyed reads outside the loop are pinned ─────────────────────


def test_rule_d_pins_a_reader_outside_the_scope(tree, monkeypatch, capsys):
    add(tree, "src/maxim/planning/planner.py", 'def ctx(state):\n    return state.data.get("mode", "")')
    add(tree, "src/maxim/memory/hip.py", 'def cap(state_snapshot):\n    return state_snapshot.get("mode")')
    add(tree, "src/maxim/agents/mem.py", 'def on(p):\n    return p.maxim_runtime["mode"]')
    rc, out = run(capsys)
    assert rc == 1 and out.count("[D]") == 3
    allow = [dict(e) for e in _ALLOW] + [
        {"file": f, "qualname": q, "count": 1, "axis": "r-site", "ref": "#1193", "reason": "r"}
        for f, q in (("planning/planner.py", "ctx"), ("memory/hip.py", "cap"), ("agents/mem.py", "on"))
    ]
    monkeypatch.setattr(L, "ALLOWLIST", allow)
    assert run(capsys)[0] == 0


def test_rule_d_ignores_a_report_dict_named_data(tree, capsys):
    add(tree, "src/maxim/session2.py", 'def load(data):\n    return data.get("mode", "")')
    assert run(capsys)[0] == 0


# ── rule E: every capability sink takes the accessor ─────────────────────────

_RULE_E = {
    "hard_coded_followup": (
        'def f(name):\n    return get_tool_followup_type(name, "active")',
        "get_tool_followup_type",
    ),
    "followup_no_mode": ("def f(name):\n    return get_tool_followup_type(name)", "without a mode argument"),
    "action_followup_literal": ('def f():\n    return ActionFollowup(tool="t", mode="live")', "ActionFollowup"),
    "log_action_missing": ('def f(ac):\n    ac.log_action(action_type="executed")', "without a mode argument"),
    "log_action_run_mode_name": ('def f(ac, s):\n    m = s.data.get("run")\n    ac.log_action(mode=m)', "log_action"),
    "mixed_assignments": (
        "def f(ex, s, x):\n    m = operational_mode(ex, s)\n    if x:\n        m = x\n    return get_mode(m)",
        "get_mode",
    ),
    "parameter": ("def f(mode_name):\n    return get_mode(mode_name)", "get_mode"),
    "dn_hard_coded": ('def f(ctrl):\n    ctrl.configure_dn_for_mode("active")', "configure_dn_for_mode"),
    "modeinfo": ('def f():\n    return ModeInfo(name="observe")', "ModeInfo"),
    "structured_context": ("def f(x):\n    return StructuredContext(mode=x)", "StructuredContext"),
    "roster_with_a_mode": ('def f(r):\n    return r.get_available_tools(mode="active")', "get_available_tools"),
    "dn_controller_configure": (
        'def f(ctrl):\n    ctrl.dn_ctrl.configure_for_mode("singularity")',
        "configure_for_mode",
    ),
    "dn_controller_inhibit": ("def f(ctrl):\n    return ctrl.dn_ctrl.inhibit_for_tool('live')", "inhibit_for_tool"),
    "accessor_with_no_executor": (
        "def f(ctrl, state):\n    ctrl.configure_dn_for_mode(operational_mode(None, state))",
        "configure_dn_for_mode",
    ),
    "accessor_with_a_computed_executor": (
        "def f(name, mk, state):\n    return get_tool_followup_type(name, operational_mode(mk(), state))",
        "get_tool_followup_type",
    ),
    "a_rebound_parameter_of_a_sink": (
        'class C:\n    def configure_dn_for_mode(self, mode_name):\n        mode_name = "active"\n'
        "        self.dn.configure_for_mode(mode_name)",
        "configure_for_mode",
    ),
    "a_defaulted_extra_parameter_of_a_sink": (
        'class C:\n    def configure_dn_for_mode(self, mode_name, fallback="singularity"):\n'
        "        self.dn_ctrl.configure_for_mode(fallback)",
        "configure_for_mode",
    ),
    "accessor_on_another_object": (
        "def f(ctrl, state):\n    ctrl.configure_dn_for_mode(operational_mode(ctrl.dn_ctrl, state))",
        "configure_dn_for_mode",
    ),
    "effective_mode_of_another_object": (
        "def f(other):\n    return get_mode(other.effective_operational_mode())",
        "get_mode",
    ),
    "import_a_sink_as": (
        "from maxim.modes.definitions import get_tool_followup_type as _gtf",
        "imports the sink get_tool_followup_type as _gtf",
    ),
    "bind_a_sink_method": (
        "def f(ctrl):\n    _cfg = ctrl.configure_dn_for_mode\n    return _cfg",
        "a non-call reference to the sink configure_dn_for_mode",
    ),
    "pass_a_sink_as_a_callback": (
        "def f(reg):\n    reg.register(get_mode)",
        "a non-call reference to the sink get_mode",
    ),
}


@pytest.mark.parametrize("shape", list(_RULE_E))
def test_rule_e_fails_each_sink_off_the_accessor(tree, capsys, shape):
    code, what = _RULE_E[shape]
    add(tree, "src/maxim/runtime/tool_dispatch.py", code)
    rc, out = run(capsys)
    assert rc == 1 and "[E]" in out and what in out, out


def test_rule_e_accepts_a_parameter_only_in_a_function_that_is_itself_a_sink(tree, capsys):
    """``LoopController.configure_dn_for_mode`` forwards its parameter to the DN controller, whose ``configure_for_mode``
    forwards it to ``get_mode``: each is a sink, so each caller is checked where it calls (Architecture S1)."""
    add(
        tree,
        "src/maxim/runtime/loop_gates.py",
        "class Ctrl:\n    def configure_dn_for_mode(self, mode_name):\n        self.dn_ctrl.configure_for_mode(mode_name)\n\n"
        "class Dn:\n    def configure_for_mode(self, mode_name):\n        return get_mode(mode_name)\n\n"
        "    def inhibit_for_tool(self, mode_name):\n        return get_mode(mode_name)",
    )
    assert run(capsys)[0] == 0


def test_rule_e_accepts_the_accessor_a_name_bound_only_to_it_and_the_passive_fallback(tree, capsys):
    add(
        tree,
        "src/maxim/runtime/tool_dispatch.py",
        "def f(executor, s, ctrl):\n"
        "    m = operational_mode(executor, s)\n"
        "    ctrl.configure_dn_for_mode(m)\n"
        '    return get_mode(m) or get_mode("passive"), ModeInfo(name=operational_mode(ctrl.executor, s))',
    )
    assert run(capsys)[0] == 0


# ── the allowlist: strict equality, orphans, shape ───────────────────────────


def test_an_allowlisted_count_is_strict_both_ways(tree, monkeypatch, capsys):
    allow = [dict(e) for e in _ALLOW]
    allow[0]["count"] = 2
    monkeypatch.setattr(L, "ALLOWLIST", allow)
    rc, out = run(capsys)
    assert rc == 1 and "1 finding(s), allowlist pins 2 (strict equality)" in out
    add(tree, "src/maxim/runtime/loop_state.py", "")
    allow[0]["count"] = 1
    tree.joinpath("src/maxim/runtime/loop_state.py").write_text(
        _LOOP_STATE.replace('mode = state.data.get("mode")', 'mode = state.data.get("mode") or state.data.get("mode")')
    )
    rc, out = run(capsys)
    assert rc == 1 and "2 finding(s), allowlist pins 1" in out


def test_an_orphan_entry_fails(tree, monkeypatch, capsys):
    allow = [dict(e) for e in _ALLOW] + [
        {
            "file": "runtime/tool_dispatch.py",
            "qualname": "gone",
            "count": 1,
            "axis": "unrelated",
            "ref": "#1",
            "reason": "r",
        }
    ]
    monkeypatch.setattr(L, "ALLOWLIST", allow)
    rc, out = run(capsys)
    assert rc == 1 and "orphan" in out


@pytest.mark.parametrize(
    ("field", "value", "msg"),
    [("axis", "trusted", "axis"), ("ref", "1193", "ref"), ("reason", " ", "reason"), ("count", 0, "count")],
)
def test_a_malformed_entry_fails(tree, monkeypatch, capsys, field, value, msg):
    allow = [dict(e) for e in _ALLOW]
    allow[0][field] = value
    monkeypatch.setattr(L, "ALLOWLIST", allow)
    rc, out = run(capsys)
    assert rc == 1 and f"allowlist[0]: {msg}" in out


def test_an_allowlist_entry_does_not_cover_rules_b_or_c(tree, monkeypatch, capsys):
    add(tree, "src/maxim/agents/other.py", "def f(ex):\n    return ex._operational_override")
    allow = [dict(e) for e in _ALLOW] + [
        {"file": "agents/other.py", "qualname": "f", "count": 1, "axis": "unrelated", "ref": "#1", "reason": "r"}
    ]
    monkeypatch.setattr(L, "ALLOWLIST", allow)
    rc, out = run(capsys)
    assert rc == 1 and "[B]" in out and "orphan" in out


# ── the scope: exit 2, never a pass ──────────────────────────────────────────


@pytest.mark.parametrize(
    "seed",
    ["runtime/agent_loop.py", "runtime/tool_dispatch.py", "runtime/executor.py", "simulation/instrumented_executor.py"],
)
def test_a_missing_seed_exits_2(tree, capsys, seed):
    tree.joinpath("src/maxim", seed).unlink()
    rc, out = run(capsys)
    assert rc == 2 and "required seed(s) missing" in out and seed in out


def test_a_scope_below_the_floor_exits_2(tree, monkeypatch, capsys):
    monkeypatch.setattr(L, "MIN_SCOPE", SCOPE + 1)
    rc, out = run(capsys)
    assert rc == 2 and "below MIN_SCOPE" in out


def test_an_unparsable_file_exits_2(tree, capsys):
    add(tree, "src/maxim/runtime/tool_dispatch.py", "def broken(:\n")
    rc, out = run(capsys)
    assert rc == 2 and "cannot parse" in out


# ── this checkout ────────────────────────────────────────────────────────────

_REPO = Path(__file__).resolve().parents[2]


def test_this_checkout_is_clean(monkeypatch, capsys):
    monkeypatch.setattr(L, "REPO_ROOT", _REPO)
    rc, out = run(capsys)
    assert rc == 0, out
    assert "lint_loop_mode_reads: OK" in out


def test_the_deletion_probe_restoring_the_pre_963_follow_up_read_fails(tmp_path, monkeypatch, capsys):
    """The #963 defect, restored on a copy of this checkout: the follow-up reads the raw state mode again."""
    shutil.copytree(
        _REPO / "src" / "maxim",
        tmp_path / "src" / "maxim",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "data"),
    )
    td = tmp_path / "src/maxim/runtime/tool_dispatch.py"
    text = td.read_text()
    assert text.count("current_mode = operational_mode(executor, state)") == 1
    td.write_text(
        text.replace(
            "current_mode = operational_mode(executor, state)", 'current_mode = state.data.get("mode", "live")'
        )
    )
    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)
    rc, out = run(capsys)
    assert rc == 1, out
    assert "tool_dispatch.py" in out and 'the constant "mode"' in out and "get_tool_followup_type" in out
