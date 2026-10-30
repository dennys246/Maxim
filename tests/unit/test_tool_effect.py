"""``Tool.effect`` and the registry ceiling (#1315 item 2; ``docs/plans/atonia_gate_approach_notes.md`` §2), in-process.

The strict red gates in ``test_atonia_red_gates_effect_policy.py`` pin the behaviour the notes name; these cover the
rest of the new code paths directly: ``Effect`` itself, the record and its mismatch check, ``declare_effect``, the
shared ``EffectAdmission`` (replacement under ``ALL``, the batch paths), ``log_effect_refusal``'s structured record,
and the three sites that catch a refusal (the DM's scene registration, the executor's entity acquisition, the user
tool injection).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from maxim.exceptions import ToolEffectRefused
from maxim.tools.base import (
    Effect,
    Tool,
    ToolOutput,
    declare_effect,
    effect_mismatch,
    effect_names,
    effect_of,
    exceeds,
)
from maxim.tools.registry import ToolRegistry, log_effect_refusal


def _cls(name: str, effect: Any = None, *, declare: bool = True) -> type[Tool]:
    body: dict[str, Any] = {
        "name": name,
        "description": name,
        "input_schema": {},
        "execute": lambda self, **kw: ToolOutput(success=True, output=name),
    }
    if declare:
        body["effect"] = effect
    return type(f"T_{name}", (Tool,), body)


# ── Effect ──────────────────────────────────────────────────────────────────


def test_effect_iterates_the_seven_flags_and_all_is_their_union() -> None:
    names = [m.name for m in Effect]
    assert names == ["READ", "SELF_SCOPE", "WRITE_LOCAL", "NETWORK", "COMMS", "ACTUATE", "GOVERN"]
    assert effect_names(Effect.ALL) == sorted(names)
    assert effect_names(Effect.READ | Effect.NETWORK) == ["NETWORK", "READ"]


def test_exceeds_is_the_subset_check() -> None:
    assert not exceeds(Effect.READ, Effect.READ)
    assert exceeds(Effect.WRITE_LOCAL, Effect.READ | Effect.NETWORK)
    assert not exceeds(Effect.READ | Effect.NETWORK, Effect.ALL)


# ── the record ──────────────────────────────────────────────────────────────


def test_undeclared_records_all_and_declared_records_its_own() -> None:
    assert effect_of(_cls("undeclared", declare=False)()) == Effect.ALL
    assert effect_of(_cls("declared", Effect.READ | Effect.SELF_SCOPE)()) == Effect.READ | Effect.SELF_SCOPE


@pytest.mark.parametrize("bad", [None, Effect(0), 1, "READ"])
def test_a_non_effect_or_empty_declaration_is_refused_at_class_creation(bad: Any) -> None:
    with pytest.raises(TypeError, match="non-empty"):
        _cls("bad", bad)


def test_mismatch_only_for_a_present_value_that_differs_from_the_record() -> None:
    declared = _cls("m_declared", Effect.READ)
    undeclared = _cls("m_undeclared", declare=False)
    assert effect_mismatch(declared()) is None
    assert effect_mismatch(undeclared()) is None
    assert effect_mismatch(SimpleNamespace(effect=Effect.READ)) is None  # no record: effect_of says ALL anyway
    undeclared.effect = Effect.ALL  # matches the recorded ALL
    assert effect_mismatch(undeclared()) is None
    declared.effect = "READ"
    assert "reassigned" in (effect_mismatch(declared()) or "")


# ── declare_effect (register_tool(effect=)) ─────────────────────────────────


def test_declare_effect_declares_once_and_never_rewrites() -> None:
    cls = _cls("late", declare=False)
    declare_effect(cls, Effect.READ)
    assert effect_of(cls()) == Effect.READ and effect_mismatch(cls()) is None
    declare_effect(cls, Effect.READ)  # the same declaration again is a no-op
    with pytest.raises(TypeError, match="cannot rewrite"):
        declare_effect(cls, Effect.ALL)


def test_declare_effect_refuses_a_bad_effect_or_a_non_tool() -> None:
    with pytest.raises(TypeError, match="non-empty"):
        declare_effect(_cls("late_bad", declare=False), Effect(0))
    with pytest.raises(TypeError, match="Tool instance"):
        declare_effect(SimpleNamespace, Effect.READ)


# ── the registry ceiling ────────────────────────────────────────────────────


def test_a_ceiling_must_be_an_effect_and_is_read_only() -> None:
    with pytest.raises(TypeError, match="ceiling"):
        ToolRegistry(ceiling="READ")  # type: ignore[arg-type]
    reg = ToolRegistry(ceiling=Effect.READ)
    assert reg.ceiling == Effect.READ
    with pytest.raises(AttributeError):
        reg.ceiling = Effect.ALL  # type: ignore[misc]


def test_a_refusal_carries_the_tool_effect_and_ceiling() -> None:
    reg = ToolRegistry(ceiling=Effect.READ)
    with pytest.raises(ToolEffectRefused) as info:
        reg.register(_cls("net", Effect.READ | Effect.NETWORK)())
    assert (info.value.tool_name, info.value.effect, info.value.ceiling) == ("net", ["NETWORK", "READ"], ["READ"])
    assert reg.find("net") is None


def test_under_all_a_replacement_by_another_class_is_allowed_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    reg = ToolRegistry()
    first, second = _cls("dup", Effect.READ)(), _cls("dup", Effect.READ)()
    reg.register(first)
    with caplog.at_level(logging.WARNING, logger="maxim.tools.registry"):
        reg.register(type(first)())  # same class: a refresh, not logged
        assert not caplog.records
        reg.register(second)
    assert reg.get("dup") is second
    (rec,) = caplog.records
    assert rec.event == "tool_replaced" and rec.data["tool"] == "dup"  # type: ignore[attr-defined]


def test_below_all_the_same_instance_again_is_not_a_replacement() -> None:
    reg = ToolRegistry(ceiling=Effect.READ)
    t = _cls("again", Effect.READ)()
    reg.register(t)
    reg.register(t)
    assert reg.get("again") is t


def test_register_many_is_all_or_nothing() -> None:
    reg = ToolRegistry(ceiling=Effect.READ)
    ok, wide = _cls("ok", Effect.READ)(), _cls("wide", Effect.ALL)()
    with pytest.raises(ToolEffectRefused):
        reg.register_many([ok, wide])
    assert reg.list_all() == []
    reg.register_many([ok])
    assert reg.list_all() == ["ok"]


def test_a_refused_scene_registers_nothing_and_evicts_nothing() -> None:
    reg = ToolRegistry(ceiling=Effect.READ, active_tool_cap=1)
    reg.register_scene_tools([_cls("s1", Effect.READ)()], "scene_a")
    with pytest.raises(ToolEffectRefused):
        reg.register_scene_tools([_cls("s2", Effect.READ)(), _cls("s3", Effect.ALL)()], "scene_b")
    assert reg.list_all() == ["s1"] and reg.get_active_scenes() == ["scene_a"]


def test_sim_registry_admits_and_keeps_its_redirect_only_under_all() -> None:
    from maxim.simulation.tools import SimToolRegistry, _FallbackRedirectTool

    narrow = SimToolRegistry(ceiling=Effect.READ)
    with pytest.raises(ToolEffectRefused):
        narrow.register(_cls("sim_wide", Effect.ALL)())
    narrow.register(_cls("sim_read", Effect.READ)())
    assert narrow.list() == ["sim_read"]
    with pytest.raises(KeyError):
        narrow.get("missing")
    assert isinstance(SimToolRegistry().get("missing"), _FallbackRedirectTool)


# ── the sites that catch a refusal ──────────────────────────────────────────


def _refusal() -> ToolEffectRefused:
    return ToolEffectRefused("x", tool_name="t", effect=["ALL"], ceiling=["READ"])


def test_log_effect_refusal_is_one_structured_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="maxim.tools.registry"):
        log_effect_refusal(_refusal(), site="here")
    (rec,) = caplog.records
    assert rec.levelno == logging.WARNING and rec.event == "tool_effect_refused"  # type: ignore[attr-defined]
    assert rec.data == {"site": "here", "tool": "t", "effect": ["ALL"], "ceiling": ["READ"]}  # type: ignore[attr-defined]


def test_the_dm_scene_logs_a_refused_entity_and_records_no_tools(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import maxim.embodiment.tool_bridge as tool_bridge
    from maxim.simulation.dm_runtime import SceneState

    def _refuse(*_a: Any, **_k: Any) -> list[Tool]:
        raise _refusal()

    monkeypatch.setattr(tool_bridge, "generate_tools_for_entity", _refuse)
    scene = SceneState({"goblin": object()}, tool_registry=ToolRegistry(ceiling=Effect.READ))
    with caplog.at_level(logging.WARNING, logger="maxim.tools.registry"):
        scene._register_entity_tools("goblin")
    assert scene._registered_tools == {}
    assert [r.event for r in caplog.records] == ["tool_effect_refused"]  # type: ignore[attr-defined]
    assert "goblin" in caplog.records[0].data["site"]  # type: ignore[attr-defined]


def test_entity_acquisition_logs_a_refusal(caplog: pytest.LogCaptureFixture) -> None:
    from maxim.runtime.executor import Executor

    entity = SimpleNamespace(name="sword", reparent=lambda root: None)

    def _refuse(_entity: Any) -> list[Tool]:
        raise _refusal()

    fake = SimpleNamespace(
        _entity_map=SimpleNamespace(resolve=lambda n: entity, is_self=lambda e: False, transfer_to_self=lambda e: None),
        embodiment=SimpleNamespace(root=object()),
        generate_entity_tools=_refuse,
    )
    with caplog.at_level(logging.WARNING, logger="maxim.tools.registry"):
        Executor._handle_entity_acquisition(fake, {"entity_acquired": "sword"})  # type: ignore[arg-type]
    assert [r.event for r in caplog.records] == ["tool_effect_refused"]  # type: ignore[attr-defined]


def test_tool_decorator_refuses_a_bad_effect_and_registers_nothing() -> None:
    import maxim.api as api

    def _bad_effect_probe() -> str:
        return "x"

    with pytest.raises(TypeError):
        api.tool(effect="READ")(_bad_effect_probe)
    assert "_bad_effect_probe" not in api.list_registered_tools()


# ── §2.1: the pinned literal-name helpers are called with literal names only ─

_PINNED_HELPERS = ("require_optional_dependency", "optional_dependency_available", "resolve_backend_class")
# The call sites that pass a variable, each reading a module-level table of literals (atonia notes §2.1: the reach
# test does not descend into these helpers, so a model-chosen name must never reach one).
_TABLE_SITES = {
    ("maxim/cli_utils.py", "_missing_backend_dependency"): ("maxim/cli_utils.py", "_CLOUD_BACKEND_DEPENDENCY"),
    ("maxim/runtime/leader_proxy.py", "_try_import"): ("maxim/runtime/leader_proxy.py", "_EXTRA_IMPORT_MAP"),
    ("maxim/models/language/router.py", "_maybe_dispatch_backend_class"): (
        "maxim/runtime/lane_backends.py",
        "BACKEND_CLASSES",
    ),
}


def _literal_table(path: Any, name: str) -> bool:
    import ast

    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        target = (
            node.target
            if isinstance(node, ast.AnnAssign)
            else (node.targets[0] if isinstance(node, ast.Assign) else None)
        )
        if isinstance(target, ast.Name) and target.id == name and isinstance(node.value, ast.Dict):
            leaves = [v for val in node.value.values for v in (val.elts if isinstance(val, ast.Tuple) else [val])]
            return all(isinstance(v, ast.Constant) and isinstance(v.value, str) for v in leaves)
    return False


def test_pinned_import_helpers_take_only_literal_or_table_names() -> None:
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src"
    sites: dict[tuple[str, str], str] = {}
    for path in sorted((src / "maxim").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
                f = call.func
                called = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                if called not in _PINNED_HELPERS:
                    continue
                first = call.args[0] if call.args else None
                if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                    sites[(str(path.relative_to(src)), fn.name)] = called
    assert set(sites) == set(_TABLE_SITES), f"non-literal pinned-helper call sites changed: {sorted(sites)}"
    for table_path, table in _TABLE_SITES.values():
        assert _literal_table(src / table_path, table), f"{table_path}::{table} is not a dict of string literals"
