"""I3: motor dispatch never reads an injected clock (world clock A-pre, D-WC3; docs/plans/world_clock_a_pre.md).

Motor and IO rate limits stay on the wall clock on purpose: a robot's servos move in real seconds, so a motor path
that read an embedder's world clock could command motion faster or slower than the hardware allows. This AST test
refuses, in the motor-dispatch modules, any import of ``maxim.time.agent_time`` and any ``MachineryClock`` /
``PerceivedTime`` annotation, or parameter named for either collaborator. Each form is proven by planting it.

Stated blind spot: an untyped object passed in (a parameter named something else, holding a clock). Nothing static
can see that without types; the rule is that motor code takes no clock collaborator at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "maxim"
MOTOR_DISPATCH = [
    *sorted((SRC / "motion").rglob("*.py")),
    SRC / "hardware" / "controller.py",
    SRC / "hardware" / "reachy" / "controller.py",
]
_MODULE = "maxim.time.agent_time"
_TYPES = {"MachineryClock", "PerceivedTime", "HostMachineryClock", "HostPerceivedTime", "NoCalendar"}
_PARAMS = {"machinery_clock", "perceived_time"}


def _names_in(node: ast.AST | None) -> set[str]:
    """Every bare or dotted name in an annotation, string annotations included."""
    if node is None:
        return set()
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _names_in(ast.parse(node.value, mode="eval").body)
        except SyntaxError:
            return set()
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            found.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            found.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub is not node:
            found |= _names_in(sub)
    return found


def injected_clock_reads(source: str) -> list[str]:
    """Each way *source* takes an injected clock, as ``line: what``."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _MODULE or alias.name.startswith(_MODULE + "."):
                    found.append(f"{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == _MODULE or (module == "maxim.time" and any(a.name == "agent_time" for a in node.names)):
                found.append(f"{node.lineno}: from {module} import ...")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            every += [a for a in (args.vararg, args.kwarg) if a is not None]
            for arg in every:
                if arg.arg in _PARAMS:
                    found.append(f"{node.lineno}: parameter {arg.arg}")
                if _names_in(arg.annotation) & _TYPES:
                    found.append(f"{node.lineno}: parameter {arg.arg} annotated with a clock collaborator")
            if not isinstance(node, ast.Lambda) and _names_in(node.returns) & _TYPES:
                found.append(f"{node.lineno}: returns a clock collaborator")
        elif isinstance(node, ast.AnnAssign) and _names_in(node.annotation) & _TYPES:
            found.append(f"{node.lineno}: annotated with a clock collaborator")
    return found


def test_the_scanned_modules_exist() -> None:
    """A renamed motor module must not silently drop out of the scan."""
    missing = [str(p) for p in MOTOR_DISPATCH if not p.is_file()]
    assert not missing, missing
    assert len(MOTOR_DISPATCH) >= 3


@pytest.mark.parametrize("path", MOTOR_DISPATCH, ids=lambda p: str(p.relative_to(SRC)))
def test_motor_dispatch_reads_no_injected_clock(path: Path) -> None:
    assert injected_clock_reads(path.read_text(encoding="utf-8")) == [], (
        f"{path.relative_to(SRC)} takes an injected clock: motor and IO rate limits stay on the wall (I3)"
    )


@pytest.mark.parametrize(
    ("planted", "expect"),
    [
        ("import maxim.time.agent_time\n", "import maxim.time.agent_time"),
        ("import maxim.time.agent_time as at\n", "import maxim.time.agent_time"),
        ("from maxim.time.agent_time import HOST_MACHINERY_CLOCK\n", "from maxim.time.agent_time import"),
        ("from maxim.time import agent_time\n", "from maxim.time import"),
        ("def move(clock: MachineryClock) -> None: ...\n", "parameter clock annotated"),
        ("def move(clock: 'agent_time.MachineryClock') -> None: ...\n", "parameter clock annotated"),
        ("def move(*, port: PerceivedTime | None = None) -> None: ...\n", "parameter port annotated"),
        ("def move(machinery_clock=None) -> None: ...\n", "parameter machinery_clock"),
        ("def clock() -> MachineryClock: ...\n", "returns a clock collaborator"),
        ("class C:\n    clock: MachineryClock\n", "annotated with a clock collaborator"),
    ],
)
def test_each_injected_clock_form_is_caught(planted: str, expect: str) -> None:
    found = injected_clock_reads(planted)
    assert any(expect in f for f in found), (planted, found)


def test_a_plain_wall_read_is_not_an_injected_clock() -> None:
    """Motor code keeps its wall reads: the ban is on the injected collaborators, not on ``time``."""
    assert (
        injected_clock_reads(
            "import time\n\ndef rate_limit(last: float) -> bool:\n    return time.monotonic() - last > 0.1\n"
        )
        == []
    )
