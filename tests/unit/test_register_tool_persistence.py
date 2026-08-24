"""D18 guard — ``register_tool`` registration is PERSISTENT, not one-shot.

``register_tool`` is a 1.0-stable extension surface whose contract says a tool
is "available to all agents" (docs/user/extension_api.md §2). The pre-fix
``_inject_pending_tools`` cleared the pending list on injection, so a tool was
visible to exactly ONE registry build. Two consequences, both silent:

* the second ``run()``/``imagine()``/``campaign()`` call lost the tool;
* ``orchestrator.py`` injects into the AUT registry from a second site, so
  whichever site ran first consumed the registrations and the other got none.

Every test below fails against the pre-fix code.
"""

from __future__ import annotations

import threading

import pytest

import maxim
from maxim.api import _inject_registered_tools, _registered_tools
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry


class _Probe(Tool):
    name = "d18_probe"
    description = "D18 guard probe"
    input_schema: dict = {}

    def execute(self, **kwargs: object) -> ToolOutput:
        return ToolOutput(success=True, output="ok")


class _Other(_Probe):
    name = "d18_other"
    description = "second probe"


@pytest.fixture(autouse=True)
def _clean_registrations():
    """Persistent registration must not leak between tests."""
    maxim.clear_registered_tools()
    yield
    maxim.clear_registered_tools()


def test_tool_survives_a_second_injection():
    """The core D18 defect: one registration, two registries, both see it."""
    maxim.register_tool(_Probe())

    first, second = ToolRegistry(), ToolRegistry()
    assert _inject_registered_tools(first) == 1
    assert _inject_registered_tools(second) == 1

    assert "d18_probe" in first.list_all()
    assert "d18_probe" in second.list_all(), "second registry lost the tool (D18)"


def test_registration_survives_many_injections():
    maxim.register_tool(_Probe())
    for i in range(5):
        reg = ToolRegistry()
        assert _inject_registered_tools(reg) == 1, f"injection {i} saw no tools"
        assert "d18_probe" in reg.list_all()


def test_two_injection_sites_do_not_starve_each_other():
    """api.run() and orchestrator.py both inject; neither may consume the list."""
    maxim.register_tool(_Probe())
    api_registry, aut_registry = ToolRegistry(), ToolRegistry()

    _inject_registered_tools(api_registry)  # api.py site
    _inject_registered_tools(aut_registry)  # orchestrator.py site

    assert "d18_probe" in api_registry.list_all()
    assert "d18_probe" in aut_registry.list_all(), "orchestrator site was starved (D18)"


def test_injection_does_not_mutate_the_registration_list():
    maxim.register_tool(_Probe())
    before = list(_registered_tools)
    _inject_registered_tools(ToolRegistry())
    assert list(_registered_tools) == before


def test_same_name_registration_is_last_wins_in_the_registry():
    """Re-injection must be idempotent by name, not accumulate duplicates."""
    maxim.register_tool(_Probe())
    maxim.register_tool(_Probe())

    reg = ToolRegistry()
    _inject_registered_tools(reg)
    assert reg.list_all().count("d18_probe") == 1


def test_reregistering_a_name_replaces_rather_than_accumulates():
    """Persistent registration must not grow without bound on re-registration."""

    class _V2(_Probe):
        description = "replacement"

    maxim.register_tool(_Probe())
    maxim.register_tool(_V2())
    assert len(_registered_tools) == 1, "same-name registration must replace in the list"
    assert _registered_tools[0].description == "replacement"

    reg = ToolRegistry()
    _inject_registered_tools(reg)
    assert reg.get("d18_probe").description == "replacement"


def test_unregister_removes_a_replaced_registration_completely():
    class _V2(_Probe):
        description = "replacement"

    maxim.register_tool(_Probe())
    maxim.register_tool(_V2())
    assert maxim.unregister_tool("d18_probe") is True
    assert maxim.unregister_tool("d18_probe") is False, "a stale duplicate survived"


def test_unregister_tool_removes_and_reports():
    maxim.register_tool(_Probe())
    maxim.register_tool(_Other())

    assert maxim.unregister_tool("d18_probe") is True
    assert maxim.unregister_tool("d18_probe") is False, "second removal must report False"

    reg = ToolRegistry()
    _inject_registered_tools(reg)
    assert "d18_probe" not in reg.list_all()
    assert "d18_other" in reg.list_all(), "unregister must not remove unrelated tools"


def test_clear_registered_tools_returns_count():
    maxim.register_tool(_Probe())
    maxim.register_tool(_Other())
    assert maxim.clear_registered_tools() == 2
    assert maxim.clear_registered_tools() == 0

    reg = ToolRegistry()
    assert _inject_registered_tools(reg) == 0


def test_decorator_registration_is_also_persistent():
    @maxim.tool
    def d18_decorated(value: str) -> str:
        """Decorated probe."""
        return value

    first, second = ToolRegistry(), ToolRegistry()
    _inject_registered_tools(first)
    _inject_registered_tools(second)
    assert "d18_decorated" in first.list_all()
    assert "d18_decorated" in second.list_all(), "@tool lost on second injection (D18)"


def test_concurrent_injection_is_thread_safe():
    maxim.register_tool(_Probe())
    seen: list[int] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            seen.append(_inject_registered_tools(ToolRegistry()))
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert seen == [1] * 8, f"some threads saw an emptied list: {seen}"


# ---------------------------------------------------------------------------
# Review-fold guards (two-lens round, 2026-08-23)
# ---------------------------------------------------------------------------


def test_nameless_tool_is_rejected_at_registration():
    """Now that registration is permanent, a nameless object would be appended
    unboundedly and warn inside registry.register on EVERY later run()."""

    class _Nameless:
        description = "no name attribute"

    with pytest.raises(TypeError, match="non-empty string"):
        maxim.register_tool(_Nameless())
    assert maxim.list_registered_tools() == []


def test_empty_name_is_rejected_by_the_tool_abc():
    """Documents which layer owns this: Tool.__init__ rejects it before
    register_tool ever sees the instance, so api.py guards only the
    duck-typed/non-Tool case above."""

    class _Blank(_Probe):
        name = ""

    with pytest.raises(ValueError, match="non-empty name"):
        _Blank()
    assert maxim.list_registered_tools() == []


def test_list_registered_tools_reports_what_will_be_injected():
    maxim.register_tool(_Probe())
    maxim.register_tool(_Other())
    assert maxim.list_registered_tools() == ["d18_probe", "d18_other"]

    maxim.unregister_tool("d18_probe")
    assert maxim.list_registered_tools() == ["d18_other"]

    reg = ToolRegistry()
    _inject_registered_tools(reg)
    assert reg.list_all() == ["d18_other"]
