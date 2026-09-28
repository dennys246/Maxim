"""The documented `maxim.create.agent` example runs, and `capture` says what it wants (roadmap 1.3.1).

The docstring once passed `capture(perception="dark cave ahead")`, which failed deep inside with
"'str' object has no attribute 'salience'". The example is now RUN from the docstring itself, so the
docs cannot drift from the API again, and a wrong argument type is refused up front."""

from __future__ import annotations

import inspect
import re
import textwrap

import pytest

import maxim
from maxim.memory.encoding import EncodingSignals


def _docstring_example(func) -> str:
    doc = inspect.getdoc(func) or ""
    block = doc.split("Example::", 1)[1]
    lines = []
    for line in block.splitlines()[1:]:
        if line.strip() and not line.startswith("    "):
            break
        lines.append(line)
    return textwrap.dedent(chr(10).join(lines))


def test_the_documented_create_agent_example_runs() -> None:
    source = _docstring_example(maxim.create.agent)
    assert "capture(" in source, "the example no longer exercises capture -- update this test"
    exec(compile(source, "<create.agent docstring>", "exec"), {"maxim": maxim})


@pytest.mark.parametrize(
    ("name", "value"),
    [("perception", "dark cave ahead"), ("action", {"tool_name": "look"}), ("outcome", 1.0)],
)
def test_capture_refuses_a_wrong_argument_type_up_front(name, value, tmp_path) -> None:
    # Its own home per case: create.agent refuses a home that already holds state (#939).
    agent = maxim.create.agent("typed_capture", remembers=True, persistence_dir=str(tmp_path / "typed_capture"))
    try:
        before = len(agent.hippocampus)
        with pytest.raises(TypeError, match=rf"capture\(\): {name} must be a"):
            agent.hippocampus.capture(**{name: value}, encoding=EncodingSignals.unmeasured("api"))
        assert len(agent.hippocampus) == before  # refused before any mutation
    finally:
        agent.shutdown()


def test_the_text_hint_names_the_fix() -> None:
    agent = maxim.create.agent("typed_capture_hint", remembers=True)
    try:
        with pytest.raises(TypeError, match=re.escape("Perception(observations={'text': ...})")):
            agent.hippocampus.capture(perception="dark cave ahead", encoding=EncodingSignals.unmeasured("api"))
    finally:
        agent.shutdown()
