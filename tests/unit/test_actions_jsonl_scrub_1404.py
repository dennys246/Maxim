"""#1404 item 2 (owner decision 2026-11-10: scrub at write): ``actions.jsonl`` keeps no URL credentials or paths.

``simulation/report.py::save_action_log`` wrote tool args and results verbatim, so a URL the model chose (a fetch
target with a token in its query) or an httpx error string carrying a request URL with ``user:pass@`` reached a
record that is committed with experiment evidence. Every string it writes now goes through
``utils/http.py::redact_urls`` (the #997 scrub: a URL keeps ``scheme://host[:port]``), nested params included; text
that holds no URL is unchanged, so the benchmark scripts' ``params`` comparisons (entity names) still match.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

_RED = pytest.mark.xfail(strict=True, reason="red gate: #1404 item 2 not fixed yet")

SECRET_URL = "https://user:s3cret@files.example.com/private/report.pdf?token=t0ken"


def _action(**over):
    base = dict(
        timestamp=1.0,
        tool_name="http_fetch",
        tool_args={},
        result_success=False,
        result_output=None,
        result_error=None,
        blocked=False,
        block_reason=None,
        agent_id=None,
        session_id=None,
        entity_class=None,
        rpe=None,
        pain=None,
        nac_failure_booked=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _written(tmp_path, *actions) -> list[dict]:
    from maxim.simulation.report import save_action_log

    bridge = SimpleNamespace(get_all_actions=lambda: list(actions))
    path = save_action_log(bridge, str(tmp_path), "sess")
    assert path is not None
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in lines if r.get("_record_kind") != "header"]


@_RED
def test_a_url_in_tool_args_keeps_only_its_origin(tmp_path) -> None:
    rec = _written(tmp_path, _action(tool_args={"url": SECRET_URL, "nested": {"links": [SECRET_URL]}}))[0]
    assert rec["params"]["url"] == "https://files.example.com"
    assert rec["params"]["nested"]["links"] == ["https://files.example.com"]


@_RED
@pytest.mark.parametrize(
    "field, attr", [("output", "result_output"), ("error", "result_error"), ("block_reason", "block_reason")]
)
def test_a_url_in_free_text_fields_keeps_only_its_origin(tmp_path, field: str, attr: str) -> None:
    text = f"ConnectError while fetching {SECRET_URL}: refused"
    rec = _written(tmp_path, _action(**{attr: text}))[0]
    assert rec[field] == "ConnectError while fetching https://files.example.com: refused"


@_RED
def test_no_credential_survives_anywhere_in_the_written_file(tmp_path) -> None:
    from maxim.simulation.report import save_action_log

    action = _action(
        tool_args={"url": SECRET_URL}, result_output=SECRET_URL, result_error=SECRET_URL, block_reason=SECRET_URL
    )
    bridge = SimpleNamespace(get_all_actions=lambda: [action])
    raw = save_action_log(bridge, str(tmp_path), "sess").read_text()
    for leak in ("s3cret", "t0ken", "/private/report.pdf"):
        assert leak not in raw


def test_control_text_and_values_without_a_url_are_unchanged(tmp_path) -> None:
    args = {"entity": "fire_pit", "amount": 3, "flag": True, "items": ["sharp_rock", None]}
    rec = _written(tmp_path, _action(tool_name="touch", tool_args=args, result_output="You touch the fire pit."))[0]
    assert rec["params"] == args
    assert rec["output"] == "You touch the fire pit."
