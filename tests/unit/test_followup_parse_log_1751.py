"""#1751: an unparseable follow-up input is logged by shape, never by content.

The input can be real tool output, including output of PRIVATE or SECRET tools, and log files
(``MAXIM_LOG_FILE``) persist it. The WARNING must carry only a length and a stable digest.
"""

from __future__ import annotations

import logging

from maxim.agents.prompt_builder import PromptBuilder
from maxim.utils.seeding import stable_hash_32

SENTINEL = "SENTINEL_LOCKER_1751"


def _builder() -> PromptBuilder:
    from maxim.agents.llm_fallback import ReasoningCarryover
    from maxim.models.language.token_counter import CharEstimateCounter

    return PromptBuilder(
        llm=None,
        reasoning_carryover=ReasoningCarryover(),
        n_ctx=32000,
        token_counter=CharEstimateCounter(),
    )


def _warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.WARNING and "followup" in r.getMessage().lower()]


def test_unparseable_followup_warning_carries_no_content(caplog) -> None:
    raw = f"{SENTINEL} private tool output that matches no follow-up format"
    with caplog.at_level(logging.DEBUG, logger="maxim"):
        _builder()._build_followup_prompt(raw)
    records = _warnings(caplog)
    assert records, "the unparseable input must still be reported"
    for record in caplog.records:  # no level, not only WARNING, may carry the content
        assert SENTINEL not in record.getMessage()
        assert SENTINEL not in str(record.args)


def test_warning_identifies_the_input_by_length_and_stable_digest(caplog) -> None:
    raw = f"{SENTINEL} body"
    with caplog.at_level(logging.WARNING, logger="maxim"):
        _builder()._build_followup_prompt(raw)
    message = _warnings(caplog)[0].getMessage()
    assert f"len={len(raw)}" in message
    assert f"{stable_hash_32(raw):08x}" in message


def test_unparseable_input_still_reaches_the_prompt_as_a_private_result() -> None:
    raw = f"{SENTINEL} body"
    prompt = _builder()._build_followup_prompt(raw)
    assert SENTINEL in prompt  # the model still sees it; only the log is shape-only
