"""Who supplied a tick's input, and so who answered a gating prompt (#1185, owner decisions 2026-11-21).

One enum for the answer's SOURCE, read where an approval is judged: a reason that needs a human's approval
(``ActionVerdict.satisfied_by``) is satisfied only by an approver in its set. It lives in ``utils`` so every layer
that labels or reads input (``runtime``, ``agents``, ``memory``) may import it (``utils/audit.py::LAYER_RULES``).

It is the approver axis, not the channel axis: the loop's ``pending_user_input_source`` label ("CLI" / "voice")
says how the words arrived, and stays as it is.
"""

from __future__ import annotations

from enum import Enum


class InputSource(str, Enum):
    """Who an input came from."""

    OPERATOR = "operator"  # a person: the interactive CLI / API / robot mailbox
    VOICE = "voice"  # a voice turn; reserved for the voice-turn gate (#1519, #1729), no producer or reader yet
    SIM = "sim"  # a simulation: its narrator's words, or its response policy's answer (human-involved, decision B)
    LOOP = "loop"  # the loop answering itself: the non-interactive auto-yes, an adapter's default "yes"
    SELF = "self"  # the agent's own tool result (an ``[ACTION_FOLLOWUP`` line)
    UNKNOWN = "unknown"


# Who may satisfy a reason that needs a human's approval (a confirmable autonomy reason). The loop's own reply
# never does: a machine auto-approve lifts only the PLANNING gate. Listed member by member, never derived from the
# enum, so adding a source can never lift a gate by accident. VOICE is NOT in it: a room microphone's "yes" must not
# confirm a write, so a VOICE yes leaves the action parked (the gate note, S7); #1729 decides the set when it makes
# voice its own source.
HUMAN_APPROVERS: frozenset[InputSource] = frozenset({InputSource.OPERATOR, InputSource.SIM})
# Who may lift the PLANNING level gate alone: any approval, a machine's included (the loop's own reply). Not VOICE
# (above), SELF (the agent's own tool result) or UNKNOWN.
PLANNING_GATE_APPROVERS: frozenset[InputSource] = HUMAN_APPROVERS | frozenset({InputSource.LOOP})

# How an approver that cannot satisfy a reason is named to the person (``needs_person_detail``).
_WHO = {
    InputSource.LOOP: "this run answered its own prompt, and no person did",
    InputSource.SELF: "the agent's own output answered the prompt, and no person did",
    InputSource.UNKNOWN: "the prompt's answer came from no known person",
    InputSource.OPERATOR: "the operator's answer cannot satisfy it",
    InputSource.SIM: "the simulation's answer cannot satisfy it",
    InputSource.VOICE: "a voice answer cannot approve it, so it stays parked until a person confirms",
}


def needs_person_detail(cause: str, approver: InputSource) -> str:
    """Why an approved action needs a person, from its cause and who approved it (#1185): true for the loop's own
    non-interactive answer, an interactive run nobody answered, and a future voice-only reason alike."""
    return f"{cause} needs a person's approval; {_WHO.get(approver, 'no person approved it')}"
