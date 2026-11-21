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

    OPERATOR = "operator"  # a person: the interactive CLI / API / robot mailbox, a wake-word voice turn
    SIM = "sim"  # a simulation: its narrator's words, or its response policy's answer (human-involved, decision B)
    LOOP = "loop"  # the loop answering itself: the non-interactive auto-yes, an adapter's default "yes"
    SELF = "self"  # the agent's own tool result (an ``[ACTION_FOLLOWUP`` line)
    UNKNOWN = "unknown"


# Who may satisfy a reason that needs a human's approval (a confirmable autonomy reason). The loop's own reply
# never does: a machine auto-approve lifts only the PLANNING gate.
HUMAN_APPROVERS: frozenset[InputSource] = frozenset({InputSource.OPERATOR, InputSource.SIM})
