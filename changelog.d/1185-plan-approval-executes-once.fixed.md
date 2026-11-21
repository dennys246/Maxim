- **A "yes" at the plan-approval prompt now runs the plan, exactly once** (#1185): the approved plan was handed back
  to the loop still marked as needing approval, so it was parked and the prompt shown again, and in production
  an approval never executed anything.
  - It now runs once on the same step as an approved PLANNING proposal. A repeated "yes" approves nothing, and
    "no" or an ambiguous reply never runs it.
  - The prompt shows the tool and parameters that will run, with the model's description labelled as such.
  - While a plan waits for its answer, any new plan is refused (`awaiting_human`), so the answer always applies
    to the plan that was shown.
  - The approval stands in for the SUPERVISED confirmation (no second prompt), but the supervision policy's
    `allowed_tools`, confidence and sandbox/CWD rules still apply. Forbidden tools, the pause and the critical
    safety constraints still refuse. Each refusal is typed and booked with no aversion learned.
  - The same rule now applies to an approval drained from the PLANNING proposal queue (an embedder's
    `proposal_queue.approve`, or the non-interactive auto-approval). Before, such an approval ran even when the
    policy's allow-list, confidence threshold or sandbox/CWD rules refused the tool; it is now refused
    `not_permitted`.
  - A simulation's response-policy "yes" now takes the same path, and counts as human-involved in the audit log
    as a confirmation's auto-yes already did.
- **A change request at a confirmation or plan prompt is no longer stranded** (#1185): the answer was stored with
  the original action, but nothing woke the loop or sent it, so it waited for the next unrelated input, and in
  substrate-primary mode nothing ever read it. With a model, it is now sent on the next turn, once, with the
  original action, and no longer also as the person's turn. In substrate-primary mode the person is told the
  mode can't apply a change to the proposed action, and it is cleared.
