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
  - A simulation's response-policy "yes" now takes the same path, and counts as human-involved in the audit log
    as a confirmation's auto-yes already did.
  - A change request at the plan prompt reaches the model as the person's turn, now with the original action
    attached.
