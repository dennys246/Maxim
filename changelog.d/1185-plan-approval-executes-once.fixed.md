- **A "yes" at the plan-approval prompt now runs the plan, exactly once** (#1185): the approved plan was handed back
  to the loop still marked as needing approval, so it was parked and the prompt shown again, and in production
  an approval never executed anything. It now runs once on the same step as an approved PLANNING proposal
  (forbidden tools, the pause and the critical safety constraints still refuse it, typed, as a machine refusal),
  bound to the plan that was shown: a repeated "yes" approves nothing, and "no" or an ambiguous reply never runs it.
