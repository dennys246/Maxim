- **A "yes" at the plan-approval prompt now runs the plan, exactly once** (#1185): the approved plan was handed back
  to the loop still marked as needing approval, so it was parked and the prompt shown again, and in production
  an approval never executed anything.
  - It now runs once on the same step as an approved PLANNING proposal. A repeated "yes" approves nothing, and
    "no" or an ambiguous reply never runs it.
  - The prompt shows the tool and parameters that will run, with the model's description labelled as such.
  - While a plan waits for its answer, any new plan is refused (`awaiting_human`), so the answer always applies
    to the plan that was shown.
  - One rule for every approval (a plan approved at the prompt, an entry drained from the PLANNING queue, and a
    "yes" at a SUPERVISED confirmation): a person's approval satisfies everything it is asked for (outside the
    allow-list, a write outside the workspace, a sandbox execute), as a confirmation "yes" always did. The pause,
    forbidden tools and the confidence floor still refuse it. A confirmation "yes" is now checked too, so a pause
    or a forbid that lands while it waits is honoured. Each refusal is typed, shown to the person, and booked with
    no aversion learned; none is worded "requires approval".
  - **Headless PLANNING runs now act, within the allow-list.** A headless (`--interactive false`) PLANNING run used
    to execute nothing, because its auto-approve was dead code. It now executes allow-listed tools and refuses
    anything that needs a person ("it needs a person's approval, and this run is non-interactive"). The loop's own
    "yes" is a machine approval: a plan is decided at once instead of being parked for nobody (each later plan used
    to be refused while the first waited). At SUPERVISED, the non-interactive auto-yes to a confirmation is
    refused the same way instead of running the action.
  - At SUPERVISED the confidence floor is now checked before the allow-list, so a low-confidence action outside
    the allow-list is a hard rejection, not a confirmation prompt.
  - The default supervision policy's confidence floor is 0.7 (was 0.8), the router's default proposal confidence,
    so the default policy no longer refuses the default proposal.
  - A simulation's response-policy "yes" now takes the same path, and counts as human-involved in the audit log
    as a confirmation's auto-yes already did.
- **A change request at a confirmation or plan prompt is no longer stranded** (#1185): the answer was stored with
  the original action, but nothing woke the loop or sent it, so it waited for the next unrelated input, and in
  substrate-primary mode nothing ever read it. With a model, it is now sent on the next turn, once, with the
  original action, and no longer also as the person's turn. In substrate-primary mode the person is told the
  mode can't apply a change to the proposed action, and it is cleared.
