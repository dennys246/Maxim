- **Plan approval no longer reads a refusal as a "yes"** (#1185): `runtime/approval.py::detect_approval_intent`
  approved "stop, don't run it", "not ok", "that is not fine" and "no good" on live plan-approval input, read
  "No." as unknown, and flipped mixed replies with `PYTHONHASHSEED`. It now approves only when the whole reply
  (punctuation stripped) is an approval phrase; a negation anywhere means it never approves, and a contradictory
  reply comes out unknown, so the prompt asks again.
