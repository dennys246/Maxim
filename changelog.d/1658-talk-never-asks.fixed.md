- **The console's talk, and any model build off the main thread, no longer stop to ask about a download on the
  server's terminal** (#1658, #1677). A talk request against `maxim serve` started from a terminal used to wait 30 s on
  "Proceed? [y/N]" on the server's stdin, where a line typed there for something else would count as a yes. The robot's
  wake-word rebuild had the same problem. Now one rule decides whether a download may be offered: only on the main
  thread, which is the operator's own startup. Anywhere else the build logs how to allow downloads and falls back to a
  smaller model you have. A model the caller named still stops the run. Talk never offers a download, so `maxim serve`
  is grant-only. When talk has no model, its error names the model talk's router could not use and how to fix it
  (for a model that is not downloaded: `maxim config set llm.auto_download true`, or
  `python -m maxim.models.download --llm <model>`), and the server log says so when another model serves talk. Talk now builds like the other run sites, so when no lane resolves
  a model it can get the `llm.json` fallback router, where it used to stop with "configure one first". Two cases on the
  main thread can still ask: the bare-`maxim` menu's live display, and the robot's first build after it wakes.
