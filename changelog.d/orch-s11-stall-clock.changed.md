- **The simulation stall detector is its own module, with an injectable clock** (orchestrator §11 slice; for #1259
  and #1488): `simulation/stall_detector.py::StallDetector` replaces the closure nested in `start_simulation_mode`,
  behaviour unchanged (the orchestrator characterization golden is byte-identical). Every "now" it reads, its idle
  timers and the ages of in-flight LLM calls, goes through `StallDetector.CLOCK`; `stall_threshold.stall_suppression`
  and the call registry's readers take an optional `now` for that. Its suppression failure now reports under the site
  `stall_detector.py:StallDetector._suppressed:stall_suppression`.
