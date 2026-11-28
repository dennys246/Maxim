- **`maxim.time.agent_time`: the machinery clock and the perceived-time port** (world clock A-pre PR 1a). An agent's
  machinery seconds (`MachineryClock`) and the calendar it may perceive (`PerceivedTime`) become two collaborators an
  embedder can supply. The host defaults (`HOST_MACHINERY_CLOCK`, `HOST_PERCEIVED_TIME`) read today's wall clock and
  local calendar at call time; `NO_CALENDAR` declares that there is no calendar, so a render renders nothing. Nothing
  reads them yet: later A-pre PRs thread them through the prompt, the context pool, the SCN-derived renders and the
  Hippocampus window, each keeping the host default. This is capability until OD-A10's embedded entry point passes a
  non-default.
