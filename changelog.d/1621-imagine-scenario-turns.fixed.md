- **`maxim.imagine(scenario=)` delivers the scenario's lines** (#1621): it read `percepts[].text` while scenarios
  carry `cli_input`, so every scripted turn reached the agent as an empty string, silently, and the run still made a
  report. `imagine` and research mode's campaign now read turns through `experiment.load_campaign_turns`, the one
  scripted-turn loader; `imagine` refuses a scenario with no `cli_input` line (`ValueError`) before any run starts,
  and the loader logs one WARNING naming each percept it does not deliver. Research turns are no longer
  whitespace-stripped, matching every other path.
