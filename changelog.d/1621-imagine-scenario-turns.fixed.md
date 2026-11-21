- **`maxim.imagine(scenario=)` delivers the scenario's lines, or refuses the scenario** (#1621): it read
  `percepts[].text` while scenarios carry `cli_input`, so every scripted turn reached the agent as an empty string,
  silently, and the run still made a report. Scripted turns now have one loader, `experiment.load_campaign_turns`,
  which parses through `scenario_source.load_scenario`, so its expectation and scoring checks still run for
  `imagine` and research mode.
  - `imagine` loads strictly: a scenario whose percepts are not all non-blank `cli_input` lines (for example one
    with a pain or vision percept, such as `scenarios/malware_with_pain.yaml`) raises `ValueError` before any run
    starts. The message names the percept, points at `maxim --sim <scenario>` for percept scenarios, and points at
    #1393, which covers delivering those percepts.
  - Research mode (`maxim.research(campaign=)`, `--research --campaign`) loads its campaign the same strict way.
    A campaign that fails to load or has no line now raises `maxim.exceptions.CampaignUnloadable`, a `ValueError`,
    before any session directory or LLM is built, and the CLI exits 2 with a fix hint instead of a traceback. It no
    longer falls back to a free-form sim and writes a paper about a different experiment. A DM campaign
    (`scenarios/campaigns/*.yaml`) is refused this way too.
  - Research turns are no longer whitespace-stripped, which matches every other path: a block-scalar
    `cli_input: |` line keeps its trailing newline.
  - The lenient loader that `run_campaign` uses skips, and names in one WARNING, every percept that is not a
    non-blank `cli_input` string. A blank line used to arrive as an empty turn, and `cli_input: 42` crashed the
    delivery. Only a null `metadata` reads as `{}`; any other non-mapping `metadata` is skipped and named.
  - In both modes the loader raises `ValueError` for a file whose top level is not a mapping, whose `percepts` is
    not a list, or that is not YAML. `load_scenario` also raises a `ValueError` naming the file, rather than an
    untyped `AttributeError` or `TypeError`, for `expectations` that are not a list of mappings (null reads as
    none), an expectation's `params`, or a `scoring` section that is not a mapping, and a non-numeric
    `estimated_duration_s`. Strict mode also refuses a percept-level `phase:` key (the old `imagine`
    format); the lenient mode warns that the key is ignored.
