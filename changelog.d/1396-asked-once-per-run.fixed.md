- A "no" (or a timeout) at the model-download prompt is remembered for the rest of that `maxim` run, so
  `--sim interactive` and `--auto-curate` no longer ask the same question twice (#1396). Each bare-`maxim` menu
  choice is its own run; the library, the console and every subcommand still ask every time.
