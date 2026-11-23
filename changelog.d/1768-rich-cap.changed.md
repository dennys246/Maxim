- **pymaxim allows rich 14 and 15** (#1768): the core dependency is `rich>=13.0.0,<16.0`, up from `<14.0`, so an
  embedder can use tools that need a newer rich (Overdrive's import-linter 2.7+ needs 14.2). Maxim's interactive and
  display tests pass on 13.9.4, 14.3.4 and 15.0.0.
