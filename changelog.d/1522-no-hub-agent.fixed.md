- **An agent without a Hippocampus or a NAc no longer reports a corrupt memory hub** (#1522): the hub needs both, so
  an agent made with `remembers=False` or `learns=False` has none, by design. Its shutdown saves its stores itself;
  it gets no session-end decay or consolidation (#1366 tracks whether it should). The false "Failed to create
  MemoryHub" and "Corrupt memory_hub state" warnings are gone, and `on_corrupt="raise"` no longer refuses to build
  such an agent with `MemoryCorruptionError`. It also no longer reads the hub's own files (`ec.json`, `scn.json`):
  a bad `ec.json` beside a learns-only agent reset its NAc on every load.
- The Python API guide's `shutdown()` example now describes what shutdown saves, and in what order (#1521).
