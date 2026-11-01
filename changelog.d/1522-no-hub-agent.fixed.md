- **An agent without a Hippocampus or a NAc no longer reports a corrupt memory hub** (#1522): the hub needs both, so
  an agent made with `remembers=False` or `learns=False` has none, by design. Its shutdown saves its stores itself;
  it gets no session-end decay or consolidation (#1366 tracks whether it should). The false "Failed to create
  MemoryHub" and "Corrupt memory_hub state" warnings are gone, and `on_corrupt="raise"` no longer refuses to build
  such an agent with `MemoryCorruptionError`. An unreadable `scn.json` or `ec.json` is still reported.
- The Python API guide's `shutdown()` example now describes what shutdown saves, and in what order (#1521).
