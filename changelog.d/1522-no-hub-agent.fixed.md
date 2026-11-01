- **An agent made with `remembers=False` no longer reports a corrupt memory hub** (#1522): with no Hippocampus the
  factory builds no MemoryHub, instead of attempting one that could only fail. The false "Failed to create
  MemoryHub" and "Corrupt memory_hub state" warnings are gone, and `on_corrupt="raise"` no longer refuses to build
  such an agent with `MemoryCorruptionError`.
- `maxim.run()`'s shutdown comment and the Python API guide's `shutdown()` example now describe what shutdown saves
  since the session-ownership change (#1521).
