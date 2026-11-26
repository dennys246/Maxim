- **Saved id indexes no longer follow `PYTHONHASHSEED`** (#1789). The Hippocampus `context_index`, NAc's
  `outcome_index` and the EC's inverted indices are sets of ids, and their saved lists used to come out in a
  per-process order, so saving the same store in two processes wrote two different files. They are now written
  sorted, and a whole Hippocampus, NAc or EC save is byte-identical across processes. `SimilarityIndex`'s bands (no
  production caller yet) and the SCN's `priors` (small ints, already seed-independent) are sorted too. Loading is
  unchanged, since every loader rebuilds the set. A store saved earlier is rewritten in sorted order on its next
  save.
- **`atomic_write_json` writes a set as a sorted list** (#1789). It used to write the set's `str` (`"{'b', 'a'}"`),
  which a loader read back as a set of characters. A writer that passes a set unconverted now gets a stable,
  loadable list.
