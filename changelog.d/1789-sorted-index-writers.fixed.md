- **Saved stores are byte-stable across processes** (#1789). The Hippocampus `context_index`, NAc's `outcome_index`,
  the EC's inverted indices and `SimilarityIndex`'s bands are sets of ids, and their lists used to follow
  `PYTHONHASHSEED`, so saving the same store in two processes wrote two different files. Each is now written sorted,
  as is the SCN's `priors`. Loading is unchanged: every loader rebuilds the set. A store saved before this change is
  rewritten in sorted order the next time it is saved.
