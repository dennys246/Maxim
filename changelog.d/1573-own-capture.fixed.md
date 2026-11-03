- **A repeated question no longer loses its answer** (#1573): recall skipped every record holding the turn's exact words,
  after its five-result limit, so an earlier identical question was never recalled and a fact was lost once its question
  had been stored five times. The turn's own records are now excluded by identity, before the limit, on enrichment's paths,
  and an episode stored many times (the same words, tool and outcome) takes one place on enrichment's paths and in
  `memory_recall`. The same line with different outcomes stays two memories. `enrichment_trace` gains `own_ids`.
