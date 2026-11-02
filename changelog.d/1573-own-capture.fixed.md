- **A repeated question no longer loses its answer** (#1573): recall skipped every record holding the turn's exact words
  after its five-result limit, so an earlier identical question was never recalled and a fact was lost once its question
  had been stored five times. Only the turn's own capture is excluded now, before the limit, and on enrichment's paths a
  line heard many times takes one place.
