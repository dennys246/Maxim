- **A fallback reply no longer teaches NAc** (#1691): when the model's reply is missing or unreadable, times out, or
  no model is available, the stock reply is still spoken, but it no longer books a causal link, goal credit, cluster
  reward, energy cost or plan outcome under the synthetic goal `llm_fallback` / `llm_timeout`. The outcome window, the
  model's carryover and the episode record still keep the turn. A new `LLMProposal.degraded` flag, set only by the
  fallback builders, marks these replies; a model writing `"strategy": "fallback"` cannot claim it.
  - This also changes sims: a parse-failure fallback was already dropped there, but a timeout's "try again with more
    time?" reply still runs in a sim, and it no longer books a positive `respond` link.
