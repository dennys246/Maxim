- **A fallback reply no longer teaches NAc** (#1691): outside a sim, when the model's reply is missing or unreadable,
  times out, or no model is available, the stock reply is still spoken, but it no longer books a causal link, goal
  credit, cluster reward, energy cost or plan outcome under the synthetic goal `llm_fallback`. The outcome window and
  the model's carryover still record the turn.
