- **A fact a person told the agent is no longer deleted after a few days** (#1710, #1137): sleep consolidation
  removed an aged record holding a person's words outright, instead of compressing it, unless that record had been
  promoted to long-term when it formed. Promotion depended on a race: the agent's novelty measure compared vision
  tracks, so any idle tick of the talk loop before the words arrived made them read as already seen, and they were
  not promoted. Sleep now compresses such a record before it can be removed (the compressed record keeps the words and
  is removed later only if its own score falls), and a text percept's novelty now compares its words with recent
  utterances. A repeated sentence still reads as familiar.
