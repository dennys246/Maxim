- **Recall matches the regular inflections of a word** (#1620): the memory content keys now bring "move", "moved",
  "moving" and "moves" to one key (and "stop"/"stopped", "store"/"stored", "place"/"placing", "close"/"closed"…), so
  a cue in one form reaches a memory stored in another. The frozen tokenizer (`normalize_tokens`, which ATL concept
  names are persisted from) is unchanged; the keys converge on top of it, so a persisted concept name still matches
  its words. Words that differ only by a final "e" now share a key ("plan" and "plane", "made" and "mad", "care" and
  "car"; "done" and the "don" of "don't"). Only words converge: numbers and symbols keep their keys. Still apart: "used" against "use", "agreed" against "agree", "accessing" against "access", and irregular
  forms ("hid", "slept"). Exp 63b's
  judge keeps the keying that run executed, so a new 63b-style run on this code is refused by its instrument checks;
  Exp 63c has its own port.
