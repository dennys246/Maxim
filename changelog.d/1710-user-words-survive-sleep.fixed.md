- **Sleep no longer deletes a record of what a person said without compressing it first** (#1710, #1137): an
  aged record holding a person's words that scored below the retention threshold was removed outright, unless it had
  been promoted to long-term when it formed, and that promotion depended on a race (the talk loop ticking before the
  words arrived). Sleep now compresses such a record instead, keeping the words; the compressed record follows the
  ordinary retention rule, so it stays while recall uses it and is removed at a later sleep when nothing does.
