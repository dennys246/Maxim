- **Short questions reach the model** (#1689): the planning path no longer answers time, date, identity, greeting,
  "how are you" or simple-maths questions of 8 words or fewer with a canned reply before calling the model. The
  keyword match dropped real requests ("hi, can you look to your left" got "Hello! How can I help you?"), answered
  the wrong question ("what time is it in Tokyo" got the local time), bypassed persona and memory, and the canned
  reply was booked to NAc and the causal links as a `plan:direct_answer` decision. That applied to robot speech as
  well as typed input. The matcher is deleted (`generate_simple_answer`, `matches_phrase`, `normalize_phrases`), so
  a fallback reply never keyword-answers either. A request with no tool prompt (`ANSWER_ONLY`) now gives the model the
  date and time, which the tool-aware prompt already did.
