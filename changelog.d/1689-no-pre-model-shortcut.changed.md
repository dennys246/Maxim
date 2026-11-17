- **Short questions reach the model** (#1689): the planning path no longer answers time, date, identity, greeting,
  "how are you" or simple-maths questions of 8 words or fewer with a canned reply before calling the model. The
  keyword match dropped real requests ("hi, can you look to your left" got "Hello! How can I help you?"), answered
  the wrong question ("what time is it in Tokyo" got the local time), bypassed persona and memory, and the canned
  reply was credited to NAc and the causal links as if it had been decided. That applied to robot speech as well as
  typed input. The model's prompt already carries the date and time. The canned answers now run only when the model
  is unavailable (`llm_fallback.generate_llm_fallback`), as a labelled fallback, and the worker no longer accepts a
  JSON action disguised as a prompt (`'{"action":...'`) or swallows its parse failure.
