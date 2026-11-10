- **A prompt too large for the model no longer takes a healthy self-hosted provider out of rotation** (#1369): a
  context-overflow refusal (the leader proxy's 413, or the 400 from llama-cpp-python, vLLM or the llama.cpp server)
  used to read as a provider failure and put the provider into cooldown, so the next, smaller prompt was refused
  too. It is now its own error, carrying the requested and window token counts. The router logs it as a WARNING,
  with those counts and the router's own n_ctx belief, does not cool the provider down, names it in the run
  report when the overflow was its last outcome, and tries the next provider.
