- **Offline call sites refuse first** (#1315 item 10b): under `llm.offline`, each library path that would fetch or
  spawn refuses early with a typed `OfflineRefusal` instead of failing deep in a library: docker
  (`container_runner`), ddgs web search (a native client the latch cannot see), ultralytics weight downloads and
  `YOLO(name)` (only an existing weight file loads), twilio SMS and calls (before anything is recorded as sent), model
  and TTS-voice downloads (before any consent prompt), RTM vision weights and faster-whisper models that are not
  cached. Hugging Face loads (sentence-transformers, transformers, faster-whisper) pass `local_files_only`, and
  transformers never runs remote code offline. A missing encoder or semantic model falls back as before, but the run
  now says "degraded" once. A robot run, an NPC line or a benchmark stopped by an offline refusal reports it as a
  refusal (exit 2), not as a crash or a failed scenario. CI now refuses an HTTP client (`urllib3`, `requests`,
  `httpx.Client`) or a raw `socket.socket(` outside `utils/http.py`, with the four non-HTTP socket sites named.
