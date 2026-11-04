- **Offline call sites refuse first** (#1315 item 10b): under `llm.offline`, each library path that would fetch or
  spawn refuses early with a typed `OfflineRefusal` instead of failing deep in a library: docker (an explicit
  `--sandbox docker` refuses; `--sandbox auto` runs in a tmpdir sandbox), ddgs web search (a native client the latch
  cannot see), ultralytics weight downloads and `YOLO(name)` (only an existing weight file loads; a missing pose
  weight only turns pose off), twilio SMS and calls (before anything is recorded as sent), model and TTS-voice
  downloads (before any consent prompt), RTM vision weights, and a faster-whisper model that is not cached. Hugging
  Face loads (sentence-transformers, faster-whisper) pass `local_files_only` where the library takes it, and
  transformers loads local-only and never runs remote code offline.
- **A missing model offline** (owner decision V25): a profile you chose (`--llm`, `llm.profile`) is refused by name
  and never swapped; an auto-detected tier falls back to a model already on disk, and the run records that as
  degraded.
- **An offline run says what it could not do** (owner decision V27): the sim's `report.json` provenance gains an
  `offline` block (the refusals, the exact count, and each degraded fallback), and `lane_decisions.jsonl` records
  `offline` and `offline_degraded`. A missing encoder or semantic model still falls back to hashing, now said once.
- **Refusals stay refusals**: a robot run or an NPC line stopped by an offline refusal reports it as a refusal (exit
  2), not a crash. A benchmark records a refused model (offline, or a leader refusal) as refused, not failed: it is
  not scored or ranked, and the other models still run (owner decision V26). `maxim doctor` keeps a check's own rows
  and adds a "refused (offline)" row.
