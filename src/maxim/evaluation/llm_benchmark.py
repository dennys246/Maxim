from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from maxim.models.language import LLMRouter


def _normalize(text: str) -> list[str]:
    raw = str(text or "").strip().lower()
    if not raw:
        return []
    cleaned = re.sub(r"[^\w\s]", " ", raw, flags=re.UNICODE)
    tokens = [t for t in cleaned.split() if t and t != "s"]
    aliases = {"maximum": "maxim", "maximums": "maxim", "maxims": "maxim"}
    return [aliases.get(t, t) for t in tokens]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m maxim.evaluation.llm_benchmark")
    p.add_argument("--transcript-dir", default="data/transcript")
    p.add_argument("--limit", type=int, default=50)
    args = p.parse_args(argv)

    router = LLMRouter()
    if not router.enabled():
        print("LLM disabled (set `MAXIM_LLM_ENABLED=1` or enable in `data/util/llm.json`).")
        return 0

    allowed_tools = {"maxim_command", "read_file", "write_file", "execute_file"}
    allowed_commands = {
        "center_vision",
        "mark_trainable_moment",
        "label_outcome",
        "request_sleep",
        "request_observe",
        "request_shutdown",
    }

    results: list[dict] = []
    for path in sorted(Path(args.transcript_dir).glob("*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if len(results) >= int(args.limit):
                    break
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                text = str(rec.get("text", "") or "").strip()
                tokens = _normalize(text)
                if "maxim" not in tokens:
                    continue
                if "sleep" in tokens or "observe" in tokens or "shutdown" in tokens or ("shut" in tokens and "down" in tokens):
                    continue
                t0 = time.perf_counter()
                action = router.route(text, allowed_tools=allowed_tools, allowed_commands=allowed_commands)
                dt = time.perf_counter() - t0
                results.append(
                    {
                        "transcript_path": path.as_posix(),
                        "chunk_index": rec.get("chunk_index"),
                        "text": text,
                        "latency_s": float(dt),
                        "action": action,
                        "valid": bool(isinstance(action, dict) and action.get("tool_name")),
                    }
                )
        except Exception:
            continue

    print(json.dumps({"count": len(results), "results": results}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

