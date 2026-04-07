from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from maxim.environment.base import Environment


def _tail_line(path: Path) -> str | None:
    try:
        with path.open("rb") as fp:
            fp.seek(0, os.SEEK_END)
            pos = fp.tell()
            if pos <= 0:
                return None
            block = 4096
            data = b""
            while pos > 0 and b"\n" not in data:
                read_size = block if pos >= block else pos
                pos -= read_size
                fp.seek(pos)
                data = fp.read(read_size) + data
            line = data.splitlines()[-1] if data else b""
            return line.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _latest_file(paths: list[Path]) -> str | None:
    best = None
    best_ts = None
    for path in paths:
        if not path.is_file():
            continue
        try:
            ts = path.stat().st_mtime
        except Exception:
            continue
        if best is None or (best_ts is not None and ts > best_ts) or best_ts is None:
            best = path
            best_ts = ts
    return best.as_posix() if best is not None else None


class ReachyEnv(Environment):
    """
    Observation-only environment over Reachy Mini artifacts stored under `data/`.
    """

    def __init__(self, *, repo_root: str | os.PathLike[str] | None = None, data_dir: str | os.PathLike[str] = "data"):
        self.repo_root = Path(repo_root or os.getcwd()).resolve()
        data_path = Path(data_dir)
        if not data_path.is_absolute():
            data_path = self.repo_root / data_path
        self.data_dir = data_path.resolve()
        self._done = False
        self._last_cli_line: str | None = None
        self._last_cli_mtime: float | None = None

    def reset(self) -> dict[str, Any]:
        self._done = False
        return self.observe()

    def observe(self) -> dict[str, Any]:
        base = self.data_dir
        videos = sorted((base / "videos").glob("*.mp4")) if (base / "videos").exists() else []
        audio = sorted((base / "audio").glob("*.wav")) if (base / "audio").exists() else []
        transcripts = sorted((base / "transcript").glob("*.jsonl")) if (base / "transcript").exists() else []
        logs = sorted((base / "logs").glob("*.log")) if (base / "logs").exists() else []
        training = sorted((base / "training").glob("*.jsonl")) if (base / "training").exists() else []
        cli_logs = sorted((base / "cli").glob("*.jsonl")) if (base / "cli").exists() else []
        vision_events = sorted((base / "vision").glob("*.jsonl")) if (base / "vision").exists() else []

        latest_transcript = _latest_file(transcripts)
        latest_log = _latest_file(logs)
        latest_training = _latest_file(training)
        latest_video = _latest_file(videos)
        latest_audio = _latest_file(audio)
        latest_cli_log = _latest_file(cli_logs)
        latest_vision_event_path = _latest_file(vision_events)
        latest_transcript_record = None
        latest_transcript_text = None
        latest_transcript_chunk_index = None
        latest_vision_event = None

        if latest_transcript:
            try:
                line = _tail_line(Path(latest_transcript))
                if line:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        latest_transcript_record = rec
                        latest_transcript_text = rec.get("text")
                        latest_transcript_chunk_index = rec.get("chunk_index")
            except Exception:
                latest_transcript_record = None

        if latest_vision_event_path:
            try:
                line = _tail_line(Path(latest_vision_event_path))
                if line:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        latest_vision_event = rec
            except Exception:
                latest_vision_event = None

        # Check for new CLI input
        cli_input = None
        if latest_cli_log:
            try:
                cli_path = Path(latest_cli_log)
                cli_mtime = cli_path.stat().st_mtime
                # Only read if file is new or modified
                if self._last_cli_mtime is None or cli_mtime > self._last_cli_mtime:
                    line = _tail_line(cli_path)
                    if line and line != self._last_cli_line:
                        try:
                            rec = json.loads(line)
                            if isinstance(rec, dict):
                                cli_input = rec.get("input") or rec.get("text") or rec.get("command")
                        except json.JSONDecodeError:
                            cli_input = line  # Plain text input
                        self._last_cli_line = line
                        self._last_cli_mtime = cli_mtime
            except Exception:
                pass

        def rel(path: str | None) -> str | None:
            if not path:
                return None
            try:
                return Path(path).resolve().relative_to(self.repo_root).as_posix()
            except Exception:
                return path

        return {
            "repo_root": self.repo_root.as_posix(),
            "data_dir": self.data_dir.as_posix(),
            "latest_transcript": rel(latest_transcript),
            "latest_transcript_chunk_index": latest_transcript_chunk_index,
            "latest_transcript_text": latest_transcript_text,
            "latest_transcript_record": latest_transcript_record,
            "latest_vision_event": latest_vision_event,
            "latest_vision_event_path": rel(latest_vision_event_path),
            "latest_log": rel(latest_log),
            "latest_training": rel(latest_training),
            "latest_video": rel(latest_video),
            "latest_audio": rel(latest_audio),
            "cli_input": cli_input,  # New CLI input if any
            "counts": {
                "videos": len(videos),
                "audio": len(audio),
                "transcripts": len(transcripts),
                "logs": len(logs),
                "training": len(training),
                "cli_logs": len(cli_logs),
                "vision_events": len(vision_events),
            },
        }

    def step(self, event: Any) -> dict[str, Any]:
        return self.observe()

    def is_done(self) -> bool:
        return bool(self._done)
