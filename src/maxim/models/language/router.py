from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from maxim.utils.logging import warn


_PROFILE_ALIASES: dict[str, str] = {
    "mistral": "mistral-7b-instruct-v0.2",
    "mistral-7b": "mistral-7b-instruct-v0.2",
    "mistral-7b-instruct": "mistral-7b-instruct-v0.2",
    "smollm": "smollm-1.7b-instruct",
    "smollm-1.7b": "smollm-1.7b-instruct",
    "smollm-1.7b-instruct": "smollm-1.7b-instruct",
}

_BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "mistral-7b-instruct-v0.2": {
        "backend": "llama_cpp",
        "model": "mistral-7b-instruct-v0.2",
        "model_path": "data/models/LLM/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "prompt_style": "mistral_instruct",
        "stop": ["</s>"],
        "n_ctx": 4096,
    },
    "smollm-1.7b-instruct": {
        "backend": "llama_cpp",
        "model": "smollm-1.7b-instruct",
        "model_path": "data/models/LLM/smollm-1.7b-instruct.Q4_K_M.gguf",
        "prompt_style": "chatml",
        "stop": ["<|im_end|>", "</s>"],
        "n_ctx": 4096,
    },
}


def _normalize_profile(name: Any) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""
    key = raw.strip().lower().replace("_", "-").replace(" ", "")
    return _PROFILE_ALIASES.get(key, raw.strip())

def normalize_llm_profile(name: Any) -> str:
    return _normalize_profile(name)


def list_llm_profiles() -> list[str]:
    profiles = set(_BUILTIN_PROFILES.keys())

    candidates: list[str] = []
    env_path = str(os.getenv("MAXIM_LLM_CONFIG", "")).strip()
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(os.getcwd(), "data", "util", "llm.json"))
    candidates.append(os.path.join(os.getcwd(), "llm.json"))
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        candidates.append(os.path.join(repo_root, "data", "util", "llm.json"))
        candidates.append(os.path.join(repo_root, "llm.json"))
    except Exception:
        pass

    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        loaded = _read_json(path)
        if not isinstance(loaded, dict):
            continue
        cfg_profiles = loaded.get("profiles")
        if isinstance(cfg_profiles, dict):
            profiles.update(str(k) for k in cfg_profiles.keys() if isinstance(k, str) and k.strip())
        break

    return sorted(profiles)


@dataclass(frozen=True, slots=True)
class LLMConfig:
    enabled: bool = False
    backend: str = "llama_cpp"
    profile: str = "mistral-7b-instruct-v0.2"
    model: str = "mistral-7b-instruct-v0.2"
    model_path: str = "data/models/LLM/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    prompt_style: str = "mistral_instruct"
    stop: tuple[str, ...] = ("</s>",)
    n_ctx: int = 4096
    max_tokens: int = 128
    temperature: float = 0.0


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in ("1", "true", "t", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "f", "no", "n", "off"):
        return False
    return None


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def load_llm_config() -> LLMConfig:
    default = LLMConfig()

    candidates: list[str] = []
    env_path = str(os.getenv("MAXIM_LLM_CONFIG", "")).strip()
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(os.getcwd(), "data", "util", "llm.json"))
    candidates.append(os.path.join(os.getcwd(), "llm.json"))
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        candidates.append(os.path.join(repo_root, "data", "util", "llm.json"))
        candidates.append(os.path.join(repo_root, "llm.json"))
    except Exception:
        pass

    raw: dict[str, Any] = {}
    for path in candidates:
        if path and os.path.isfile(path):
            loaded = _read_json(path)
            if isinstance(loaded, dict):
                raw = loaded
            break

    profile_raw = os.getenv("MAXIM_LLM_PROFILE")
    if profile_raw is None:
        profile_raw = raw.get("profile") or raw.get("model") or default.profile
    profile = _normalize_profile(profile_raw) or default.profile

    profiles_raw = raw.get("profiles")
    profiles = profiles_raw if isinstance(profiles_raw, dict) else {}
    profile_cfg = profiles.get(profile)
    if not isinstance(profile_cfg, dict):
        profile_cfg = profiles.get(str(profile_raw or "").strip())
    if not isinstance(profile_cfg, dict):
        profile_cfg = {}

    builtin = _BUILTIN_PROFILES.get(profile) or {}

    enabled = _as_bool(os.getenv("MAXIM_LLM_ENABLED"))
    if enabled is None:
        enabled = _as_bool(raw.get("enabled"))
    if enabled is None:
        enabled = bool(default.enabled)

    backend = str(
        os.getenv(
            "MAXIM_LLM_BACKEND",
            profile_cfg.get("backend", raw.get("backend", builtin.get("backend", default.backend))),
        )
        or default.backend
    ).strip()

    model = str(
        os.getenv(
            "MAXIM_LLM_MODEL",
            profile_cfg.get("model", raw.get("model", builtin.get("model", default.model))),
        )
        or default.model
    ).strip()
    model_path = str(
        os.getenv(
            "MAXIM_LLM_MODEL_PATH",
            profile_cfg.get("model_path", raw.get("model_path", builtin.get("model_path", default.model_path))),
        )
        or default.model_path
    ).strip()

    def _as_int(env_key: str, raw_key: str, fallback: int) -> int:
        val = os.getenv(env_key)
        if val is None:
            val = profile_cfg.get(raw_key, raw.get(raw_key, builtin.get(raw_key)))
        try:
            return int(val)
        except Exception:
            return int(fallback)

    def _as_float(env_key: str, raw_key: str, fallback: float) -> float:
        val = os.getenv(env_key)
        if val is None:
            val = profile_cfg.get(raw_key, raw.get(raw_key, builtin.get(raw_key)))
        try:
            return float(val)
        except Exception:
            return float(fallback)

    n_ctx = _as_int("MAXIM_LLM_N_CTX", "n_ctx", default.n_ctx)
    max_tokens = _as_int("MAXIM_LLM_MAX_TOKENS", "max_tokens", default.max_tokens)
    temperature = _as_float("MAXIM_LLM_TEMPERATURE", "temperature", default.temperature)

    prompt_style = str(
        os.getenv(
            "MAXIM_LLM_PROMPT_STYLE",
            profile_cfg.get("prompt_style", raw.get("prompt_style", builtin.get("prompt_style", default.prompt_style))),
        )
        or default.prompt_style
    ).strip()

    stop_val = profile_cfg.get("stop", raw.get("stop", builtin.get("stop")))
    stop: tuple[str, ...]
    if isinstance(stop_val, (list, tuple)) and stop_val:
        stop = tuple(str(s) for s in stop_val if isinstance(s, (str, int, float)) and str(s).strip())
    elif isinstance(stop_val, str) and stop_val.strip():
        stop = tuple(s.strip() for s in stop_val.split(",") if s.strip())
    else:
        stop = tuple(default.stop)

    return LLMConfig(
        enabled=bool(enabled),
        backend=backend or default.backend,
        profile=str(profile),
        model=model or default.model,
        model_path=model_path or default.model_path,
        prompt_style=prompt_style or default.prompt_style,
        stop=stop or default.stop,
        n_ctx=int(n_ctx),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
    )


def _mistral_instruct_prompt(system: str, user: str) -> str:
    merged = (str(system or "").strip() + "\n\n" + str(user or "").strip()).strip()
    return f"<s>[INST] {merged} [/INST]"


def _chatml_prompt(system: str, user: str) -> str:
    sys_text = str(system or "").strip()
    user_text = str(user or "").strip()
    return (
        "<|im_start|>system\n"
        + sys_text
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + user_text
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def _build_prompt(cfg: LLMConfig, system: str, user: str) -> str:
    style = str(getattr(cfg, "prompt_style", "") or "").strip().lower().replace("-", "_")
    if style in ("chatml", "im_start"):
        return _chatml_prompt(system, user)
    return _mistral_instruct_prompt(system, user)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    raw = raw.replace("```json", "```").replace("```JSON", "```")
    if "```" in raw:
        parts = raw.split("```")
        raw = "".join(parts[1:-1]).strip() if len(parts) >= 3 else raw.replace("```", "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


class _LlamaCppBackend:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._llm = None

    def _ensure(self) -> bool:
        if self._llm is not None:
            return True
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            warn("LLM backend unavailable (install `llama-cpp-python`): %s", e)
            return False

        model_path = str(self.cfg.model_path or "").strip()
        if not model_path or not os.path.exists(model_path):
            warn("LLM model_path not found: %s", model_path)
            return False

        try:
            self._llm = Llama(
                model_path=model_path,
                n_ctx=int(self.cfg.n_ctx),
                verbose=False,
            )
            return True
        except Exception as e:
            warn("Failed to load LLM model (%s): %s", model_path, e)
            self._llm = None
            return False

    def complete(self, prompt: str, *, max_tokens: int, temperature: float, stop: tuple[str, ...]) -> str:
        if not self._ensure():
            return ""
        out = self._llm(
            str(prompt),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            stop=list(stop) if stop else None,
        )
        try:
            return str(out["choices"][0]["text"])
        except Exception:
            return ""


class LLMRouter:
    """
    Small, swappable transcript → action router.

    Returns a single Maxim action dict in the canonical schema:
    {"tool_name": str, "params": dict}
    """

    def __init__(self, cfg: LLMConfig | None = None) -> None:
        self.cfg = cfg or load_llm_config()
        self._backend = None

    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "enabled", False))

    def _get_backend(self) -> Any | None:
        if self._backend is not None:
            return self._backend
        if not self.enabled():
            self._backend = None
            return None

        backend = str(getattr(self.cfg, "backend", "") or "").strip().lower().replace("-", "_")
        if backend in ("llama", "llama_cpp", "llamacpp"):
            self._backend = _LlamaCppBackend(self.cfg)
            return self._backend

        warn("Unknown LLM backend: %s", backend)
        self._backend = None
        return None

    def route(
        self,
        transcript_text: str,
        *,
        allowed_tools: set[str],
        allowed_commands: set[str],
    ) -> dict[str, Any] | None:
        backend = self._get_backend()
        if backend is None:
            return None

        tools = ", ".join(sorted(allowed_tools))
        commands = ", ".join(sorted(allowed_commands))

        system = (
            "You are Maxim, a local robot assistant. "
            "Return ONLY a single JSON object (no prose) describing the next action."
        )
        user = f"""
Transcript:
{transcript_text}

Allowed tools: {tools}
If tool_name == "maxim_command", command must be one of: {commands}

Return JSON exactly like:
{{"tool_name":"...","params":{{...}}}}
""".strip()
        prompt = _build_prompt(self.cfg, system, user)

        try:
            text = backend.complete(
                prompt,
                max_tokens=int(self.cfg.max_tokens),
                temperature=float(self.cfg.temperature),
                stop=tuple(getattr(self.cfg, "stop", ("</s>",))),
            )
        except Exception as e:
            warn("LLM route failed: %s", e)
            return None

        obj = _extract_json_object(text)
        if not isinstance(obj, dict):
            return None

        tool_name = obj.get("tool_name")
        params = obj.get("params") if isinstance(obj.get("params"), dict) else {}
        if not isinstance(tool_name, str) or not tool_name or tool_name not in allowed_tools:
            return None

        if tool_name == "maxim_command":
            cmd = params.get("command")
            if not isinstance(cmd, str) or not cmd or cmd not in allowed_commands:
                return None

        return {"tool_name": tool_name, "params": dict(params)}
