from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Any

from maxim.utils.logging import warn


# Quantization levels ordered by quality (higher = better quality, larger size)
QUANTIZATION_LEVELS: dict[str, dict[str, Any]] = {
    "Q2_K": {"bits": 2, "description": "Smallest, lowest quality", "suffix": "Q2_K"},
    "Q3_K_S": {"bits": 3, "description": "Very small, low quality", "suffix": "Q3_K_S"},
    "Q3_K_M": {"bits": 3, "description": "Small, low quality", "suffix": "Q3_K_M"},
    "Q3_K_L": {"bits": 3, "description": "Small, better quality", "suffix": "Q3_K_L"},
    "Q4_0": {"bits": 4, "description": "Medium, legacy format", "suffix": "Q4_0"},
    "Q4_K_S": {"bits": 4, "description": "Medium, good balance", "suffix": "Q4_K_S"},
    "Q4_K_M": {"bits": 4, "description": "Medium, recommended default", "suffix": "Q4_K_M"},
    "Q5_0": {"bits": 5, "description": "Large, legacy format", "suffix": "Q5_0"},
    "Q5_K_S": {"bits": 5, "description": "Large, high quality", "suffix": "Q5_K_S"},
    "Q5_K_M": {"bits": 5, "description": "Large, higher quality", "suffix": "Q5_K_M"},
    "Q6_K": {"bits": 6, "description": "Very large, very high quality", "suffix": "Q6_K"},
    "Q8_0": {"bits": 8, "description": "Largest, near-original quality", "suffix": "Q8_0"},
    "F16": {"bits": 16, "description": "Full precision float16", "suffix": "F16"},
    "F32": {"bits": 32, "description": "Full precision float32", "suffix": "F32"},
}

DEFAULT_QUANTIZATION = "Q4_K_M"


def list_quantization_levels() -> list[str]:
    """Return available quantization levels ordered by size (smallest first)."""
    return sorted(QUANTIZATION_LEVELS.keys(), key=lambda k: (QUANTIZATION_LEVELS[k]["bits"], k))


def get_quantization_info(level: str) -> dict[str, Any] | None:
    """Get info about a quantization level."""
    normalized = str(level or "").strip().upper().replace("-", "_")
    return QUANTIZATION_LEVELS.get(normalized)


_PROFILE_ALIASES: dict[str, str] = {
    "mistral": "mistral-7b-instruct-v0.2",
    "mistral-7b": "mistral-7b-instruct-v0.2",
    "mistral-7b-instruct": "mistral-7b-instruct-v0.2",
    "smollm": "smollm-1.7b-instruct",
    "smollm-1.7b": "smollm-1.7b-instruct",
    "smollm-1.7b-instruct": "smollm-1.7b-instruct",
    # Llama models
    "llama2": "llama-2-7b-chat",
    "llama2-7b": "llama-2-7b-chat",
    "llama2-13b": "llama-2-13b-chat",
    "llama3": "llama-3-8b-instruct",
    "llama3-8b": "llama-3-8b-instruct",
    # Phi models
    "phi2": "phi-2",
    "phi3": "phi-3-mini-4k-instruct",
    "phi3-mini": "phi-3-mini-4k-instruct",
    # Qwen models
    "qwen": "qwen2-7b-instruct",
    "qwen2": "qwen2-7b-instruct",
    "qwen2-7b": "qwen2-7b-instruct",
    # Gemma models
    "gemma": "gemma-2b-it",
    "gemma-2b": "gemma-2b-it",
    "gemma-7b": "gemma-7b-it",
}

_BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "mistral-7b-instruct-v0.2": {
        "backend": "llama_cpp",
        "model": "mistral-7b-instruct-v0.2",
        "model_base": "mistral-7b-instruct-v0.2",
        "prompt_style": "mistral_instruct",
        "stop": ["</s>"],
        "n_ctx": 4096,
    },
    "smollm-1.7b-instruct": {
        "backend": "llama_cpp",
        "model": "smollm-1.7b-instruct",
        "model_base": "smollm-1.7b-instruct",
        "prompt_style": "chatml",
        "stop": ["<|im_end|>", "</s>"],
        "n_ctx": 4096,
    },
    "llama-2-7b-chat": {
        "backend": "llama_cpp",
        "model": "llama-2-7b-chat",
        "model_base": "llama-2-7b-chat",
        "prompt_style": "llama2_chat",
        "stop": ["</s>"],
        "n_ctx": 4096,
    },
    "llama-2-13b-chat": {
        "backend": "llama_cpp",
        "model": "llama-2-13b-chat",
        "model_base": "llama-2-13b-chat",
        "prompt_style": "llama2_chat",
        "stop": ["</s>"],
        "n_ctx": 4096,
    },
    "llama-3-8b-instruct": {
        "backend": "llama_cpp",
        "model": "llama-3-8b-instruct",
        "model_base": "Meta-Llama-3-8B-Instruct",
        "prompt_style": "llama3_instruct",
        "stop": ["<|eot_id|>"],
        "n_ctx": 8192,
    },
    "phi-2": {
        "backend": "llama_cpp",
        "model": "phi-2",
        "model_base": "phi-2",
        "prompt_style": "phi",
        "stop": ["<|endoftext|>"],
        "n_ctx": 2048,
    },
    "phi-3-mini-4k-instruct": {
        "backend": "llama_cpp",
        "model": "phi-3-mini-4k-instruct",
        "model_base": "Phi-3-mini-4k-instruct",
        "prompt_style": "phi3",
        "stop": ["<|end|>", "<|endoftext|>"],
        "n_ctx": 4096,
    },
    "qwen2-7b-instruct": {
        "backend": "llama_cpp",
        "model": "qwen2-7b-instruct",
        "model_base": "Qwen2-7B-Instruct",
        "prompt_style": "chatml",
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "n_ctx": 8192,
    },
    "gemma-2b-it": {
        "backend": "llama_cpp",
        "model": "gemma-2b-it",
        "model_base": "gemma-2b-it",
        "prompt_style": "gemma",
        "stop": ["<end_of_turn>"],
        "n_ctx": 8192,
    },
    "gemma-7b-it": {
        "backend": "llama_cpp",
        "model": "gemma-7b-it",
        "model_base": "gemma-7b-it",
        "prompt_style": "gemma",
        "stop": ["<end_of_turn>"],
        "n_ctx": 8192,
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


def build_model_path(
    model_base: str,
    quantization: str = DEFAULT_QUANTIZATION,
    models_dir: str = "data/models/LLM",
) -> str:
    """Build the model path from base name and quantization level."""
    quant = str(quantization or DEFAULT_QUANTIZATION).strip().upper().replace("-", "_")
    if quant not in QUANTIZATION_LEVELS:
        quant = DEFAULT_QUANTIZATION
    base = str(model_base or "").strip()
    return os.path.join(models_dir, f"{base}.{quant}.gguf")


@dataclass(frozen=True, slots=True)
class LLMConfig:
    enabled: bool = False
    backend: str = "llama_cpp"
    profile: str = "mistral-7b-instruct-v0.2"
    model: str = "mistral-7b-instruct-v0.2"
    model_base: str = "mistral-7b-instruct-v0.2"
    model_path: str = "data/models/LLM/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    quantization: str = "Q4_K_M"
    prompt_style: str = "mistral_instruct"
    stop: tuple[str, ...] = ("</s>",)
    n_ctx: int = 4096
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    n_gpu_layers: int = -1  # -1 = auto (use all available GPU layers)
    n_threads: int | None = None  # None = auto-detect
    seed: int = -1  # -1 = random


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
    # Get quantization level (default Q4_K_M)
    quantization = str(
        os.getenv(
            "MAXIM_LLM_QUANTIZATION",
            profile_cfg.get("quantization", raw.get("quantization", builtin.get("quantization", default.quantization))),
        )
        or default.quantization
    ).strip().upper().replace("-", "_")
    if quantization not in QUANTIZATION_LEVELS:
        quantization = DEFAULT_QUANTIZATION

    # Get model_base for path construction
    model_base = str(
        os.getenv(
            "MAXIM_LLM_MODEL_BASE",
            profile_cfg.get("model_base", raw.get("model_base", builtin.get("model_base", default.model_base))),
        )
        or default.model_base
    ).strip()

    # Get model_path - if not explicitly set, build from model_base + quantization
    explicit_path = os.getenv("MAXIM_LLM_MODEL_PATH")
    if explicit_path is None:
        explicit_path = profile_cfg.get("model_path", raw.get("model_path", builtin.get("model_path")))

    if explicit_path:
        model_path = str(explicit_path).strip()
    else:
        # Build path from model_base and quantization
        model_path = build_model_path(model_base, quantization)

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
    top_p = _as_float("MAXIM_LLM_TOP_P", "top_p", default.top_p)
    top_k = _as_int("MAXIM_LLM_TOP_K", "top_k", default.top_k)
    repeat_penalty = _as_float("MAXIM_LLM_REPEAT_PENALTY", "repeat_penalty", default.repeat_penalty)
    n_gpu_layers = _as_int("MAXIM_LLM_N_GPU_LAYERS", "n_gpu_layers", default.n_gpu_layers)
    seed = _as_int("MAXIM_LLM_SEED", "seed", default.seed)

    # n_threads: None means auto-detect
    n_threads_val = os.getenv("MAXIM_LLM_N_THREADS")
    if n_threads_val is None:
        n_threads_val = profile_cfg.get("n_threads", raw.get("n_threads", builtin.get("n_threads")))
    n_threads: int | None = None
    if n_threads_val is not None:
        try:
            n_threads = int(n_threads_val)
        except Exception:
            n_threads = None

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
        model_base=model_base or default.model_base,
        model_path=model_path or default.model_path,
        quantization=quantization,
        prompt_style=prompt_style or default.prompt_style,
        stop=stop or default.stop,
        n_ctx=int(n_ctx),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repeat_penalty=float(repeat_penalty),
        n_gpu_layers=int(n_gpu_layers),
        n_threads=n_threads,
        seed=int(seed),
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


def _llama2_chat_prompt(system: str, user: str) -> str:
    sys_text = str(system or "").strip()
    user_text = str(user or "").strip()
    if sys_text:
        return f"<s>[INST] <<SYS>>\n{sys_text}\n<</SYS>>\n\n{user_text} [/INST]"
    return f"<s>[INST] {user_text} [/INST]"


def _llama3_instruct_prompt(system: str, user: str) -> str:
    sys_text = str(system or "").strip()
    user_text = str(user or "").strip()
    prompt = "<|begin_of_text|>"
    if sys_text:
        prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{sys_text}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


def _phi_prompt(system: str, user: str) -> str:
    sys_text = str(system or "").strip()
    user_text = str(user or "").strip()
    if sys_text:
        return f"Instruct: {sys_text}\n{user_text}\nOutput:"
    return f"Instruct: {user_text}\nOutput:"


def _phi3_prompt(system: str, user: str) -> str:
    sys_text = str(system or "").strip()
    user_text = str(user or "").strip()
    prompt = ""
    if sys_text:
        prompt += f"<|system|>\n{sys_text}<|end|>\n"
    prompt += f"<|user|>\n{user_text}<|end|>\n<|assistant|>\n"
    return prompt


def _gemma_prompt(system: str, user: str) -> str:
    sys_text = str(system or "").strip()
    user_text = str(user or "").strip()
    if sys_text:
        user_text = f"{sys_text}\n\n{user_text}"
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"


def _alpaca_prompt(system: str, user: str) -> str:
    sys_text = str(system or "").strip()
    user_text = str(user or "").strip()
    if sys_text:
        return f"### System:\n{sys_text}\n\n### Instruction:\n{user_text}\n\n### Response:\n"
    return f"### Instruction:\n{user_text}\n\n### Response:\n"


def _vicuna_prompt(system: str, user: str) -> str:
    sys_text = str(system or "").strip()
    user_text = str(user or "").strip()
    if sys_text:
        return f"{sys_text}\n\nUSER: {user_text}\nASSISTANT:"
    return f"USER: {user_text}\nASSISTANT:"


_PROMPT_BUILDERS: dict[str, callable] = {
    "mistral_instruct": _mistral_instruct_prompt,
    "chatml": _chatml_prompt,
    "im_start": _chatml_prompt,
    "llama2_chat": _llama2_chat_prompt,
    "llama3_instruct": _llama3_instruct_prompt,
    "phi": _phi_prompt,
    "phi3": _phi3_prompt,
    "gemma": _gemma_prompt,
    "alpaca": _alpaca_prompt,
    "vicuna": _vicuna_prompt,
}


def list_prompt_styles() -> list[str]:
    """Return available prompt styles."""
    return sorted(_PROMPT_BUILDERS.keys())


def _build_prompt(cfg: LLMConfig, system: str, user: str) -> str:
    style = str(getattr(cfg, "prompt_style", "") or "").strip().lower().replace("-", "_")
    builder = _PROMPT_BUILDERS.get(style, _mistral_instruct_prompt)
    return builder(system, user)


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
    """llama.cpp backend with full configuration support."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._llm = None
        self._lock = threading.Lock()

    def _ensure(self) -> bool:
        if self._llm is not None:
            return True

        with self._lock:
            # Double-check after acquiring lock
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
                # Build kwargs with all supported options
                llama_kwargs: dict[str, Any] = {
                    "model_path": model_path,
                    "n_ctx": int(self.cfg.n_ctx),
                    "verbose": False,
                }

                # GPU layers (-1 = all available)
                n_gpu_layers = getattr(self.cfg, "n_gpu_layers", -1)
                if n_gpu_layers is not None:
                    llama_kwargs["n_gpu_layers"] = int(n_gpu_layers)

                # Thread count (None = auto)
                n_threads = getattr(self.cfg, "n_threads", None)
                if n_threads is not None:
                    llama_kwargs["n_threads"] = int(n_threads)

                # Seed (-1 = random)
                seed = getattr(self.cfg, "seed", -1)
                if seed is not None and seed != -1:
                    llama_kwargs["seed"] = int(seed)

                self._llm = Llama(**llama_kwargs)
                return True
            except Exception as e:
                warn("Failed to load LLM model (%s): %s", model_path, e)
                self._llm = None
                return False

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: tuple[str, ...],
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
    ) -> str:
        if not self._ensure():
            return ""

        # Build generation kwargs
        gen_kwargs: dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stop": list(stop) if stop else None,
        }

        # Optional parameters from config
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        elif hasattr(self.cfg, "top_p"):
            gen_kwargs["top_p"] = float(self.cfg.top_p)

        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        elif hasattr(self.cfg, "top_k"):
            gen_kwargs["top_k"] = int(self.cfg.top_k)

        if repeat_penalty is not None:
            gen_kwargs["repeat_penalty"] = float(repeat_penalty)
        elif hasattr(self.cfg, "repeat_penalty"):
            gen_kwargs["repeat_penalty"] = float(self.cfg.repeat_penalty)

        out = self._llm(str(prompt), **gen_kwargs)
        try:
            return str(out["choices"][0]["text"])
        except Exception:
            return ""

    def unload(self) -> None:
        """Unload the model to free memory."""
        with self._lock:
            if self._llm is not None:
                del self._llm
                self._llm = None


class LLMRouter:
    """
    Small, swappable transcript → action router.

    Returns a single Maxim action dict in the canonical schema:
    {"tool_name": str, "params": dict}
    """

    # Sentinel value to distinguish "init failed" from "not yet initialized"
    _INIT_FAILED = object()

    def __init__(self, cfg: LLMConfig | None = None) -> None:
        self.cfg = cfg or load_llm_config()
        self._backend: Any | None = None
        self._backend_lock = threading.Lock()

    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "enabled", False))

    def _get_backend(self) -> Any | None:
        # Fast path: already initialized
        if self._backend is not None:
            return None if self._backend is LLMRouter._INIT_FAILED else self._backend

        # Thread-safe initialization
        with self._backend_lock:
            # Double-check after acquiring lock
            if self._backend is not None:
                return None if self._backend is LLMRouter._INIT_FAILED else self._backend

            if not self.enabled():
                self._backend = LLMRouter._INIT_FAILED
                return None

            backend = str(getattr(self.cfg, "backend", "") or "").strip().lower().replace("-", "_")
            if backend in ("llama", "llama_cpp", "llamacpp"):
                self._backend = _LlamaCppBackend(self.cfg)
                return self._backend

            warn("Unknown LLM backend: %s", backend)
            self._backend = LLMRouter._INIT_FAILED
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

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> dict[str, Any] | None:
        """Generate a JSON response from a prompt.

        Used by LLMWorker for agentic goal proposal.

        Args:
            prompt: The full prompt to send to the LLM (used as user message).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Parsed JSON dict or None if generation failed.
        """
        backend = self._get_backend()
        if backend is None:
            return None

        # Wrap prompt in the appropriate format for this model (ChatML, Mistral, etc.)
        system = "You are a helpful robot assistant. Always respond with valid JSON only, no other text."
        wrapped_prompt = _build_prompt(self.cfg, system, prompt)

        try:
            text = backend.complete(
                wrapped_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=tuple(getattr(self.cfg, "stop", ("</s>",))),
            )
            if text:
                warn("LLM raw response (first 200 chars): %s", text[:200] if len(text) > 200 else text)
        except Exception as e:
            warn("LLM generate_json failed: %s", e)
            return None

        obj = _extract_json_object(text)
        if not isinstance(obj, dict):
            warn("LLM returned non-dict: %s (raw text was: %s)", type(obj), text[:100] if text else "empty")
            return None

        return obj
