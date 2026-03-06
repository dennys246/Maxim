from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from maxim.utils.logging import info, warn
from maxim.utils.structured_logging import log_agentic
from maxim.models.language.cost_tracker import CostTracker, CostTrackerConfig, ModelPricing
from maxim.utils.cloud_redaction import CloudRedactionFilter, RedactionResult
from maxim.utils.cloud_audit import CloudAuditEntry, CloudAuditLogger


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
    # PyTorch/Transformers variants (for Blackwell GPU support)
    "smollm-torch": "smollm-1.7b-instruct-torch",
    "mistral-torch": "mistral-7b-instruct-torch",
    "llama3-torch": "llama3-8b-instruct-torch",
    "phi3-torch": "phi3-mini-torch",
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
    # PyTorch/Transformers profiles (for Blackwell GPU support)
    "smollm-1.7b-instruct-torch": {
        "backend": "pytorch",
        "model": "smollm-1.7b-instruct",
        "model_base": "HuggingFaceTB/SmolLM-1.7B-Instruct",
        "prompt_style": "chatml",
        "stop": ["<|im_end|>"],
        "n_ctx": 2048,
        "quantization": "F16",
    },
    "mistral-7b-instruct-torch": {
        "backend": "pytorch",
        "model": "mistral-7b-instruct-v0.2",
        "model_base": "mistralai/Mistral-7B-Instruct-v0.2",
        "prompt_style": "mistral_instruct",
        "stop": ["</s>"],
        "n_ctx": 4096,
        "quantization": "F16",
    },
    "llama3-8b-instruct-torch": {
        "backend": "pytorch",
        "model": "llama-3-8b-instruct",
        "model_base": "meta-llama/Meta-Llama-3-8B-Instruct",
        "prompt_style": "llama3_instruct",
        "stop": ["<|eot_id|>"],
        "n_ctx": 8192,
        "quantization": "F16",
    },
    "phi3-mini-torch": {
        "backend": "pytorch",
        "model": "phi-3-mini-4k-instruct",
        "model_base": "microsoft/Phi-3-mini-4k-instruct",
        "prompt_style": "phi3",
        "stop": ["<|end|>"],
        "n_ctx": 4096,
        "quantization": "F16",
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


# ─────────────────────────────────────────────────────────────────────────────
# Token counting
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for counting tokens in text."""

    def count_tokens(self, text: str) -> int: ...


class CharEstimateCounter:
    """Fallback token counter using ~3 chars per token estimate.

    Structured prompts (JSON, headers, formatting) average ~3.0-3.5 chars/token
    on common tokenizers. Using //3 is intentionally conservative to prevent
    context overflow when the real tokenizer isn't available.
    """

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 3)


class LlamaCppTokenCounter:
    """Token counter backed by llama-cpp-python's actual tokenizer."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def count_tokens(self, text: str) -> int:
        try:
            return len(self._llm.tokenize(text.encode("utf-8")))
        except Exception:
            return max(1, len(text) // 3)


class _LazyTokenCounter:
    """Token counter that upgrades to the real tokenizer when the model loads.

    At construction time, the LLM model may still be warming up in a background
    thread. This counter starts with CharEstimateCounter and transparently
    upgrades to LlamaCppTokenCounter once the backend model is loaded.
    """

    def __init__(self, router: Any) -> None:
        self._router = router
        self._real_counter: TokenCounter | None = None

    def _try_upgrade(self) -> TokenCounter:
        if self._real_counter is not None:
            return self._real_counter
        # Check if backend model has loaded since construction
        backend = self._router._get_tokenizer_backend()
        if backend is not None and hasattr(backend, "_llm") and backend._llm is not None:
            self._real_counter = LlamaCppTokenCounter(backend._llm)
            return self._real_counter
        return CharEstimateCounter()

    def count_tokens(self, text: str) -> int:
        return self._try_upgrade().count_tokens(text)


# ─────────────────────────────────────────────────────────────────────────────
# LLM response metadata (for cloud usage + cost tracking)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class LLMResponse:
    """Structured response from any LLM backend."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0
    provider: str = ""
    stop_reason: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    cached_input_tokens: int = 0
    uncached_input_tokens: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# LLM configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LLMConfig:
    enabled: bool = False
    backend: str = "llama_cpp"
    cloud_enabled: bool = False
    profile: str = "mistral-7b-instruct-v0.2"
    model: str = "mistral-7b-instruct-v0.2"
    model_base: str = "mistral-7b-instruct-v0.2"
    model_path: str = "data/models/LLM/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    quantization: str = "Q4_K_M"
    prompt_style: str = "mistral_instruct"
    stop: tuple[str, ...] = ("</s>",)
    n_ctx: int = 16384
    max_tokens: int = 512  # Increased from 128 to support full JSON tool responses
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    n_gpu_layers: int = -1  # -1 = auto (use all available GPU layers)
    n_threads: int | None = None  # None = auto-detect
    seed: int = -1  # -1 = random
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)
    agent_profiles: dict[str, str] = field(default_factory=dict)
    prompt_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    pricing: dict[str, dict[str, Any]] = field(default_factory=dict)
    redaction: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoutingPolicy:
    """Governs routing and budget enforcement for providers."""

    provider_priority: list[str] = field(default_factory=list)
    fallback_on_rate_limit: bool = True
    fallback_on_timeout: bool = True
    fallback_on_budget_exceeded: str = "local"
    require_cloud_opt_in: bool = True
    context_window_routing: bool = True
    max_cost_per_request: float = 0.50
    max_cost_per_hour: float = 1.00
    max_cost_per_day: float = 10.00
    max_cost_per_month: float = 100.00
    cost_warning_threshold: float = 0.80
    cost_critical_threshold: float = 0.95


@dataclass
class ProviderState:
    """Tracks provider health and backoff state."""

    backoff_until: float = 0.0
    consecutive_errors: int = 0
    last_error: str = ""
    last_success: float = 0.0


_DEFAULT_PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-4-5-20250514": ModelPricing(3.00, 15.00, 0.30),
    "claude-haiku-4-5-20251001": ModelPricing(0.80, 4.00, 0.08),
    "claude-opus-4-5-20250514": ModelPricing(15.00, 75.00, 1.50),
    "gpt-4o": ModelPricing(2.50, 10.00, 1.25),
    "gpt-4o-mini": ModelPricing(0.15, 0.60, 0.075),
    "local": ModelPricing(0.0, 0.0, 0.0),
}

_MODEL_DOWNGRADE_MAP: dict[str, str] = {
    "claude-opus-4-5-20250514": "claude-sonnet-4-5-20250514",
    "claude-sonnet-4-5-20250514": "claude-haiku-4-5-20251001",
    "gpt-4o": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4o-mini",
}


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


def load_llm_config(profile_override: str | None = None) -> LLMConfig:
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

    profile_raw = profile_override or os.getenv("MAXIM_LLM_PROFILE")
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

    cloud_enabled = _as_bool(os.getenv("MAXIM_LLM_CLOUD_ENABLED"))
    if cloud_enabled is None:
        cloud_enabled = _as_bool(raw.get("cloud_enabled"))
    if cloud_enabled is None:
        cloud_enabled = False

    backend = str(
        os.getenv(
            "MAXIM_LLM_BACKEND",
            profile_cfg.get("backend", raw.get("backend", builtin.get("backend", default.backend))),
        )
        or default.backend
    ).strip()
    provider_override = profile_cfg.get("provider", raw.get("provider"))
    if provider_override and not os.getenv("MAXIM_LLM_BACKEND"):
        backend = str(provider_override).strip()

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

    providers = raw.get("providers") if isinstance(raw.get("providers"), dict) else {}
    routing = raw.get("routing") if isinstance(raw.get("routing"), dict) else {}
    agent_profiles = raw.get("agent_profiles") if isinstance(raw.get("agent_profiles"), dict) else {}
    prompt_profiles = raw.get("prompt_profiles") if isinstance(raw.get("prompt_profiles"), dict) else {}
    pricing = raw.get("pricing") if isinstance(raw.get("pricing"), dict) else {}
    redaction = raw.get("redaction") if isinstance(raw.get("redaction"), dict) else {}

    return LLMConfig(
        enabled=bool(enabled),
        backend=backend or default.backend,
        cloud_enabled=bool(cloud_enabled),
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
        providers=providers,
        routing=routing,
        agent_profiles=agent_profiles,
        prompt_profiles=prompt_profiles,
        pricing=pricing,
        redaction=redaction,
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


def _sanitize_json_string(text: str) -> str:
    """Escape control characters inside JSON strings.

    LLMs sometimes output literal newlines/tabs inside JSON string values,
    which is invalid JSON. This function escapes them properly.
    """
    result = []
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            continue

        if char == '\\':
            result.append(char)
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            result.append(char)
            continue

        if in_string:
            # Escape control characters inside strings
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            elif ord(char) < 32:
                # Other control characters - escape as unicode
                result.append(f'\\u{ord(char):04x}')
            else:
                result.append(char)
        else:
            result.append(char)

    return ''.join(result)


def _find_first_json_object(text: str) -> str | None:
    """Find the first complete JSON object in text by matching braces.

    This handles cases where the LLM outputs multiple JSON objects or
    extra content after the first valid JSON.
    """
    if not text or not text.startswith("{"):
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                # Found the matching closing brace
                return text[: i + 1]

    # No matching brace found - return the whole thing for repair attempt
    return text


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    # Strip ChatML tokens that LLMs sometimes include in output
    chatml_tokens = ["<|im_end|>", "<|im_start|>", "<|assistant|>", "<|user|>", "<|system|>"]
    for token in chatml_tokens:
        if token in raw:
            # Take only content before the first ChatML token
            raw = raw.split(token)[0].strip()

    # Detect commentary patterns - LLM is explaining instead of outputting JSON
    commentary_patterns = [
        "the provided",
        "the answer",
        "to answer",
        "i will",
        "i would",
        "here is",
        "here's",
        "let me",
        "this is",
        "the question",
        "the response",
        "as requested",
        "the json",
        "valid json",
        "the format",
        "to create",
        "we can",
        "you can",
    ]
    raw_lower = raw.lower()
    for pattern in commentary_patterns:
        if raw_lower.startswith(pattern):
            # LLM is commenting, not outputting JSON
            return None

    # Extract FIRST code block only (not all of them)
    raw = raw.replace("```json", "```").replace("```JSON", "```")
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            # Take only the FIRST code block content
            raw = parts[1].strip()
        else:
            raw = raw.replace("```", "").strip()

    start = raw.find("{")
    if start < 0:
        warn("JSON extraction failed: no opening brace found")
        return None

    # Find the MATCHING closing brace for the first opening brace
    # This handles cases where there's extra content after the first JSON object
    json_candidate = _find_first_json_object(raw[start:])

    if json_candidate is None:
        # No opening brace or completely malformed
        warn("JSON extraction failed: couldn't find JSON object")
        return None

    # Check if we got a complete JSON (ends with })
    if not json_candidate.rstrip().endswith("}"):
        # Truncated - try to repair by adding missing braces
        # Single-pass counting (6x faster than 6 separate .count() calls)
        open_braces = close_braces = open_brackets = close_brackets = 0
        quote_count = 0
        prev_char = ""
        for char in json_candidate:
            if char == "{":
                open_braces += 1
            elif char == "}":
                close_braces += 1
            elif char == "[":
                open_brackets += 1
            elif char == "]":
                close_brackets += 1
            elif char == '"' and prev_char != "\\":
                quote_count += 1
            prev_char = char

        missing_braces = open_braces - close_braces
        missing_brackets = max(0, open_brackets - close_brackets)

        # Strip trailing incomplete content
        json_candidate = json_candidate.rstrip().rstrip(",")

        # Close unclosed strings
        if quote_count % 2 == 1:
            json_candidate += '"'

        # Add closing brackets/braces
        json_candidate += "]" * missing_brackets + "}" * missing_braces
        info("JSON repair: added %d braces, %d brackets", missing_braces, missing_brackets)

    try:
        obj = json.loads(json_candidate)
    except json.JSONDecodeError as e:
        # Try sanitizing control characters inside strings
        if "control character" in str(e).lower() or "invalid" in str(e).lower():
            try:
                sanitized = _sanitize_json_string(json_candidate)
                obj = json.loads(sanitized)
                info("JSON parse succeeded after sanitizing control characters")
            except Exception as e2:
                warn("JSON parse failed after sanitization: %s | len=%d | last_50: %s",
                     str(e2)[:80], len(json_candidate), json_candidate[-50:] if len(json_candidate) > 50 else json_candidate)
                return None
        else:
            warn("JSON parse failed: %s | len=%d | last_50: %s",
                 str(e)[:80], len(json_candidate), json_candidate[-50:] if len(json_candidate) > 50 else json_candidate)
            return None
    except Exception as e:
        warn("JSON parse failed: %s | len=%d | last_50: %s",
             str(e)[:80], len(json_candidate), json_candidate[-50:] if len(json_candidate) > 50 else json_candidate)
        return None

    return obj if isinstance(obj, dict) else None


class _LlamaCppBackend:
    """llama.cpp backend with full configuration support."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._llm = None
        self._lock = threading.Lock()
        self.requires_prompt_formatting = True

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
                # Sanity-check n_ctx against model name hints to avoid
                # GGML_ASSERT crashes from over-sized KV cache allocations.
                n_ctx = int(self.cfg.n_ctx)
                model_lower = os.path.basename(model_path).lower()
                ctx_hints = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384}
                for hint, limit in ctx_hints.items():
                    if hint in model_lower and n_ctx > limit:
                        warn(
                            "n_ctx=%d exceeds model's advertised %s context (%s). "
                            "Clamping to %d to avoid backend allocation failure.",
                            n_ctx, hint, model_lower, limit,
                        )
                        n_ctx = limit
                        break

                # Build kwargs with all supported options
                llama_kwargs: dict[str, Any] = {
                    "model_path": model_path,
                    "n_ctx": n_ctx,
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

    def warmup(self) -> bool:
        """Pre-load the model. Call at startup to avoid first-request latency."""
        return self._ensure()

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
        self._backends: dict[str, Any] = {}
        self._backend_lock = threading.Lock()
        self._providers = self._normalize_providers(self.cfg)
        self._provider_states: dict[str, ProviderState] = {
            key: ProviderState() for key in self._providers.keys()
        }
        self._routing_policy = self._load_routing_policy(self.cfg.routing)
        self._cost_tracker = CostTracker(
            pricing=self._load_pricing_table(self.cfg),
            config=self._load_cost_config(self.cfg),
        )
        self._audit_logger = CloudAuditLogger()
        self._cloud_allowed, self._cloud_block_reason = self._validate_cloud_config()
        try:
            import atexit
            atexit.register(self._cost_tracker.flush)
        except Exception:
            pass

    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "enabled", False))

    @property
    def n_ctx(self) -> int:
        """Return the model's context window size in tokens."""
        return int(getattr(self.cfg, "n_ctx", 4096))

    @property
    def model_name(self) -> str:
        """Return the configured model name (if any)."""
        return str(getattr(self.cfg, "model", "") or "")

    def get_token_counter(self) -> TokenCounter:
        """Return a token counter that lazily upgrades to the real tokenizer.

        At startup the model may still be loading in a background thread.
        The returned counter starts with CharEstimateCounter and transparently
        switches to the actual tokenizer once it's available.
        """
        return _LazyTokenCounter(self)

    def _get_tokenizer_backend(self) -> Any | None:
        if self._backend is not None:
            return self._backend
        for backend in self._backends.values():
            if backend is not None and hasattr(backend, "_llm") and backend._llm is not None:
                return backend
        return None

    @staticmethod
    def _normalize_providers(cfg: LLMConfig) -> dict[str, dict[str, Any]]:
        providers = cfg.providers if isinstance(cfg.providers, dict) else {}
        if providers:
            return {
                str(k): v for k, v in providers.items()
                if isinstance(k, str) and isinstance(v, dict)
            }
        # Backward-compat: synthesize a local provider from base config
        return {
            "local": {
                "type": cfg.backend or "llama_cpp",
                "model": cfg.model,
                "model_base": cfg.model_base,
                "model_path": cfg.model_path,
                "n_ctx": cfg.n_ctx,
            }
        }

    @staticmethod
    def _load_routing_policy(raw: dict[str, Any]) -> RoutingPolicy:
        data = raw if isinstance(raw, dict) else {}
        return RoutingPolicy(
            provider_priority=list(data.get("provider_priority", [])) if isinstance(data.get("provider_priority"), list) else [],
            fallback_on_rate_limit=bool(data.get("fallback_on_rate_limit", True)),
            fallback_on_timeout=bool(data.get("fallback_on_timeout", True)),
            fallback_on_budget_exceeded=str(data.get("fallback_on_budget_exceeded", "local") or "local"),
            require_cloud_opt_in=bool(data.get("require_cloud_opt_in", True)),
            context_window_routing=bool(data.get("context_window_routing", True)),
            max_cost_per_request=float(data.get("max_cost_per_request", 0.50) or 0.0),
            max_cost_per_hour=float(data.get("max_cost_per_hour", 1.00) or 0.0),
            max_cost_per_day=float(data.get("max_cost_per_day", 10.00) or 0.0),
            max_cost_per_month=float(data.get("max_cost_per_month", 100.00) or 0.0),
            cost_warning_threshold=float(data.get("cost_warning_threshold", 0.80) or 0.0),
            cost_critical_threshold=float(data.get("cost_critical_threshold", 0.95) or 0.0),
        )

    @staticmethod
    def _load_pricing_table(cfg: LLMConfig) -> dict[str, ModelPricing]:
        pricing: dict[str, ModelPricing] = dict(_DEFAULT_PRICING)
        raw = cfg.pricing if isinstance(cfg.pricing, dict) else {}
        for model, entry in raw.items():
            if not isinstance(model, str) or not isinstance(entry, dict):
                continue
            try:
                pricing[model] = ModelPricing(
                    input_price=float(entry.get("input_price", entry.get("input", 0.0)) or 0.0),
                    output_price=float(entry.get("output_price", entry.get("output", 0.0)) or 0.0),
                    cached_input_price=float(entry.get("cached_input_price", entry.get("cached_input", 0.0)) or 0.0),
                )
            except Exception:
                continue
        return pricing

    @staticmethod
    def _load_cost_config(cfg: LLMConfig) -> CostTrackerConfig:
        raw = cfg.routing if isinstance(cfg.routing, dict) else {}
        return CostTrackerConfig(
            state_path=str(raw.get("cost_state_path", "data/util/cost_state.json")),
            persistence_interval_s=float(raw.get("cost_persistence_interval_s", 10.0) or 10.0),
            persistence_interval_n=int(raw.get("cost_persistence_interval_n", 5) or 5),
            reserved_budget_ratio=float(raw.get("reserved_budget_ratio", 0.2) or 0.2),
            min_spend_samples=int(raw.get("min_spend_samples", 5) or 5),
        )

    def _validate_cloud_config(self) -> tuple[bool, str]:
        if not self.cfg.cloud_enabled:
            return False, "cloud_enabled false"

        redaction_present = False
        redaction_cfg = self.cfg.redaction if isinstance(self.cfg.redaction, dict) else {}
        if redaction_cfg.get("policy"):
            redaction_present = True
        if not redaction_present:
            for provider in self._providers.values():
                if isinstance(provider, dict) and provider.get("redaction_policy"):
                    redaction_present = True
                    break
        if not redaction_present:
            warn("Cloud enabled but no redaction policy configured; cloud dispatch disabled")
            return False, "redaction missing"

        policy = self._routing_policy
        if (
            policy.max_cost_per_request <= 0
            and policy.max_cost_per_hour <= 0
            and policy.max_cost_per_day <= 0
            and policy.max_cost_per_month <= 0
        ):
            warn("Cloud enabled but all cost limits are zero; cloud dispatch disabled")
            return False, "cost limits missing"

        return True, ""

    def _provider_type(self, provider_cfg: dict[str, Any]) -> str:
        raw = provider_cfg.get("type") or provider_cfg.get("backend") or self.cfg.backend
        return str(raw or "").strip().lower().replace("-", "_")

    def _provider_is_cloud(self, provider_cfg: dict[str, Any]) -> bool:
        provider_type = self._provider_type(provider_cfg)
        return provider_type in ("anthropic", "claude", "openai", "openai_compatible", "openai_compat")

    def _provider_n_ctx(self, provider_cfg: dict[str, Any]) -> int:
        try:
            return int(provider_cfg.get("n_ctx", self.cfg.n_ctx))
        except Exception:
            return int(self.cfg.n_ctx)

    def _provider_model(self, provider_cfg: dict[str, Any]) -> str:
        model = provider_cfg.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
        return str(self.cfg.model or "")

    def _provider_cost_visible(self, provider_cfg: dict[str, Any]) -> bool:
        if "cost_visible" in provider_cfg:
            return bool(provider_cfg.get("cost_visible"))
        return self._provider_is_cloud(provider_cfg)

    def _provider_pricing_required(self, provider_cfg: dict[str, Any]) -> bool:
        if "pricing_required" in provider_cfg:
            return bool(provider_cfg.get("pricing_required"))
        return self._provider_is_cloud(provider_cfg)

    def _get_backend_for_provider(self, provider_key: str) -> Any | None:
        backend = self._backends.get(provider_key)
        if backend is not None:
            return None if backend is LLMRouter._INIT_FAILED else backend

        provider_cfg = self._providers.get(provider_key, {})
        provider_type = self._provider_type(provider_cfg)

        # Local backends
        if provider_type in ("llama", "llama_cpp", "llamacpp"):
            cfg = self._build_provider_config(provider_cfg)
            backend = _LlamaCppBackend(cfg)
        elif provider_type in ("pytorch", "torch", "transformers", "huggingface", "hf"):
            from maxim.models.language.transformers_backend import _PyTorchTransformersBackend
            cfg = self._build_provider_config(provider_cfg)
            backend = _PyTorchTransformersBackend(cfg)
        # Cloud backends
        elif provider_type in ("anthropic", "claude"):
            from maxim.models.language.anthropic_backend import _AnthropicBackend
            backend = _AnthropicBackend(self.cfg, provider_key=provider_key)
        elif provider_type in ("openai", "gpt"):
            from maxim.models.language.openai_backend import _OpenAIBackend
            backend = _OpenAIBackend(self.cfg, provider_key=provider_key)
        elif provider_type in ("openai_compatible", "openai_compat"):
            from maxim.models.language.openai_backend import _OpenAIBackend
            backend = _OpenAIBackend(self.cfg, provider_key=provider_key)
        else:
            warn("Unknown LLM provider type: %s (%s)", provider_type, provider_key)
            backend = LLMRouter._INIT_FAILED

        self._backends[provider_key] = backend
        return None if backend is LLMRouter._INIT_FAILED else backend

    def _build_provider_config(self, provider_cfg: dict[str, Any]) -> LLMConfig:
        """Override base config for local provider-specific settings."""
        model = provider_cfg.get("model", self.cfg.model)
        model_base = provider_cfg.get("model_base", self.cfg.model_base)
        model_path = provider_cfg.get("model_path", self.cfg.model_path)
        n_ctx = provider_cfg.get("n_ctx", self.cfg.n_ctx)
        backend = provider_cfg.get("type", self.cfg.backend)
        prompt_style = provider_cfg.get("prompt_style", self.cfg.prompt_style)
        stop_val = provider_cfg.get("stop", self.cfg.stop)
        stop: tuple[str, ...]
        if isinstance(stop_val, (list, tuple)) and stop_val:
            stop = tuple(str(s) for s in stop_val if isinstance(s, (str, int, float)) and str(s).strip())
        elif isinstance(stop_val, str) and stop_val.strip():
            stop = tuple(s.strip() for s in stop_val.split(",") if s.strip())
        else:
            stop = tuple(self.cfg.stop)
        return replace(
            self.cfg,
            backend=str(backend or self.cfg.backend),
            model=str(model or self.cfg.model),
            model_base=str(model_base or self.cfg.model_base),
            model_path=str(model_path or self.cfg.model_path),
            n_ctx=int(n_ctx or self.cfg.n_ctx),
            prompt_style=str(prompt_style or self.cfg.prompt_style),
            stop=stop,
        )

    def _default_provider(self) -> str:
        if self._routing_policy.provider_priority:
            for name in self._routing_policy.provider_priority:
                if name in self._providers:
                    return name
        return next(iter(self._providers.keys()))

    def _provider_order(self) -> list[str]:
        if self._routing_policy.provider_priority:
            return [p for p in self._routing_policy.provider_priority if p in self._providers]
        return list(self._providers.keys())

    def _local_providers(self) -> list[str]:
        locals_only = []
        for key, cfg in self._providers.items():
            if not self._provider_is_cloud(cfg):
                locals_only.append(key)
        return locals_only

    def _estimate_prompt_tokens(self, system: str, user: str) -> int:
        counter = self.get_token_counter()
        combined = f"{system}\n{user}".strip()
        try:
            return int(counter.count_tokens(combined))
        except Exception:
            return max(1, len(combined) // 3)

    def _budget_status(self, now: float) -> tuple[str, dict[str, float]]:
        policy = self._routing_policy
        totals = self._cost_tracker.get_totals(now)
        ratios: dict[str, float] = {}
        if policy.max_cost_per_hour > 0:
            ratios["hourly"] = totals["hourly"] / policy.max_cost_per_hour
        if policy.max_cost_per_day > 0:
            ratios["daily"] = totals["daily"] / policy.max_cost_per_day
        if policy.max_cost_per_month > 0:
            ratios["monthly"] = totals["monthly"] / policy.max_cost_per_month
        if not ratios:
            return "normal", totals
        max_ratio = max(ratios.values())
        if max_ratio >= 1.0:
            return "blocked", totals
        if max_ratio >= policy.cost_critical_threshold:
            return "critical", totals
        if max_ratio >= policy.cost_warning_threshold:
            return "warning", totals
        return "normal", totals

    def _model_for_tier(self, current_model: str, provider_cfg: dict[str, Any], tier: str) -> str:
        if tier not in ("warning", "critical"):
            return current_model
        tiers = provider_cfg.get("model_tiers")
        if isinstance(tiers, list) and tiers:
            model_list = [str(m) for m in tiers if isinstance(m, str)]
            if current_model in model_list:
                idx = model_list.index(current_model)
                if tier == "warning":
                    return model_list[min(idx + 1, len(model_list) - 1)]
                return model_list[-1]
            if tier == "critical":
                return model_list[-1]
        if tier == "warning":
            return _MODEL_DOWNGRADE_MAP.get(current_model, current_model)
        # critical: downgrade until stable
        model = current_model
        while model in _MODEL_DOWNGRADE_MAP and _MODEL_DOWNGRADE_MAP[model] != model:
            model = _MODEL_DOWNGRADE_MAP[model]
        return model

    def _candidate_providers(
        self,
        prompt_tokens: int,
        max_tokens: int,
        now: float,
    ) -> tuple[list[str], str, dict[str, float]]:
        policy = self._routing_policy
        budget_tier, totals = self._budget_status(now)

        providers = self._provider_order()
        filtered: list[str] = []
        for key in providers:
            cfg = self._providers.get(key, {})
            if self._provider_is_cloud(cfg):
                if not self._cloud_allowed or (policy.require_cloud_opt_in and not self.cfg.cloud_enabled):
                    continue
            state = self._provider_states.get(key)
            if state and state.backoff_until > now:
                continue
            if policy.context_window_routing:
                n_ctx = self._provider_n_ctx(cfg)
                if prompt_tokens + max_tokens > n_ctx:
                    continue
            filtered.append(key)

        if budget_tier == "blocked":
            if policy.fallback_on_budget_exceeded == "local":
                locals_only = []
                for key in self._local_providers():
                    cfg = self._providers.get(key, {})
                    if policy.context_window_routing:
                        n_ctx = self._provider_n_ctx(cfg)
                        if prompt_tokens + max_tokens > n_ctx:
                            continue
                    locals_only.append(key)
                return locals_only, budget_tier, totals
            return [], budget_tier, totals

        return filtered, budget_tier, totals

    def _note_provider_success(self, provider_key: str) -> None:
        state = self._provider_states.get(provider_key)
        if state is None:
            return
        state.consecutive_errors = 0
        state.backoff_until = 0.0
        state.last_error = ""
        state.last_success = time.time()

    def _note_provider_failure(self, provider_key: str, error: str) -> None:
        state = self._provider_states.get(provider_key)
        if state is None:
            return
        state.consecutive_errors += 1
        state.last_error = error[:200]
        backoff = min(60.0, 1.0 * (2 ** max(state.consecutive_errors - 1, 0)))
        state.backoff_until = time.time() + backoff

    def _emit_cloud_audit(
        self,
        *,
        provider_key: str,
        model: str,
        usage: dict[str, Any],
        cost_usd: float,
        redaction: RedactionResult,
        request_context: dict[str, Any] | None,
    ) -> None:
        ctx = request_context or {}
        entry = CloudAuditEntry(
            timestamp=time.time(),
            provider=provider_key,
            model=model,
            data_categories_sent=redaction.categories_sent,
            data_categories_redacted=redaction.categories_redacted,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            estimated_cost_usd=float(cost_usd or 0.0),
            request_id=str(ctx.get("request_id", "")),
            agent=str(ctx.get("agent", "")),
            redaction_policy=redaction.policy,
        )
        try:
            self._audit_logger.write(entry)
        except Exception as e:
            warn("Failed to write cloud audit entry: %s", e)

    def _complete_text(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
        provider_hint: str | None = None,
        request_context: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """Complete text with optional usage metadata."""
        if not self.enabled():
            return "", None

        prompt_tokens = self._estimate_prompt_tokens(system, user)
        now = time.time()

        providers, budget_tier, totals = self._candidate_providers(
            prompt_tokens, max_tokens, now
        )
        if provider_hint and provider_hint in providers:
            providers = [provider_hint] + [p for p in providers if p != provider_hint]

        if not providers:
            warn("No eligible LLM providers for request")
            return "", None

        policy = self._routing_policy
        for provider_key in providers:
            provider_cfg = self._providers.get(provider_key, {})
            if self._provider_is_cloud(provider_cfg) and not self._cloud_allowed:
                continue

            backend = self._get_backend_for_provider(provider_key)
            if backend is None:
                self._note_provider_failure(provider_key, "backend_init_failed")
                continue

            model = self._provider_model(provider_cfg)
            model_override = self._model_for_tier(model, provider_cfg, budget_tier)

            if self._provider_is_cloud(provider_cfg):
                if policy.max_cost_per_request > 0:
                    estimate = self._cost_tracker.estimate_cost(
                        model_override,
                        prompt_tokens,
                        max_tokens,
                    )
                    if estimate is None and self._provider_pricing_required(provider_cfg):
                        warn("Missing pricing for model %s; skipping provider %s", model_override, provider_key)
                        continue
                    if estimate is not None and estimate > policy.max_cost_per_request:
                        warn("Estimated cost %.4f exceeds per-request limit", estimate)
                        if policy.fallback_on_budget_exceeded == "reject":
                            return "", None
                        continue

            redaction_result: RedactionResult | None = None
            redacted_system = system
            redacted_user = user
            if self._provider_is_cloud(provider_cfg):
                redactor = CloudRedactionFilter.from_config(
                    provider_cfg=provider_cfg,
                    global_cfg=self.cfg.redaction,
                )
                redaction_result = redactor.redact(system, user)
                redacted_system = redaction_result.system
                redacted_user = redaction_result.user

            try:
                if getattr(backend, "requires_prompt_formatting", True):
                    prompt_cfg = getattr(backend, "cfg", self.cfg)
                    stop = tuple(getattr(prompt_cfg, "stop", ("</s>",)))
                    prompt = _build_prompt(prompt_cfg, redacted_system, redacted_user)
                    text = backend.complete(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop,
                    )
                    if text:
                        self._note_provider_success(provider_key)
                        return text, None
                else:
                    if hasattr(backend, "complete_with_usage"):
                        kwargs: dict[str, Any] = {
                            "system": redacted_system,
                            "user": redacted_user,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "stop": tuple(getattr(self.cfg, "stop", ("</s>",))),
                        }
                        if model_override and getattr(backend, "supports_model_override", False):
                            kwargs["model_override"] = model_override
                        resp = backend.complete_with_usage(**kwargs)
                        if isinstance(resp, LLMResponse) and resp.content:
                            self._note_provider_success(provider_key)
                            usage = {
                                "input_tokens": resp.input_tokens,
                                "output_tokens": resp.output_tokens,
                                "model": resp.model,
                                "provider": resp.provider,
                                "cached_input_tokens": resp.cached_input_tokens,
                                "uncached_input_tokens": resp.uncached_input_tokens,
                            }
                            cost_usd = 0.0
                            try:
                                cost_usd = self._cost_tracker.record(
                                    provider=resp.provider or provider_key,
                                    model=resp.model or model_override,
                                    input_tokens=resp.input_tokens,
                                    output_tokens=resp.output_tokens,
                                    cached_input_tokens=resp.cached_input_tokens,
                                    uncached_input_tokens=resp.uncached_input_tokens,
                                    timestamp=now,
                                )
                            except Exception:
                                cost_usd = 0.0
                            usage["cost_usd"] = cost_usd

                            if redaction_result is None:
                                redaction_result = RedactionResult(
                                    system=redacted_system,
                                    user=redacted_user,
                                    categories_sent=[],
                                    categories_redacted=[],
                                    policy="unknown",
                                )
                            self._emit_cloud_audit(
                                provider_key=provider_key,
                                model=resp.model or model_override,
                                usage=usage,
                                cost_usd=cost_usd,
                                redaction=redaction_result,
                                request_context=request_context,
                            )
                            return resp.content, usage
                    # Fallback: treat user string as prompt
                    text = backend.complete(
                        redacted_user,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=tuple(getattr(self.cfg, "stop", ("</s>",))),
                    )
                    if text:
                        self._note_provider_success(provider_key)
                        return text, None
            except Exception as e:
                warn("LLM complete failed (%s): %s", provider_key, e)

            self._note_provider_failure(provider_key, "call_failed")

        return "", None

    def preview_provider(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Preview which provider would likely be used for this prompt."""
        prompt_tokens = self._estimate_prompt_tokens(system, user)
        now = time.time()
        providers, budget_tier, totals = self._candidate_providers(
            prompt_tokens, max_tokens, now
        )
        if not providers:
            return {
                "provider": "",
                "is_cloud": False,
                "model": "",
                "budget_tier": "blocked",
                "prompt_tokens": prompt_tokens,
                "n_ctx": 0,
            }
        provider_key = providers[0]
        provider_cfg = self._providers.get(provider_key, {})
        model = self._provider_model(provider_cfg)
        model_override = self._model_for_tier(model, provider_cfg, budget_tier)
        return {
            "provider": provider_key,
            "is_cloud": self._provider_is_cloud(provider_cfg),
            "model": model_override,
            "budget_tier": budget_tier,
            "prompt_tokens": prompt_tokens,
            "n_ctx": self._provider_n_ctx(provider_cfg),
            "cost_visible": self._provider_cost_visible(provider_cfg),
            "totals": totals,
        }

    def get_cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    def cloud_allowed(self) -> bool:
        return bool(self._cloud_allowed)

    def get_routing_policy(self) -> RoutingPolicy:
        return self._routing_policy

    def cloud_block_reason(self) -> str:
        return self._cloud_block_reason

    def has_cost_visible_provider(self) -> bool:
        for cfg in self._providers.values():
            if self._provider_is_cloud(cfg) and self._provider_cost_visible(cfg):
                return True
        return False

    def get_provider_configs(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self._providers.items()}

    def warmup(self) -> bool:
        """
        Pre-load the LLM model at startup to avoid first-request latency.

        Call this after other initialization is complete. The model loading
        happens in a background thread so it doesn't block startup.

        Returns True if warmup was initiated, False if LLM is disabled.
        """
        if not self.enabled():
            return False

        import threading
        import time

        def _warmup_thread():
            start_time = time.time()
            log_agentic("llm_router", "startup", {"status": "loading", "model": str(getattr(self.cfg, "model_path", ""))[-50:]})
            backend = self._get_backend()
            if backend is not None and hasattr(backend, "warmup"):
                info("Warming up LLM backend...")
                if backend.warmup():
                    elapsed = time.time() - start_time
                    info("LLM model loaded and ready")
                    log_agentic("llm_router", "startup", {"status": "ready", "load_time_s": round(elapsed, 1)})
                else:
                    warn("LLM warmup failed")
                    log_agentic("llm_router", "error", {"context": "warmup", "error": "warmup returned false"}, level="WARNING")
            elif backend is not None:
                # Backend doesn't have warmup, try a minimal completion to load
                info("Warming up LLM backend (no warmup method, using test prompt)...")
                try:
                    backend.complete(
                        prompt="Hello",
                        max_tokens=1,
                        temperature=0.0,
                        stop=(),
                    )
                    elapsed = time.time() - start_time
                    info("LLM model loaded and ready")
                    log_agentic("llm_router", "startup", {"status": "ready", "load_time_s": round(elapsed, 1)})
                except Exception as e:
                    warn("LLM warmup failed: %s", e)
                    log_agentic("llm_router", "error", {"context": "warmup", "error": str(e)[:50]}, level="WARNING")

        thread = threading.Thread(target=_warmup_thread, daemon=True, name="LLMWarmup")
        thread.start()
        return True

    def _get_backend(self) -> Any | None:
        # Fast path: already initialized
        if self._backend is not None:
            return None if self._backend is LLMRouter._INIT_FAILED else self._backend

        # Thread-safe initialization
        with self._backend_lock:
            if self._backend is not None:
                return None if self._backend is LLMRouter._INIT_FAILED else self._backend

            if not self.enabled():
                self._backend = LLMRouter._INIT_FAILED
                return None

            provider_key = self._default_provider()
            backend = self._get_backend_for_provider(provider_key)
            self._backend = backend if backend is not None else LLMRouter._INIT_FAILED
            return backend

    def route(
        self,
        transcript_text: str,
        *,
        allowed_tools: set[str],
        allowed_commands: set[str],
    ) -> dict[str, Any] | None:
        if not self.enabled():
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
        text, _usage = self._complete_text(
            system,
            user,
            temperature=float(self.cfg.temperature),
            max_tokens=int(self.cfg.max_tokens),
        )

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
        *,
        provider_hint: str | None = None,
        request_context: dict[str, Any] | None = None,
        system_override: str | None = None,
    ) -> dict[str, Any] | None:
        """Generate a JSON response from a prompt.

        Used by LLMWorker for agentic goal proposal.

        Args:
            prompt: The full prompt to send to the LLM (used as user message).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            provider_hint: Optional provider key to prefer.
            request_context: Metadata for audit logs (agent, request_id, lane).
            system_override: Override the default JSON-only system prompt.

        Returns:
            Parsed JSON dict or None if generation failed.
        """
        if not self.enabled():
            return None
        backend = self._get_backend()

        # Two-stage approach for ANSWER_ONLY prompts
        # Stage 1: Get plain text answer (easier for small models)
        # Stage 2: Wrap in JSON programmatically
        if prompt.startswith("ANSWER_ONLY|"):
            question = prompt[len("ANSWER_ONLY|"):].strip()
            return self._generate_answer_only(
                backend,
                question,
                temperature,
                max_tokens,
                provider_hint=provider_hint,
                request_context=request_context,
            )

        # Tool-aware prompt with full context
        if prompt.startswith("TOOL_PROMPT|"):
            tool_prompt = prompt[len("TOOL_PROMPT|"):].strip()
            return self._generate_tool_response(
                backend,
                tool_prompt,
                temperature,
                max_tokens,
                provider_hint=provider_hint,
                request_context=request_context,
            )

        # Standard JSON generation path
        # Use a strict system prompt
        system = system_override or (
            "You are a JSON-only response system. "
            "Output ONLY valid JSON. No explanations, no code, no markdown. "
            "If the user prompt contains partial JSON, complete it. "
            "Never explain how to create JSON - just output the JSON directly."
        )

        text, usage = self._complete_text(
            system,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_hint=provider_hint,
            request_context=request_context,
        )
        if text:
            info("LLM raw response (first 200 chars): %s", text[:200] if len(text) > 200 else text)

        obj = _extract_json_object(text)
        if not isinstance(obj, dict):
            warn("LLM returned non-dict: %s (raw text was: %s)", type(obj), text[:100] if text else "empty")
            return None

        if usage:
            if "usage" not in obj:
                obj["usage"] = usage
            else:
                obj["_usage"] = usage

        return obj

    def _generate_answer_only(
        self,
        backend: Any,
        question: str,
        temperature: float,
        max_tokens: int,
        *,
        provider_hint: str | None = None,
        request_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Two-stage answer generation: get plain text, then wrap in JSON.

        This approach works better with small local models that struggle
        with JSON formatting instructions.
        """
        # Stage 1: Ask for a plain text answer (no JSON complexity)
        system = (
            "You are Maxim, a helpful robot assistant. "
            "Answer the user's question directly and concisely. "
            "Give only the answer, no explanations or preamble."
        )
        user = f"Question: {question}\nAnswer:"
        text, usage = self._complete_text(
            system,
            user,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_hint=provider_hint,
            request_context=request_context,
        )

        if not text or not text.strip():
            warn("LLM returned empty answer for question: %s", question)
            return None

        # Clean up the answer
        answer = text.strip()

        # Remove common prefixes the LLM might add
        prefixes_to_remove = [
            "answer:", "the answer is:", "response:", "here is the answer:",
            "a:", "q:", "the answer:",
        ]
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break

        # Note: Response length is controlled by max_tokens in config
        # No truncation here - let the full response through

        info("LLM answer_only response: %s", answer[:500] if len(answer) > 500 else answer)

        # Stage 2: Wrap in JSON programmatically (no LLM needed)
        # Escape the answer for JSON
        escaped_answer = answer.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        # Field order: action → confidence → reasoning (most important first)
        result = {
            "action": {
                "tool_name": "respond",
                "params": {"message": escaped_answer},
            },
            "confidence": 0.85,
            "reasoning": "answer_only",
        }
        if usage:
            result["usage"] = usage
        return result

    def _generate_tool_response(
        self,
        backend: Any,
        tool_prompt: str,
        temperature: float,
        max_tokens: int,
        *,
        provider_hint: str | None = None,
        request_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Generate a tool-aware response with full context.

        This method handles complex prompts that include:
        - Mode context and goals
        - Available tools
        - Percepts and observations
        - Agent states

        Returns a structured response with action, reasoning, and optional next_actions.
        """
        # Detect PLANNING mode from the prompt banner
        # Match against key phrase that appears in both old and new banner formats
        is_planning_mode = "PLANNING MODE" in tool_prompt and "APPROVAL" in tool_prompt

        if is_planning_mode:
            # PLANNING MODE: Allow proposal text followed by JSON
            system = (
                "You are Maxim, an intelligent robot assistant. "
                "You are in PLANNING mode and need user approval before acting. "
                "Follow the response format instructions in the prompt exactly."
            )
            user = f"""{tool_prompt}

IMPORTANT: Follow the PLANNING MODE format exactly as shown above.
First write your proposal in plain text, then <|action_json|>, then the JSON."""
        else:
            # NORMAL MODE: JSON-only response
            system = (
                "You are Maxim, an intelligent robot assistant. "
                "You MUST respond with valid JSON only. No explanations outside JSON. "
                "Select the most appropriate tool based on the context and user request. "
                "If unsure, use 'respond' to communicate with the user."
            )
            user = f"""{tool_prompt}

Respond with ONLY a valid JSON object. No text before or after the JSON."""

        if is_planning_mode:
            info("PLANNING mode detected - expecting proposal + <|action_json|> + JSON format")

        text, usage = self._complete_text(
            system,
            user,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_hint=provider_hint,
            request_context=request_context,
        )
        if text:
            info("LLM tool_response raw (first 300 chars): %s", text[:300] if len(text) > 300 else text)

        if not text or not text.strip():
            warn("LLM returned empty tool response")
            return None

        # Check for planning mode delimiter: plan text followed by <|action_json|> followed by JSON
        plan_text = None
        action_json_delimiter = "<|action_json|>"
        if action_json_delimiter in text:
            parts = text.split(action_json_delimiter, 1)
            plan_text = parts[0].strip()
            text = parts[1].strip() if len(parts) > 1 else ""
            info("Planning mode: extracted plan_text (%d chars) and JSON", len(plan_text))
        elif is_planning_mode:
            # PLANNING MODE FALLBACK: LLM output raw JSON without the delimiter
            # We'll generate a synthetic proposal after extracting the JSON
            info("PLANNING mode but no delimiter found - will generate synthetic proposal")

        # Extract JSON from response
        obj = _extract_json_object(text)
        if not isinstance(obj, dict):
            warn("LLM tool_response returned non-dict: %s", type(obj))
            # Try to salvage by extracting message from JSON-like text
            clean_text = text.strip()

            # If text looks like JSON with a respond action, extract the message
            # Pattern: "message": "actual message content"
            if '"message"' in clean_text:
                import re
                msg_match = re.search(r'"message"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', clean_text)
                if msg_match:
                    # Extract and unescape the message
                    extracted_msg = msg_match.group(1).replace('\\"', '"')
                    if extracted_msg and len(extracted_msg) > 10:
                        info("Salvaged message from failed JSON parse: %s", extracted_msg[:50])
                        return {
                            "action": {
                                "tool_name": "respond",
                                "params": {"message": extracted_msg},
                            },
                            "confidence": 0.5,
                            "reasoning": "salvaged_from_json",
                        }

            # Last resort: wrap non-JSON text (but NOT raw JSON)
            if clean_text and not clean_text.startswith("{"):
                return {
                    "action": {
                        "tool_name": "respond",
                        "params": {"message": clean_text[:500]},
                    },
                    "confidence": 0.5,
                    "reasoning": "fallback_parse",
                }
            return None

        # Validate action structure
        action = obj.get("action")
        if action and isinstance(action, dict):
            if "tool_name" not in action:
                # Try to extract tool_name from flat structure
                if "tool_name" in obj:
                    action = {"tool_name": obj["tool_name"], "params": obj.get("params", {})}
                    obj["action"] = action

        # Ensure required fields
        if "confidence" not in obj:
            obj["confidence"] = 0.7
        if "reasoning" not in obj:
            obj["reasoning"] = ""
        if "mode_goal_achieved" not in obj:
            obj["mode_goal_achieved"] = False

        # Add planning mode fields if present
        if plan_text:
            obj["_plan_text"] = plan_text
            obj["_requires_approval"] = True
        elif is_planning_mode:
            # PLANNING MODE FALLBACK: Generate synthetic proposal from the action
            # This handles cases where the LLM outputs raw JSON despite instructions
            action = obj.get("action", {})
            tool_name = action.get("tool_name", "unknown") if isinstance(action, dict) else "unknown"
            params = action.get("params", {}) if isinstance(action, dict) else {}

            # Generate a user-friendly proposal based on the tool
            if tool_name == "internet_search":
                query = params.get("query", "information")
                synthetic_plan = f"I'd like to search the internet for: {query}\n\nMay I proceed with this search?"
            elif tool_name == "read_file":
                path = params.get("path", "a file")
                synthetic_plan = f"I'd like to read the file: {path}\n\nMay I proceed?"
            elif tool_name == "write_file":
                path = params.get("path", "a file")
                synthetic_plan = f"I'd like to write to the file: {path}\n\nMay I proceed?"
            elif tool_name == "http_fetch":
                url = params.get("url", "a URL")
                synthetic_plan = f"I'd like to fetch content from: {url}\n\nMay I proceed?"
            elif tool_name == "respond":
                # For respond, no approval needed - just pass through
                synthetic_plan = None
            else:
                synthetic_plan = f"I'd like to execute the '{tool_name}' action.\n\nMay I proceed?"

            if synthetic_plan:
                info("Generated synthetic proposal for PLANNING mode: %s", synthetic_plan[:100])
                obj["_plan_text"] = synthetic_plan
                obj["_requires_approval"] = True

        if usage:
            if "usage" not in obj:
                obj["usage"] = usage
            else:
                obj["_usage"] = usage

        return obj
