"""LLM configuration — profiles, quantization, and config loading.

Extracted from router.py for modularity. Contains all config data
(builtin profiles, quantization levels, aliases) and the
load_llm_config() function that assembles LLMConfig from JSON files,
environment variables, and builtin defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Quantization levels
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Profile aliases and builtin profiles
# ─────────────────────────────────────────────────────────────────────────────

_PROFILE_ALIASES: dict[str, str] = {
    "mistral": "mistral-7b-instruct-v0.2",
    "mistral-7b": "mistral-7b-instruct-v0.2",
    "mistral-7b-instruct": "mistral-7b-instruct-v0.2",
    "mixtral": "mixtral-8x7b-instruct",
    "mixtral-8x7b": "mixtral-8x7b-instruct",
    "smollm": "smollm-1.7b-instruct",
    "smollm-1.7b": "smollm-1.7b-instruct",
    "smollm-1.7b-instruct": "smollm-1.7b-instruct",
    # Llama models
    "llama2": "llama-2-7b-chat",
    "llama2-7b": "llama-2-7b-chat",
    "llama2-13b": "llama-2-13b-chat",
    "llama3": "llama-3-8b-instruct",
    "llama3-8b": "llama-3-8b-instruct",
    "llama-3.1-70b": "llama-3.1-70b-instruct",
    "llama3.1-70b": "llama-3.1-70b-instruct",
    "llama70b": "llama-3.1-70b-instruct",
    # Phi models
    "phi2": "phi-2",
    "phi3": "phi-3-mini-4k-instruct",
    "phi3-mini": "phi-3-mini-4k-instruct",
    # Qwen models
    "qwen": "qwen2-7b-instruct",
    "qwen2": "qwen2-7b-instruct",
    "qwen2-7b": "qwen2-7b-instruct",
    "qwen2.5-14b": "qwen2.5-14b-instruct",
    "qwen2.5": "qwen2.5-14b-instruct",
    "qwen14b": "qwen2.5-14b-instruct",
    "qwen2.5-32b": "qwen2.5-32b-instruct",
    "qwen32b": "qwen2.5-32b-instruct",
    # Gemma models
    "gemma": "gemma-2b-it",
    "gemma-2b": "gemma-2b-it",
    "gemma-7b": "gemma-7b-it",
    # PyTorch/Transformers variants (for Blackwell GPU support)
    "smollm-torch": "smollm-1.7b-instruct-torch",
    "mistral-torch": "mistral-7b-instruct-torch",
    "llama3-torch": "llama3-8b-instruct-torch",
    "phi3-torch": "phi3-mini-torch",
    # Cloud providers (Anthropic)
    "claude": "claude-sonnet-4-6",
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-haiku": "claude-haiku-4-5-20251001",
    "claude-opus": "claude-opus-4-6",
    # Cloud providers (OpenAI)
    "gpt4o": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",
    "gpt-4o-mini": "gpt-4o-mini",
}

_BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "mistral-7b-instruct-v0.2": {
        "backend": "llama_cpp",
        "model": "mistral-7b-instruct-v0.2",
        "model_base": "mistral-7b-instruct-v0.2",
        "prompt_style": "mistral_instruct",
        "stop": ["</s>"],
        "n_ctx": 4096,
        # Architecture metadata for P4c dynamic n_ctx sizing (see
        # peer_leader_flexibility_plan). weights_gb is the Q4_K_M GGUF
        # size (4.07 GB measured from bartowski/TheBloke HF metadata).
        # GQA: 32 attention heads / 4 = 8 KV heads. kv_type_bytes=2
        # is f16 KV cache (llama.cpp default without --type_k/--type_v).
        "arch": {
            "n_layers": 32,
            "n_kv_heads": 8,
            "head_dim": 128,
            "kv_type_bytes": 2,
            "weights_gb": 4.1,
        },
    },
    "mixtral-8x7b-instruct": {
        "backend": "llama_cpp",
        "model": "mixtral-8x7b-instruct",
        "model_base": "Mixtral-8x7B-Instruct-v0.1",
        "prompt_style": "mistral_instruct",
        "stop": ["</s>"],
        "n_ctx": 32768,
        # Mistral Mixtral-8x7B-Instruct-v0.1 published arch: 32 layers,
        # 32 attention heads, 8 KV heads (GQA ratio 4), hidden_size 4096
        # → head_dim 128. MoE: 8 experts with top-2 routing per token,
        # but ALL experts must reside in memory simultaneously, so the
        # VRAM/RAM footprint reflects the full ~26.4 GB Q4_K_M weights.
        # Q4_K_M GGUF: 26.44 GB measured (bartowski).
        "arch": {
            "n_layers": 32,
            "n_kv_heads": 8,
            "head_dim": 128,
            "kv_type_bytes": 2,
            "weights_gb": 26.4,
        },
    },
    "smollm-1.7b-instruct": {
        "backend": "llama_cpp",
        "model": "smollm-1.7b-instruct",
        "model_base": "smollm-1.7b-instruct",
        "prompt_style": "chatml",
        "stop": ["<|im_end|>", "</s>"],
        "n_ctx": 4096,
        # HuggingFaceTB/SmolLM-1.7B-Instruct config: 24 layers,
        # 32 attention heads, 32 KV heads (MHA, no GQA on the
        # smaller SmolLM variants), hidden_size 2048 → head_dim 64.
        # Q4_K_M GGUF: 0.98 GB measured.
        "arch": {
            "n_layers": 24,
            "n_kv_heads": 32,
            "head_dim": 64,
            "kv_type_bytes": 2,
            "weights_gb": 1.0,
        },
    },
    "llama-2-7b-chat": {
        "backend": "llama_cpp",
        "model": "llama-2-7b-chat",
        "model_base": "llama-2-7b-chat",
        "prompt_style": "llama2_chat",
        "stop": ["</s>"],
        "n_ctx": 4096,
        # Llama 2 7B published arch: 32 layers, 32 attention heads,
        # 32 KV heads (MHA — GQA wasn't introduced until Llama 2 70B),
        # hidden_size 4096 → head_dim 128.
        # Q4_K_M GGUF: ~3.83 GB (TheBloke; not in our HF scrape because
        # meta-llama repos are gated behind auth).
        "arch": {
            "n_layers": 32,
            "n_kv_heads": 32,
            "head_dim": 128,
            "kv_type_bytes": 2,
            "weights_gb": 3.9,
        },
    },
    "llama-2-13b-chat": {
        "backend": "llama_cpp",
        "model": "llama-2-13b-chat",
        "model_base": "llama-2-13b-chat",
        "prompt_style": "llama2_chat",
        "stop": ["</s>"],
        "n_ctx": 4096,
        # Llama 2 13B published arch: 40 layers, 40 attention heads,
        # 40 KV heads (MHA), hidden_size 5120 → head_dim 128.
        # Q4_K_M GGUF: ~7.37 GB (TheBloke).
        "arch": {
            "n_layers": 40,
            "n_kv_heads": 40,
            "head_dim": 128,
            "kv_type_bytes": 2,
            "weights_gb": 7.4,
        },
    },
    "llama-3-8b-instruct": {
        "backend": "llama_cpp",
        "model": "llama-3-8b-instruct",
        "model_base": "Meta-Llama-3-8B-Instruct",
        "prompt_style": "llama3_instruct",
        "stop": ["<|eot_id|>"],
        "n_ctx": 8192,
        # Meta Llama 3 8B published arch: 32 layers, 32 attention
        # heads, 8 KV heads (GQA ratio 4), hidden_size 4096 → head_dim 128.
        # Q4_K_M GGUF: 4.58 GB measured (QuantFactory).
        "arch": {
            "n_layers": 32,
            "n_kv_heads": 8,
            "head_dim": 128,
            "kv_type_bytes": 2,
            "weights_gb": 4.6,
        },
    },
    "llama-3.1-70b-instruct": {
        "backend": "llama_cpp",
        "model": "llama-3.1-70b-instruct",
        "model_base": "Meta-Llama-3.1-70B-Instruct",
        "prompt_style": "llama3_instruct",
        "stop": ["<|eot_id|>"],
        "n_ctx": 32768,
        # Meta Llama 3.1 70B published arch: 80 layers, 64 attention
        # heads, 8 KV heads (GQA ratio 8), hidden_size 8192 → head_dim 128.
        # The training-time n_ctx ceiling is 131072 (128K); the 32K cap
        # above is the conservative production default — long-context
        # inference at 128K on 70B-Q4 needs 64 GB+ of unified memory.
        # Q4_K_M GGUF: 42.52 GB measured (bartowski). Borderline on
        # 48 GB Apple Silicon (leaves ~6 GB for KV cache + OS); fully
        # comfortable on 64 GB+.
        "arch": {
            "n_layers": 80,
            "n_kv_heads": 8,
            "head_dim": 128,
            "kv_type_bytes": 2,
            "weights_gb": 42.5,
        },
    },
    "phi-2": {
        "backend": "llama_cpp",
        "model": "phi-2",
        "model_base": "phi-2",
        "prompt_style": "phi",
        "stop": ["<|endoftext|>"],
        "n_ctx": 2048,
        # Microsoft phi-2 published arch: 32 layers, 32 attention heads,
        # 32 KV heads (MHA), hidden_size 2560 → head_dim 80.
        # Q4_K_M GGUF: 1.67 GB measured (TheBloke).
        "arch": {
            "n_layers": 32,
            "n_kv_heads": 32,
            "head_dim": 80,
            "kv_type_bytes": 2,
            "weights_gb": 1.7,
        },
    },
    "phi-3-mini-4k-instruct": {
        "backend": "llama_cpp",
        "model": "phi-3-mini-4k-instruct",
        "model_base": "Phi-3-mini-4k-instruct",
        "prompt_style": "phi3",
        "stop": ["<|end|>", "<|endoftext|>"],
        "n_ctx": 4096,
        # Microsoft Phi-3-mini-4k published arch: 32 layers, 32
        # attention heads, 32 KV heads (MHA for the mini variant;
        # the medium variant uses GQA), hidden_size 3072 → head_dim 96.
        # Q4_K_M GGUF: ~2.3 GB (microsoft/Phi-3-mini-4k-instruct-gguf).
        "arch": {
            "n_layers": 32,
            "n_kv_heads": 32,
            "head_dim": 96,
            "kv_type_bytes": 2,
            "weights_gb": 2.3,
        },
    },
    "qwen2.5-14b-instruct": {
        "backend": "llama_cpp",
        "model": "qwen2.5-14b-instruct",
        "model_base": "Qwen2.5-14B-Instruct",
        "prompt_style": "chatml",
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "n_ctx": 32768,
        # Alibaba Qwen2.5-14B-Instruct published arch: 48 layers,
        # 40 attention heads, 8 KV heads (GQA ratio 5), hidden_size
        # 5120 → head_dim 128.
        # Q4_K_M GGUF: 8.37 GB measured (bartowski). The 32K declared
        # n_ctx above is the training-time ceiling; actual runtime
        # n_ctx on tight hardware is computed dynamically by P4c's
        # estimate_max_ctx (e.g., 16 GB VRAM caps it at ~4K f16 KV).
        "arch": {
            "n_layers": 48,
            "n_kv_heads": 8,
            "head_dim": 128,
            "kv_type_bytes": 2,
            "weights_gb": 8.4,
        },
    },
    "qwen2.5-32b-instruct": {
        "backend": "llama_cpp",
        "model": "qwen2.5-32b-instruct",
        "model_base": "Qwen2.5-32B-Instruct",
        "prompt_style": "chatml",
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "n_ctx": 32768,
        # Alibaba Qwen2.5-32B-Instruct published arch: 64 layers,
        # 40 attention heads, 8 KV heads (GQA ratio 5), hidden_size
        # 5120 → head_dim 128. Same GQA shape as the 14B sibling.
        # Q4_K_M GGUF: 19.85 GB measured (bartowski). Comfortable on
        # 48 GB Apple Silicon; the "I bought a Mac with unified memory
        # for this" default per the leader-UX plan doc.
        "arch": {
            "n_layers": 64,
            "n_kv_heads": 8,
            "head_dim": 128,
            "kv_type_bytes": 2,
            "weights_gb": 19.9,
        },
    },
    "qwen2-7b-instruct": {
        "backend": "llama_cpp",
        "model": "qwen2-7b-instruct",
        "model_base": "Qwen2-7B-Instruct",
        "prompt_style": "chatml",
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "n_ctx": 8192,
        # Alibaba Qwen2-7B-Instruct published arch: 28 layers,
        # 28 attention heads, 4 KV heads (GQA ratio 7), hidden_size
        # 3584 → head_dim 128.
        # Q4_K_M GGUF: 4.36 GB measured (Qwen/Qwen2-7B-Instruct-GGUF).
        "arch": {
            "n_layers": 28,
            "n_kv_heads": 4,
            "head_dim": 128,
            "kv_type_bytes": 2,
            "weights_gb": 4.4,
        },
    },
    "gemma-2b-it": {
        "backend": "llama_cpp",
        "model": "gemma-2b-it",
        "model_base": "gemma-2b-it",
        "prompt_style": "gemma",
        "stop": ["<end_of_turn>"],
        "n_ctx": 8192,
        # Google Gemma 1 2B published arch: 18 layers, 8 attention
        # heads, 1 KV head (MQA — a single shared KV head), hidden_size
        # 2048, head_dim 256 (head_dim * num_heads != hidden_size
        # because Gemma uses separate projection). kv_type_bytes is
        # still f16. Weights ~1.5 GB.
        "arch": {
            "n_layers": 18,
            "n_kv_heads": 1,
            "head_dim": 256,
            "kv_type_bytes": 2,
            "weights_gb": 1.5,
        },
    },
    "gemma-7b-it": {
        "backend": "llama_cpp",
        "model": "gemma-7b-it",
        "model_base": "gemma-7b-it",
        "prompt_style": "gemma",
        "stop": ["<end_of_turn>"],
        "n_ctx": 8192,
        # Google Gemma 1 7B published arch: 28 layers, 16 attention
        # heads, 16 KV heads (MHA), hidden_size 3072, head_dim 256
        # (Gemma uses the same "large heads, non-standard ratio"
        # pattern as Gemma 2B). Weights ~5.0 GB Q4_K_M.
        "arch": {
            "n_layers": 28,
            "n_kv_heads": 16,
            "head_dim": 256,
            "kv_type_bytes": 2,
            "weights_gb": 5.0,
        },
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
    # Cloud providers (Anthropic)
    "claude-sonnet-4-6": {
        "backend": "anthropic",
        "model": "claude-sonnet-4-6",
        "model_base": "claude-sonnet-4-6",
        "prompt_style": "chatml",
        "n_ctx": 200000,
        "cloud": True,
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "claude-haiku-4-5-20251001": {
        "backend": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "model_base": "claude-haiku-4-5-20251001",
        "prompt_style": "chatml",
        "n_ctx": 200000,
        "cloud": True,
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "claude-opus-4-6": {
        "backend": "anthropic",
        "model": "claude-opus-4-6",
        "model_base": "claude-opus-4-6",
        "prompt_style": "chatml",
        "n_ctx": 200000,
        "cloud": True,
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    # Cloud providers (OpenAI)
    "gpt-4o": {
        "backend": "openai",
        "model": "gpt-4o",
        "model_base": "gpt-4o",
        "prompt_style": "chatml",
        "n_ctx": 128000,
        "cloud": True,
        "api_key_env": "OPENAI_API_KEY",
    },
    "gpt-4o-mini": {
        "backend": "openai",
        "model": "gpt-4o-mini",
        "model_base": "gpt-4o-mini",
        "prompt_style": "chatml",
        "n_ctx": 128000,
        "cloud": True,
        "api_key_env": "OPENAI_API_KEY",
    },
    # ─── Cloud providers (OpenAI-compatible endpoints) ────────────────
    # These use the openai backend with custom base_url.
    # Users just set the API key env var and use the profile name.
    "gemini-2.5-flash": {
        "backend": "openai",
        "model": "gemini-2.5-flash-preview-05-20",
        "model_base": "gemini-2.5-flash-preview-05-20",
        "prompt_style": "chatml",
        "n_ctx": 1000000,
        "cloud": True,
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "gemini-2.5-pro": {
        "backend": "openai",
        "model": "gemini-2.5-pro-preview-05-06",
        "model_base": "gemini-2.5-pro-preview-05-06",
        "prompt_style": "chatml",
        "n_ctx": 1000000,
        "cloud": True,
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "groq-llama3-70b": {
        "backend": "openai",
        "model": "llama-3.3-70b-versatile",
        "model_base": "llama-3.3-70b-versatile",
        "prompt_style": "chatml",
        "n_ctx": 128000,
        "cloud": True,
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "groq-mixtral": {
        "backend": "openai",
        "model": "mixtral-8x7b-32768",
        "model_base": "mixtral-8x7b-32768",
        "prompt_style": "chatml",
        "n_ctx": 32768,
        "cloud": True,
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "together-llama3-70b": {
        "backend": "openai",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "model_base": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "prompt_style": "chatml",
        "n_ctx": 128000,
        "cloud": True,
        "api_key_env": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
    },
    "fireworks-llama3-70b": {
        "backend": "openai",
        "model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "model_base": "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "prompt_style": "chatml",
        "n_ctx": 128000,
        "cloud": True,
        "api_key_env": "FIREWORKS_API_KEY",
        "base_url": "https://api.fireworks.ai/inference/v1",
    },
    "mistral-large": {
        "backend": "openai",
        "model": "mistral-large-latest",
        "model_base": "mistral-large-latest",
        "prompt_style": "chatml",
        "n_ctx": 128000,
        "cloud": True,
        "api_key_env": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
    },
    "mistral-small": {
        "backend": "openai",
        "model": "mistral-small-latest",
        "model_base": "mistral-small-latest",
        "prompt_style": "chatml",
        "n_ctx": 128000,
        "cloud": True,
        "api_key_env": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
    },
    "deepseek-chat": {
        "backend": "openai",
        "model": "deepseek-chat",
        "model_base": "deepseek-chat",
        "prompt_style": "chatml",
        "n_ctx": 64000,
        "cloud": True,
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
    },
    "deepseek-reasoner": {
        "backend": "openai",
        "model": "deepseek-reasoner",
        "model_base": "deepseek-reasoner",
        "prompt_style": "chatml",
        "n_ctx": 64000,
        "cloud": True,
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
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
    models_dir: str | None = None,
) -> str:
    """Build the model path from base name and quantization level.

    Tries the canonical path first, then falls back to case-insensitive
    search in the models directory. This handles mismatches between
    profile model_base casing and downloader output filenames.
    """
    if models_dir is None:
        from maxim.utils.paths import model_dir

        models_dir = str(model_dir() / "LLM")
    quant = str(quantization or DEFAULT_QUANTIZATION).strip().upper().replace("-", "_")
    if quant not in QUANTIZATION_LEVELS:
        quant = DEFAULT_QUANTIZATION
    base = str(model_base or "").strip()
    canonical = os.path.join(models_dir, f"{base}.{quant}.gguf")

    # If canonical path exists, use it
    if os.path.isfile(canonical):
        return canonical

    # Case-insensitive fallback: scan the directory for a match
    try:
        target = f"{base}.{quant}.gguf".lower()
        # Also try with hyphens instead of dots for quantization
        target_alt = f"{base}-{quant.lower()}.gguf".lower()
        for filename in os.listdir(models_dir):
            if filename.lower() == target or filename.lower() == target_alt:
                return os.path.join(models_dir, filename)
    except OSError:
        pass

    # No match found — return canonical (caller handles missing file)
    return canonical


# ─────────────────────────────────────────────────────────────────────────────
# LLMConfig dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LLMConfig:
    enabled: bool = False
    backend: str = "llama_cpp"
    cloud_enabled: bool = False
    profile: str = "mistral-7b-instruct-v0.2"
    model: str = "mistral-7b-instruct-v0.2"
    model_base: str = "mistral-7b-instruct-v0.2"
    model_path: str = ""  # resolved at runtime via build_model_path()
    quantization: str = "Q4_K_M"
    prompt_style: str = "mistral_instruct"
    stop: tuple[str, ...] = ("</s>",)
    n_ctx: int = 16384
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    n_gpu_layers: int = -1
    n_threads: int | None = None
    seed: int = -1
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)
    agent_profiles: dict[str, str] = field(default_factory=dict)
    prompt_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    pricing: dict[str, dict[str, Any]] = field(default_factory=dict)
    redaction: dict[str, Any] = field(default_factory=dict)
    contemplation: tuple[tuple[str, Any], ...] = ()


# ─────────────────────────────────────────────────────────────────────────────
# Config loading helpers
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# load_llm_config — the main config assembly function
# ─────────────────────────────────────────────────────────────────────────────


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
    quantization = (
        str(
            os.getenv(
                "MAXIM_LLM_QUANTIZATION",
                profile_cfg.get(
                    "quantization", raw.get("quantization", builtin.get("quantization", default.quantization))
                ),
            )
            or default.quantization
        )
        .strip()
        .upper()
        .replace("-", "_")
    )
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
    # Allow env var to set a default redaction policy (used by --cloud-* CLI flags)
    env_redaction_policy = os.environ.get("MAXIM_LLM_REDACTION_POLICY", "").strip()
    if env_redaction_policy and not redaction.get("policy"):
        redaction = dict(redaction)
        redaction["policy"] = env_redaction_policy
    contemplation_raw = raw.get("contemplation")
    contemplation = tuple(contemplation_raw.items()) if isinstance(contemplation_raw, dict) else ()

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
        contemplation=contemplation,
    )
