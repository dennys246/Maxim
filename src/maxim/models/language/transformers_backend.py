"""PyTorch/Transformers backend for LLM inference.

Supports HuggingFace models with GPU acceleration via PyTorch.
Designed for Blackwell GPUs (RTX 50 series) where other backends may crash.
GPU detection uses PyTorch exclusively (not TensorFlow).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

from maxim.utils.optional_deps import require_optional_dependency

if TYPE_CHECKING:
    from maxim.models.language.config import LLMConfig

logger = logging.getLogger(__name__)


class _PyTorchTransformersBackend:
    """HuggingFace Transformers backend with PyTorch inference."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"
        self._lock = threading.Lock()
        self._init_attempted = False
        self.requires_prompt_formatting = True
        self._trust_remote_code = self._get_bool_config("hf_trust_remote_code", False)
        self._require_review = self._get_bool_config("hf_require_review", True)
        self._revision = self._get_str_config("hf_revision", None)
        self._local_files_only = self._get_bool_config("hf_local_files_only", False)
        self._allowlist = self._get_list_config("hf_allowlist", [])
        self._fear_agent: Any = None  # Lazy-loaded

    def _get_bool_config(self, key: str, default: bool) -> bool:
        """Get boolean config from cfg or environment."""
        # Check environment first
        env_key = f"MAXIM_LLM_{key.upper()}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")

        # Check config
        val = getattr(self.cfg, key, None)
        if val is not None:
            if isinstance(val, bool):
                return val
            return str(val).lower() in ("1", "true", "yes", "on")

        return default

    def _get_str_config(self, key: str, default: str | None) -> str | None:
        """Get string config from cfg or environment."""
        env_key = f"MAXIM_LLM_{key.upper()}"
        env_val = os.getenv(env_key)
        if env_val:
            return env_val.strip()

        val = getattr(self.cfg, key, None)
        if val is not None:
            return str(val).strip() or default

        return default

    def _get_list_config(self, key: str, default: list[str]) -> list[str]:
        """Get list config from cfg or environment."""
        env_key = f"MAXIM_LLM_{key.upper()}"
        env_val = os.getenv(env_key)
        if env_val:
            return [s.strip() for s in env_val.split(",") if s.strip()]

        val = getattr(self.cfg, key, None)
        if val is not None:
            if isinstance(val, (list, tuple)):
                return [str(s).strip() for s in val if s]
            if isinstance(val, str):
                return [s.strip() for s in val.split(",") if s.strip()]

        return default

    def _get_fear_agent(self) -> Any:
        """Get or create FearAgent instance."""
        if self._fear_agent is None:
            try:
                from maxim.agents.fear_agent import FearAgent

                # FearAgent without LLM for bootstrap (avoid circular dep)
                self._fear_agent = FearAgent(llm=None)
            except ImportError:
                logger.warning("FearAgent not available")
                return None
        return self._fear_agent

    def _review_remote_code(self, model_id: str) -> bool:
        """Review remote code before loading a model.

        Uses FearAgent to analyze Python files that will execute
        when loading a HuggingFace model with trust_remote_code=True.

        Args:
            model_id: HuggingFace model identifier (e.g., "org/model-name")

        Returns:
            True if code is approved for execution, False otherwise.
        """
        logger.info(f"FearAgent reviewing remote code for: {model_id}")

        # Check allowlist first
        if self._allowlist and model_id not in self._allowlist:
            logger.warning(f"Model {model_id} not in allowlist, blocking")
            return False

        # Require pinned revision for trust_remote_code
        if not self._revision:
            logger.warning(f"No revision pinned for {model_id}, blocking trust_remote_code")
            return False

        try:
            from huggingface_hub import snapshot_download

            # Download repo snapshot (or use cached)
            local_dir = snapshot_download(
                model_id,
                revision=self._revision,
                local_files_only=self._local_files_only,
                allow_patterns=["*.py", "*.json"],
            )

            # Identify Python files that will execute
            # HuggingFace executes: auto_map targets, modeling_*.py,
            # configuration_*.py, tokenization_*.py, __init__.py
            files_to_review = []
            for root, dirs, files in os.walk(local_dir):
                for fname in files:
                    if fname.endswith(".py"):
                        # Check if this is an executable file
                        if any(
                            pat in fname
                            for pat in [
                                "modeling_",
                                "configuration_",
                                "tokenization_",
                                "__init__",
                                "auto_",
                                "processing_",
                            ]
                        ):
                            files_to_review.append(os.path.join(root, fname))

            # Also check config.json for auto_map references
            config_path = os.path.join(local_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                auto_map = config.get("auto_map", {})
                for key, module_path in auto_map.items():
                    # module_path like "modeling_custom.CustomModel"
                    if "." in module_path:
                        module_file = module_path.split(".")[0] + ".py"
                        full_path = os.path.join(local_dir, module_file)
                        if os.path.exists(full_path) and full_path not in files_to_review:
                            files_to_review.append(full_path)

            if not files_to_review:
                logger.info(f"No executable Python files found in {model_id}")
                return True

            # Review each file with FearAgent
            fear_agent = self._get_fear_agent()
            if not fear_agent:
                # No FearAgent available - fail closed
                logger.warning("FearAgent unavailable, blocking trust_remote_code")
                return False

            all_findings = []
            for fpath in files_to_review:
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        code = f.read()

                    rel_path = os.path.relpath(fpath, local_dir)
                    result = fear_agent.review_code(
                        code,
                        source=f"huggingface:{model_id}/{rel_path}",
                        context=f"HuggingFace model code, revision={self._revision}",
                    )

                    all_findings.extend(result.findings)

                    if not result.allow:
                        logger.warning(f"FearAgent blocked {rel_path}: {result.summary}")
                        return False

                except Exception as e:
                    logger.error(f"Failed to review {fpath}: {e}")
                    # Fail closed on review errors
                    return False

            # Log summary
            if all_findings:
                logger.info(
                    f"FearAgent found {len(all_findings)} concerns in {model_id}, "
                    f"but all were LOW/MEDIUM severity - allowing"
                )
            else:
                logger.info(f"FearAgent approved {model_id} - no concerns found")

            return True

        except Exception as e:
            logger.error(f"Remote code review failed for {model_id}: {e}")
            # Fail closed
            return False

    def _ensure(self) -> bool:
        """Lazy-load model on first use (thread-safe)."""
        if self._model is not None:
            return True
        if self._init_attempted:
            return False

        with self._lock:
            if self._model is not None:
                return True
            if self._init_attempted:
                return False

            self._init_attempted = True
            # Requested-but-missing dependency is a SETUP error: raise loudly
            # with an actionable hint BEFORE the broad try below (which is for
            # genuine model-load failures, not missing packages). Otherwise a
            # missing torch/transformers collapses into "Failed to load PyTorch
            # model" + return False — degraded and unactionable.
            require_optional_dependency("torch", extra="llm-torch", feature="Transformers backend")
            require_optional_dependency("transformers", extra="llm-torch", feature="Transformers backend")
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # Determine device - PyTorch ONLY (no TensorFlow)
                if torch.cuda.is_available():
                    self._device = "cuda"
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"PyTorch backend using CUDA: {gpu_name}")
                else:
                    self._device = "cpu"
                    logger.info("PyTorch backend using CPU (no CUDA available)")

                # Get model identifier
                model_id = self._get_model_id()
                logger.info(f"Loading model: {model_id}")

                # FearAgent review for trust_remote_code
                if self._trust_remote_code and self._require_review:
                    if not self._review_remote_code(model_id):
                        logger.error(f"FearAgent blocked loading {model_id}")
                        return False

                # Load tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=self._trust_remote_code,
                    revision=self._revision,
                    local_files_only=self._local_files_only,
                )

                # Load model with appropriate dtype/quantization
                load_kwargs = self._get_load_kwargs()
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=self._trust_remote_code,
                    revision=self._revision,
                    local_files_only=self._local_files_only,
                    use_safetensors=True,
                    **load_kwargs,
                )

                if self._device == "cuda" and not load_kwargs.get("device_map"):
                    self._model = self._model.to(self._device)

                logger.info(f"Model loaded successfully on {self._device}")
                return True

            except Exception as e:
                logger.error(f"Failed to load PyTorch model: {e}")
                return False

    def _get_model_id(self) -> str:
        """Get HuggingFace model identifier from config."""
        # Check for explicit HF model path
        model_base = getattr(self.cfg, "model_base", "") or ""
        if "/" in str(model_base):
            return str(model_base)  # Already a HF identifier

        # Map common names to HF identifiers
        MODEL_MAP = {
            "mistral-7b-instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
            "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
            "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
            "phi-2": "microsoft/phi-2",
            "phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
            "gemma-2b-it": "google/gemma-2b-it",
            "gemma-7b-it": "google/gemma-7b-it",
            "qwen2-7b-instruct": "Qwen/Qwen2-7B-Instruct",
            "smollm-1.7b-instruct": "HuggingFaceTB/SmolLM-1.7B-Instruct",
        }
        return MODEL_MAP.get(str(model_base), str(model_base))

    def _get_load_kwargs(self) -> dict[str, Any]:
        """Get model loading kwargs based on config."""
        import torch

        kwargs: dict[str, Any] = {}
        quant = getattr(self.cfg, "quantization", "F16") or "F16"

        if self._device == "cuda":
            # GPU loading options
            if quant in ("Q4_K_M", "Q4_0", "Q4_K_S", "int4"):
                # 4-bit quantization via bitsandbytes
                try:
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    kwargs["device_map"] = "auto"
                except ImportError:
                    logger.warning("bitsandbytes not available, using float16")
                    kwargs["torch_dtype"] = torch.float16

            elif quant in ("Q8_0", "int8"):
                # 8-bit quantization via bitsandbytes
                try:
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    kwargs["device_map"] = "auto"
                except ImportError:
                    logger.warning("bitsandbytes not available, using float16")
                    kwargs["torch_dtype"] = torch.float16

            elif quant in ("F16", "float16"):
                kwargs["torch_dtype"] = torch.float16

            elif quant in ("BF16", "bfloat16"):
                kwargs["torch_dtype"] = torch.bfloat16

            else:
                # Default to float16 for GPU
                kwargs["torch_dtype"] = torch.float16

        else:
            # CPU loading - use float32 for compatibility
            kwargs["torch_dtype"] = torch.float32

        return kwargs

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
        """Generate text completion."""
        if not self._ensure():
            return ""

        try:
            import torch

            # Tokenize
            n_ctx = getattr(self.cfg, "n_ctx", 4096)
            max_length = max(1, n_ctx - max_tokens)
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            if self._device == "cuda":
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Build generation kwargs
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self._tokenizer.eos_token_id,
            }

            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                if top_p is not None:
                    gen_kwargs["top_p"] = top_p
                elif hasattr(self.cfg, "top_p"):
                    gen_kwargs["top_p"] = float(self.cfg.top_p)
                if top_k is not None:
                    gen_kwargs["top_k"] = top_k
                elif hasattr(self.cfg, "top_k"):
                    gen_kwargs["top_k"] = int(self.cfg.top_k)

            if repeat_penalty is not None and repeat_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = repeat_penalty
            elif hasattr(self.cfg, "repeat_penalty"):
                rp = float(self.cfg.repeat_penalty)
                if rp != 1.0:
                    gen_kwargs["repetition_penalty"] = rp

            # Add stopping criteria for stop tokens
            if stop:
                from transformers import StoppingCriteria, StoppingCriteriaList

                class StopOnTokens(StoppingCriteria):
                    def __init__(self, stop_strings: tuple[str, ...], tokenizer: Any) -> None:
                        self.stop_strings = stop_strings
                        self.tokenizer = tokenizer

                    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                        generated = self.tokenizer.decode(input_ids[0][-20:])
                        return any(s in generated for s in self.stop_strings)

                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokens(stop, self._tokenizer)])

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)

            # Decode (only new tokens)
            input_len = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][input_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Handle stop tokens (belt and suspenders)
            for stop_token in stop:
                if stop_token in text:
                    text = text.split(stop_token)[0]
                    break

            return text.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def unload(self) -> None:
        """Unload model from memory."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None

            # Clear CUDA cache
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            self._init_attempted = False
