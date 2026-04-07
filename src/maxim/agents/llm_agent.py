"""LLM Agent - A flexible agent powered by llama.cpp with easy model switching."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from maxim.agents.base import Agent
from maxim.models.language.router import (
    DEFAULT_QUANTIZATION,
    LLMConfig,
    QUANTIZATION_LEVELS,
    _LlamaCppBackend,
    _build_prompt,
    _extract_json_object,
    build_model_path,
    list_llm_profiles,
    list_prompt_styles,
    list_quantization_levels,
    load_llm_config,
)
from maxim.utils.logging import warn


@dataclass(frozen=True)
class LLMAgentConfig:
    """Configuration for LLMAgent with sensible defaults."""

    # Model selection
    profile: str = "mistral-7b-instruct-v0.2"
    quantization: str = DEFAULT_QUANTIZATION  # Default Q4_K_M
    model_path: str | None = None  # Override auto-generated path

    # Generation parameters
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1

    # System prompt for the agent
    system_prompt: str = "You are a helpful AI assistant."

    # Hardware settings
    n_gpu_layers: int = -1  # -1 = auto (use all available)
    n_threads: int | None = None  # None = auto-detect
    n_ctx: int = 4096

    # Behavior
    enabled: bool = True
    json_mode: bool = False  # If True, parse responses as JSON


class LLMAgent(Agent):
    """
    A flexible LLM-powered agent using llama.cpp.

    Features:
    - Easy model switching via profiles or direct path
    - Quantization support (defaults to Q4_K_M)
    - Thread-safe lazy initialization
    - Configurable generation parameters
    - Optional JSON mode for structured outputs

    Example usage:
        # Use default Mistral 7B with Q4 quantization
        agent = LLMAgent()

        # Use a specific profile
        agent = LLMAgent(profile="llama3-8b")

        # Use custom quantization
        agent = LLMAgent(profile="mistral-7b", quantization="Q5_K_M")

        # Full custom config
        config = LLMAgentConfig(
            profile="phi3-mini",
            quantization="Q4_K_M",
            temperature=0.8,
            system_prompt="You are a coding assistant.",
        )
        agent = LLMAgent(config=config)
    """

    agent_name = "llm_agent"

    # Sentinel for failed initialization
    _INIT_FAILED = object()

    def __init__(
        self,
        *,
        name: str | None = None,
        enabled: bool = True,
        config: LLMAgentConfig | None = None,
        profile: str | None = None,
        quantization: str | None = None,
        model_path: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> None:
        """
        Initialize the LLM agent.

        Args:
            name: Agent name (optional)
            enabled: Whether the agent is enabled
            config: Full LLMAgentConfig (overrides other params if provided)
            profile: Model profile to use (e.g., "mistral-7b", "llama3-8b")
            quantization: Quantization level (default: Q4_K_M)
            model_path: Direct path to model file (overrides profile)
            system_prompt: System prompt for the agent
            temperature: Generation temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            json_mode: If True, parse responses as JSON
        """
        super().__init__(name=name, enabled=enabled)

        # Build config from parameters or use provided config
        if config is not None:
            self._agent_config = config
        else:
            self._agent_config = LLMAgentConfig(
                profile=profile or "mistral-7b-instruct-v0.2",
                quantization=quantization or DEFAULT_QUANTIZATION,
                model_path=model_path,
                system_prompt=system_prompt or "You are a helpful AI assistant.",
                temperature=temperature if temperature is not None else 0.7,
                max_tokens=max_tokens if max_tokens is not None else 512,
                json_mode=json_mode,
                enabled=enabled,
            )

        self._backend: _LlamaCppBackend | None = None
        self._backend_lock = threading.Lock()
        self._llm_config: LLMConfig | None = None

        # Conversation history for multi-turn support
        self._history: list[dict[str, str]] = []
        self._max_history: int = 10

    def _get_llm_config(self) -> LLMConfig:
        """Build LLMConfig from agent config."""
        if self._llm_config is not None:
            return self._llm_config

        # Load base config (reads from files/env)
        load_llm_config()

        # Determine model path
        model_path = self._agent_config.model_path
        if not model_path:
            # Build from profile and quantization
            from maxim.models.language.router import _BUILTIN_PROFILES, _normalize_profile

            profile = _normalize_profile(self._agent_config.profile)
            builtin = _BUILTIN_PROFILES.get(profile, {})
            model_base = builtin.get("model_base", profile)
            model_path = build_model_path(model_base, self._agent_config.quantization)

        # Get prompt style from profile
        from maxim.models.language.router import _BUILTIN_PROFILES, _normalize_profile

        profile = _normalize_profile(self._agent_config.profile)
        builtin = _BUILTIN_PROFILES.get(profile, {})
        prompt_style = builtin.get("prompt_style", "mistral_instruct")
        stop = tuple(builtin.get("stop", ["</s>"]))
        n_ctx = builtin.get("n_ctx", self._agent_config.n_ctx)

        self._llm_config = LLMConfig(
            enabled=self._agent_config.enabled,
            backend="llama_cpp",
            profile=profile,
            model=profile,
            model_base=builtin.get("model_base", profile),
            model_path=model_path,
            quantization=self._agent_config.quantization,
            prompt_style=prompt_style,
            stop=stop,
            n_ctx=n_ctx,
            max_tokens=self._agent_config.max_tokens,
            temperature=self._agent_config.temperature,
            top_p=self._agent_config.top_p,
            top_k=self._agent_config.top_k,
            repeat_penalty=self._agent_config.repeat_penalty,
            n_gpu_layers=self._agent_config.n_gpu_layers,
            n_threads=self._agent_config.n_threads,
        )
        return self._llm_config

    def _get_backend(self) -> _LlamaCppBackend | None:
        """Get or create the LLM backend (thread-safe)."""
        if self._backend is not None:
            if self._backend is LLMAgent._INIT_FAILED:
                return None
            return self._backend

        with self._backend_lock:
            if self._backend is not None:
                if self._backend is LLMAgent._INIT_FAILED:
                    return None
                return self._backend

            if not self._agent_config.enabled:
                self._backend = LLMAgent._INIT_FAILED
                return None

            try:
                cfg = self._get_llm_config()
                self._backend = _LlamaCppBackend(cfg)
                return self._backend
            except Exception as e:
                warn("Failed to initialize LLM backend: %s", e)
                self._backend = LLMAgent._INIT_FAILED
                return None

    def generate(
        self,
        user_input: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        add_to_history: bool = True,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            user_input: The user's input/question
            system_prompt: Override the default system prompt
            temperature: Override the default temperature
            max_tokens: Override the default max tokens
            add_to_history: Whether to add this exchange to history

        Returns:
            The generated response text
        """
        backend = self._get_backend()
        if backend is None:
            return ""

        cfg = self._get_llm_config()
        sys_prompt = system_prompt or self._agent_config.system_prompt

        # Build prompt
        prompt = _build_prompt(cfg, sys_prompt, user_input)

        try:
            response = backend.complete(
                prompt,
                max_tokens=max_tokens or self._agent_config.max_tokens,
                temperature=temperature if temperature is not None else self._agent_config.temperature,
                stop=cfg.stop,
            )

            if add_to_history:
                self._history.append({"role": "user", "content": user_input})
                self._history.append({"role": "assistant", "content": response})
                # Trim history if needed
                while len(self._history) > self._max_history * 2:
                    self._history.pop(0)
                    self._history.pop(0)

            return response.strip()
        except Exception as e:
            warn("LLM generation failed: %s", e)
            return ""

    def generate_json(
        self,
        user_input: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Generate a JSON response from the LLM.

        Args:
            user_input: The user's input/question
            system_prompt: Override the default system prompt
            temperature: Override the default temperature
            max_tokens: Override the default max tokens

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        # Enhance system prompt for JSON output
        sys_prompt = system_prompt or self._agent_config.system_prompt
        json_sys_prompt = f"{sys_prompt}\n\nIMPORTANT: Respond ONLY with valid JSON. No prose, no explanations."

        response = self.generate(
            user_input,
            system_prompt=json_sys_prompt,
            temperature=temperature if temperature is not None else 0.0,  # Lower temp for JSON
            max_tokens=max_tokens,
            add_to_history=False,
        )

        if not response:
            return None

        return _extract_json_object(response)

    def propose_intent(self, state: Any, memory: Any, **kwargs: Any) -> dict[str, Any] | None:
        """
        Propose an intent based on state (Agent interface).

        Override this method in subclasses for custom behavior.
        """
        # Default implementation - can be overridden
        return None

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    def get_history(self) -> list[dict[str, str]]:
        """Get the conversation history."""
        return list(self._history)

    def switch_model(
        self,
        profile: str | None = None,
        quantization: str | None = None,
        model_path: str | None = None,
    ) -> bool:
        """
        Switch to a different model.

        Args:
            profile: New model profile
            quantization: New quantization level
            model_path: Direct path to new model

        Returns:
            True if switch was successful
        """
        with self._backend_lock:
            # Unload current model
            if self._backend is not None and self._backend is not LLMAgent._INIT_FAILED:
                self._backend.unload()

            # Update config (frozen dataclass — use replace)
            import dataclasses as _dc

            replacements: dict[str, Any] = {}
            if profile:
                replacements["profile"] = profile
            if quantization:
                replacements["quantization"] = quantization
            if model_path:
                replacements["model_path"] = model_path
            if replacements:
                self._agent_config = _dc.replace(self._agent_config, **replacements)

            # Reset for re-initialization
            self._backend = None
            self._llm_config = None

            # Try to initialize with new config
            backend = self._get_backend()
            return backend is not None

    def on_stop(self, **kwargs: Any) -> None:
        """Clean up when agent stops."""
        with self._backend_lock:
            if self._backend is not None and self._backend is not LLMAgent._INIT_FAILED:
                self._backend.unload()
            self._backend = None

    @staticmethod
    def list_available_profiles() -> list[str]:
        """List available model profiles."""
        return list_llm_profiles()

    @staticmethod
    def list_quantization_levels() -> list[str]:
        """List available quantization levels."""
        return list_quantization_levels()

    @staticmethod
    def list_prompt_styles() -> list[str]:
        """List available prompt styles."""
        return list_prompt_styles()

    @staticmethod
    def get_quantization_info(level: str) -> dict[str, Any] | None:
        """Get info about a quantization level."""
        normalized = str(level or "").strip().upper().replace("-", "_")
        return QUANTIZATION_LEVELS.get(normalized)


class ChatLLMAgent(LLMAgent):
    """
    An LLM agent optimized for chat/conversation.

    Includes conversation history in prompts for multi-turn dialogue.
    """

    agent_name = "chat_llm_agent"

    def generate(
        self,
        user_input: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        add_to_history: bool = True,
    ) -> str:
        """Generate a response with conversation history context."""
        backend = self._get_backend()
        if backend is None:
            return ""

        cfg = self._get_llm_config()
        sys_prompt = system_prompt or self._agent_config.system_prompt

        # Build user input with history context
        history_text = ""
        if self._history:
            for msg in self._history[-self._max_history * 2 :]:
                role = msg["role"].capitalize()
                content = msg["content"]
                history_text += f"{role}: {content}\n"
            history_text += f"User: {user_input}"
        else:
            history_text = user_input

        prompt = _build_prompt(cfg, sys_prompt, history_text)

        try:
            response = backend.complete(
                prompt,
                max_tokens=max_tokens or self._agent_config.max_tokens,
                temperature=temperature if temperature is not None else self._agent_config.temperature,
                stop=cfg.stop,
            )

            if add_to_history:
                self._history.append({"role": "user", "content": user_input})
                self._history.append({"role": "assistant", "content": response.strip()})
                while len(self._history) > self._max_history * 2:
                    self._history.pop(0)
                    self._history.pop(0)

            return response.strip()
        except Exception as e:
            warn("LLM generation failed: %s", e)
            return ""


class TaskLLMAgent(LLMAgent):
    """
    An LLM agent optimized for task execution.

    Parses LLM output as structured actions/intents.
    """

    agent_name = "task_llm_agent"

    def __init__(
        self,
        *,
        name: str | None = None,
        enabled: bool = True,
        config: LLMAgentConfig | None = None,
        allowed_tools: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, config=config, json_mode=True, **kwargs)
        self.allowed_tools = allowed_tools or set()

    def propose_intent(self, state: Any, memory: Any, **kwargs: Any) -> dict[str, Any] | None:
        """
        Propose an intent based on state.

        Queries the LLM to determine the next action based on current state.
        """
        if not self.allowed_tools:
            return None

        # Extract relevant state info
        data = getattr(state, "data", {}) if hasattr(state, "data") else {}

        tools_list = ", ".join(sorted(self.allowed_tools))
        state_summary = str(data)[:1000]  # Truncate for context

        user_prompt = f"""Current state:
{state_summary}

Available tools: {tools_list}

Based on the current state, what action should be taken?
Return JSON: {{"tool_name": "...", "params": {{...}}, "confidence": 0.0-1.0}}
If no action needed, return: {{"tool_name": null, "confidence": 0.0}}"""

        result = self.generate_json(user_prompt, temperature=0.0)

        if not result or not isinstance(result, dict):
            return None

        tool_name = result.get("tool_name")
        if not tool_name or tool_name not in self.allowed_tools:
            return None

        return {
            "goal": {
                "tool_name": tool_name,
                "params": result.get("params", {}),
            },
            "confidence": float(result.get("confidence", 1.0)),
        }
