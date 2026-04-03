from __future__ import annotations


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


def _build_prompt(cfg: object, system: str, user: str) -> str:
    style = str(getattr(cfg, "prompt_style", "") or "").strip().lower().replace("-", "_")
    builder = _PROMPT_BUILDERS.get(style, _mistral_instruct_prompt)
    return builder(system, user)
