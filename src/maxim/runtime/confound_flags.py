"""Opt-in disable flags for default-on prompt/state scaffolds.

Centralizes the env-var gates used by the V1 substrate-attribution
phased re-run (per ``docs/plans/confound_quarantine.md``). Each flag
is read through one helper here so:

1. Env-var names are typo-safe (one source of truth, ``ALL_FLAGS``).
2. ``tests/conftest.py`` can iterate ``ALL_FLAGS`` for the autouse
   scrub fixture — adding a flag here automatically scrubs it in
   tests, per ``feedback_opt_in_env_in_hot_paths.md``.
3. There is one grep target ("scaffold-disable gates for V1
   attribution") for the audit surface.

All four named flags are debug/experimental per CC4. Defaults preserve
current behavior — when the env var is unset, every gated injector
fires exactly as today. Setting ``MAXIM_DISABLE_<NAME>=1`` (or
``MAXIM_NO_DEFAULT_PERSONA=1``) disables the named scaffold. Per R4
in the plan, this module is scoped exclusively to scaffold-disable
flags whose impact on the V1 attribution claim is being measured.
Unrelated debug toggles do NOT belong here.
"""

from __future__ import annotations

import os

_TRUTHY = ("1", "true", "yes")


def _flag(name: str) -> bool:
    """Return True when ``name`` is set to a truthy value.

    Mirrors the parse semantics of ``maxim.utils.env``-style flag
    helpers: case-insensitive, whitespace-trimmed, accepting
    ``1``/``true``/``yes``.
    """
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def pfc_preamble_enabled() -> bool:
    """True when the PFC deliberation preamble should be injected.

    Gated at ``PromptBuilder._add_pfc_preamble_section``. Unsetting
    ``MAXIM_DISABLE_PFC_PREAMBLE`` (the default) preserves the
    pre-quarantine behavior: the ~1k-token preamble fires whenever any
    bio-signal is present on the request context (sim mode, working
    memory thoughts, causal context, etc.).
    """
    return not _flag("MAXIM_DISABLE_PFC_PREAMBLE")


def acting_coach_enabled() -> bool:
    """True when the Acting Coach + embodied identity rewrite should fire.

    Gated at two sites in ``prompt_builder.py``:

    - ``build_identity_section`` — when False, identity stays the
      generic "robot assistant" string regardless of
      ``request.acting_coach``.
    - ``_add_acting_coach_section`` — when False, the budgeter never
      receives the coach text.

    Also consulted by ``simulation/orchestrator.py`` to suppress the
    ``aut_llm_worker.acting_coach`` attachment so the worker never
    holds the config in the first place.
    """
    return not _flag("MAXIM_DISABLE_ACTING_COACH")


def sim_sandbox_text_enabled() -> bool:
    """True when the "SIMULATION ENVIRONMENT: ..." block should be emitted.

    Gated at ``build_identity_section`` in ``prompt_builder.py``. The
    INTERACTIVE MODE block is unaffected by this flag.
    """
    return not _flag("MAXIM_DISABLE_SIM_SANDBOX_TEXT")


def default_persona_enabled() -> bool:
    """True when an absent ``--persona`` should fall back to ``adversarial``.

    Gated in ``cli.py``'s persona dispatch sites. When False, callers
    should treat absent ``--persona`` as ``None`` (true neutral)
    instead of using ``DEFAULT_PERSONA``.
    """
    return not _flag("MAXIM_NO_DEFAULT_PERSONA")


# Consumed by tests/conftest.py to autouse-scrub every flag in this
# module. Adding a new flag here automatically scrubs it in tests; the
# pin-test pattern in tests/unit/test_confound_flags.py catches a
# refactor that drops a gate site.
ALL_FLAGS: tuple[str, ...] = (
    "MAXIM_DISABLE_PFC_PREAMBLE",
    "MAXIM_DISABLE_ACTING_COACH",
    "MAXIM_DISABLE_SIM_SANDBOX_TEXT",
    "MAXIM_NO_DEFAULT_PERSONA",
)


__all__ = [
    "ALL_FLAGS",
    "acting_coach_enabled",
    "default_persona_enabled",
    "pfc_preamble_enabled",
    "sim_sandbox_text_enabled",
]
