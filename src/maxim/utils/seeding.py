"""Global deterministic seeding for reproducible simulation runs.

S4 of simulator_upgrades_plan. Provides a single entry point to seed
all RNG sources (stdlib random, numpy, torch, PYTHONHASHSEED) from one
integer seed. Also provides per-agent RNG derivation to prevent
cross-agent decision correlation in multi-agent sims.

Usage:
    from maxim.utils.seeding import seed_all, make_agent_rng

    seed_all(42)  # Call once at startup before heavy imports
    rng = make_agent_rng(42, "agent_0")  # Per-agent RNG
"""

from __future__ import annotations

import hashlib
import logging
import os
import random

logger = logging.getLogger(__name__)

# Module-level flag to track whether seeding has been applied
_global_seed: int | None = None


def seed_all(seed: int) -> None:
    """Seed all RNG sources for deterministic execution.

    Should be called as early as possible in the process — before any
    imports that construct RNG state (numpy, torch, etc.).

    Sets:
        - os.environ["PYTHONHASHSEED"] (must be set before interpreter start
          for full effect, but we set it anyway for subprocess propagation)
        - random.seed()
        - numpy.random.seed() (if numpy is importable)
        - torch.manual_seed() + torch.cuda.manual_seed_all() (if torch is importable)
    """
    global _global_seed
    _global_seed = seed

    # PYTHONHASHSEED — set for subprocess propagation. Note: changing this
    # after interpreter start doesn't affect the current process's hash()
    # behavior, but child processes (like S3's persistence_child) will pick
    # it up.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # stdlib random
    random.seed(seed)

    # numpy
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    # torch (optional — only installed with llm-torch extra)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logger.info("Global seed set: %d", seed)


def get_global_seed() -> int | None:
    """Return the global seed if set, else None."""
    return _global_seed


def make_agent_rng(base_seed: int, agent_id: str) -> random.Random:
    """Create a per-agent RNG stream derived from base seed + agent_id.

    Each agent gets a deterministic but uncorrelated RNG stream. This
    prevents cross-agent decision correlation that would look like
    information leak even without any shared state.

    The derivation uses SHA-256 to mix seed + agent_id into a new seed,
    avoiding simple seed+offset patterns that can produce correlated
    subsequences in linear congruential generators.

    Args:
        base_seed: The global seed from --seed.
        agent_id: Unique agent identifier (e.g., "agent_0", "npc_bartender").

    Returns:
        A random.Random instance seeded deterministically from the inputs.
    """
    # Mix seed and agent_id via SHA-256 for uncorrelated derivation
    h = hashlib.sha256(f"{base_seed}:{agent_id}".encode()).hexdigest()
    derived_seed = int(h[:16], 16)  # Use first 64 bits
    rng = random.Random(derived_seed)
    logger.debug("Agent RNG: agent_id=%s derived_seed=%d", agent_id, derived_seed)
    return rng
