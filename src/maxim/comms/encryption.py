"""Memory encryption at rest — middleware approach.

All memory layers persist to disk and may contain sensitive data.
Rather than modifying save/load in each layer, this module provides
``save_encrypted`` / ``load_encrypted`` as drop-in replacements for
``json.dump`` / ``json.load`` with transparent AES-256-GCM encryption.

Key from env var ``MAXIM_MEMORY_KEY`` (base64-encoded AES-256 key).
Falls back to unencrypted JSON if no key is configured (dev mode).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_memory_key() -> bytes | None:
    """Get the AES key from environment, or None if not configured."""
    key_str = os.environ.get("MAXIM_MEMORY_KEY")
    if not key_str:
        return None
    try:
        import base64

        return base64.b64decode(key_str)
    except Exception:
        logger.warning("Invalid MAXIM_MEMORY_KEY — falling back to unencrypted")
        return None


def save_encrypted(data: dict[str, Any], path: str) -> None:
    """Encrypt and write memory data. Key from env var, never on disk.

    Middleware: All memory layers call this instead of their own
    json.dump(). Each layer serializes to dict as before — this
    function handles the encryption transparently.
    """
    plaintext = json.dumps(data).encode()
    key = _get_memory_key()

    if key:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import (
                AESGCM,
            )

            nonce = os.urandom(12)
            cipher = AESGCM(key)
            ciphertext = cipher.encrypt(nonce, plaintext, None)
            Path(path).write_bytes(nonce + ciphertext)
        except ImportError:
            logger.warning(
                "cryptography package not installed — saving unencrypted"
            )
            Path(path).write_bytes(plaintext)
    else:
        Path(path).write_bytes(plaintext)

    # Owner read/write only
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # Windows doesn't support Unix permissions


def load_encrypted(path: str) -> dict[str, Any]:
    """Read and decrypt memory data. Auto-detects encrypted vs plain."""
    raw = Path(path).read_bytes()
    key = _get_memory_key()

    if key:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import (
                AESGCM,
            )

            nonce, ciphertext = raw[:12], raw[12:]
            cipher = AESGCM(key)
            plaintext = cipher.decrypt(nonce, ciphertext, None)
            return json.loads(plaintext)
        except ImportError:
            pass  # No cryptography package — try plain
        except Exception:
            pass  # May be unencrypted legacy file — try plain

    return json.loads(raw)
