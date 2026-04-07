"""Peer-side config: where to find the leader + credentials.

Stored at ~/.config/maxim/peer.yml on POSIX, %APPDATA%\\maxim\\peer.yml on
Windows. Mirrors the layout of the leader's ~/.config/maxim/api_key file.

Fields:
  url          — leader's OpenAI-compatible endpoint (e.g., tunnel URL or LAN IP)
  api_key      — Bearer token shared by the leader
  model        — optional; model name to send (defaults to whatever /v1/models returns)
  is_cloud     — optional; forces MAXIM_MAX_CLOUD_LANES=1 for public URLs

`build_primary_router` consults this file at startup and uses its values
as defaults for the infer-lane remote config. Env vars (MAXIM_LANE_INFER_*)
still override these — so users can do per-session tweaks without touching
the file.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path


def peer_config_path() -> Path:
    """Return the platform-appropriate peer config path."""
    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "maxim" / "peer.yml"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "maxim" / "peer.yml"


@dataclass(frozen=True)
class PeerConfig:
    url: str
    api_key: str
    model: str | None = None
    is_cloud: bool = False

    def to_yaml(self) -> str:
        """Minimal YAML serialization (stable field order, no PyYAML dep)."""
        lines = [
            f"url: {self.url}",
            f"api_key: {self.api_key}",
        ]
        if self.model:
            lines.append(f"model: {self.model}")
        if self.is_cloud:
            lines.append("is_cloud: true")
        return "\n".join(lines) + "\n"


def read_peer_config() -> PeerConfig | None:
    """Load the peer config, or None if missing / malformed."""
    path = peer_config_path()
    if not path.is_file():
        return None
    try:
        content = path.read_text()
    except OSError:
        return None
    fields: dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition(":")
        fields[key.strip()] = value.strip()
    url = fields.get("url", "")
    api_key = fields.get("api_key", "")
    if not url or not api_key:
        return None
    is_cloud = fields.get("is_cloud", "").lower() in ("true", "1", "yes")
    model = fields.get("model") or None
    return PeerConfig(url=url, api_key=api_key, model=model, is_cloud=is_cloud)


def write_peer_config(cfg: PeerConfig) -> Path:
    """Persist a peer config with 0600 perms on POSIX."""
    path = peer_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg.to_yaml())
    if platform.system() != "Windows":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return path


def delete_peer_config() -> bool:
    """Remove the peer config file. Returns True if a file was deleted."""
    path = peer_config_path()
    if not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


def apply_peer_config_to_env(cfg: PeerConfig) -> None:
    """Set MAXIM_LANE_INFER_* env vars from a peer config, without overwriting
    any that are already set (env wins over file for per-session overrides)."""
    os.environ.setdefault("MAXIM_LANE_INFER_REMOTE_URL", cfg.url)
    os.environ.setdefault("MAXIM_LANE_INFER_REMOTE_API_KEY", cfg.api_key)
    if cfg.model:
        os.environ.setdefault("MAXIM_LANE_INFER_REMOTE_MODEL", cfg.model)
    # Peer connections are to your own infrastructure — even if the URL
    # resolves to a public IP (Cloudflare tunnel), it's not a cloud
    # provider. Mark it as peer-owned so lane_backends treats it as
    # "self-hosted" (no cloud gate, no redaction policy required).
    os.environ.setdefault("MAXIM_MAX_CLOUD_LANES", "1")


def truncate_key(key: str, *, keep: int = 6) -> str:
    if len(key) <= keep * 2 + 1:
        return key
    return f"{key[:keep]}…{key[-keep:]}"


__all__ = [
    "PeerConfig",
    "peer_config_path",
    "read_peer_config",
    "write_peer_config",
    "delete_peer_config",
    "apply_peer_config_to_env",
    "truncate_key",
]
