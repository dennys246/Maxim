"""Generate and inspect ~/.cloudflared/config.yml for Maxim's leader-mode tunnel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CONFIG_PATH = Path.home() / ".cloudflared" / "config.yml"


@dataclass(frozen=True)
class TunnelConfig:
    tunnel_id: str
    credentials_file: str
    hostname: str
    local_service: str  # e.g., "http://localhost:8099" (LeaderProxy)


def config_exists() -> bool:
    return CONFIG_PATH.is_file()


def render_config_yml(cfg: TunnelConfig) -> str:
    """Serialize a TunnelConfig to the cloudflared YAML format.

    We emit valid YAML by hand rather than pulling in PyYAML — the format is
    simple and stable, and Maxim already has many optional dependencies.
    """
    return (
        f"tunnel: {cfg.tunnel_id}\n"
        f"credentials-file: {cfg.credentials_file}\n"
        f"ingress:\n"
        f"  - hostname: {cfg.hostname}\n"
        f"    service: {cfg.local_service}\n"
        f"  - service: http_status:404\n"
    )


def write_config_yml(cfg: TunnelConfig, *, overwrite: bool = False) -> Path:
    """Write config.yml to ~/.cloudflared/. Returns the written path.

    If the file exists and overwrite=False, raises FileExistsError.
    The parent directory is created if missing.
    """
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists() and not overwrite:
        raise FileExistsError(
            f"{CONFIG_PATH} already exists. Pass overwrite=True to replace it, or back it up manually first."
        )
    CONFIG_PATH.write_text(render_config_yml(cfg))
    return CONFIG_PATH


def read_config_summary() -> dict[str, str] | None:
    """Return a lightweight dict of the existing config, or None if not present.

    Not a full YAML parser — pulls the fields we care about via line parsing.
    Used by `maxim tunnel status` to show the user what's configured.
    """
    if not config_exists():
        return None
    summary: dict[str, str] = {}
    try:
        for line in CONFIG_PATH.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("tunnel:"):
                summary["tunnel"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("credentials-file:"):
                summary["credentials_file"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("- hostname:"):
                summary.setdefault("hostname", stripped.split(":", 1)[1].strip())
            elif stripped.startswith("service:") and "http" in stripped:
                summary.setdefault("service", stripped.split(":", 1)[1].strip())
    except OSError:
        return None
    return summary or None


__all__ = [
    "CONFIG_PATH",
    "TunnelConfig",
    "config_exists",
    "render_config_yml",
    "write_config_yml",
    "read_config_summary",
]
