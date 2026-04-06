"""Maxim package."""

__version__ = "0.1.0"


def get_version_info() -> dict[str, str]:
    """Return version + git hash for debug/version endpoint."""
    import subprocess

    info: dict[str, str] = {"version": __version__}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            info["git_hash"] = result.stdout.strip()
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            info["git_message"] = result.stdout.strip()
    except Exception:
        pass
    return info
