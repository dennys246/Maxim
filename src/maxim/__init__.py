"""Maxim — bio-inspired cognitive architecture.

Public API (verb-based):

    import maxim

    maxim.configure(verbosity=2)
    maxim.run(model="mistral-7b")
    maxim.imagine(goal="test safety", persona="adversarial")
    maxim.connect("simulated")
    maxim.diagnose()
    maxim.observe("memory")
    maxim.introspect("causal")   # alias for observe
"""

__version__ = "0.2.0"

# Verb-based public API — lazy-loaded to keep ``import maxim`` fast.
_API_VERBS = frozenset(
    {
        "configure",
        "connect",
        "diagnose",
        "imagine",
        "introspect",
        "observe",
        "run",
        # Phase 8 additions:
        "campaign",
        "benchmark",
        "research",
        "on",
        "register_tool",
        "register_persona",
        "tool",
    }
)

# Also expose key types for library users who need them directly.
_API_TYPES = {
    "DiagnosticReport": "maxim.api",
    "CampaignResult": "maxim.api",
    "BenchmarkResult": "maxim.api",
    "ResearchResult": "maxim.api",
    "EventHandle": "maxim.api",
    # Error hierarchy — users can catch these
    "MaximError": "maxim.exceptions",
    "ConfigurationError": "maxim.exceptions",
    "ModelError": "maxim.exceptions",
    "ModelLoadError": "maxim.exceptions",
    "ToolExecutionError": "maxim.exceptions",
}

__all__ = [
    "__version__",
    "get_version_info",
    # Original verbs
    "configure",
    "connect",
    "diagnose",
    "imagine",
    "introspect",
    "observe",
    "run",
    # Phase 8 verbs
    "campaign",
    "benchmark",
    "research",
    "on",
    "register_tool",
    "register_persona",
    "tool",
    # Types
    "DiagnosticReport",
    "CampaignResult",
    "BenchmarkResult",
    "ResearchResult",
    "EventHandle",
    # Errors
    "MaximError",
    "ConfigurationError",
    "ModelError",
    "ModelLoadError",
    "ToolExecutionError",
]


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


def __getattr__(name: str):
    """Lazy-load API verbs and types on first access."""
    if name in _API_VERBS:
        from maxim import api

        func = getattr(api, name)
        globals()[name] = func  # Cache for subsequent calls
        return func
    if name in _API_TYPES:
        import importlib

        mod = importlib.import_module(_API_TYPES[name])
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'maxim' has no attribute {name!r}")
