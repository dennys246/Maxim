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

__version__ = "0.7.0"

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
        "list_models",
        "download_model",
        "delete_model",
        # (get_session/list_sessions removed — use maxim.load.session/sessions)
    }
)

# Sub-module namespaces — lazy-loaded as ``maxim.create``, ``maxim.load``.
_API_SUBMODULES = {
    "create": "maxim.create",
    "load": "maxim.load",
}

# Also expose key types for library users who need them directly.
_API_TYPES = {
    "DiagnosticReport": "maxim.api",
    "CampaignResult": "maxim.api",
    "BenchmarkResult": "maxim.api",
    "ResearchResult": "maxim.api",
    "EventHandle": "maxim.api",
    "ModelInfo": "maxim.api",
    # Event payload types (for on() callbacks)
    "ToolCallEvent": "maxim.api",
    "MemoryCaptureEvent": "maxim.api",
    "PainSignalEvent": "maxim.api",
    "PromptEvent": "maxim.api",
    # Error hierarchy — category-level exceptions users can catch
    "MaximError": "maxim.exceptions",
    "ConfigurationError": "maxim.exceptions",
    "MaximConnectionError": "maxim.exceptions",
    "ModelError": "maxim.exceptions",
    "ModelLoadError": "maxim.exceptions",
    "ToolExecutionError": "maxim.exceptions",
    "ToolNotFoundError": "maxim.exceptions",
    "MaximMemoryError": "maxim.exceptions",
    "PlanningError": "maxim.exceptions",
    "HardwareError": "maxim.exceptions",
    "MaximRuntimeError": "maxim.exceptions",
    # Return types from API verbs
    "RobotController": "maxim.hardware.controller",
    # SEM types — stable public types for embodiment composition
    "Entity": "maxim.embodiment.sem",
    # Session + Report types
    "Session": "maxim.session",
    "Report": "maxim.report",
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
    "list_models",
    "download_model",
    "delete_model",
    # Types
    "DiagnosticReport",
    "CampaignResult",
    "BenchmarkResult",
    "ResearchResult",
    "EventHandle",
    "ModelInfo",
    # Event payload types
    "ToolCallEvent",
    "MemoryCaptureEvent",
    "PainSignalEvent",
    "PromptEvent",
    "Session",
    "Report",
    # SEM types (Entity is constructable; Sensor/Modulator are Protocols — import from maxim.embodiment.sem)
    "Entity",
    # Errors (category-level)
    "MaximError",
    "ConfigurationError",
    "MaximConnectionError",
    "ModelError",
    "ModelLoadError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "MaximMemoryError",
    "PlanningError",
    "HardwareError",
    "MaximRuntimeError",
    # Return types
    "RobotController",
    # Sub-module namespaces
    "create",
    "load",
]


def get_version_info() -> dict[str, str]:
    """Return version + git hash for debug/version endpoint."""
    import os

    info: dict[str, str] = {"version": __version__}

    # Guard: only shell out to git if we're in a repo (PyPI installs won't have .git)
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(os.path.dirname(_pkg_dir))  # src/maxim -> src -> repo
    if not os.path.isdir(os.path.join(_repo_root, ".git")):
        return info

    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=_repo_root,
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
            cwd=_repo_root,
        )
        if result.returncode == 0:
            info["git_message"] = result.stdout.strip()
    except Exception:
        pass
    return info


def __dir__():
    """Expose all public names for tab-completion and dir()."""
    return sorted(set(__all__) | set(globals().keys()))


def __getattr__(name: str):
    """Lazy-load API verbs, types, and sub-modules on first access."""
    if name in _API_VERBS:
        from maxim import api

        func = getattr(api, name)
        globals()[name] = func  # Cache for subsequent calls
        return func
    if name in _API_SUBMODULES:
        import importlib

        mod = importlib.import_module(_API_SUBMODULES[name])
        globals()[name] = mod
        return mod
    if name in _API_TYPES:
        import importlib

        mod = importlib.import_module(_API_TYPES[name])
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'maxim' has no attribute {name!r}")
