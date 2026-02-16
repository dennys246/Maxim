"""Posterior Parietal Lobe — Mathematical Cognition Module.

Provides two complementary number systems:

  **IPS (Intraparietal Sulcus)** — Approximate Number System (ANS).
  Fast magnitude comparison, trend detection, anomaly detection.

  **Angular Gyrus** — Exact computation + math memory (MemoryLayer).
  Precise arithmetic, statistical analysis, dual verbal/code output.

Plus supporting infrastructure:

  **NumericalWorkspace** — Session-scoped working memory for agents.
  **MathTool** — ToolRegistry entry routing to IPS or Angular Gyrus.
"""

from maxim.math.angular_gyrus import AngularGyrus, AngularGyrusConfig
from maxim.math.ips import IPS, IPSConfig
from maxim.math.math_types import CompressedMathMemory, MathMemory
from maxim.math.types import (
    AnalysisResult,
    AnomalyResult,
    ApproximateResult,
    ComparisonResult,
    ExactResult,
    MagnitudeCategory,
    MathCategory,
    NumberSystem,
    NumericalValue,
    PatternType,
    RandomnessResult,
    TrendDirection,
    TrendResult,
)
from maxim.math.workspace import NumericalWorkspace, WorkspaceConfig

# Linear algebra (lazy access via maxim.math.linalg)
from maxim.math import linalg as linalg  # noqa: F401

__all__ = [
    # Core modules
    "IPS",
    "IPSConfig",
    "AngularGyrus",
    "AngularGyrusConfig",
    "NumericalWorkspace",
    "WorkspaceConfig",
    # Record types
    "MathMemory",
    "CompressedMathMemory",
    # Result types
    "ApproximateResult",
    "ExactResult",
    "TrendResult",
    "AnomalyResult",
    "AnalysisResult",
    "RandomnessResult",
    # Enums
    "NumberSystem",
    "MagnitudeCategory",
    "ComparisonResult",
    "TrendDirection",
    "MathCategory",
    "PatternType",
    # Value container
    "NumericalValue",
    # Linear algebra submodule
    "linalg",
]
