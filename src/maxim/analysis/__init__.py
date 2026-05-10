"""Analysis utilities — read-only diff/inspection over persisted session state.

Roy harness component (R2 of 5). See
:mod:`maxim.analysis.substrate_diff` for the substrate divergence
analyzer that compares two sim_reports session directories.
"""

from __future__ import annotations

from maxim.analysis.substrate_diff import (
    AtlDiff,
    EcDiff,
    HippocampusDiff,
    NacDiff,
    SubstrateDiff,
    atl_diff,
    ec_diff,
    hippocampus_diff,
    nac_diff,
    substrate_diff,
)

__all__ = [
    "AtlDiff",
    "EcDiff",
    "HippocampusDiff",
    "NacDiff",
    "SubstrateDiff",
    "atl_diff",
    "ec_diff",
    "hippocampus_diff",
    "nac_diff",
    "substrate_diff",
]
