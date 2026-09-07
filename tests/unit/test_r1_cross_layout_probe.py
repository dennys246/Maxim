"""Guard test for the R1 cross-layout structural-null probe.

R1's finding is a STRUCTURAL null: a want taught at layout S1 reads out
learned-bias-decisive at S1 (and at a same-cluster perturbation) but COLLAPSES
at a genuinely distinct layout, because the cluster-keyed learned-bias readout is
an exact-key lookup with no cross-cluster similarity channel. This pins that
mechanism through the real ingest + recommend_action path, offline (no bridge, no
LLM). If a 1.3 build adds a generalization channel, `at_S_distinct` flips to
decisive and this test fails — the signal R1 becomes a live experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import r1_cross_layout_probe as R1  # noqa: E402


def test_cache_confirmed_structural_null(tmp_path):
    rc = R1.main(["--allow-dirty", "--out", str(tmp_path / "r1.json")])
    assert rc == 0
    import json

    rec = json.loads((tmp_path / "r1.json").read_text())
    assert rec["verdict"] == "CACHE-CONFIRMED"
    assert rec["cache_confirmed"] is True
    ro = rec["readout"]
    # Control: the taught want fires at the trained layout.
    assert ro["at_S1"]["bias_decisive"] is True
    # The null: it COLLAPSES at a genuinely distinct layout...
    assert ro["at_S_distinct"]["bias_decisive"] is False
    assert rec["layouts_distinct"] is True
    # ...and "transfers" only when the layout is not actually different.
    assert ro["at_S_same_cluster"]["bias_decisive"] is True
    assert rec["same_perturbation_stays_in_cluster"] is True


def test_distinct_layout_is_a_different_cluster(tmp_path):
    # The load-bearing premise: the distinct layout really maps to a DIFFERENT
    # cluster id than S1 (else "collapse" would be a vacuous same-cluster read).
    rc = R1.main(["--allow-dirty", "--out", str(tmp_path / "r1.json")])
    assert rc == 0
    import json

    rec = json.loads((tmp_path / "r1.json").read_text())
    s1 = rec["readout"]["at_S1"]["cluster_id"]
    sk = rec["readout"]["at_S_distinct"]["cluster_id"]
    ss = rec["readout"]["at_S_same_cluster"]["cluster_id"]
    assert sk != s1  # distinct layout -> distinct cluster (the interesting case)
    assert ss == s1  # perturbation -> same cluster (the vacuous case)
