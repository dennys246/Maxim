"""#818 -- an inherent-class cluster bias does not decay while the agent is offline.

``NAc.decay_cluster_reward_biases`` (per tick) skips ``_inherent_bias_keys``: a Queen-curated bias is
decay-EXEMPT ("innate fears do not extinguish the way learned ones do"). ``apply_wall_clock_decay`` --
what ``NAc.load()`` runs over the time since ``saved_at`` -- decayed every cluster bias, so an inherent one
halved for every day the agent was off (``cluster_bias_wall_decay_half_life_s`` = 1 day).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from maxim.decisions.nac import NAc, NACConfig

DAY = 86400.0
INHERENT = ("agent", "cluster-hot", "touch")
LEARNED = ("agent", "cluster-cold", "touch")


def _nac_with_biases(path: str | None = None) -> NAc:
    nac = NAc(NACConfig(persistence_path=path))
    nac._cluster_reward_bias[INHERENT] = -0.8
    nac._cluster_reward_bias[LEARNED] = -0.8
    nac.mark_inherent_bias(*INHERENT)
    return nac


def test_wall_clock_decay_leaves_an_inherent_bias_exactly_as_it_was() -> None:
    nac = _nac_with_biases()
    nac.apply_wall_clock_decay(3 * DAY)
    assert nac._cluster_reward_bias[INHERENT] == -0.8  # exempt, like the per-tick decay
    assert nac._cluster_reward_bias[LEARNED] == pytest.approx(-0.8 * 0.5**3)  # learned ones still age


def test_wall_clock_decay_never_prunes_an_inherent_bias() -> None:
    """Long enough to prune a learned bias of the same size: the inherent one stays, marker and all."""
    nac = _nac_with_biases()
    result = nac.apply_wall_clock_decay(30 * DAY)
    assert LEARNED not in nac._cluster_reward_bias
    assert result.get("cluster_reward_bias_pruned") == 1
    assert nac._cluster_reward_bias[INHERENT] == -0.8
    assert INHERENT in nac.inherent_bias_keys


def test_an_agent_off_for_three_days_reloads_its_inherent_bias_intact(tmp_path: Path) -> None:
    """The path the defect lived on: save, then NAc.load() with decay-on-load across a real gap."""
    path = tmp_path / "nac.json"
    _nac_with_biases(str(path)).save()
    data = json.loads(path.read_text())
    data["saved_at"] = time.time() - 3 * DAY
    path.write_text(json.dumps(data))
    reloaded = NAc(NACConfig(persistence_path=str(path)))
    reloaded.load()
    assert reloaded._cluster_reward_bias[INHERENT] == pytest.approx(-0.8)
    assert reloaded._cluster_reward_bias[LEARNED] == pytest.approx(-0.8 * 0.5**3, rel=1e-3)
    assert INHERENT in reloaded.inherent_bias_keys
