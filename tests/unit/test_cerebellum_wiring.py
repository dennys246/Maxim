"""The cerebellum trains from the PRODUCTION tool path, not just its own API.

Why this file exists (2026-09-01). `ModulatorAffordanceTool.execute` called
``Cerebellum.observe_from_action(entity_path=..., actual_sensors=...)`` while
the signature is ``entity=`` / ``actual=``. Every call raised ``TypeError``
into an enclosing ``except Exception: log_swallowed_exception()``, so the
forward model **never trained in production** between 2026-04-07 and
2026-09-01 — roughly five months — while
``prompts/acting_coach.py::_compose_cerebellum_predictions`` rendered an
empty section into every prompt.

`tests/integration/test_affordance_transfer.py` passed throughout, because it
calls ``observe_from_action`` **correctly and directly**. That is the lesson:
a unit test of an API proves the API works and says nothing about whether the
one production caller agrees with it. These tests drive the real path.

A second swallow in ``Cerebellum.observe``'s telemetry block hid the same
class of bug: it read ``key.entity`` / ``key.params_hash`` against a comment
describing a five-field ``ModelKey`` that never existed (the real fields are
``entity_path``/``param_bucket``), so ``sim_cerebellum_train`` had never
fired either.
"""

from __future__ import annotations

import pytest

from maxim.embodiment.body import Embodiment
from maxim.embodiment.cerebellum import Cerebellum, ModelKey
from maxim.embodiment.spec import _parse_entity
from maxim.embodiment.tool_bridge import ModulatorAffordanceTool


def _rig():
    body = _parse_entity(
        {
            "name": "agent",
            "entity_type": "body",
            "sensors": {"hunger": {"unit": "ratio", "range": [0, 1], "initial": 0.8}},
        }
    )
    food = _parse_entity(
        {
            "name": "food",
            "entity_type": "item",
            # The target entity MUST carry sensors: the production block is
            # guarded `if self._cerebellum is not None and entity_state:`, and
            # entity_state is the TARGET's readings. The forward model learns
            # "what happens to this object when I act on it" — an entity with
            # no sensors trains nothing, by design.
            "sensors": {"portions": {"unit": "count", "range": [0, 5], "initial": 5.0}},
            "modulators": {
                "nutrition": {
                    "abstract": True,
                    "affordances": {
                        "eat": {"params": {}, "description": "Eat", "self_effect": {"hunger": -0.4}},
                    },
                }
            },
        }
    )
    emb = Embodiment(body)
    mod = food.modulators["nutrition"]
    cereb = Cerebellum()
    tool = ModulatorAffordanceTool(
        food, mod, "eat", mod.affordances["eat"], "food_eat", embodiment=emb, cerebellum=cereb
    )
    return tool, cereb


class TestCerebellumTrainsFromProductionPath:
    def test_executing_an_affordance_trains_the_forward_model(self):
        """The regression. Pre-fix this asserted 0 models after execution."""
        tool, cereb = _rig()
        assert len(cereb._models) == 0

        tool.execute()

        assert len(cereb._models) > 0, (
            "executing an affordance did not train the forward model — the "
            "production call path and Cerebellum.observe_from_action disagree"
        )

    def test_the_trained_key_is_the_real_modelkey_shape(self):
        """Pins the field names whose drift caused the telemetry half."""
        tool, cereb = _rig()
        tool.execute()

        key = next(iter(cereb._models))
        assert isinstance(key, ModelKey)
        assert ModelKey._fields == ("entity_path", "modulator", "affordance", "param_bucket")
        assert key.modulator == "nutrition"
        assert key.affordance == "eat"

    def test_repeated_execution_accumulates_observations(self):
        """Training must actually integrate, not just allocate a model."""
        tool, cereb = _rig()
        for _ in range(3):
            tool.execute()

        model = next(iter(cereb._models.values()))
        assert model.observations >= 3, f"expected >=3 observations, got {model.observations}"

    def test_every_call_site_kwarg_exists_in_the_signature(self):
        """The structural guard that WOULD have caught this.

        Parses the real call in ``tool_bridge`` and checks each keyword
        against the real signature. My first draft of this test compared the
        signature to itself, which is tautological — it passed against the
        broken code. The failure was never in the signature; it was in the
        caller disagreeing with it, so that is what has to be read.
        """
        import ast
        import inspect
        import pathlib as _pl

        src = _pl.Path(inspect.getfile(ModulatorAffordanceTool)).read_text()
        calls = [
            n
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "observe_from_action"
        ]
        assert calls, "no observe_from_action call found in tool_bridge — did it move?"

        params = set(inspect.signature(Cerebellum.observe_from_action).parameters)
        for call in calls:
            used = {kw.arg for kw in call.keywords if kw.arg}
            unknown = used - params
            assert not unknown, (
                f"tool_bridge calls observe_from_action with {sorted(unknown)}, "
                f"which the signature does not accept {sorted(params)}. This is "
                f"a TypeError into a swallow — exactly the 2026-04-07 break."
            )

    def test_telemetry_block_does_not_raise(self, caplog):
        """The swallowed AttributeError that hid the whole thing."""
        tool, cereb = _rig()
        with caplog.at_level("WARNING"):
            tool.execute()
        swallowed = [r for r in caplog.records if "swallowed" in r.getMessage().lower()]
        assert not swallowed, f"a swallowed exception fired during training: {swallowed}"


@pytest.mark.parametrize("bad_kw", ["entity_path", "actual_sensors"])
def test_the_pre_fix_kwargs_are_rejected_loudly(bad_kw):
    """Documents that the pre-fix call was a TypeError, not a silent no-op —
    it only LOOKED silent because of the swallow around it."""
    cereb = Cerebellum()
    kwargs = {"entity": "e", "modulator": "m", "affordance": "a", "params": {}, "actual": {"s": 1.0}}
    kwargs[bad_kw] = kwargs.pop("entity" if bad_kw == "entity_path" else "actual")
    with pytest.raises(TypeError):
        cereb.observe_from_action(**kwargs)
