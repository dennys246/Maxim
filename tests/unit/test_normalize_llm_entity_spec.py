"""C4-followup-1: `normalize_llm_entity_spec` adds `abstract: True` to
capability-only modulators in LLM-generated entity specs.

Without this normalizer, the C4 hard-error flip (v1_refinement §1.1-T4
follow-up) would reject every LLM-imagined or foundry-generated entity
at parse time. Bundled YAMLs are audited; LLM specs are not.

The normalizer is non-mutating + idempotent + recursive (walks children).
"""

from __future__ import annotations

from maxim.embodiment.spec import _parse_entity, normalize_llm_entity_spec


# ---------------------------------------------------------------------------
# (a) Bare modulator → marked abstract
# ---------------------------------------------------------------------------


class TestBareModulatorMarked:
    def test_capability_only_modulator_gets_abstract_true(self):
        data = {
            "name": "guard",
            "entity_type": "npc",
            "sensors": {"hp": {"unit": "points", "range": [0, 20], "initial": 20}},
            "modulators": {
                "social": {
                    "affordances": {
                        "speak": {"params": {"message": "str"}, "description": "say"},
                    },
                },
            },
        }
        out = normalize_llm_entity_spec(data)
        assert out["modulators"]["social"]["abstract"] is True

    def test_multiple_bare_modulators_all_marked(self):
        data = {
            "name": "mage",
            "modulators": {
                "social": {"affordances": {}},
                "combat": {"affordances": {}},
                "magic": {"affordances": {}},
            },
        }
        out = normalize_llm_entity_spec(data)
        for mod_name in ("social", "combat", "magic"):
            assert out["modulators"][mod_name]["abstract"] is True


# ---------------------------------------------------------------------------
# (b) Modulator with sensors → untouched (not marked)
# ---------------------------------------------------------------------------


class TestSensorBearingModulatorUntouched:
    def test_modulator_with_sensors_not_marked_abstract(self):
        data = {
            "name": "dragon",
            "modulators": {
                "wing": {
                    "sensors": {
                        "integrity": {"unit": "ratio", "range": [0, 1], "initial": 1.0},
                    },
                    "affordances": {},
                },
            },
        }
        out = normalize_llm_entity_spec(data)
        assert "abstract" not in out["modulators"]["wing"]


# ---------------------------------------------------------------------------
# (c) Explicit abstract: False → respected (not silently overridden)
# ---------------------------------------------------------------------------


class TestExplicitAbstractFalseRespected:
    def test_explicit_abstract_false_on_bare_modulator_not_overridden(self):
        # If an author EXPLICITLY says abstract=False on a bare modulator,
        # the normalizer must NOT silently flip it to True. The intended
        # outcome is that the C4 check downstream still rejects this entity
        # (no sensors AND not abstract). The normalizer is a "fill-in
        # missing marker" helper, not a "force-correct authorial intent" tool.
        data = {
            "name": "broken_thing",
            "modulators": {
                "weird": {
                    "abstract": False,
                    "affordances": {},
                },
            },
        }
        out = normalize_llm_entity_spec(data)
        assert out["modulators"]["weird"]["abstract"] is False


# ---------------------------------------------------------------------------
# (d) Idempotency: re-normalizing produces the same result
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_normalize_twice_equals_normalize_once(self):
        data = {
            "name": "guard",
            "modulators": {"social": {"affordances": {}}},
        }
        once = normalize_llm_entity_spec(data)
        twice = normalize_llm_entity_spec(once)
        assert once == twice


# ---------------------------------------------------------------------------
# (e) Non-mutation: input dict is not modified
# ---------------------------------------------------------------------------


class TestNonMutating:
    def test_input_dict_not_mutated(self):
        data = {
            "name": "guard",
            "modulators": {"social": {"affordances": {}}},
        }
        original_modulators = data["modulators"]
        original_social = data["modulators"]["social"]
        normalize_llm_entity_spec(data)
        # Same object references — input untouched.
        assert data["modulators"] is original_modulators
        assert "abstract" not in original_social

    def test_input_modulator_dict_not_shared_with_output(self):
        data = {
            "name": "guard",
            "modulators": {"social": {"affordances": {}}},
        }
        out = normalize_llm_entity_spec(data)
        # The output's modulator dict must be a new object so the abstract
        # assignment doesn't leak back to the input.
        assert out["modulators"]["social"] is not data["modulators"]["social"]


# ---------------------------------------------------------------------------
# (f) Recursion: children entities also normalized
# ---------------------------------------------------------------------------


class TestRecursesChildren:
    def test_bare_modulator_on_child_marked_abstract(self):
        data = {
            "name": "carrier",
            "entity_type": "vehicle",
            "children": [
                {
                    "name": "pilot",
                    "entity_type": "npc",
                    "modulators": {"social": {"affordances": {}}},
                },
            ],
        }
        out = normalize_llm_entity_spec(data)
        assert out["children"][0]["modulators"]["social"]["abstract"] is True

    def test_grandchild_modulator_normalized(self):
        data = {
            "name": "fleet",
            "children": [
                {
                    "name": "ship",
                    "children": [
                        {
                            "name": "crew_member",
                            "modulators": {"social": {"affordances": {}}},
                        },
                    ],
                },
            ],
        }
        out = normalize_llm_entity_spec(data)
        assert out["children"][0]["children"][0]["modulators"]["social"]["abstract"] is True


# ---------------------------------------------------------------------------
# (g) End-to-end: post-normalize, _parse_entity loads without raising C4
# ---------------------------------------------------------------------------


class TestNoC4ErrorAfterNormalize:
    def test_bare_llm_spec_parses_without_c4_error_after_normalize(self):
        # Pre-normalize this would raise ConfigurationError from
        # SpecModulator.__init__ (C4 — flipped from warning in PR-A.3).
        # Post-normalize it should load cleanly.
        data = {
            "name": "imagined_guard",
            "entity_type": "npc",
            "sensors": {"hp": {"unit": "points", "range": [0, 20], "initial": 20}},
            "modulators": {
                "social": {
                    "affordances": {
                        "speak": {"params": {"message": "str"}, "description": "say"},
                    },
                },
            },
        }
        # Must not raise.
        _parse_entity(normalize_llm_entity_spec(data))

    def test_bare_llm_spec_raises_c4_error_WITHOUT_normalize(self):
        # Sanity check that the C4 error IS still raised on bare specs
        # when the normalizer is bypassed — otherwise this whole followup
        # is testing nothing. Post-PR-A.3 (C4 hard-error flip) this is a
        # `ConfigurationError`, not a `DeprecationWarning`.
        import pytest

        from maxim.exceptions import ConfigurationError

        data = {
            "name": "imagined_guard",
            "entity_type": "npc",
            "sensors": {"hp": {"unit": "points", "range": [0, 20], "initial": 20}},
            "modulators": {
                "social": {
                    "affordances": {
                        "speak": {"params": {"message": "str"}, "description": "say"},
                    },
                },
            },
        }
        with pytest.raises(ConfigurationError, match=r"declares no sensors"):
            _parse_entity(data)  # NOT normalized


# ---------------------------------------------------------------------------
# (h) Edge cases: non-dict inputs, missing modulators key, etc.
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_modulators_key_returns_copy_unchanged(self):
        data = {"name": "primitive", "sensors": {"hp": {}}}
        out = normalize_llm_entity_spec(data)
        assert out == data

    def test_empty_modulators_dict_returns_copy_unchanged(self):
        data = {"name": "primitive", "modulators": {}}
        out = normalize_llm_entity_spec(data)
        assert out == data

    def test_non_dict_input_returned_unchanged(self):
        # Defensive — _parse_entity itself rejects non-dict inputs, but the
        # normalizer should pass through gracefully without crashing.
        assert normalize_llm_entity_spec("not a dict") == "not a dict"  # type: ignore[arg-type]

    def test_modulator_with_empty_sensors_dict_treated_as_capability_only(self):
        # `sensors: {}` is semantically equivalent to no sensors — the
        # check is "has at least one sensor."
        data = {
            "name": "guard",
            "modulators": {
                "social": {
                    "sensors": {},
                    "affordances": {},
                },
            },
        }
        out = normalize_llm_entity_spec(data)
        assert out["modulators"]["social"]["abstract"] is True
