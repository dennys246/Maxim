"""Guards for Phase 2 — the three-tier orient reactivity + per-entity profile.

Sound reactivity is a data-driven, per-entity gate (not hardcoded thresholds):
  - < escalate_above           → ignored (perceived, not delivered)
  - escalate_above .. reflex   → deliberative CHOICE (reaches the LLM)
  - > BOTH reflex thresholds   → REFLEX (automatic orient, bypasses the LLM)
Plus a physical-reach limit (max_orient_azimuth) so a body never "orients" past
what it can physically turn. A body declares its own profile under `orienting:`
in YAML → entity.metadata; absent → defaults. See
docs/plans/deferred/productive_orienting_affordance.md + thalamus_hypothalamus_framing.md.
"""

from __future__ import annotations

from maxim.embodiment.audio_localization import (
    OrientingProfile,
    is_audio_escalation,
    is_orienting_reflex,
    reflex_oriented_azimuth,
    resolve_orienting_profile,
)


# ── the three tiers (default profile) ───────────────────────────────────────


def test_default_tiers():
    d = OrientingProfile()
    # ignored
    assert is_audio_escalation(0.4, d) is False
    # deliberative: escalates but not reflex
    assert is_audio_escalation(0.7, d) is True
    assert is_orienting_reflex(0.7, 0.7, d) is False
    # reflex: loud AND sudden
    assert is_orienting_reflex(0.95, 0.95, d) is True


def test_reflex_needs_both_loud_and_sudden():
    d = OrientingProfile()
    assert is_orienting_reflex(0.95, 0.5, d) is False  # loud but not sudden
    assert is_orienting_reflex(0.5, 0.95, d) is False  # sudden but not loud


def test_boundaries_are_strict_greater_than():
    d = OrientingProfile()  # escalate 0.5, reflex 0.9/0.9
    assert is_audio_escalation(0.5, d) is False  # exactly at → not escalated
    assert is_orienting_reflex(0.9, 0.9, d) is False  # exactly at → not reflex


# ── per-entity sensitivity (the elderly_humanoid case) ──────────────────────


def test_reduced_sensitivity_profile_raises_thresholds():
    elderly = OrientingProfile(escalate_above=0.7, reflex_salience_above=0.97, reflex_novelty_above=0.97)
    # a 0.6 sound that a normal body would escalate is ignored by the elderly one
    assert is_audio_escalation(0.6, elderly) is False
    # a 0.95/0.95 sound that reflexes a normal body does NOT reflex the elderly one
    assert is_orienting_reflex(0.95, 0.95, elderly) is False
    # but a genuinely loud+sudden 0.98 still reaches the reflex
    assert is_orienting_reflex(0.98, 0.98, elderly) is True


# ── physical reach limit ────────────────────────────────────────────────────


def test_reflex_full_reach_centers():
    d = OrientingProfile()  # max_orient_azimuth 0.0 → full reach
    assert reflex_oriented_azimuth(-0.8, d) == 0.0
    assert reflex_oriented_azimuth(0.8, d) == 0.0


def test_reflex_limited_reach_leaves_residual():
    limited = OrientingProfile(max_orient_azimuth=0.3)
    # sound farther out than the body can turn → residual offset, sign preserved
    assert reflex_oriented_azimuth(-0.8, limited) == -0.3
    assert reflex_oriented_azimuth(0.8, limited) == 0.3
    # sound within reach → fully centered
    assert reflex_oriented_azimuth(-0.2, limited) == 0.0


# ── resolve from entity metadata (data-driven, no per-body code) ────────────


class _Root:
    def __init__(self, metadata):
        self.metadata = metadata


class _Emb:
    def __init__(self, metadata):
        self.root = _Root(metadata)


def test_resolve_from_declared_metadata():
    emb = _Emb({"orienting": {"escalate_above": 0.8, "max_orient_azimuth": 0.25}})
    p = resolve_orienting_profile(emb)
    assert p.escalate_above == 0.8
    assert p.max_orient_azimuth == 0.25
    assert p.reflex_salience_above == 0.9  # unspecified → default kept


def test_resolve_defaults_when_absent_or_malformed():
    assert resolve_orienting_profile(None) == OrientingProfile()
    assert resolve_orienting_profile(_Emb({})) == OrientingProfile()
    assert resolve_orienting_profile(_Emb({"orienting": "nope"})) == OrientingProfile()
    # malformed numeric → that field falls back to default
    p = resolve_orienting_profile(_Emb({"orienting": {"escalate_above": "loud"}}))
    assert p.escalate_above == 0.5


def test_resolve_clamps_max_orient_to_unit():
    p = resolve_orienting_profile(_Emb({"orienting": {"max_orient_azimuth": 5.0}}))
    assert p.max_orient_azimuth == 1.0
