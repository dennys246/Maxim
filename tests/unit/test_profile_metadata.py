"""Drift-guard tests for llama_cpp profile architecture metadata.

Every profile in ``_BUILTIN_PROFILES`` with ``backend=="llama_cpp"`` must
declare an ``arch`` dict with the fields that ``estimate_max_ctx`` (P4c)
needs. This module enforces three levels of protection:

1. **Presence guard**: every llama_cpp profile has ``arch`` with all
   required keys. A new profile can't land without metadata.
2. **Sanity guard**: each field value is within a plausible range.
   Catches structural typos like ``n_layers=320`` (missing a decimal)
   or ``head_dim=0``.
3. **Spot-check guard**: a handful of well-known profiles have their
   exact values asserted, so an accidental edit to ``qwen2.5-14b``'s
   row (e.g., copy-pasting Mistral's n_layers=32 over Qwen's 48)
   produces a loud CI failure with a clear error message.

Combined, these three tests make it hard to ship wrong metadata
without being told about it. That matters because the P4c formula
produces silently-wrong n_ctx estimates if the metadata is off —
the resulting sim would OOM or underuse memory without any clear
signal that the metadata is the problem.
"""

from __future__ import annotations

import pytest

from maxim.models.download import LLM_MODELS
from maxim.models.language.config import _BUILTIN_PROFILES, normalize_llm_profile

_LLAMA_CPP_PROFILES = sorted(name for name, data in _BUILTIN_PROFILES.items() if data.get("backend") == "llama_cpp")

_REQUIRED_ARCH_FIELDS = (
    "n_layers",
    "n_kv_heads",
    "head_dim",
    "kv_type_bytes",
    "weights_gb",
)


@pytest.mark.parametrize("profile_name", _LLAMA_CPP_PROFILES)
def test_profile_has_arch_metadata(profile_name: str):
    """Every llama_cpp profile must declare an ``arch`` dict with all
    fields ``estimate_max_ctx`` needs to compute KV cache budget."""
    data = _BUILTIN_PROFILES[profile_name]
    arch = data.get("arch")
    assert arch is not None, (
        f"{profile_name!r} is missing 'arch' metadata. "
        f"Add it with n_layers, n_kv_heads, head_dim, kv_type_bytes, weights_gb. "
        f"See peer_leader_flexibility_plan.md P4b for field definitions."
    )
    for field in _REQUIRED_ARCH_FIELDS:
        assert field in arch, (
            f"{profile_name!r} arch missing required field {field!r}. Required: {list(_REQUIRED_ARCH_FIELDS)}"
        )


@pytest.mark.parametrize("profile_name", _LLAMA_CPP_PROFILES)
def test_profile_arch_fields_sane(profile_name: str):
    """Each arch field must be within a plausible range. Catches
    typos that the presence test misses (e.g., missing decimal point,
    copy-paste from wrong model)."""
    arch = _BUILTIN_PROFILES[profile_name]["arch"]

    n_layers = arch["n_layers"]
    assert isinstance(n_layers, int), f"{profile_name}.arch.n_layers must be int"
    assert 8 <= n_layers <= 128, (
        f"{profile_name}.arch.n_layers={n_layers} out of range [8, 128]. "
        f"Currently supported architectures span this range — if you're "
        f"intentionally adding a larger model, raise this bound."
    )

    n_kv_heads = arch["n_kv_heads"]
    assert isinstance(n_kv_heads, int), f"{profile_name}.arch.n_kv_heads must be int"
    assert 1 <= n_kv_heads <= 128, (
        f"{profile_name}.arch.n_kv_heads={n_kv_heads} out of range [1, 128]. "
        f"MQA models have 1 KV head; MHA has n_kv_heads == n_attention_heads "
        f"(e.g., SmolLM-1.7B has 32); GQA has fewer KV heads than attention "
        f"heads. The upper bound of 128 is a soft sanity check — no current "
        f"supported model has more KV heads than that."
    )

    head_dim = arch["head_dim"]
    assert isinstance(head_dim, int), f"{profile_name}.arch.head_dim must be int"
    assert 32 <= head_dim <= 256, (
        f"{profile_name}.arch.head_dim={head_dim} out of range [32, 256]. "
        f"Typical values: 64 (SmolLM), 80 (Phi-2), 96 (Phi-3), 128 (Llama/Qwen/Mistral), "
        f"256 (Gemma). If your model is outside this range, verify the HF config."
    )

    kv_type_bytes = arch["kv_type_bytes"]
    assert kv_type_bytes in (0.5, 1, 2, 4), (
        f"{profile_name}.arch.kv_type_bytes={kv_type_bytes} must be one of "
        f"(0.5 for q4_0, 1 for q8_0, 2 for f16, 4 for f32). "
        f"P4c's estimate_max_ctx uses this to scale KV cache byte cost."
    )

    weights_gb = arch["weights_gb"]
    assert isinstance(weights_gb, (int, float)), f"{profile_name}.arch.weights_gb must be numeric"
    assert 0.3 <= weights_gb <= 100, (
        f"{profile_name}.arch.weights_gb={weights_gb} out of range [0.3, 100]. "
        f"Smallest supported model is SmolLM (~1 GB Q4_K_M); largest is "
        f"14B class (~9 GB Q4_K_M). Values outside this range are almost "
        f"certainly a typo."
    )


class TestKnownProfileSpotChecks:
    """Explicit assertions on well-known profile values.

    These catch the subtle error case that the presence + sanity guards
    miss: valid-shape metadata that happens to be wrong (e.g., someone
    copy-pastes Mistral's n_layers=32 onto Qwen2.5-14b's row, which
    should be 48). Spot-checking each top-tier model ensures those
    don't drift silently.

    If a model's upstream config changes in a way that invalidates
    these values, update the test alongside the profile. These values
    are the "committed record" of what we believe each model's arch
    to be; any change needs a corresponding test update.
    """

    def test_mistral_7b_v0_2(self):
        arch = _BUILTIN_PROFILES["mistral-7b-instruct-v0.2"]["arch"]
        assert arch["n_layers"] == 32
        assert arch["n_kv_heads"] == 8  # GQA
        assert arch["head_dim"] == 128
        assert arch["kv_type_bytes"] == 2

    def test_llama_3_8b_instruct(self):
        arch = _BUILTIN_PROFILES["llama-3-8b-instruct"]["arch"]
        assert arch["n_layers"] == 32
        assert arch["n_kv_heads"] == 8  # GQA ratio 4
        assert arch["head_dim"] == 128

    def test_llama_2_13b_chat(self):
        arch = _BUILTIN_PROFILES["llama-2-13b-chat"]["arch"]
        assert arch["n_layers"] == 40
        assert arch["n_kv_heads"] == 40  # MHA — Llama 2 pre-70B uses MHA
        assert arch["head_dim"] == 128

    def test_qwen2_5_14b_instruct(self):
        arch = _BUILTIN_PROFILES["qwen2.5-14b-instruct"]["arch"]
        assert arch["n_layers"] == 48
        assert arch["n_kv_heads"] == 8  # GQA ratio 5 (40/8)
        assert arch["head_dim"] == 128
        assert arch["weights_gb"] == pytest.approx(8.4, abs=0.5)

    def test_qwen2_5_32b_instruct(self):
        """Qwen2.5-32B uses the same GQA shape as the 14B sibling
        (8 KV heads, head_dim 128) but doubles the layer count to 64.
        Spot-check pins both, since copy-pasting the 14B row would
        silently break VRAM estimation."""
        arch = _BUILTIN_PROFILES["qwen2.5-32b-instruct"]["arch"]
        assert arch["n_layers"] == 64
        assert arch["n_kv_heads"] == 8  # GQA ratio 5 (40/8)
        assert arch["head_dim"] == 128
        assert arch["weights_gb"] == pytest.approx(19.9, abs=1.0)

    def test_llama_3_1_70b_instruct(self):
        """Llama 3.1 70B uses GQA ratio 8 (64 attention / 8 KV) — the
        same KV-head count as the 8B sibling, but the larger model has
        10x the layers (80 vs 32). Spot-check pins the layer count
        because mis-copying 32 here would cause a >2x VRAM underestimate."""
        arch = _BUILTIN_PROFILES["llama-3.1-70b-instruct"]["arch"]
        assert arch["n_layers"] == 80
        assert arch["n_kv_heads"] == 8  # GQA ratio 8 (64/8)
        assert arch["head_dim"] == 128
        assert arch["weights_gb"] == pytest.approx(42.5, abs=2.0)

    def test_mixtral_8x7b_instruct(self):
        """Mixtral-8x7B-Instruct-v0.1 uses GQA ratio 4 (32/8). MoE
        with 8 experts top-2 routed; all experts reside in memory,
        so weights_gb reflects the full ~26.4 GB Q4_K_M footprint,
        NOT the 2-expert active subset."""
        arch = _BUILTIN_PROFILES["mixtral-8x7b-instruct"]["arch"]
        assert arch["n_layers"] == 32
        assert arch["n_kv_heads"] == 8  # GQA ratio 4 (32/8)
        assert arch["head_dim"] == 128
        assert arch["weights_gb"] == pytest.approx(26.4, abs=1.5)

    def test_qwen2_7b_instruct(self):
        arch = _BUILTIN_PROFILES["qwen2-7b-instruct"]["arch"]
        assert arch["n_layers"] == 28
        assert arch["n_kv_heads"] == 4  # GQA ratio 7
        assert arch["head_dim"] == 128

    def test_smollm_1_7b(self):
        arch = _BUILTIN_PROFILES["smollm-1.7b-instruct"]["arch"]
        assert arch["n_layers"] == 24
        assert arch["n_kv_heads"] == 32  # MHA
        assert arch["head_dim"] == 64

    def test_phi_2(self):
        arch = _BUILTIN_PROFILES["phi-2"]["arch"]
        assert arch["n_layers"] == 32
        assert arch["head_dim"] == 80

    def test_gemma_2b_it_uses_mqa(self):
        """Gemma 1 2B uses MQA (single KV head) — a gotcha if someone
        assumes MHA by default. Also uses unusually large head_dim=256."""
        arch = _BUILTIN_PROFILES["gemma-2b-it"]["arch"]
        assert arch["n_kv_heads"] == 1  # MQA
        assert arch["head_dim"] == 256


class TestL1LeaderUXAdditions:
    """Cross-link guards for the three L1 profiles added per
    [docs/plans/archive/leader_ux_profile_management.md]. Pins (1) alias
    resolution to the canonical slug, (2) the profile's ``model``
    field maps to a ``LLM_MODELS`` download entry, (3) the prompt
    style matches the family. Catches the most likely
    drop-something-on-rename failure modes."""

    L1_PROFILES = (
        ("qwen2.5-32b-instruct", ("qwen2.5-32b", "qwen32b"), "chatml"),
        ("llama-3.1-70b-instruct", ("llama-3.1-70b", "llama3.1-70b", "llama70b"), "llama3_instruct"),
        ("mixtral-8x7b-instruct", ("mixtral", "mixtral-8x7b"), "mistral_instruct"),
    )

    @pytest.mark.parametrize("slug,aliases,prompt_style", L1_PROFILES)
    def test_alias_resolves_to_canonical_slug(self, slug, aliases, prompt_style):
        del prompt_style  # unused in this test
        for alias in aliases:
            assert normalize_llm_profile(alias) == slug, (
                f"alias {alias!r} should resolve to {slug!r}, got "
                f"{normalize_llm_profile(alias)!r}. Check _PROFILE_ALIASES "
                f"in src/maxim/models/language/config.py."
            )

    @pytest.mark.parametrize("slug,aliases,prompt_style", L1_PROFILES)
    def test_profile_model_field_has_download_url(self, slug, aliases, prompt_style):
        del aliases, prompt_style  # unused in this test
        model_key = _BUILTIN_PROFILES[slug]["model"]
        assert model_key in LLM_MODELS, (
            f"profile {slug!r} declares model={model_key!r} but no "
            f"LLM_MODELS entry exists. Add the download URL to "
            f"src/maxim/models/download.py."
        )
        entry = LLM_MODELS[model_key]
        assert entry["url"].startswith("https://huggingface.co/"), (
            f"L1 download URLs are HuggingFace resolve URLs; got {entry['url']!r}."
        )

    @pytest.mark.parametrize("slug,aliases,prompt_style", L1_PROFILES)
    def test_prompt_style_matches_family(self, slug, aliases, prompt_style):
        del aliases  # unused in this test
        assert _BUILTIN_PROFILES[slug]["prompt_style"] == prompt_style, (
            f"profile {slug!r} expected prompt_style={prompt_style!r} "
            f"(family default); got {_BUILTIN_PROFILES[slug]['prompt_style']!r}."
        )


def test_all_required_profiles_have_metadata():
    """Regression guard: all expected llama_cpp profiles (P4b baseline
    plus the L1 leader-UX additions per leader_ux_profile_management.md)
    must exist and have metadata. If someone renames or removes a
    profile, this test produces a clear failure rather than a cryptic
    "KeyError" elsewhere."""
    expected = {
        "mistral-7b-instruct-v0.2",
        "smollm-1.7b-instruct",
        "llama-2-7b-chat",
        "llama-2-13b-chat",
        "llama-3-8b-instruct",
        "phi-2",
        "phi-3-mini-4k-instruct",
        "qwen2.5-14b-instruct",
        "qwen2-7b-instruct",
        "gemma-2b-it",
        "gemma-7b-it",
        # L1 additions (leader_ux_profile_management.md):
        "mixtral-8x7b-instruct",
        "llama-3.1-70b-instruct",
        "qwen2.5-32b-instruct",
    }
    actual = set(_LLAMA_CPP_PROFILES)
    assert expected <= actual, (
        f"Missing llama_cpp profiles since baseline: {expected - actual}. "
        f"If a profile was intentionally renamed or removed, update both "
        f"this test's expected set AND ensure the replacement has arch "
        f"metadata via test_profile_has_arch_metadata."
    )
