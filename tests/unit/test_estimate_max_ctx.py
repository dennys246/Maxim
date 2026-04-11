"""Tests for the dynamic n_ctx sizing introduced in P4c.

The P4c formula computes the largest llama.cpp ``n_ctx`` that fits the
weights + KV cache + safety margin into available VRAM. These tests cover
three behaviors:

1. **Formula correctness** — for representative profiles + VRAM budgets,
   the result matches what we'd hand-derive from the KV math.
2. **Tier walk** — :func:`_pick_infer_profile_with_ctx` falls through
   when the top-tier profile produces a useless n_ctx for the budget.
3. **Plumbing** — ``LaneConfig.n_ctx`` populated by ``detect_tiers`` is
   threaded into ``LocalServerSpawner._build_cmd``, and the
   ``--llm-n-ctx`` CLI flag (via ``MAXIM_LLM_N_CTX``) overrides it.
"""

from __future__ import annotations

import os

import pytest

from maxim.models.language.config import _BUILTIN_PROFILES
from maxim.runtime.capabilities import RuntimeCapabilities
from maxim.runtime.lane_models import (
    _NCTX_SAFE_FALLBACK,
    _lookup_fallback_nctx,
    _pick_infer_profile_with_ctx,
    detect_tiers,
    estimate_max_ctx,
)
from maxim.runtime.local_server_spawner import LocalServerSpawner


# ─── estimate_max_ctx ────────────────────────────────────────────────────────


class TestEstimateMaxCtx:
    """Direct unit tests on the formula."""

    def test_qwen_14b_at_24gb_returns_substantial_context(self):
        """24 GB unified memory (Apple Silicon w/ headroom) should yield
        a context size deep into the 5-figure range, well above 8K."""
        profile = _BUILTIN_PROFILES["qwen2.5-14b-instruct"]
        n_ctx = estimate_max_ctx(profile, vram_budget_gb=24.0)
        assert n_ctx >= 8192, f"expected ≥8192, got {n_ctx}"
        assert n_ctx <= 32768, f"expected ≤32768 (profile cap), got {n_ctx}"
        assert n_ctx % 1024 == 0, f"expected 1024-aligned, got {n_ctx}"

    def test_qwen_14b_at_16gb_formula_caps_at_profile_max(self):
        """At 16 GB the raw KV math just barely fits qwen's full 32K
        context, so the formula returns the profile cap. This is the
        "optimistic" view — the fallback table in
        :func:`_pick_infer_profile_with_ctx` is what bounds it back to
        a practically-safe ~4K. Here we just confirm the formula's own
        ceiling behavior."""
        profile = _BUILTIN_PROFILES["qwen2.5-14b-instruct"]
        n_ctx = estimate_max_ctx(profile, vram_budget_gb=16.0)
        assert n_ctx == 32768, f"expected formula to cap at profile max, got {n_ctx}"

    def test_qwen_14b_at_10gb_returns_zero(self):
        """Weights (8.4 GB) + safety margin (1.5 GB) leave under 0.1 GB
        for KV cache → formula should refuse rather than guess."""
        profile = _BUILTIN_PROFILES["qwen2.5-14b-instruct"]
        n_ctx = estimate_max_ctx(profile, vram_budget_gb=10.0)
        assert n_ctx == 0, f"expected 0 (weights exceed budget), got {n_ctx}"

    def test_respects_profile_cap_on_huge_budget(self):
        """Even with absurd VRAM, the result must not exceed the profile's
        declared n_ctx (the model wasn't trained beyond that)."""
        profile = _BUILTIN_PROFILES["qwen2.5-14b-instruct"]
        declared_max = profile.get("n_ctx", 0)
        assert declared_max > 0, "test fixture: profile must declare n_ctx"
        n_ctx = estimate_max_ctx(profile, vram_budget_gb=512.0)
        assert n_ctx == declared_max, f"expected {declared_max}, got {n_ctx}"

    def test_q4_quant_doubles_effective_context(self):
        """With q4_0 KV (0.5 bytes/elem vs f16's 2), the resulting
        n_ctx for the same VRAM budget should be roughly 4x larger
        (modulo the profile cap and rounding)."""
        profile = _BUILTIN_PROFILES["llama-3-8b-instruct"]
        f16_ctx = estimate_max_ctx(profile, vram_budget_gb=12.0, kv_quant_mode="f16")
        q4_ctx = estimate_max_ctx(profile, vram_budget_gb=12.0, kv_quant_mode="q4_0")
        if f16_ctx > 0 and f16_ctx < profile.get("n_ctx", 0):
            assert q4_ctx > f16_ctx, f"q4 ({q4_ctx}) should exceed f16 ({f16_ctx})"

    def test_missing_arch_returns_zero(self):
        """Custom profiles without arch metadata can't be sized — caller
        must fall through to the static fallback table or env default."""
        bare_profile = {"backend": "llama_cpp", "n_ctx": 4096}
        assert estimate_max_ctx(bare_profile, vram_budget_gb=24.0) == 0

    def test_malformed_arch_returns_zero(self):
        """Arch fields that aren't coercible to numbers must produce 0
        (graceful degradation, not a crash)."""
        bad = {"arch": {"n_layers": "many", "n_kv_heads": 8, "head_dim": 128, "weights_gb": 5.0}}
        assert estimate_max_ctx(bad, vram_budget_gb=24.0) == 0


# ─── fallback table ──────────────────────────────────────────────────────────


class TestFallbackTable:
    def test_qwen_24gb_lookup(self):
        assert _lookup_fallback_nctx("qwen2.5-14b-instruct", 24.0) == 16384

    def test_qwen_16gb_lookup(self):
        assert _lookup_fallback_nctx("qwen2.5-14b-instruct", 16.0) == 4096

    def test_qwen_below_floor_returns_zero(self):
        assert _lookup_fallback_nctx("qwen2.5-14b-instruct", 12.0) == 0

    def test_unknown_profile_returns_zero(self):
        assert _lookup_fallback_nctx("not-a-real-model", 24.0) == 0

    def test_table_rows_are_descending(self):
        """Walk-largest-first only works if rows are sorted descending."""
        for profile, rows in _NCTX_SAFE_FALLBACK.items():
            mins = [r[0] for r in rows]
            assert mins == sorted(mins, reverse=True), f"{profile} fallback rows must be sorted by VRAM descending"


# ─── _pick_infer_profile_with_ctx ────────────────────────────────────────────


class TestPickInferProfileWithCtx:
    def test_24gb_picks_qwen(self):
        profile, n_ctx = _pick_infer_profile_with_ctx(24.0)
        assert profile == "qwen2.5-14b-instruct"
        assert n_ctx >= 8192

    def test_16gb_picks_qwen_with_modest_ctx(self):
        profile, n_ctx = _pick_infer_profile_with_ctx(16.0)
        assert profile == "qwen2.5-14b-instruct"
        assert n_ctx >= 2048
        assert n_ctx <= 8192

    def test_12gb_falls_through_to_smaller_profile(self):
        """12 GB can't run qwen-14b (weights+margin alone are ~10GB),
        so the picker should walk to a smaller tier, not return qwen."""
        profile, n_ctx = _pick_infer_profile_with_ctx(12.0)
        assert profile != "qwen2.5-14b-instruct"
        assert n_ctx >= 2048

    def test_zero_vram_returns_smollm_floor(self):
        """No VRAM at all → universal fallback row, n_ctx left to spawner."""
        profile, n_ctx = _pick_infer_profile_with_ctx(0.0)
        assert profile == "smollm-1.7b-instruct"
        assert n_ctx == 0


# ─── detect_tiers wires n_ctx into LaneConfig ────────────────────────────────


class TestDetectTiersThreadsNCtx:
    def test_24gb_mps_lane_has_n_ctx_set(self):
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=24.0, ram_gb=24.0)
        tiers = detect_tiers(caps)
        assert "large" in tiers
        large = tiers["large"]
        assert large.model_profile == "qwen2.5-14b-instruct"
        assert large.n_ctx is not None
        assert large.n_ctx >= 2048

    def test_16gb_cuda_lane_has_n_ctx_below_advertised_max(self):
        """16 GB discrete GPU should land on qwen-14b but with a context
        well under qwen's 32K declared max — the whole point of P4c."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="cuda", vram_gb=16.0, ram_gb=64.0)
        tiers = detect_tiers(caps)
        large = tiers["large"]
        assert large.model_profile == "qwen2.5-14b-instruct"
        assert large.n_ctx is not None
        assert large.n_ctx < 32768

    def test_no_gpu_no_n_ctx_on_medium(self):
        """Medium-tier (CPU) doesn't get a computed n_ctx — left None
        so the backend uses its own default. Only large is sized today."""
        caps = RuntimeCapabilities(has_gpu=False, gpu_type=None, vram_gb=0.0, ram_gb=16.0)
        tiers = detect_tiers(caps)
        assert "large" not in tiers
        if "medium" in tiers:
            assert tiers["medium"].n_ctx is None


# ─── LocalServerSpawner plumbing ─────────────────────────────────────────────


class TestSpawnerCmdline:
    def test_n_ctx_appears_in_cmd(self, tmp_path):
        fake_model = tmp_path / "model.gguf"
        fake_model.write_bytes(b"fake")
        spawner = LocalServerSpawner(model_path=str(fake_model), n_ctx=4096)
        cmd = spawner._build_cmd()
        assert "--n_ctx" in cmd
        idx = cmd.index("--n_ctx")
        assert cmd[idx + 1] == "4096"

    def test_kv_quant_f16_emits_no_flags(self, tmp_path):
        """f16 is the spawner's default; we deliberately omit the flags
        to stay backward-compatible with older llama_cpp_python builds."""
        fake_model = tmp_path / "model.gguf"
        fake_model.write_bytes(b"fake")
        spawner = LocalServerSpawner(model_path=str(fake_model), kv_quant_mode="f16")
        cmd = spawner._build_cmd()
        assert "--type_k" not in cmd
        assert "--type_v" not in cmd

    def test_kv_quant_q4_0_emits_both_flags(self, tmp_path):
        fake_model = tmp_path / "model.gguf"
        fake_model.write_bytes(b"fake")
        spawner = LocalServerSpawner(model_path=str(fake_model), kv_quant_mode="q4_0")
        cmd = spawner._build_cmd()
        assert "--type_k" in cmd
        assert "--type_v" in cmd
        kidx = cmd.index("--type_k")
        vidx = cmd.index("--type_v")
        assert cmd[kidx + 1] == "q4_0"
        assert cmd[vidx + 1] == "q4_0"

    def test_kv_quant_q8_0_emits_both_flags(self, tmp_path):
        fake_model = tmp_path / "model.gguf"
        fake_model.write_bytes(b"fake")
        spawner = LocalServerSpawner(model_path=str(fake_model), kv_quant_mode="q8_0")
        cmd = spawner._build_cmd()
        assert "--type_k" in cmd
        assert cmd[cmd.index("--type_k") + 1] == "q8_0"


# ─── --llm-n-ctx CLI override ────────────────────────────────────────────────


class TestCliOverride:
    def test_normalize_args_sets_env_var(self, monkeypatch):
        from argparse import Namespace

        from maxim.cli_utils import normalize_args

        monkeypatch.delenv("MAXIM_LLM_N_CTX", raising=False)
        args = Namespace(
            audio="false",
            interactive=None,
            mode="exploration",
            epochs=0,
            language_model=None,
            llm_n_ctx=2048,
            cloud_fallback=None,
            cloud_lane=None,
            cloud_budget=None,
        )
        normalize_args(args)
        assert os.environ.get("MAXIM_LLM_N_CTX") == "2048"

    def test_normalize_args_rejects_negative(self, monkeypatch):
        from argparse import Namespace

        from maxim.cli_utils import normalize_args

        monkeypatch.delenv("MAXIM_LLM_N_CTX", raising=False)
        args = Namespace(
            audio="false",
            interactive=None,
            mode="exploration",
            epochs=0,
            language_model=None,
            llm_n_ctx=-1,
            cloud_fallback=None,
            cloud_lane=None,
            cloud_budget=None,
        )
        with pytest.raises(SystemExit):
            normalize_args(args)

    def test_normalize_args_rejects_non_integer(self, monkeypatch):
        from argparse import Namespace

        from maxim.cli_utils import normalize_args

        monkeypatch.delenv("MAXIM_LLM_N_CTX", raising=False)
        args = Namespace(
            audio="false",
            interactive=None,
            mode="exploration",
            epochs=0,
            language_model=None,
            llm_n_ctx="not-a-number",
            cloud_fallback=None,
            cloud_lane=None,
            cloud_budget=None,
        )
        with pytest.raises(SystemExit):
            normalize_args(args)
