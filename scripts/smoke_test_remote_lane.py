"""Diagnostic: verify build_primary_router → HTTP call works end-to-end.

Usage:
    MAXIM_LANE_INFER_REMOTE_URL=http://127.0.0.1:8000/v1 \
    MAXIM_LANE_INFER_REMOTE_MODEL=mistral-7b-instruct-v0.2 \
    python scripts/smoke_test_remote_lane.py
"""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from maxim.runtime.lane_backends import build_primary_router

router, manager = build_primary_router()

print(f"\nrouter type: {type(router).__name__}")
print(f"cfg.backend: {router.cfg.backend}")
print(f"cfg.enabled: {router.cfg.enabled}")
print(f"cfg.cloud_enabled: {router.cfg.cloud_enabled}")
print(f"_cloud_allowed: {router._cloud_allowed}")
print(f"_cloud_block_reason: {router._cloud_block_reason}")

print("\nProviders:")
for k, v in router._providers.items():
    is_cloud = router._provider_is_cloud(v)
    allow_local = v.get("allow_local_endpoints")
    print(f"  {k}: is_cloud={is_cloud} allow_local_endpoints={allow_local}")
    print(f"    cfg: {v}")

print("\nCandidate filter:")
providers, tier, totals = router._candidate_providers(100, 20, time.time())
print(f"  candidate providers: {providers}")
print(f"  budget_tier: {tier}")

if not providers:
    print("\n!!! NO CANDIDATES — request would be rejected. Diagnose filter chain.")
else:
    print("\nCalling _complete_text...")
    t0 = time.time()
    text, usage = router._complete_text(
        system="You are a helpful assistant.",
        user="What is 2+2? Answer in one word.",
        temperature=0.0,
        max_tokens=20,
    )
    print(f"  elapsed={time.time() - t0:.2f}s")
    print(f"  text: {text!r}")
    print(f"  usage: {usage}")
