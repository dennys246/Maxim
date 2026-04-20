# E2.5 ComponentIndex Discovery — PoC Experiment Protocol

**Date:** 2026-04-19
**Experiment:** Validate that the ComponentIndex's two-layer architecture (alias table + semantic embedding) enables natural language discovery of SEM components.
**Hypothesis:** Natural language queries should resolve to the correct component ref ≥ 90% of the time via alias table (Layer 1), and ≥ 70% of the time via embedding similarity (Layer 2) when sentence-transformers is available.

## Setup

```bash
# Ensure you're on the E2.5 branch
git checkout feat/07-e25-component-index

# Run with PYTHONPATH pointing to src
export PYTHONPATH=src
```

## Layer 1 (Alias Table) Validation

```python
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.embodiment.component_index import ComponentIndex
from maxim.utils.paths import bundled_data

# Build from bundled seed components
registry = ComponentRegistry(
    search_paths=[bundled_data() / "components"],
    include_defaults=False,
)
index = ComponentIndex(registry)
print(f"Indexed: {index.stats()}")

# PoC queries — these MUST all pass for the gate
alias_tests = [
    ("old sword", "weapons/rusty_sword"),
    ("healing draught", "items/healing_potion"),
    ("wild dog", "creatures/wolf"),
    ("timber wolf", "creatures/wolf"),
    ("corroded sword", "weapons/rusty_sword"),
    ("health potion", "items/healing_potion"),
    ("restorative elixir", "items/healing_potion"),
    ("grey wolf", "creatures/wolf"),
    ("blaster pistol", None),  # Should NOT match (truly novel)
]

passed = 0
for query, expected in alias_tests:
    match = index.find(query)
    ref = match.ref if match else None
    layer = match.layer if match else "none"
    ok = ref == expected
    passed += ok
    status = "PASS" if ok else "FAIL"
    print(f"  {status}: find('{query}') → {ref} [{layer}] (expected {expected})")

print(f"\nAlias layer: {passed}/{len(alias_tests)} passed")
```

## Layer 2 (Embedding Similarity) Validation

```python
# These require sentence-transformers for meaningful results
# With fallback bag-of-words, only exact word overlap will match

embedding_tests = [
    ("sharp blade for combat", "weapons/"),  # should match a weapon
    ("magical restoration drink", "items/"),  # should match a potion/item
    ("hostile beast in the wild", "creatures/"),  # should match a creature
    ("dark corridor underground", "environments/"),  # should match an environment
]

print("\n--- Embedding Layer (find_similar top-3) ---")
for query, expected_prefix in embedding_tests:
    results = index.find_similar(query, k=3)
    top_refs = [r.ref for r in results]
    has_match = any(r.startswith(expected_prefix) for r in top_refs)
    status = "PASS" if has_match else "FAIL"
    scores = [(r.ref, f"{r.score:.3f}") for r in results]
    print(f"  {status}: '{query}' → {scores}")
```

## Dedup Check Validation

```python
# Near-duplicate wolf spec
near_wolf = {
    "entity": {
        "name": "timber_wolf",
        "entity_type": "creature",
        "sensors": {"hp": {"range": [0, 15], "initial": 15}},
        "modulators": {"combat": {"affordances": {"bite": {"description": "Bite prey"}}}},
        "failure_modes": [{"name": "death"}],
    }
}

# Truly novel spec
novel_spec = {
    "entity": {
        "name": "quantum_computer",
        "entity_type": "device",
        "sensors": {"qubits": {}, "temperature": {}},
        "modulators": {"computation": {"affordances": {"entangle": {"description": "Entangle qubits"}}}},
    }
}

dup = index.dedup_check(near_wolf, threshold=0.5)
novel = index.dedup_check(novel_spec, threshold=0.9)
print(f"\n--- Dedup Check ---")
print(f"  Near-wolf: {'DETECTED' if dup else 'MISSED'} — {dup}")
print(f"  Novel device: {'CLEAN' if not novel else 'FALSE POSITIVE'} — {novel}")
```

## Pass Criteria

| Layer | Metric | Threshold |
|-------|--------|-----------|
| Alias (L1) | Exact match accuracy | ≥ 8/9 (89%) |
| Embedding (L2) | Category-correct in top-3 | ≥ 3/4 with sentence-transformers |
| Embedding (L2) | Category-correct in top-3 | ≥ 1/4 with fallback (bag-of-words) |
| Dedup | Near-duplicate detected | YES with sentence-transformers |
| Dedup | Novel spec clean | YES |

## Reproduction

```bash
# Quick run (fallback mode — no sentence-transformers needed):
PYTHONPATH=src python -c "
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.embodiment.component_index import ComponentIndex
from maxim.utils.paths import bundled_data

registry = ComponentRegistry(search_paths=[bundled_data() / 'components'], include_defaults=False)
index = ComponentIndex(registry)
stats = index.stats()
print(f'Components: {stats[\"component_count\"]}, Aliases: {stats[\"alias_count\"]}, Fallback: {stats[\"using_fallback\"]}')

for q, exp in [('old sword', 'weapons/rusty_sword'), ('healing draught', 'items/healing_potion'), ('wild dog', 'creatures/wolf')]:
    m = index.find(q)
    ref = m.ref if m else None
    print(f'  find(\"{q}\") → {ref} [{m.layer if m else \"none\"}] {\"PASS\" if ref == exp else \"FAIL\"}')
"
```
