# `mesh.yml` parser dialect is FROZEN

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

**[engineering] `mesh.yml` parser dialect is FROZEN** (Plan 4 C1, 2026-04-14): `peer/mesh_config.py::parse_mesh_config` is a deliberately trivial hand-rolled YAML-ish parser — flat top-level `key: value` scalars plus a single nested `nodes:` list of `- name: foo` blocks with indented continuation lines. **DO NOT bolt features onto it.** It rejects tabs, bare `- ` entries, duplicate node names, and strips `#` inline comments ONLY when preceded by whitespace (so `cluster_key: sk-abc#literal` is preserved, not silently truncated — round 2 review E1). If you need quoted strings (URL fragments beyond what the whitespace-# rule handles), YAML anchors, multi-line values, or tab indentation: **do not extend this parser.** The two escape hatches are (a) switch `mesh.yml` to TOML and use stdlib `tomllib`, or (b) promote PyYAML from optional extra to core dep. Either change is a C2/C3 architectural decision, not a drive-by patch. Round 1 review flagged five silent-mis-parse classes the original implementation tolerated; round 2 review flagged a sixth (E1 silent `#` truncation). A seventh finding is highly likely if the dialect grows. Regression guard: [src/maxim/peer/mesh_config.py::parse_mesh_config](src/maxim/peer/mesh_config.py) + corresponding unit tests in [tests/unit/test_mesh_config.py](tests/unit/test_mesh_config.py).
