"""Maxim Hivemind — substrate-sharing layer.

Top-level package for the Hivemind shareability infrastructure shipped in
v1.0 per ``v1_refinement.md`` §B5 and ``maxim_hivemind.md``. The 1.0
surface is purely the data + math layer:

- :mod:`maxim.hivemind.merge` (PR B) — pure-function Bayesian-aggregation
  utilities for NAc + EC state dicts.
- (PR C) — identity-bearing concept detection (ported from the old
  Mother plan).
- (PR D) — substrate snapshot bundle format + ``maxim substrate
  export/import`` CLI verbs.

The Oasis software (1.1) and the peer-to-peer Hivemind protocol (1.2)
will build on this package. Provenance + substrate-domain tagging on
the per-link / per-node level shipped in PR A (``CausalLink.source`` /
``CausalLink.domain`` / ``CausalLink.contributors`` and the parallel
``_substrate_node_sources`` / ``_substrate_node_domains`` dicts on
``EntorhinalCortex``).

Public surface is re-exported here so callers write
``from maxim.hivemind import nac_merge, ec_merge`` rather than reaching
into individual modules.
"""

from __future__ import annotations

from maxim.hivemind.bundle import (
    BUNDLE_KIND,
    BUNDLE_SCHEMA_VERSION,
    BundleBodyMismatch,
    BundleBodyUnverifiable,
    assert_bundle_body_compatible,
    compose_bundle,
    extract_bundle,
    read_bundle_manifest,
)
from maxim.hivemind.identity import (
    IDENTITY_DOMAIN_MARKER,
    filter_identity_bearing_links,
    is_identity_bearing,
    is_identity_domain,
)
from maxim.hivemind.merge import (
    CONSENSUS_SOURCE,
    ec_merge,
    nac_merge,
)

__all__ = [
    "BUNDLE_KIND",
    "BUNDLE_SCHEMA_VERSION",
    "BundleBodyMismatch",
    "BundleBodyUnverifiable",
    "assert_bundle_body_compatible",
    "CONSENSUS_SOURCE",
    "IDENTITY_DOMAIN_MARKER",
    "compose_bundle",
    "ec_merge",
    "extract_bundle",
    "filter_identity_bearing_links",
    "is_identity_bearing",
    "is_identity_domain",
    "nac_merge",
    "read_bundle_manifest",
]
