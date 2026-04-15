"""P4 Stage 2 — mug test fixture loader.

Loads ``scenarios/substrate/p4_mug_test.yaml`` and provides the
``(PIL.Image, class_name)`` pairs keyed by ``(class_name,
sample_index)``. Used by ``test_p4_fixture_validation.py`` for the
SHA-256 pin + per-class retrieval validation, and by Phase 2D's
subprocess mug test for the real-image cross-modal binding
experiment.

Reusing this module between the validation test and the mug test
means the fixture path + parsing logic has one canonical site. If
the YAML format ever evolves (v2+), a single loader change updates
every consumer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_YAML = _REPO_ROOT / "scenarios" / "substrate" / "p4_mug_test.yaml"

# **PINNED SHA-256 of the fixture YAML.**
#
# If this hash drifts, tests/substrate/test_p4_fixture_validation.py
# will fail loudly. A legitimate change to the fixture requires:
#
#   1. An explicit `change-fixture` commit message describing WHY the
#      fixture was regenerated (re-calibration, class swap, etc.)
#   2. Re-running scripts/p4_clip_calibration_sweep.py and updating
#      docs/experiments/p4_clip_calibration.md with the new numbers
#   3. Bumping this constant with the new hash
#
# Per docs/plans/substrate_p4_cross_modal_binding.md Stage 2, the
# "no band-aid fixture tweaks" rule forbids silent edits.
FIXTURE_SHA256 = "f137862a5588a6e514e197d6cfef49872db2107d522ce9b636c656495aff3cf6"


@dataclass(frozen=True)
class FixtureClass:
    name: str
    class_idx: int
    clip_zero_shot_accuracy: float
    sample_indices: tuple[int, ...]


@dataclass(frozen=True)
class FixtureDescriptor:
    fixture_version: int
    dataset: str
    split: str
    samples_per_class: int
    total_pairs: int
    classes: tuple[FixtureClass, ...]


def compute_fixture_sha256() -> str:
    """Return the SHA-256 hex digest of the fixture YAML bytes."""
    return hashlib.sha256(FIXTURE_YAML.read_bytes()).hexdigest()


def load_fixture_descriptor() -> FixtureDescriptor:
    """Parse ``p4_mug_test.yaml`` into a ``FixtureDescriptor``.

    Uses PyYAML if available, otherwise a hand-rolled parser covering
    the subset of YAML the fixture actually uses (the P4 fixture is
    strict-schema so the parser doesn't need to be general).
    """
    raw = FIXTURE_YAML.read_text()
    try:
        import yaml

        data = yaml.safe_load(raw)
    except ImportError:
        data = _parse_simple_yaml(raw)

    classes = tuple(
        FixtureClass(
            name=c["name"],
            class_idx=int(c["class_idx"]),
            clip_zero_shot_accuracy=float(c["clip_zero_shot_accuracy"]),
            sample_indices=tuple(int(i) for i in c["sample_indices"]),
        )
        for c in data["classes"]
    )
    return FixtureDescriptor(
        fixture_version=int(data["fixture_version"]),
        dataset=str(data["dataset"]),
        split=str(data["split"]),
        samples_per_class=int(data["samples_per_class"]),
        total_pairs=int(data["total_pairs"]),
        classes=classes,
    )


def _parse_simple_yaml(raw: str) -> dict:
    """Fallback parser for the P4 fixture YAML subset. Handles:

    - Top-level ``key: value`` scalars (int/float/string)
    - ``classes:`` list with ``- name: X`` entries and
      indented continuation lines.
    - Inline lists like ``sample_indices: [1, 2, 3]``
    """
    result: dict = {}
    lines = raw.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        if line.startswith("classes:"):
            classes, consumed = _parse_simple_class_list(lines, i + 1)
            result["classes"] = classes
            i = consumed
            continue
        key, _, value = line.partition(":")
        result[key.strip()] = _coerce_scalar(value.strip())
        i += 1
    return result


def _parse_simple_class_list(lines: list[str], start: int) -> tuple[list[dict], int]:
    classes: list[dict] = []
    current: dict | None = None
    i = start
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        # Top-level key (no leading whitespace) terminates the list.
        if line and not line[0].isspace():
            break
        if stripped.startswith("- "):
            if current is not None:
                classes.append(current)
            current = {}
            first_key_line = stripped[2:]
            if ":" in first_key_line:
                key, _, value = first_key_line.partition(":")
                current[key.strip()] = _coerce_scalar(value.strip())
        elif current is not None and ":" in stripped:
            key, _, value = stripped.partition(":")
            current[key.strip()] = _coerce_scalar(value.strip())
        i += 1
    if current is not None:
        classes.append(current)
    return classes, i


def _coerce_scalar(value: str):
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(p.strip()) for p in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_fixture_images() -> dict[tuple[str, int], "Image"]:
    """Return a dict mapping ``(class_name, dataset_index) -> PIL.Image``
    for every (class, sample) pair in the fixture. Loads images via
    torchvision's Flowers102 cache. Requires the dataset to already
    be present in ``~/.cache/maxim/p4_flowers`` (downloaded by
    ``scripts/p4_clip_calibration_sweep.py`` on first run).

    Caller is responsible for closing images or letting PIL GC them.
    """
    from torchvision.datasets import Flowers102

    cache_root = Path.home() / ".cache" / "maxim" / "p4_flowers"
    dataset = Flowers102(root=str(cache_root), split="test", download=False)

    descriptor = load_fixture_descriptor()
    images: dict[tuple[str, int], Image] = {}
    for cls in descriptor.classes:
        for sample_idx in cls.sample_indices:
            image, label = dataset[sample_idx]
            expected_class = Flowers102.classes[label]
            if expected_class != cls.name:
                raise ValueError(
                    f"fixture drift: sample_index {sample_idx} expected class "
                    f"{cls.name!r} but torchvision returned {expected_class!r}. "
                    f"Regenerate the fixture via scripts/p4_clip_calibration_sweep.py "
                    f"and bump FIXTURE_SHA256 in this module."
                )
            images[(cls.name, sample_idx)] = image
    return images
