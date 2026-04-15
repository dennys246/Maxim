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
FIXTURE_SHA256 = "967e83ed18851e1dfcad418be57f3275cf04a961462e6dc4dd055b6b71c8920b"


# **PINNED class-name list for torchvision.datasets.Flowers102.**
#
# Stage 2 v2 fold / Arch #2: the original loader read class names
# from ``torchvision.datasets.Flowers102.classes`` as a class
# attribute — an implementation detail that torchvision does not
# document as public API. The next torchvision bump could silently
# remove it (or reorder it) and break the fixture's class_idx
# mapping without raising until retrieval fails in a confusing way.
#
# The fix is to pin the 102 names in-repo and assert against whatever
# torchvision reports at load time. If torchvision ever drifts, the
# assertion raises immediately with an actionable error; if
# torchvision disappears the attribute entirely, the assertion falls
# through to a fallback path that uses the pinned list directly.
#
# The 102 names are the canonical Oxford VGG Flowers-102 class labels
# from Nilsback & Zisserman 2008, as torchvision ships them in
# torchvision>=0.15. Re-dump via:
#
#     PYTHONPATH=src python -c "from torchvision.datasets import Flowers102; print(Flowers102.classes)"
#
# If you edit this list, update FIXTURE_SHA256 above too — the
# fixture's class_idx fields are indexed against this list.
FLOWERS102_CLASS_NAMES: tuple[str, ...] = (
    "pink primrose",  # 0
    "hard-leaved pocket orchid",  # 1
    "canterbury bells",  # 2
    "sweet pea",  # 3
    "english marigold",  # 4
    "tiger lily",  # 5
    "moon orchid",  # 6
    "bird of paradise",  # 7
    "monkshood",  # 8
    "globe thistle",  # 9
    "snapdragon",  # 10
    "colt's foot",  # 11
    "king protea",  # 12
    "spear thistle",  # 13
    "yellow iris",  # 14
    "globe-flower",  # 15
    "purple coneflower",  # 16
    "peruvian lily",  # 17
    "balloon flower",  # 18
    "giant white arum lily",  # 19
    "fire lily",  # 20
    "pincushion flower",  # 21
    "fritillary",  # 22
    "red ginger",  # 23
    "grape hyacinth",  # 24
    "corn poppy",  # 25
    "prince of wales feathers",  # 26
    "stemless gentian",  # 27
    "artichoke",  # 28
    "sweet william",  # 29
    "carnation",  # 30
    "garden phlox",  # 31
    "love in the mist",  # 32
    "mexican aster",  # 33
    "alpine sea holly",  # 34
    "ruby-lipped cattleya",  # 35
    "cape flower",  # 36
    "great masterwort",  # 37
    "siam tulip",  # 38
    "lenten rose",  # 39
    "barbeton daisy",  # 40
    "daffodil",  # 41
    "sword lily",  # 42
    "poinsettia",  # 43
    "bolero deep blue",  # 44
    "wallflower",  # 45
    "marigold",  # 46
    "buttercup",  # 47
    "oxeye daisy",  # 48
    "common dandelion",  # 49
    "petunia",  # 50
    "wild pansy",  # 51
    "primula",  # 52
    "sunflower",  # 53
    "pelargonium",  # 54
    "bishop of llandaff",  # 55
    "gaura",  # 56
    "geranium",  # 57
    "orange dahlia",  # 58
    "pink-yellow dahlia?",  # 59
    "cautleya spicata",  # 60
    "japanese anemone",  # 61
    "black-eyed susan",  # 62
    "silverbush",  # 63
    "californian poppy",  # 64
    "osteospermum",  # 65
    "spring crocus",  # 66
    "bearded iris",  # 67
    "windflower",  # 68
    "tree poppy",  # 69
    "gazania",  # 70
    "azalea",  # 71
    "water lily",  # 72
    "rose",  # 73
    "thorn apple",  # 74
    "morning glory",  # 75
    "passion flower",  # 76
    "lotus",  # 77
    "toad lily",  # 78
    "anthurium",  # 79
    "frangipani",  # 80
    "clematis",  # 81
    "hibiscus",  # 82
    "columbine",  # 83
    "desert-rose",  # 84
    "tree mallow",  # 85
    "magnolia",  # 86
    "cyclamen",  # 87
    "watercress",  # 88
    "canna lily",  # 89
    "hippeastrum",  # 90
    "bee balm",  # 91
    "ball moss",  # 92
    "foxglove",  # 93
    "bougainvillea",  # 94
    "camellia",  # 95
    "mallow",  # 96
    "mexican petunia",  # 97
    "bromelia",  # 98
    "blanket flower",  # 99
    "trumpet creeper",  # 100
    "blackberry lily",  # 101
)


def assert_torchvision_classes_match_pin() -> None:
    """Cross-check torchvision's Flowers102.classes against our pinned
    list. Raises AssertionError with a diff-ish message if torchvision
    has reindexed or renamed any class. Called from load_fixture_images
    at load time so drift surfaces immediately, not after the fixture
    silently binds to the wrong images.
    """
    try:
        from torchvision.datasets import Flowers102
    except ImportError as e:
        raise RuntimeError("torchvision is not installed — cannot cross-check Flowers102 class names") from e

    tv_classes = getattr(Flowers102, "classes", None)
    if tv_classes is None:
        # torchvision removed the attribute entirely; fall back to the
        # pinned list and proceed. Downstream loaders will use
        # FLOWERS102_CLASS_NAMES directly.
        return

    if tuple(tv_classes) != FLOWERS102_CLASS_NAMES:
        mismatches = [
            (i, expected, actual)
            for i, (expected, actual) in enumerate(zip(FLOWERS102_CLASS_NAMES, tv_classes))
            if expected != actual
        ]
        raise AssertionError(
            f"torchvision.datasets.Flowers102.classes drifted from the pinned "
            f"FLOWERS102_CLASS_NAMES in tests/substrate/p4_fixture_loader.py.\n"
            f"torchvision length: {len(tv_classes)}, pinned length: {len(FLOWERS102_CLASS_NAMES)}\n"
            f"Mismatches (idx, pinned, torchvision):\n"
            + "\n".join(f"  {i}: {exp!r} != {act!r}" for i, exp, act in mismatches[:10])
            + ("\n  ..." if len(mismatches) > 10 else "")
            + "\n\nIf torchvision re-indexed Flowers102, update FLOWERS102_CLASS_NAMES\n"
            "AND re-run scripts/p4_clip_calibration_sweep.py to regenerate the\n"
            "fixture's class_idx mapping under a `change-fixture` commit."
        )


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
    # Stage 2 v2 — canonical ship-shape BuildConfig parameters loaded
    # from the fixture YAML. Present for fixture_version >= 2; the v1
    # fixture had these hard-coded in scripts/p4_mug_test_sweep.py.
    # See fixture_version 2 header comment in p4_mug_test.yaml for
    # the rationale and operating point selection.
    build_noise_reps: int = 0
    build_bridges_enabled: bool = False
    build_text_ec_threshold: float = 0.60
    build_vision_ec_threshold: float = 1.01


def compute_fixture_sha256() -> str:
    """Return the SHA-256 hex digest of the fixture YAML bytes."""
    return hashlib.sha256(FIXTURE_YAML.read_bytes()).hexdigest()


def load_fixture_descriptor() -> FixtureDescriptor:
    """Parse ``p4_mug_test.yaml`` into a ``FixtureDescriptor``.

    Uses PyYAML as the single parser — no fallback. Stage 2 v1
    shipped a hand-rolled ``_parse_simple_yaml`` fallback that the
    Round 2 Executor-lens review flagged for silently dropping
    inline ``#`` comments (Exec #1) and creating a split-brain
    vs the PyYAML path (Exec #2: the SHA pin protects YAML bytes
    but not the semantic interpretation if two parsers produce
    different results). The fallback is deleted here.

    PyYAML is a core pymaxim dependency (declared in pyproject.toml
    alongside numpy/scipy/json-repair) and is also transitive via
    sentence-transformers in the ``semantic`` extra, so no caller
    can reach this function without PyYAML being importable.
    """
    import yaml

    raw = FIXTURE_YAML.read_text()
    data = yaml.safe_load(raw)

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
        build_noise_reps=int(data.get("build_noise_reps", 0)),
        build_bridges_enabled=bool(data.get("build_bridges_enabled", False)),
        build_text_ec_threshold=float(data.get("build_text_ec_threshold", 0.60)),
        build_vision_ec_threshold=float(data.get("build_vision_ec_threshold", 1.01)),
        classes=classes,
    )


def load_fixture_images() -> dict[tuple[str, int], "Image"]:
    """Return a dict mapping ``(class_name, dataset_index) -> PIL.Image``
    for every (class, sample) pair in the fixture. Loads images via
    torchvision's Flowers102 cache. Requires the dataset to already
    be present in ``~/.cache/maxim/p4_flowers`` (downloaded by
    ``scripts/p4_clip_calibration_sweep.py`` on first run).

    Caller is responsible for closing images or letting PIL GC them.

    Three drift guards, in order of strictness:

    1. ``assert_torchvision_classes_match_pin()`` cross-checks the full
       102-entry ordering against our pinned list (Arch #2). If
       torchvision ever reindexes Flowers102, the assertion fires.
    2. ``cls.class_idx`` is verified against the pinned list: the
       fixture YAML's class_idx field is assumed to match
       ``FLOWERS102_CLASS_NAMES.index(cls.name)``. If not, the fixture
       is internally inconsistent — surface as ValueError at load
       (Arch #10).
    3. ``expected_class != cls.name`` is the per-sample check from
       Stage 2 v1 — if torchvision returns a different label for the
       pinned sample_index than expected, the sample indexing drifted
       even though the class ordering stayed constant.
    """
    from torchvision.datasets import Flowers102

    # Drift guard 1: global class ordering
    assert_torchvision_classes_match_pin()

    cache_root = Path.home() / ".cache" / "maxim" / "p4_flowers"
    dataset = Flowers102(root=str(cache_root), split="test", download=False)

    descriptor = load_fixture_descriptor()

    # Drift guard 2: per-class class_idx consistency with the pinned
    # list. The fixture YAML carries class_idx for readability; the
    # guard ensures it's not silently stale. Arch #10 fold.
    for cls in descriptor.classes:
        try:
            expected_idx = FLOWERS102_CLASS_NAMES.index(cls.name)
        except ValueError as e:
            raise ValueError(
                f"fixture class {cls.name!r} is not in FLOWERS102_CLASS_NAMES — "
                f"either the pinned list is stale or the fixture references a "
                f"class that does not exist in Flowers-102"
            ) from e
        if cls.class_idx != expected_idx:
            raise ValueError(
                f"fixture class {cls.name!r} has class_idx={cls.class_idx} in "
                f"p4_mug_test.yaml but the pinned FLOWERS102_CLASS_NAMES index "
                f"is {expected_idx}. Regenerate the fixture under a "
                f"`change-fixture` commit and bump FIXTURE_SHA256."
            )

    images: dict[tuple[str, int], Image] = {}
    for cls in descriptor.classes:
        for sample_idx in cls.sample_indices:
            image, label = dataset[sample_idx]
            # Drift guard 3: per-sample label check
            expected_class = FLOWERS102_CLASS_NAMES[label]
            if expected_class != cls.name:
                raise ValueError(
                    f"fixture drift: sample_index {sample_idx} expected class "
                    f"{cls.name!r} but torchvision returned {expected_class!r}. "
                    f"Regenerate the fixture via scripts/p4_clip_calibration_sweep.py "
                    f"and bump FIXTURE_SHA256 in this module."
                )
            images[(cls.name, sample_idx)] = image
    return images
