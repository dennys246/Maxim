# Substrate P4 Stage 2 — CLIP ViT-B-32 Oxford Flowers-102 calibration

**Sweep wall clock:** 55.5s
**Samples per class:** 20
**Headroom band:** [0.50, 0.85]
**Target class count:** 10

## Pinned 10-class subset (headroom band)

| idx | class | zero-shot top-1 |
|---|---|---|
| 18 | balloon flower | 0.750 |
| 48 | oxeye daisy | 0.750 |
| 72 | water lily | 0.750 |
| 77 | lotus | 0.750 |
| 21 | pincushion flower | 0.700 |
| 71 | azalea | 0.700 |
| 22 | fritillary | 0.650 |
| 58 | orange dahlia | 0.650 |
| 75 | morning glory | 0.650 |
| 97 | mexican petunia | 0.650 |

## Full 102-class accuracy table (descending)

| idx | class | zero-shot top-1 |
|---|---|---|
| 5 | tiger lily | 1.000 |
| 7 | bird of paradise | 1.000 |
| 12 | king protea | 1.000 |
| 13 | spear thistle | 1.000 |
| 14 | yellow iris | 1.000 |
| 24 | grape hyacinth | 1.000 |
| 30 | carnation | 1.000 |
| 35 | ruby-lipped cattleya | 1.000 |
| 39 | lenten rose | 1.000 |
| 41 | daffodil | 1.000 |
| 47 | buttercup | 1.000 |
| 53 | sunflower | 1.000 |
| 56 | gaura | 1.000 |
| 59 | pink-yellow dahlia? | 1.000 |
| 62 | black-eyed susan | 1.000 |
| 73 | rose | 1.000 |
| 78 | toad lily | 1.000 |
| 79 | anthurium | 1.000 |
| 80 | frangipani | 1.000 |
| 81 | clematis | 1.000 |
| 90 | hippeastrum | 1.000 |
| 99 | blanket flower | 1.000 |
| 8 | monkshood | 0.950 |
| 9 | globe thistle | 0.950 |
| 16 | purple coneflower | 0.950 |
| 17 | peruvian lily | 0.950 |
| 19 | giant white arum lily | 0.950 |
| 27 | stemless gentian | 0.950 |
| 43 | poinsettia | 0.950 |
| 66 | spring crocus | 0.950 |
| 70 | gazania | 0.950 |
| 82 | hibiscus | 0.950 |
| 87 | cyclamen | 0.950 |
| 91 | bee balm | 0.950 |
| 95 | camellia | 0.950 |
| 98 | bromelia | 0.950 |
| 101 | blackberry lily | 0.950 |
| 25 | corn poppy | 0.900 |
| 60 | cautleya spicata | 0.900 |
| 93 | foxglove | 0.900 |
| 11 | colt's foot | 0.850 |
| 52 | primula | 0.850 |
| 64 | californian poppy | 0.850 |
| 76 | passion flower | 0.850 |
| 83 | columbine | 0.850 |
| 86 | magnolia | 0.850 |
| 29 | sweet william | 0.800 |
| 61 | japanese anemone | 0.800 |
| 65 | osteospermum | 0.800 |
| 94 | bougainvillea | 0.800 |
| 18 | balloon flower | 0.750 |
| 48 | oxeye daisy | 0.750 |
| 72 | water lily | 0.750 |
| 77 | lotus | 0.750 |
| 89 | canna lily | 0.750 |
| 21 | pincushion flower | 0.700 |
| 71 | azalea | 0.700 |
| 22 | fritillary | 0.650 |
| 58 | orange dahlia | 0.650 |
| 75 | morning glory | 0.650 |
| 97 | mexican petunia | 0.650 |
| 4 | english marigold | 0.600 |
| 10 | snapdragon | 0.600 |
| 31 | garden phlox | 0.600 |
| 34 | alpine sea holly | 0.600 |
| 67 | bearded iris | 0.600 |
| 84 | desert-rose | 0.600 |
| 33 | mexican aster | 0.550 |
| 49 | common dandelion | 0.550 |
| 1 | hard-leaved pocket orchid | 0.500 |
| 20 | fire lily | 0.450 |
| 38 | siam tulip | 0.450 |
| 0 | pink primrose | 0.400 |
| 6 | moon orchid | 0.400 |
| 28 | artichoke | 0.400 |
| 40 | barbeton daisy | 0.400 |
| 45 | wallflower | 0.400 |
| 2 | canterbury bells | 0.300 |
| 3 | sweet pea | 0.300 |
| 85 | tree mallow | 0.300 |
| 100 | trumpet creeper | 0.200 |
| 68 | windflower | 0.150 |
| 23 | red ginger | 0.100 |
| 50 | petunia | 0.050 |
| 51 | wild pansy | 0.050 |
| 69 | tree poppy | 0.050 |
| 15 | globe-flower | 0.000 |
| 26 | prince of wales feathers | 0.000 |
| 32 | love in the mist | 0.000 |
| 36 | cape flower | 0.000 |
| 37 | great masterwort | 0.000 |
| 42 | sword lily | 0.000 |
| 44 | bolero deep blue | 0.000 |
| 46 | marigold | 0.000 |
| 54 | pelargonium | 0.000 |
| 55 | bishop of llandaff | 0.000 |
| 57 | geranium | 0.000 |
| 63 | silverbush | 0.000 |
| 74 | thorn apple | 0.000 |
| 88 | watercress | 0.000 |
| 92 | ball moss | 0.000 |
| 96 | mallow | 0.000 |

## Selection rationale

The headroom band `[0.50, 0.85]` was chosen so that
Stage 3's head-to-head against OpenCLIP on these classes has meaningful
signal on BOTH sides — saturated classes (>0.85) collapse the baseline
standard deviation and make the `+2σ` margin vacuous, and impossible
classes (<0.50) compare two weak signals.

Classes in the target count are pinned verbatim in
`scenarios/substrate/p4_mug_test.yaml`. Re-running this sweep is
forbidden under the 'no band-aid fixture tweaks' rule unless an
explicit follow-up PR amends the protocol.
