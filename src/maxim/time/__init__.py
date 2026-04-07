"""Time module - SCN (Suprachiasmatic Nucleus) temporal indexing.

Provides temporal rhythm indexing and querying for memories,
plus a Kuramoto coupled oscillator network for temporal rhythm learning.
"""

from maxim.time.oscillator import OscillatorConfig, OscillatorNetwork
from maxim.time.scn import SCN
from maxim.time.temporal_signature import TemporalSignature, circular_distance

__all__ = [
    "SCN",
    "TemporalSignature",
    "circular_distance",
    "OscillatorNetwork",
    "OscillatorConfig",
]
