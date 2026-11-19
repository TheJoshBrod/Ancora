from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TokenStats:
    n_notes: int
    n_tokens: int
    n_beats: int
    n_collisions: int
    n_offgrid: int
    max_snap_err: float
    n_before_first_beat: int


@dataclass(frozen=True)
class TokenizeResult:
    tokens: np.ndarray
    sustain: np.ndarray  # sustain class per FRETS token, -1 elsewhere
    flags: np.ndarray
    stats: TokenStats
