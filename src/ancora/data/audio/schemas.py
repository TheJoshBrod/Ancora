from typing import Protocol

import numpy as np
import torch


class AudioEncoder(Protocol):
    name: str
    filename: str

    def encode(self, waveform: torch.Tensor, subdiv_times_s: np.ndarray) -> np.ndarray: ...
