from functools import cache

import numpy as np
import torch
from encodec.model import EncodecModel


@cache
def _load_model(device: str) -> EncodecModel:
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    return model.to(device).eval()


class EncodecCodes:
    name = 'encodec'
    filename = 'audio_codes'

    def __init__(self, device: str):
        self.device = device
        self.model = _load_model(device)

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, subdiv_times_s: np.ndarray) -> np.ndarray:
        frames = self.model.encode(waveform.unsqueeze(0).to(self.device))
        codes = torch.cat([f[0] for f in frames], dim=-1)
        return codes.squeeze(0).cpu().numpy().astype(np.int16)


class EncodecLatents:
    name = 'latents'
    filename = 'encodec_latents'

    def __init__(self, device: str):
        self.device = device
        self.model = _load_model(device)

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, subdiv_times_s: np.ndarray) -> np.ndarray:
        z = self.model.encoder(waveform.unsqueeze(0).to(self.device))
        return z.squeeze(0).cpu().numpy().astype(np.float16)
