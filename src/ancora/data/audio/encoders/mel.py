import numpy as np
import torch
import torchaudio

from ancora.data.audio.constants import SR

# 5 ms frames resolve 48 subdivisions per beat up to 250 BPM
HOP_S = 0.005


class BeatSyncMel:
    name = 'mel'
    filename = 'mel_beat48'

    def __init__(self, device: str, n_mels: int = 80):
        self.device = device
        self.n_mels = n_mels
        self.hop = round(SR * HOP_S)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=2048, hop_length=self.hop, n_mels=n_mels
        ).to(device)

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, subdiv_times_s: np.ndarray) -> np.ndarray:
        spec = self.mel(waveform.to(self.device)).squeeze(0)
        logmel = torch.log(spec + 1e-5).T
        pos = torch.as_tensor(
            np.asarray(subdiv_times_s) * SR / self.hop,
            dtype=torch.float32,
            device=self.device,
        )
        pos = pos.clamp(0, logmel.shape[0] - 1)
        lo = pos.floor().long()
        hi = (lo + 1).clamp(max=logmel.shape[0] - 1)
        w = (pos - lo.float()).unsqueeze(1)
        out = logmel[lo] * (1 - w) + logmel[hi] * w
        return out.cpu().numpy().astype(np.float16)
