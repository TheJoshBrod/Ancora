import logging
from collections.abc import Callable
from pathlib import Path

import av
import numpy as np
import torch
import torchaudio

from ancora.data.audio.constants import AUDIO_EXTS, SR
from ancora.data.audio.encoders.encodec import EncodecCodes, EncodecLatents
from ancora.data.audio.encoders.mel import BeatSyncMel
from ancora.data.audio.schemas import AudioEncoder

logger = logging.getLogger(__name__)

ENCODERS: dict[str, Callable[[str], AudioEncoder]] = {
    EncodecCodes.name: EncodecCodes,
    EncodecLatents.name: EncodecLatents,
    BeatSyncMel.name: BeatSyncMel,
}


def build_encoders(names) -> list[AudioEncoder]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return [ENCODERS[name](device) for name in names]


def find_audio_stems(folder_path):
    stems = []
    for p in sorted(Path(folder_path).iterdir()):
        name = p.name.lower()
        if name.endswith(AUDIO_EXTS) and not name.startswith(('preview', 'crowd')):
            stems.append(str(p))
    return stems


def find_guitar_stem(folder_path):
    for p in sorted(Path(folder_path).iterdir()):
        if p.stem.lower() == 'guitar' and p.suffix.lower() in AUDIO_EXTS:
            return str(p)
    return None


def load_and_mix(stem_paths: list[str], target_sr: int = SR):
    waveforms = []
    for path in stem_paths:
        try:
            container = av.open(path, mode='r')
            stream = container.streams.audio[0]
            frames = [frame.to_ndarray() for frame in container.decode(stream)]
            container.close()
            waveform = torch.from_numpy(np.concatenate(frames, axis=1)).float()
            if stream.sample_rate != target_sr:
                waveform = torchaudio.functional.resample(waveform, stream.sample_rate, target_sr)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveforms.append(waveform)
        except Exception as e:
            logger.warning("Error loading %s: %s", path, e)
    if not waveforms:
        return None
    mixed = torch.zeros(1, max(w.shape[1] for w in waveforms))
    for w in waveforms:
        mixed[:, : w.shape[1]] += w
    return mixed / (mixed.abs().max() + 1e-8)


def load_song_audio(folder_path, guitar_stem=False):
    stems = find_audio_stems(folder_path)
    mix = load_and_mix(stems) if stems else None
    guitar = None
    if guitar_stem:
        path = find_guitar_stem(folder_path)
        # a lone guitar.* file is the full mix, not a stem
        if path and len(stems) > 1:
            guitar = load_and_mix([path])
    return mix, guitar


def save_features(song_dir, encoders: list[AudioEncoder], waveform, subdiv_times_s, suffix=''):
    song_dir = Path(song_dir)
    song_dir.mkdir(parents=True, exist_ok=True)
    for encoder in encoders:
        np.save(
            song_dir / f'{encoder.filename}{suffix}.npy', encoder.encode(waveform, subdiv_times_s)
        )
