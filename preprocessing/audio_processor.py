import os
import numpy as np
import torch
import torchaudio
from encodec import EncodecModel

class AudioEncoder:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(6.0) # standard bandwidth for encodec
        self.model.to(self.device)
        self.model.eval()
        
    def find_audio_stems(self, folder_path):
        """Finds every audio stem to mix.

        In multi-stem charts, song.* is only the backing track (no guitar,
        drums, ...), so it must be mixed with the other stems rather than used
        on its own. In single-file charts song.* is the full mix, and this
        returns just that file. preview.* and crowd.* are always skipped.
        """
        extensions = ('.ogg', '.mp3', '.wav', '.opus', '.flac')
        stems = []
        for f in sorted(os.listdir(folder_path)):
            name = f.lower()
            if name.endswith(extensions) and not name.startswith(('preview', 'crowd')):
                stems.append(os.path.join(folder_path, f))
        return stems

    def load_and_mix(self, stem_paths):
        """Loads stems, mixes to mono, resamples to 24kHz."""
        target_sr = 24000
        mixed_waveform = None
        max_len = 0
        
        import av
        waveforms = []
        for path in stem_paths:
            try:
                container = av.open(path)
                stream = container.streams.audio[0]
                frames = []
                for frame in container.decode(stream):
                    frames.append(frame.to_ndarray())
                # frames is list of (channels, samples)
                waveform = np.concatenate(frames, axis=1)
                waveform = torch.from_numpy(waveform).float()
                sr = stream.sample_rate
                
                # Resample if needed
                if sr != target_sr:
                    waveform = torchaudio.functional.resample(waveform, sr, target_sr)
                # Mix to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                waveforms.append(waveform)
                max_len = max(max_len, waveform.shape[1])
            except Exception as e:
                print(f"Error loading {path}: {e}")
                
        if not waveforms:
            return None
            
        mixed_waveform = torch.zeros(1, max_len)
        for w in waveforms:
            mixed_waveform[:, :w.shape[1]] += w
            
        # Normalize
        mixed_waveform = mixed_waveform / (torch.max(torch.abs(mixed_waveform)) + 1e-8)
        return mixed_waveform

    def encode(self, waveform):
        """Encodes waveform using Encodec."""
        with torch.no_grad():
            waveform = waveform.unsqueeze(0).to(self.device) # Add batch dim
            encoded_frames = self.model.encode(waveform)
            # encoded_frames is a list of (codes, scale). we just want codes
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
            return codes.squeeze(0).cpu().numpy().astype(np.int16)
            
    def process_song(self, folder_path, output_dir, song_id):
        stems = self.find_audio_stems(folder_path)
        if not stems:
            return None, 0.0
            
        waveform = self.load_and_mix(stems)
        if waveform is None:
            return None, 0.0
            
        duration_s = waveform.shape[1] / 24000.0
        codes = self.encode(waveform)
        
        song_dir = os.path.join(output_dir, 'tokens', song_id)
        os.makedirs(song_dir, exist_ok=True)
        np.save(os.path.join(song_dir, f'audio_codes.npy'), codes)
        
        return codes, duration_s

