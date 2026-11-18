import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import mido
from pydub import AudioSegment

class ChartStandardizer:
    def __init__(self, target_sample_rate: int = 24000):
        """
        Initializes the standardizer.
        target_sample_rate: 24kHz is a common target for audio encoding models.
        """
        self.target_sample_rate = target_sample_rate
        self.supported_audio = ['.mp3', '.ogg', '.opus', '.wav', '.flac']
        self.supported_charts = ['.mid', '.chart']

    def process_song_folder(self, folder_path: str, output_dir: str):
        """Processes a single song folder containing arbitrary audio/chart formats."""
        folder = Path(folder_path)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        song_name = folder.name
        audio_file = self._find_file(folder, self.supported_audio)
        chart_file = self._find_file(folder, self.supported_charts)

        if not audio_file or not chart_file:
            print(f"Skipping {song_name}: Missing audio or chart file.")
            return

        print(f"Processing {song_name}...")
        
        # 1. Standardize Audio
        standard_audio_path = out_path / f"{song_name}.wav"
        self._convert_audio(audio_file, standard_audio_path)

        # 2. Standardize Chart (Extracting ONLY Medium Lead Guitar)
        notes_data = []
        if chart_file.suffix == '.mid':
            notes_data = self._parse_midi(chart_file)
        elif chart_file.suffix == '.chart':
            notes_data = self._parse_dot_chart(chart_file)

        # 3. Save Unified JSON
        standard_chart_path = out_path / f"{song_name}.json"
        with open(standard_chart_path, 'w') as f:
            json.dump({
                "song_id": song_name,
                "audio_file": standard_audio_path.name,
                "notes": notes_data
            }, f, indent=2)
            
        print(f"Successfully processed {song_name} -> {out_path}\n")

    def _find_file(self, folder: Path, extensions: List[str]) -> Path:
        """Finds the first file in the folder matching the target extensions."""
        for ext in extensions:
            files = list(folder.glob(f"*{ext}"))
            if files:
                return files[0]
        return None

    def _convert_audio(self, input_path: Path, output_path: Path):
        """Converts arbitrary audio to standard Mono WAV."""
        try:
            # Pydub uses ffmpeg under the hood to handle .opus, .mp3, etc.
            audio = AudioSegment.from_file(str(input_path))
            # Convert to mono and set frame rate
            audio = audio.set_channels(1).set_frame_rate(self.target_sample_rate)
            audio.export(str(output_path), format="wav")
        except Exception as e:
            print(f"Audio conversion failed for {input_path}: {e}")

    def _parse_midi(self, midi_path: Path) -> List[Dict[str, Any]]:
        """Parses only the Medium difficulty Lead Guitar from a MIDI file."""
        notes = []
        
        # Wrap in try/except because CH community MIDIs often violate strict MIDI specs
        try:
            # clip=True attempts to clamp out-of-range velocity/data values automatically
            mid = mido.MidiFile(midi_path, clip=True)
        except Exception as e:
            print(f"    [!] Skipping {midi_path.name}: Corrupt or non-standard MIDI data ({e})")
            return notes
        
        # Isolate the Lead Guitar track
        target_track = None
        for track in mid.tracks:
            if track.name == 'PART GUITAR':
                target_track = track
                break
                
        if not target_track:
            return notes # Skip if no guitar track exists
            
        current_time_ticks = 0
        for msg in target_track:
            current_time_ticks += msg.time
            
            # GH/CH Standard: Medium difficulty notes are 72 (Green) to 76 (Orange)
            if hasattr(msg, 'note') and 72 <= msg.note <= 76:
                # Normalize the fret index to 0-4 (Green to Orange)
                fret_mapped = msg.note - 72 
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append({
                        "time_ticks": current_time_ticks,
                        "fret": fret_mapped,
                        "type": "note_on"
                    })
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                     notes.append({
                        "time_ticks": current_time_ticks,
                        "fret": fret_mapped,
                        "type": "note_off"
                    })
        return notes

    def _parse_dot_chart(self, chart_path: Path) -> List[Dict[str, Any]]:
        """Parses only the Medium difficulty Lead Guitar from a .chart file."""
        notes = []
        in_medium_section = False
        
        with open(chart_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # We only care about Medium Lead Guitar
                if line == "[MediumSingle]":
                    in_medium_section = True
                    continue
                elif line.startswith("[") and in_medium_section:
                    break # We've hit the next section, stop parsing to save time
                    
                if in_medium_section and "=" in line:
                    parts = line.split("=")
                    tick = int(parts[0].strip())
                    event_data = parts[1].strip().split()
                    
                    # N = Note event. We map frets 0-4 (Green to Orange)
                    if len(event_data) >= 3 and event_data[0] == "N":
                        fret = int(event_data[1])
                        
                        # Only grab standard notes (0-4), ignoring forced HOPOs or open notes for now
                        if 0 <= fret <= 4:
                            duration = int(event_data[2])
                            notes.append({
                                "time_ticks": tick,
                                "fret": fret,
                                "duration_ticks": duration
                            })
        return notes

# === Execution ===
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python preprocessing.py <path_to_clone_hero_songs_folder>")
        sys.exit(1)

    # Pointing directly to your Clone Hero songs folder
    RAW_DATA_DIR = sys.argv[1]
    CLEAN_DATA_DIR = "./songs"
    
    pipeline = ChartStandardizer(target_sample_rate=24000)
    
    raw_path = Path(RAW_DATA_DIR)
    if raw_path.exists():
        print(f"Starting preprocessing from {RAW_DATA_DIR}...")
        for item in raw_path.iterdir():
            if item.is_dir():
                pipeline.process_song_folder(str(item), CLEAN_DATA_DIR)
        print("Preprocessing complete! You can now run split.py on ./clean_dataset")
    else:
        print(f"Directory {RAW_DATA_DIR} not found. Please check the path.")