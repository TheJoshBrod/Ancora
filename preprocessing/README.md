# Preprocessing

Turns your own Clone Hero charts into training data: EnCodec audio codes plus Medium lead-guitar note tokens.

## Pipeline

```
process_batch.py    ->  data/tokens/<song_id>/ + data/manifest.csv
split.py            ->  data/dataset_split.json
```

| File | Role |
| --- | --- |
| `sng_parser.py` | Unpacks `.sng` archives (`process_batch.py` calls it automatically) |
| `chart_parser.py` | Parses `notes.chart` / `notes.mid` Medium guitar into `(time_s, fret_bitmask, duration_s)`, using the tempo map and `song.ini` delay |
| `tokenizer.py` | Puts events on a 100 ms grid (0 = PAD, 1-63 = fret bitmask, 64 = BOS, 65 = EOS; see `tokenizer_config.json`) |
| `audio_processor.py` | Mixes all stems to 24 kHz mono and encodes them with EnCodec (6 kbps) |
| `process_batch.py` | Runs everything above over a folder and writes the manifest |
| `split.py` | Song-grouped train/val/test split (no leakage between versions of a song) |

## Usage

Run from this folder:

```bash
# Charts only, no torch needed: handy for checking parsing
python process_batch.py --source "/mnt/c/Users/<you>/OneDrive/Documents/Clone Hero/Songs" --out ../data --no-audio

# Full run with audio (uses the GPU if available)
python process_batch.py --source "/mnt/c/Users/<you>/OneDrive/Documents/Clone Hero/Songs" --out ../data

# A folder of .sng files from your own library works the same way
python process_batch.py --source /path/to/sng_folder --out ../data

python split.py --data ../data
```

Songs already marked `done` in the manifest are skipped on re-runs; pass `--reprocess` to redo them.

## Known limitations

- MIDI open notes (SysEx / `ENHANCED_OPENS`) are not parsed yet. `.chart` open notes are.
- On the 100 ms grid, onset timing can be off by up to ±50 ms, and two notes can fall in the same cell (flagged as `collision_100ms` in the manifest).
- HOPO/tap flags are ignored.
