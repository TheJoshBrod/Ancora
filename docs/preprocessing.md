# Preprocessing

Turns your own Clone Hero charts into training data: EnCodec audio codes plus Medium lead-guitar note tokens.

## Pipeline

```
scripts/preprocess.py  ->  data/tokens/<song_id>/ + data/manifest.csv
scripts/split.py       ->  data/dataset_split.json
```

The logic lives in `src/ancora/data/`; the scripts only parse arguments and call it.

`--source` may hold extracted song folders (`notes.chart` / `notes.mid`, `song.ini`,
audio stems) and/or packed `.sng` files, which are first extracted into
`<out>/extracted/<name>/`.

Outputs (under `--out`):

```
tokens/<song_id>/audio_codes.npy     EnCodec codes (skipped with --no-audio)
tokens/<song_id>/tokens_100ms.npy    fret-bitmask tokens on a 100 ms grid
tokens/<song_id>/sustain_100ms.npy   sustain length (s) per grid cell
tokens/<song_id>/events_raw.npz      (time_s, bitmask, duration_s) events
manifest.csv                         one row per song with QA flags
```

| File | Role |
| --- | --- |
| `sng.py` | Unpacks `.sng` archives (preprocessing calls it automatically; `scripts/extract_sng.py` for one file) |
| `chart.py` | Parses `notes.chart` / `notes.mid` Medium guitar into `(time_s, fret_bitmask, duration_s)`, using the tempo map and `song.ini` delay |
| `tokenizer.py` | Puts events on a 100 ms grid (0 = PAD, 1-63 = fret bitmask, 64 = BOS, 65 = EOS; see `tokenizer_config.json`) |
| `audio.py` | Mixes all stems to 24 kHz mono and encodes them with EnCodec (6 kbps) |
| `preprocess.py` | Runs everything above over a folder and writes the manifest |
| `split.py` | Song-grouped train/val/test split (no leakage between versions of a song) |

## Usage

Run from the repo root:

```bash
# Charts only, skips EnCodec encoding: handy for checking parsing
uv run scripts/preprocess.py --source "/mnt/c/Users/<you>/OneDrive/Documents/Clone Hero/Songs" --no-audio

# Full run with audio (uses the GPU if available)
uv run scripts/preprocess.py --source "/mnt/c/Users/<you>/OneDrive/Documents/Clone Hero/Songs"

# A folder of .sng files from your own library works the same way
uv run scripts/preprocess.py --source /path/to/sng_folder

uv run scripts/split.py --data data
```

Songs already marked `done` in the manifest are skipped on re-runs; pass `--reprocess` to redo them.

## Known limitations

- MIDI open notes (SysEx / `ENHANCED_OPENS`) are not parsed yet. `.chart` open notes are.
- On the 100 ms grid, onset timing can be off by up to ±50 ms, and two notes can fall in the same cell (flagged as `collision_100ms` in the manifest).
- HOPO/tap flags are ignored.
