# Preprocessing

Turns your own Clone Hero charts into training data: beat-aligned lead-guitar
tokens for all four difficulties, plus audio features on the same beat grid.

## Usage

Run from the repo root. `--source` takes one or more folders holding extracted song
folders (`notes.chart` / `notes.mid`, `song.ini`, audio stems) and/or packed `.sng`
files.

```bash
SONGS="/mnt/c/Users/<you>/OneDrive/Documents/Clone Hero/Songs"

# Inventory only: which charts, difficulties and stems you have
uv run scripts/preprocess.py --source "$SONGS" --catalog-only

# Full run: EnCodec codes + beat-synchronous mel, guitar stems where available
uv run scripts/preprocess.py --source "$SONGS" --features encodec mel --guitar-stem

# Song-grouped train/val/test split, then check it for leaks
uv run scripts/split.py --data data
uv run scripts/split.py --data data --check
```

- `--features` takes any of `encodec` (default), `latents` and `mel`.
- When a song has several charts, every readable copy is processed. The first
  readable copy, in `--source` order, is the song's primary chart; the others
  name it in `duplicate_of` (in `catalog.csv` and `manifest.csv`). List the folder
  you trust most first, such as your converted official charts. `--primary-only`
  processes one chart per song. All copies of a song land in the same split.
- Songs already marked `done` are skipped on re-runs; `--reprocess` redoes them.
- Keep `--out` on the WSL disk (the default `data/`): reading training data from
  `/mnt/c` is slow.

## Outputs

Under `--out` (default `data/`):

```
catalog.csv                     every chart found; non-primary copies marked
manifest.csv                    one row per processed song, with QA columns
tokenizer_config.json           token ids, vocab size and sustain classes in use
dataset_split.json              train/val/test song ids (scripts/split.py)
extracted/<name>/               unpacked .sng files
tokens/<song_id>/tokens.npz     <diff>_chart_{tokens,sustain,flags} per difficulty
tokens/<song_id>/chart.npz      raw note arrays, beat grid, downbeats, sections
tokens/<song_id>/<feature>.npy  one per --features encoder: audio_codes,
                                encodec_latents, mel_beat48 (*_guitar.npy
                                copies with --guitar-stem)
```

A dataset folder holds tokens from one vocabulary only: if its
`tokenizer_config.json` doesn't match the current tokenizer, preprocessing stops
and asks for a new `--out` folder.

## Tokens

One sequence per difficulty, on the chart's own beats:
`BOS BEAT [POS_k FRETS_m]* BEAT ... EOS`.

- Vocabulary: `PAD 0, BOS 1, EOS 2, BEAT 3, POS_k 4-51, FRETS_m 52-114` (115 tokens).
- `POS_k`: position within the beat, in 1/48ths.
- `FRETS_m`: fret bitmask, bits 0-4 = Green, Red, Yellow, Blue, Orange; bit 5 = open.
- Side arrays aligned with the tokens: `sustain` (class per `FRETS` token, -1
  elsewhere; classes are none, 1/4, 1/2, 3/4, 1, 1.5, 2, 3, 4, 6, 8+ beats) and
  `flags` (note flag bits, kept for a later HOPO/tap/star-power model).

## Manifest columns worth watching

- `difficulties`: which of E/M/H/X the chart has; `n_notes_<diff>` per difficulty.
- `offgrid_<diff>`: notes that don't land on the 1/48-beat grid.
- `collisions`: notes merged because they snapped to the same grid step.
- `alignment_suspect`: last note after the audio ends, or a minute or more before it.

## Code layout

The logic lives in `src/ancora/data/`; the scripts only parse arguments and call it.
`chart/`, `tokenizer/` and `audio/` are packages split into `constants.py`,
`schemas.py` (dataclasses and protocols) and `services.py` (functions); import from
those modules directly.

| Module | Role |
| --- | --- |
| `chart/` | Parses `notes.chart` / `notes.mid` lead guitar at all four difficulties: ticks, resolution, tempo map, time signatures, note flags (forced, tap, open, star power) and section markers |
| `tokenizer/` | Notes in beats ↔ tokens; sustain classes; lane-shift augmentation; `subdivision_times` for the 1/48-beat grid |
| `audio/` | Loads and mixes stems to 24 kHz mono, then runs the chosen encoders from `audio/encoders/`: EnCodec codes, EnCodec latents (before quantization), beat-synchronous log-mel (one frame per 1/48 beat) |
| `sng.py` | Unpacks `.sng` archives (called automatically; `scripts/extract_sng.py` for a single file) |
| `preprocess.py` | Catalogs every source folder, then runs everything above per song and writes the manifest |
| `split.py` | Song-grouped train/val/test split; `--check` fails on any leak |

### Adding an audio encoder

Each encoder turns a mono 24 kHz waveform into one array and has a `name` (its
`--features` choice) and a `filename` (the saved `.npy`). Write a class matching
`AudioEncoder` in `audio/schemas.py` and register it in `ENCODERS` in
`audio/services.py`. Its library becomes a regular dependency, imported at module
level.

## Known limitations

- Beat grids come only from the chart's tempo map; there is no beat tracker yet.
- `.chart` `[Song] Offset` is recorded (`chart_offset_s`) but not applied; only
  `song.ini` `delay` shifts note times.
- MIDI open notes are read from `[ENHANCED_OPENS]` (base - 1) and Phase Shift
  SysEx; other open-note conventions are not.
- The sustain cutoff defaults to 1/4 beat unless `song.ini`
  `sustain_cutoff_threshold` is set; Clone Hero's own default may differ.
