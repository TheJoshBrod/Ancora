# Preprocessing

Turns your own Clone Hero charts into training data: beat-aligned lead-guitar
tokens for all four difficulties, plus audio features.

## Pipeline

```
scripts/preprocess.py  ->  data/catalog.csv, data/manifest.csv, data/tokens/<song_id>/
scripts/split.py       ->  data/dataset_split.json
```

The logic lives in `src/ancora/data/`; the scripts only parse arguments and call it.
`chart/`, `tokenizer/` and `audio/` are packages split into `constants.py`,
`schemas.py` (dataclasses and protocols) and `services.py` (functions); import
from those modules directly.

| File | Role |
| --- | --- |
| `sng.py` | Unpacks `.sng` archives (preprocessing calls it automatically; `scripts/extract_sng.py` for one file) |
| `chart/` | Parses `notes.chart` / `notes.mid` lead guitar at Easy, Medium, Hard and Expert: ticks, resolution, tempo map, time signatures, flags (forced, tap, open, star power) and section markers |
| `tokenizer/` | Beat-aligned tokens: `BEAT`, `POS_k` (1/48 beat), `FRETS_m` (63 fret bitmasks); sustain classes in beats; lane-shift augmentation helper; `subdivision_times` for the 1/48-beat grid |
| `audio/` | Loads and mixes stems to 24 kHz mono, then runs the chosen encoders from `audio/encoders/`: EnCodec codes, EnCodec latents (before quantization), beat-synchronous log-mel (one frame per 1/48 beat); optional guitar-stem-only copies |
| `preprocess.py` | Catalogs every source folder, then runs everything above per song and writes the manifest |
| `split.py` | Song-grouped train/val/test split; `--check` fails on any leak |

## Outputs

Under `--out` (default `data/`):

```
catalog.csv                          every chart found, with duplicates marked
manifest.csv                         one row per processed song with QA columns
tokenizer_config.json                token ids, vocab size and sustain classes the
                                     tokens were made with
tokens/<song_id>/tokens.npz          <diff>_chart_{tokens,sustain,flags} per difficulty
tokens/<song_id>/chart.npz           raw note arrays, beat grid, downbeats, sections
tokens/<song_id>/<encoder>.npy       one per --features encoder (audio_codes,
                                     encodec_latents, mel_beat48); *_guitar.npy
                                     copies with --guitar-stem
```

A dataset folder holds tokens from one vocabulary only: if `tokenizer_config.json`
there doesn't match the current tokenizer, preprocessing stops and asks for a new
`--out` folder.

## Audio encoders

Each encoder in `audio/encoders/` turns a mono 24 kHz waveform into one array and
has a `name` (the `--features` choice) and a `filename` (the saved `.npy`). To add a
representation, write a class matching `audio/schemas.py:AudioEncoder` and register
it in `ENCODERS` in `audio/services.py`. Its library becomes a regular dependency:
imports stay at module level.

## Tokens

One sequence per difficulty: `BOS BEAT [POS_k FRETS_m]* BEAT ... EOS`.

- Vocabulary: `PAD 0, BOS 1, EOS 2, BEAT 3, POS_k 4-51, FRETS_m 52-114` (115 tokens).
- Fret bitmask: bits 0-4 = Green, Red, Yellow, Blue, Orange; bit 5 = open.
- Side targets aligned with the tokens: `sustain` (class per `FRETS` token, -1
  elsewhere; classes are none, 1/4, 1/2, 3/4, 1, 1.5, 2, 3, 4, 6, 8+ beats) and
  `flags` (note flag bits, kept for the later HOPO/tap/star-power model).
- Each difficulty is tokenized on the chart's own beats (`<diff>_chart_*`).

## Usage

Run from the repo root. Point `--source` at one or more folders; every song found
is listed in `catalog.csv`, and later copies of the same song are marked
`duplicate_of` and skipped unless `--include-duplicates`.

```bash
# Inventory only: what charts, difficulties and stems you have
uv run scripts/preprocess.py --source "/mnt/c/Users/<you>/OneDrive/Documents/Clone Hero/Songs" /mnt/d/songs --catalog-only

# Full run: EnCodec codes + beat-synchronous mel, guitar stems where available
uv run scripts/preprocess.py --source "/mnt/c/Users/<you>/OneDrive/Documents/Clone Hero/Songs" \
    --features encodec mel --guitar-stem

uv run scripts/split.py --data data
uv run scripts/split.py --data data --check
```

Keep `--out` on the WSL disk (the default `data/`); reading training data from `/mnt/c`
is slow. Songs already marked `done` are skipped on re-runs; pass
`--reprocess` to redo them.

## Manifest columns worth watching

- `difficulties`: which of E/M/H/X the chart has; `n_notes_<diff>` per difficulty.
- `offgrid_<diff>`: notes that do not land on the 1/48-beat grid (chart beats).
- `collisions`: notes merged because they snapped to the same grid step.
- `alignment_suspect`: last note after the audio ends, or a minute or more before it.

## Known limitations

- Beat grids come only from the chart's tempo map; there is no beat tracker yet.
- `.chart` `[Song] Offset` is recorded (`chart_offset_s`) but not applied; only
  `song.ini` `delay` shifts note times, as before.
- MIDI open notes are read from `[ENHANCED_OPENS]` (base - 1) and Phase Shift
  SysEx; other open-note conventions are not.
- The sustain cutoff defaults to 1/4 beat unless `song.ini`
  `sustain_cutoff_threshold` is set; Clone Hero's own default may differ.
