# Ancora

Ancora is an open-source audio-to-chart system for generating Clone Hero–compatible lead guitar charts from music audio.

Ancora ships code (and possibly model weights), never songs. You train it on charts you own.

## Goals

- Lead guitar at all four difficulties (Easy, Medium, Hard, Expert), with Medium and Hard
  quality as the priority
- Works on any song with no tempo or BPM input: beats come from an automatic beat tracker
- Trains locally on a single 16 GB GPU
- Measured, not eyeballed: timing and lane metrics, compared against an existing open model

## Approach

- **Beat-aligned tokens.** Notes are placed on a 1/48-beat grid (`BEAT`, `POS_k`,
  `FRETS_m` tokens), with sustain length predicted separately in beats.
- **Trained on tracker beats.** Each chart is tokenized on both its own tempo map and
  on the beats a tracker finds in the audio, so the model learns to cope with imperfect
  beat grids, which is all it will have at inference time.
- **One model, all difficulties.** A small encoder-decoder Transformer, conditioned on the
  target difficulty.
- **Audio features.** EnCodec codes or latents, or a log-mel spectrogram with one frame per
  1/48 beat; optionally the separate guitar stem when a chart ships one.
- **Evaluation first.** Onset precision/recall/F1 within ±100 ms (with a per-song offset
  search), lane accuracy, note contour and note-count ratio, per difficulty.

## Roadmap

1. Preprocessing: chart parsing, catalog, beat-aligned tokens, audio features, song-level
   split. **Done** (not yet run at scale on real songs).
2. Beat tracker: compare candidates against chart tempo maps and pick one.
3. Evaluation harness.
4. Baseline: score an existing open-source charter on the held-out songs.
5. Model and training.
6. Export to `.chart` + `song.ini` and playtest in Clone Hero.

Later (v2): HOPO, tap and star-power placement as a second pass; section labels as input;
a real-vs-generated critic for best-of-N sampling; preference fine-tuning.

## Status

Early-stage development. Preprocessing exists; there is no model yet.

## Development

```bash
uv sync                       # create .venv from uv.lock (CUDA 13.2 torch)
uv run pre-commit install     # ruff, pyright, semgrep and file checks on every commit
uv run pytest
```

To build a training dataset from your Clone Hero songs, see
[docs/preprocessing.md](docs/preprocessing.md).

| Path | Contents |
| --- | --- |
| `src/ancora/data/` | Chart parsing, tokenizer, audio features, preprocessing pipeline |
| `scripts/` | Thin command-line entry points (`preprocess.py`, `split.py`, ...) |
| `docs/` | Usage docs |
| `tests/` | pytest suite, mirroring `src/` |
| `.semgrep/` | Project semgrep rules and their tests |
| `references/` | Index of related papers (PDFs stay local) |
| `data/`, `outputs/`, `songs/` | Generated data, run outputs and local song files (all gitignored) |

The pre-commit hook brings its own semgrep. To test the rules themselves, install the
CLI (`pipx install semgrep`) and run
`semgrep --test --config .semgrep/rules.yml .semgrep/rules.py`.
