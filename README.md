# Ancora

Ancora is an open-source audio-to-chart system for generating Clone Hero–compatible lead guitar charts from music audio.

## Current Scope (MVP)

- Lead guitar only
- Medium difficulty charts
- Single audio input → playable Clone Hero chart output

The goal is to produce a playable, beat-aligned guitar track that can be used in Clone Hero.


## Technical Approach

- Architecture: Transformer-based sequence-to-sequence model designed to map acoustic features directly to chart events.
- Scope Optimization: Tuned specifically for **Medium difficulty** characteristics. The prediction space is streamlined by filtering out higher-order chord combinations and open notes that appear exclusively in higher difficulties, while placing an emphasis on note frequency and sustain duration prediction specific to the Medium tier.

## Output

- Clone Hero `.chart` / `.ini` compatible files (TBD)
- Tempo-synced note sequences for lead guitar

## Status

Early-stage development / proof of concept.

## Development

```bash
uv sync                       # create .venv from uv.lock (CUDA 13.2 torch)
uv run pre-commit install     # ruff, pyright, semgrep and file checks on every commit
uv run pytest
```

| Path | Contents |
| --- | --- |
| `src/ancora/data/` | Chart parsing, tokenizer, audio features, preprocessing pipeline |
| `scripts/` | Thin command-line entry points (`preprocess.py`, `split.py`, ...) |
| `tests/` | pytest suite |
| `docs/preprocessing.md` | Preprocessing outputs and usage |
| `.semgrep/` | Project semgrep rules; `semgrep --test --config .semgrep/rules.yml .semgrep/rules.py` checks them |
| `data/`, `outputs/` | Generated data and run outputs (gitignored) |
