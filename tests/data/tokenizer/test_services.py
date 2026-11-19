import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from test_factories import write_chart

from ancora.data.chart.constants import FLAG_FORCED, FLAG_TAP, OPEN_BIT
from ancora.data.chart.services import load_chart
from ancora.data.tokenizer.constants import BOS, EOS
from ancora.data.tokenizer.services import (
    detokenize,
    is_frets,
    lane_shift,
    subdivision_times,
    sustain_class,
    times_from_beat_positions,
    tokenize,
    vocab_description,
    write_vocab_config,
)


def test_tokenize_roundtrip_exact():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / 'song'
        write_chart(folder)
        c = load_chart(str(folder))
    assert c is not None
    notes = c.notes['medium']
    ticks = np.array([n.tick for n in notes])
    lengths = np.array([n.length for n in notes])
    res = tokenize(
        ticks / 192, (ticks + lengths) / 192, [n.frets for n in notes], [n.flags for n in notes]
    )
    toks = res.tokens
    assert toks[0] == BOS and toks[-1] == EOS
    assert res.stats.n_offgrid == 0 and res.stats.n_collisions == 0
    assert len(toks) == len(res.sustain) == len(res.flags)
    back = detokenize(toks, res.sustain)
    assert [round(b * 192) for b, _, _ in back] == list(ticks)
    assert [m for _, m, _ in back] == [n.frets for n in notes]
    # blue sustain (2 beats) is truncated to 1 by the next note
    assert [s for _, _, s in back] == [0.0, 0.5, 0.0, 1.0, 0.0]
    frets_flags = res.flags[[i for i, t in enumerate(toks) if is_frets(int(t))]]
    assert frets_flags[3] & FLAG_FORCED and frets_flags[4] & FLAG_TAP


def test_tokenize_collisions_and_offgrid():
    starts = np.array([0.0, 0.001, 1.0 + 1 / 96])
    res = tokenize(starts, starts, [1, 2, 4])
    assert res.stats.n_collisions == 1
    assert res.stats.n_offgrid == 1
    back = detokenize(res.tokens)
    assert back[0][1] == 0b11


def test_sustain_classes():
    cls = sustain_class([0.0, 0.2, 0.25, 0.6, 1.0, 2.4, 7.5, 20.0])
    assert list(cls) == [0, 0, 1, 2, 4, 6, 10, 10]


def test_lane_shift():
    res = tokenize([0, 1, 2], [0, 1, 2], [0b1, 0b11, 0b100 | OPEN_BIT])
    up = lane_shift(res.tokens, 0, len(res.tokens), +1)
    assert up is not None
    assert [m for _, m, _ in detokenize(up)] == [0b10, 0b110, 0b1000 | OPEN_BIT]
    assert lane_shift(res.tokens, 0, len(res.tokens), -1) is None
    assert lane_shift(tokenize([0], [0], [0b1000]).tokens, 0, 10, +1, max_fret=3) is None


def test_subdivision_times():
    grid = np.array([0.0, 0.5, 1.0, 1.6, 2.2])
    positions = np.array([-0.5, 0.0, 1.5, 2.5, 4.0, 4.5])
    assert np.allclose(
        times_from_beat_positions(positions, grid), [-0.25, 0.0, 0.75, 1.3, 2.2, 2.5]
    )
    sub = subdivision_times(grid, subdiv=48)
    assert len(sub) == 4 * 48 + 1 and np.isclose(sub[48], 0.5) and np.isclose(sub[-1], 2.2)


def test_write_vocab_config(tmp_path):
    path = write_vocab_config(tmp_path)
    assert json.loads(path.read_text()) == vocab_description()
    assert write_vocab_config(tmp_path) == path
    path.write_text(json.dumps({**vocab_description(), 'vocab_size': 1}))
    with pytest.raises(ValueError, match='different token vocabulary'):
        write_vocab_config(tmp_path)
