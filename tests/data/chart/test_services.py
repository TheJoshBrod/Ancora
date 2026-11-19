import tempfile
from pathlib import Path

import numpy as np
from test_factories import write_chart, write_mid

from ancora.data.chart.constants import (
    FLAG_FORCE_HOPO,
    FLAG_FORCED,
    FLAG_STAR_POWER,
    FLAG_TAP,
    OPEN_BIT,
)
from ancora.data.chart.services import load_chart, parse_mid


def test_chart_parse():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / 'song'
        write_chart(folder, ini_delay=100)
        c = load_chart(str(folder))
    assert c is not None
    assert c.fmt == 'chart' and c.resolution == 192
    assert c.difficulties_present() == ['medium', 'expert']
    med = c.notes['medium']
    assert [n.tick for n in med] == [0, 192, 384, 576, 768]
    assert med[1].frets == 0b110 and med[1].length == 96
    assert med[2].frets == OPEN_BIT
    assert med[3].flags & FLAG_FORCED
    assert med[4].flags & FLAG_TAP and med[4].flags & FLAG_STAR_POWER
    assert c.sections == [(0, 'Intro'), (768, 'Chorus 1')]
    t = c.ticks_to_seconds([0, 192, 768, 960])
    assert np.allclose(t, [0.1, 0.6, 2.1, 3.1]), t
    assert np.allclose(c.beat_times()[:6], [0.1, 0.6, 1.1, 1.6, 2.1, 3.1])
    assert list(c.downbeat_indices(9)) == [0, 4, 8]


def test_mid_parse():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 'notes.mid')
        write_mid(path)
        c = parse_mid(path)
    assert c.resolution == 480 and c.fmt == 'mid'
    med = c.notes['medium']
    assert [n.tick for n in med] == [0, 480, 960]
    assert med[0].frets == 0b11 and med[0].flags & FLAG_FORCE_HOPO
    assert med[1].frets == OPEN_BIT
    assert med[2].frets == 0b100 and med[2].length == 480
    assert med[2].flags & FLAG_TAP and med[2].flags & FLAG_STAR_POWER
    assert not med[0].flags & FLAG_TAP
    assert [n.tick for n in c.notes['expert']] == [0]
    assert c.notes['easy'] == [] and c.notes['hard'] == []
    assert c.sections == [(0, 'verse 1'), (960, 'chorus 1')]
    assert list(c.downbeat_indices(7)) == [0, 3, 6]
