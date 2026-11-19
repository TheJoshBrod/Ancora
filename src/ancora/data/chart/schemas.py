from dataclasses import dataclass, field

import numpy as np

from ancora.data.chart.constants import DIFFICULTIES


@dataclass
class Note:
    tick: int
    frets: int
    length: int
    flags: int = 0


@dataclass
class ChartData:
    resolution: int
    tempo_map: list[tuple[int, float]]  # (tick, microseconds per beat)
    time_signatures: list[tuple[int, int]]
    notes: dict[str, list[Note]] = field(default_factory=dict)
    star_power: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    sections: list[tuple[int, str]] = field(default_factory=list)
    offset_s: float = 0.0  # song.ini delay
    chart_offset_s: float = 0.0  # not applied: sign convention unverified
    fmt: str = ''
    ini: dict[str, str] = field(default_factory=dict)

    def ticks_to_seconds(self, ticks) -> np.ndarray:
        ticks = np.asarray(ticks, dtype=np.float64)
        starts = np.array([t for t, _ in self.tempo_map], dtype=np.float64)
        us = np.array([u for _, u in self.tempo_map], dtype=np.float64)
        sec_per_tick = us / 1e6 / self.resolution
        seg_start_s = np.concatenate([[0.0], np.cumsum(np.diff(starts) * sec_per_tick[:-1])])
        idx = np.clip(np.searchsorted(starts, ticks, side='right') - 1, 0, None)
        return seg_start_s[idx] + (ticks - starts[idx]) * sec_per_tick[idx] + self.offset_s

    def last_tick(self) -> int:
        end = 0
        for notes in self.notes.values():
            for n in notes:
                end = max(end, n.tick + n.length)
        for t, _ in self.sections:
            end = max(end, t)
        return end

    def beat_times(self, extra_beats: int = 4) -> np.ndarray:
        n_beats = self.last_tick() // self.resolution + 1 + extra_beats
        return self.ticks_to_seconds(np.arange(n_beats) * self.resolution)

    def downbeat_indices(self, n_beats: int) -> np.ndarray:
        sigs = self.time_signatures or [(0, 4)]
        out, beat = [], 0
        for i, (tick, num) in enumerate(sigs):
            start_beat = tick // self.resolution
            end_beat = sigs[i + 1][0] // self.resolution if i + 1 < len(sigs) else n_beats
            beat = max(beat, start_beat)
            while beat < end_beat:
                out.append(beat)
                beat += max(num, 1)
        return np.array([b for b in out if b < n_beats], dtype=np.int32)

    def difficulties_present(self) -> list[str]:
        return [d for d in DIFFICULTIES if self.notes.get(d)]

    def events_seconds(self, difficulty: str):
        notes = self.notes.get(difficulty) or []
        if not notes:
            return []
        ticks = np.array([n.tick for n in notes])
        ends = np.array([n.tick + n.length for n in notes])
        t0, t1 = self.ticks_to_seconds(ticks), self.ticks_to_seconds(ends)
        return [(float(a), n.frets, float(b - a)) for a, b, n in zip(t0, t1, notes, strict=True)]
