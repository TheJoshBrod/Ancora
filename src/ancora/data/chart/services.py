import contextlib
import re
from pathlib import Path

import mido

from ancora.data.chart.constants import (
    CHART_SECTIONS,
    DIFFICULTIES,
    FLAG_FORCE_HOPO,
    FLAG_FORCE_STRUM,
    FLAG_FORCED,
    FLAG_STAR_POWER,
    FLAG_TAP,
    MIDI_BASE,
    MIDI_STAR_POWER_NOTE,
    MIDI_TAP_NOTE,
    OPEN_BIT,
    PS_SYSEX_DIFF,
    PS_SYSEX_OPEN,
    PS_SYSEX_TAP,
)
from ancora.data.chart.schemas import ChartData, Note


def parse_ini(filepath):
    ini_data = {}
    if not Path(filepath).exists():
        return ini_data
    with Path(filepath).open(encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, val = line.split('=', 1)
                ini_data[key.strip().lower()] = val.strip()
    return ini_data


def _merge_chords(raw: list[Note]) -> list[Note]:
    raw.sort(key=lambda n: n.tick)
    merged: list[Note] = []
    for n in raw:
        if merged and merged[-1].tick == n.tick:
            m = merged[-1]
            m.frets |= n.frets
            m.length = max(m.length, n.length)
            m.flags |= n.flags
        else:
            merged.append(Note(n.tick, n.frets, n.length, n.flags))
    return merged


def _in_ranges(tick: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= tick < start + max(length, 1) for start, length in ranges)


def _section_name(text: str) -> str | None:
    t = text.strip().strip('"').strip()
    t = t[1:-1] if t.startswith('[') and t.endswith(']') else t
    m = re.match(r'^(?:section|prc)[ _](.+)$', t, flags=re.IGNORECASE)
    return m.group(1).replace('_', ' ').strip() if m else None


def parse_chart(filepath, delay_ms=0.0) -> ChartData:
    resolution, chart_offset = 192, 0.0
    tempo, sigs, sections = [], [], []
    raw = {d: [] for d in DIFFICULTIES}
    forced = {d: set() for d in DIFFICULTIES}
    taps = {d: set() for d in DIFFICULTIES}
    star = {d: [] for d in DIFFICULTIES}

    with Path(filepath).open(encoding='utf-8-sig', errors='ignore') as f:
        lines = f.readlines()

    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('//') or line in '{}':
            continue
        if line.startswith('[') and line.endswith(']'):
            section = line[1:-1]
            continue
        if section == 'Song':
            m = re.match(r'(\w+)\s*=\s*"?([^"]*)"?', line)
            if m:
                key, val = m.group(1).lower(), m.group(2).strip()
                if key == 'resolution':
                    resolution = int(float(val))
                elif key == 'offset':
                    with contextlib.suppress(ValueError):
                        chart_offset = float(val)
        elif section == 'SyncTrack':
            m = re.match(r'(\d+)\s*=\s*(B|TS)\s+(\d+)(?:\s+(\d+))?', line)
            if m:
                tick, kind, val = int(m.group(1)), m.group(2), int(m.group(3))
                if kind == 'B' and val > 0:
                    tempo.append((tick, 60e6 / (val / 1000.0)))  # milli-BPM -> us per beat
                elif kind == 'TS':
                    sigs.append((tick, val))
        elif section == 'Events':
            m = re.match(r'(\d+)\s*=\s*E\s+"(.*)"', line)
            if m:
                name = _section_name(m.group(2))
                if name:
                    sections.append((int(m.group(1)), name))
        elif section in CHART_SECTIONS:
            diff = CHART_SECTIONS[section]
            m = re.match(r'(\d+)\s*=\s*([NS])\s+(\d+)\s+(\d+)', line)
            if not m:
                continue
            tick, kind, a, b = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
            if kind == 'N':
                if 0 <= a <= 4:
                    raw[diff].append(Note(tick, 1 << a, b))
                elif a == 7:
                    raw[diff].append(Note(tick, OPEN_BIT, b))
                elif a == 5:
                    forced[diff].add(tick)
                elif a == 6:
                    taps[diff].add(tick)
            elif kind == 'S' and a == 2:
                star[diff].append((tick, b))

    chart = ChartData(
        resolution=resolution,
        tempo_map=sorted(tempo) or [(0, 500000.0)],
        time_signatures=sorted(sigs) or [(0, 4)],
        sections=sorted(sections),
        offset_s=delay_ms / 1000.0,
        chart_offset_s=chart_offset,
        fmt='chart',
    )
    if chart.tempo_map[0][0] != 0:
        chart.tempo_map.insert(0, (0, chart.tempo_map[0][1]))
    for d in DIFFICULTIES:
        notes = _merge_chords(raw[d])
        for n in notes:
            if n.tick in forced[d]:
                n.flags |= FLAG_FORCED
            if n.tick in taps[d]:
                n.flags |= FLAG_TAP
            if _in_ranges(n.tick, star[d]):
                n.flags |= FLAG_STAR_POWER
        chart.notes[d] = notes
        chart.star_power[d] = sorted(star[d])
    return chart


def _abs_messages(track):
    tick = 0
    for msg in track:
        tick += msg.time
        yield tick, msg


def parse_mid(filepath, delay_ms=0.0) -> ChartData:
    # clip=True tolerates the out-of-range data bytes common in community MIDIs
    try:
        mid = mido.MidiFile(filepath, clip=True)
    except Exception as e:
        raise ValueError(f"Unreadable MIDI: {e!r}") from e

    tempo, sigs, sections = [], [], []
    for tick, msg in _abs_messages(mid.tracks[0]):
        if msg.type == 'set_tempo':
            tempo.append((tick, float(msg.tempo)))
        elif msg.type == 'time_signature':
            sigs.append((tick, msg.numerator))

    guitar = None
    for track in mid.tracks:
        name = (track.name or '').strip()
        if name == 'PART GUITAR':
            guitar = track
        elif name == 'EVENTS':
            for tick, msg in _abs_messages(track):
                if msg.type in ('text', 'marker'):
                    sec = _section_name(msg.text)
                    if sec:
                        sections.append((tick, sec))

    chart = ChartData(
        resolution=mid.ticks_per_beat,
        tempo_map=sorted(tempo) or [(0, 500000.0)],
        time_signatures=sorted(sigs) or [(0, 4)],
        sections=sorted(sections),
        offset_s=delay_ms / 1000.0,
        fmt='mid',
    )
    if chart.tempo_map[0][0] != 0:
        chart.tempo_map.insert(0, (0, chart.tempo_map[0][1]))
    if guitar is None:
        chart.notes = {d: [] for d in DIFFICULTIES}
        return chart

    msgs = list(_abs_messages(guitar))
    enhanced_opens = any(m.type == 'text' and m.text.strip() == '[ENHANCED_OPENS]' for _, m in msgs)

    open_since: dict[int, int] = {}
    intervals: dict[int, list[tuple[int, int]]] = {}
    sysex_ranges: dict[tuple[str | None, int], list[tuple[int, int]]] = {}
    sysex_open_at: dict[tuple[str | None, int], int] = {}
    last_tick = 0
    for tick, msg in msgs:
        last_tick = tick
        if msg.type == 'note_on' and msg.velocity > 0:
            if msg.note in open_since:
                intervals.setdefault(msg.note, []).append((open_since[msg.note], tick))
            open_since[msg.note] = tick
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in open_since:
                intervals.setdefault(msg.note, []).append((open_since.pop(msg.note), tick))
        elif msg.type == 'sysex':
            d = tuple(msg.data)
            # PS SysEx: 50 53 00 00 <difficulty|FF> <type> <on/off>
            if len(d) >= 7 and d[0] == 0x50 and d[1] == 0x53:
                diff = None if d[4] == 0xFF else PS_SYSEX_DIFF.get(d[4])
                key = (diff, d[5])
                if d[6]:
                    sysex_open_at[key] = tick
                elif key in sysex_open_at:
                    sysex_ranges.setdefault(key, []).append((sysex_open_at.pop(key), tick))
    for note, start in open_since.items():
        intervals.setdefault(note, []).append((start, last_tick))

    def ranges(pitch):
        return [(s, e - s) for s, e in intervals.get(pitch, [])]

    tap_ranges = ranges(MIDI_TAP_NOTE)
    star_ranges = ranges(MIDI_STAR_POWER_NOTE)
    for d in DIFFICULTIES:
        base = MIDI_BASE[d]
        raw = []
        for fret in range(5):
            for s, e in intervals.get(base + fret, []):
                raw.append(Note(s, 1 << fret, e - s))
        if enhanced_opens:
            for s, e in intervals.get(base - 1, []):
                raw.append(Note(s, OPEN_BIT, e - s))
        notes = _merge_chords(raw)

        hopo, strum = ranges(base + 5), ranges(base + 6)
        ps_open = [
            (s, e - s)
            for k, v in sysex_ranges.items()
            if k[1] == PS_SYSEX_OPEN and k[0] in (d, None)
            for s, e in v
        ]
        ps_tap = [
            (s, e - s)
            for k, v in sysex_ranges.items()
            if k[1] == PS_SYSEX_TAP and k[0] in (d, None)
            for s, e in v
        ]
        for n in notes:
            if _in_ranges(n.tick, hopo):
                n.flags |= FLAG_FORCE_HOPO
            if _in_ranges(n.tick, strum):
                n.flags |= FLAG_FORCE_STRUM
            if _in_ranges(n.tick, tap_ranges) or _in_ranges(n.tick, ps_tap):
                n.flags |= FLAG_TAP
            if _in_ranges(n.tick, star_ranges):
                n.flags |= FLAG_STAR_POWER
            if ps_open and _in_ranges(n.tick, ps_open) and n.frets & 1:
                n.frets = (n.frets & ~1) | OPEN_BIT
        chart.notes[d] = notes
        chart.star_power[d] = star_ranges
    return chart


def load_chart(folder_path) -> ChartData | None:
    # notes.chart wins when a folder has both notes.chart and notes.mid
    folder = Path(folder_path)
    ini = parse_ini(folder / 'song.ini')
    try:
        delay_ms = float(ini.get('delay', 0) or 0)
    except ValueError:
        delay_ms = 0.0
    chart_path = folder / 'notes.chart'
    mid_path = folder / 'notes.mid'
    if chart_path.exists():
        chart = parse_chart(chart_path, delay_ms)
    elif mid_path.exists():
        chart = parse_mid(mid_path, delay_ms)
    else:
        return None
    chart.ini = ini
    return chart


def parse_song(folder_path, difficulty='medium'):
    chart = load_chart(folder_path)
    if chart is None:
        return None
    return chart.events_seconds(difficulty) or None
