import os
import re
import mido

def parse_ini(filepath):
    """Parses song.ini to extract delay and other metadata."""
    ini_data = {}
    if not os.path.exists(filepath):
        return ini_data
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, val = line.split('=', 1)
                ini_data[key.strip().lower()] = val.strip()
    return ini_data

def parse_chart(filepath, delay_ms=0):
    """Parses a .chart file. Extracts MediumSingle notes."""
    resolution = 192
    bpms = [] # list of (tick, bpm)
    notes = [] # list of (tick, fret_bitmask, duration_ticks)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    current_section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('//'): continue
        
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1]
            continue
            
        if current_section == 'Song':
            if line.lower().startswith('resolution'):
                match = re.search(r'=\s*(\d+)', line)
                if match: resolution = int(match.group(1))
        
        elif current_section == 'SyncTrack':
            # e.g., 0 = B 113072
            match = re.match(r'(\d+)\s*=\s*B\s+(\d+)', line)
            if match:
                tick = int(match.group(1))
                bpm = float(match.group(2)) / 1000.0
                bpms.append((tick, bpm))
                
        elif current_section == 'MediumSingle':
            # e.g., 192 = N 0 0
            match = re.match(r'(\d+)\s*=\s*N\s+(\d)\s+(\d+)', line)
            if match:
                tick = int(match.group(1))
                fret = int(match.group(2))
                duration = int(match.group(3))
                
                # Fret mapping: 0=Green, 1=Red, 2=Yellow, 3=Blue, 4=Orange, 7=Open
                if fret in [0, 1, 2, 3, 4, 7]:
                    bit = fret if fret != 7 else 5
                    notes.append((tick, 1 << bit, duration))
                    
    if not notes:
        return None # No Medium track or no notes
        
    # Process overlapping/simultaneous notes (chords)
    notes.sort(key=lambda x: x[0])
    chords = []
    current_tick = -1
    current_mask = 0
    current_dur = 0
    for tick, mask, dur in notes:
        if tick == current_tick:
            current_mask |= mask
            current_dur = max(current_dur, dur)
        else:
            if current_tick != -1:
                chords.append((current_tick, current_mask, current_dur))
            current_tick = tick
            current_mask = mask
            current_dur = dur
    if current_tick != -1:
        chords.append((current_tick, current_mask, current_dur))
        
    # Convert ticks to seconds
    bpms.sort(key=lambda x: x[0])
    if not bpms:
        bpms = [(0, 120.0)]
        
    def ticks_to_seconds(tick):
        t_sec = 0.0
        current_tick = 0
        current_bpm = bpms[0][1]
        
        for b_tick, b_bpm in bpms:
            if tick > b_tick:
                delta_ticks = b_tick - current_tick
                t_sec += delta_ticks * (60.0 / (current_bpm * resolution))
                current_tick = b_tick
                current_bpm = b_bpm
            else:
                break
                
        delta_ticks = tick - current_tick
        t_sec += delta_ticks * (60.0 / (current_bpm * resolution))
        return t_sec

    events = []
    for tick, mask, dur in chords:
        t_sec = ticks_to_seconds(tick) + (delay_ms / 1000.0)
        dur_sec = ticks_to_seconds(tick + dur) - ticks_to_seconds(tick)
        events.append((t_sec, mask, dur_sec))
        
    return events

def parse_mid(filepath, delay_ms=0):
    """Parses a .mid file. Extracts Medium difficulty from PART GUITAR."""
    # Community/Harmonix MIDIs often contain out-of-range data bytes, which strict
    # mido rejects; clip=True clamps them. Truly corrupt files (e.g. truncated)
    # still fail and are reported as errors rather than "no Medium track".
    try:
        mid = mido.MidiFile(filepath, clip=True)
    except Exception as e:
        raise ValueError(f"Unreadable MIDI: {e!r}") from e
        
    resolution = mid.ticks_per_beat
    
    # Extract tempo map
    tempo_map = [] # list of (tick, tempo)
    current_tick = 0
    
    # Assume tempo map is in the first track
    for msg in mid.tracks[0]:
        current_tick += msg.time
        if msg.type == 'set_tempo':
            tempo_map.append((current_tick, msg.tempo))
            
    if not tempo_map:
        tempo_map = [(0, 500000)] # 120 bpm default
        
    def ticks_to_seconds(tick):
        t_sec = 0.0
        c_tick = 0
        c_tempo = tempo_map[0][1]
        for t_tick, t_tempo in tempo_map:
            if tick > t_tick:
                t_sec += mido.tick2second(t_tick - c_tick, resolution, c_tempo)
                c_tick = t_tick
                c_tempo = t_tempo
            else:
                break
        t_sec += mido.tick2second(tick - c_tick, resolution, c_tempo)
        return t_sec

    notes = []
    for track in mid.tracks:
        if track.name == 'PART GUITAR':
            c_tick = 0
            active_notes = {}
            for msg in track:
                c_tick += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    pitch = msg.note
                    # Medium is 72 to 76 (Green to Orange). 77/78 are the Medium
                    # force-HOPO / force-strum markers, NOT notes, so they must be
                    # excluded. MIDI open notes are signalled via SysEx / ENHANCED_OPENS
                    # and are not handled yet.
                    if 72 <= pitch <= 76:
                        active_notes[pitch] = c_tick
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    pitch = msg.note
                    if pitch in active_notes:
                        start_tick = active_notes[pitch]
                        dur_ticks = c_tick - start_tick
                        
                        fret = pitch - 72  # 0 (Green) .. 4 (Orange)
                        notes.append((start_tick, 1 << fret, dur_ticks))
                        del active_notes[pitch]
            break
            
    if not notes:
        return None
        
    notes.sort(key=lambda x: x[0])
    chords = []
    current_tick = -1
    current_mask = 0
    current_dur = 0
    for tick, mask, dur in notes:
        if tick == current_tick:
            current_mask |= mask
            current_dur = max(current_dur, dur)
        else:
            if current_tick != -1:
                chords.append((current_tick, current_mask, current_dur))
            current_tick = tick
            current_mask = mask
            current_dur = dur
    if current_tick != -1:
        chords.append((current_tick, current_mask, current_dur))
        
    events = []
    for tick, mask, dur in chords:
        t_sec = ticks_to_seconds(tick) + (delay_ms / 1000.0)
        dur_sec = ticks_to_seconds(tick + dur) - ticks_to_seconds(tick)
        events.append((t_sec, mask, dur_sec))
        
    return events

def parse_song(folder_path):
    ini_path = os.path.join(folder_path, 'song.ini')
    ini_data = parse_ini(ini_path)
    delay_ms = float(ini_data.get('delay', 0))
    
    chart_path = os.path.join(folder_path, 'notes.chart')
    mid_path = os.path.join(folder_path, 'notes.mid')
    
    if os.path.exists(chart_path):
        return parse_chart(chart_path, delay_ms)
    elif os.path.exists(mid_path):
        return parse_mid(mid_path, delay_ms)
    else:
        return None
