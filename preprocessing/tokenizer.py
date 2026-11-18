import numpy as np
import os

def tokenize_events(events):
    """
    Quantizes events to a 100ms grid.
    events: list of (time_sec, bitmask, duration_sec)
    Returns:
      tokens_100ms: int16 array
      sustain_100ms: float32 array
      raw_events: array of (time, mask, duration)
      collision_100ms: bool
    """
    if not events:
        return np.array([64, 65], dtype=np.int16), np.array([0.0, 0.0], dtype=np.float32), np.array([]), False
        
    GRID_STEP = 0.1
    collision = False
    
    # 64 = BOS, 65 = EOS, 0 = PAD
    # find max cell
    max_time = max(e[0] for e in events)
    n_cells = int(np.round(max_time / GRID_STEP)) + 1
    
    # +2 for BOS and EOS
    tokens = np.zeros(n_cells + 2, dtype=np.int16)
    sustains = np.zeros(n_cells + 2, dtype=np.float32)
    
    tokens[0] = 64 # BOS
    tokens[-1] = 65 # EOS
    
    cell_to_event = {}
    
    for t_sec, mask, dur in events:
        cell_idx = int(np.round(t_sec / GRID_STEP))
        # shift by 1 due to BOS
        idx = cell_idx + 1
        
        if tokens[idx] != 0:
            collision = True
            
        tokens[idx] = mask
        sustains[idx] = dur
        cell_to_event[idx] = (t_sec, mask, dur)
        
    # Handle sustain truncation
    # A sustain ends when the next note starts.
    idxs = sorted(cell_to_event.keys())
    for i in range(len(idxs) - 1):
        idx = idxs[i]
        next_idx = idxs[i+1]
        
        t_sec, mask, dur = cell_to_event[idx]
        next_t_sec = cell_to_event[next_idx][0]
        
        if t_sec + dur > next_t_sec:
            # truncate
            sustains[idx] = max(0.0, next_t_sec - t_sec)
            
    raw_events = np.array(events, dtype=np.float32)
    return tokens, sustains, raw_events, collision

def save_tokens(output_dir, song_id, tokens, sustains, raw_events):
    song_dir = os.path.join(output_dir, 'tokens', song_id)
    os.makedirs(song_dir, exist_ok=True)
    
    np.save(os.path.join(song_dir, 'tokens_100ms.npy'), tokens)
    np.save(os.path.join(song_dir, 'sustain_100ms.npy'), sustains)
    np.savez_compressed(os.path.join(song_dir, 'events_raw.npz'), events=raw_events)
