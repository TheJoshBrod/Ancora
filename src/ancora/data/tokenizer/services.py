import json
from pathlib import Path

import numpy as np

from ancora.data.tokenizer.constants import (
    BEAT,
    BOS,
    DEFAULT_SUSTAIN_CUTOFF_BEATS,
    EOS,
    FRETS_OFFSET,
    PAD,
    POS_OFFSET,
    SNAP_TOLERANCE,
    SUBDIV,
    SUSTAIN_CLASSES,
    VOCAB_SIZE,
)
from ancora.data.tokenizer.schemas import TokenizeResult, TokenStats


def pos_token(k):
    return POS_OFFSET + int(k)


def frets_token(mask):
    return FRETS_OFFSET + int(mask) - 1


def is_pos(tok):
    return POS_OFFSET <= tok < FRETS_OFFSET


def is_frets(tok):
    return FRETS_OFFSET <= tok < VOCAB_SIZE


def sustain_class(length_beats, cutoff=DEFAULT_SUSTAIN_CUTOFF_BEATS):
    length_beats = np.asarray(length_beats, dtype=np.float64)
    cls = np.abs(length_beats[..., None] - SUSTAIN_CLASSES[1:]).argmin(-1) + 1
    cls = np.where(length_beats >= SUSTAIN_CLASSES[-1], len(SUSTAIN_CLASSES) - 1, cls)
    return np.where(length_beats < cutoff, 0, cls).astype(np.int8)


def tokenize(
    start_beats,
    end_beats,
    frets,
    flags=None,
    n_beats=None,
    sustain_cutoff=DEFAULT_SUSTAIN_CUTOFF_BEATS,
    subdiv=SUBDIV,
) -> TokenizeResult:
    start = np.asarray(start_beats, dtype=np.float64)
    end = np.asarray(end_beats, dtype=np.float64)
    frets = np.asarray(frets, dtype=np.int64)
    flags = (
        np.zeros(len(start), dtype=np.int64) if flags is None else np.asarray(flags, dtype=np.int64)
    )
    order = np.argsort(start, kind='stable')
    start, end, frets, flags = start[order], end[order], frets[order], flags[order]

    steps = start * subdiv
    snapped = np.round(steps)
    snap_err = np.abs(steps - snapped)
    n_before = int((snapped < 0).sum())
    snapped = np.maximum(snapped, 0).astype(np.int64)

    uniq, inv = np.unique(snapped, return_inverse=True)
    m_frets = np.zeros(len(uniq), dtype=np.int64)
    m_flags = np.zeros(len(uniq), dtype=np.int64)
    m_end = np.full(len(uniq), -np.inf)
    np.bitwise_or.at(m_frets, inv, frets)
    np.bitwise_or.at(m_flags, inv, flags)
    np.maximum.at(m_end, inv, end)
    n_collisions = len(start) - len(uniq)

    m_start = uniq / subdiv
    next_start = np.append(m_start[1:], np.inf) if len(m_start) else m_start
    length = np.clip(np.minimum(m_end, next_start) - m_start, 0, None)
    sus = sustain_class(length, sustain_cutoff)

    last_beat = int(uniq[-1] // subdiv) if len(uniq) else -1
    n_beats = max(n_beats or 0, last_beat + 1)

    tokens, sus_arr, flag_arr = [BOS], [-1], [0]
    j = 0
    for b in range(n_beats):
        tokens.append(BEAT)
        sus_arr.append(-1)
        flag_arr.append(0)
        while j < len(uniq) and uniq[j] // subdiv == b:
            tokens += [pos_token(uniq[j] % subdiv), frets_token(m_frets[j])]
            sus_arr += [-1, int(sus[j])]
            flag_arr += [0, int(m_flags[j])]
            j += 1
    tokens.append(EOS)
    sus_arr.append(-1)
    flag_arr.append(0)

    stats = TokenStats(
        n_notes=len(start),
        n_tokens=len(tokens),
        n_beats=int(n_beats),
        n_collisions=int(n_collisions),
        n_offgrid=int((snap_err > SNAP_TOLERANCE).sum()),
        max_snap_err=float(snap_err.max()) if len(snap_err) else 0.0,
        n_before_first_beat=n_before,
    )
    return TokenizeResult(
        tokens=np.asarray(tokens, dtype=np.int16),
        sustain=np.asarray(sus_arr, dtype=np.int8),
        flags=np.asarray(flag_arr, dtype=np.uint8),
        stats=stats,
    )


def detokenize(tokens, sustain=None, subdiv=SUBDIV):
    out, beat, pos = [], -1, None
    sustain = None if sustain is None else np.asarray(sustain)
    for i, tok in enumerate(np.asarray(tokens).tolist()):
        if tok == BEAT:
            beat += 1
            pos = None
        elif is_pos(tok):
            pos = tok - POS_OFFSET
        elif is_frets(tok) and pos is not None and beat >= 0:
            sus = 0.0
            if sustain is not None and sustain[i] > 0:
                sus = float(SUSTAIN_CLASSES[sustain[i]])
            out.append((beat + pos / subdiv, tok - FRETS_OFFSET + 1, sus))
            pos = None
        elif tok == EOS:
            break
    return out


def lane_shift(tokens, start, end, shift, max_fret=4):
    # None when any shifted fret would leave 0..max_fret
    toks = np.array(tokens, copy=True)
    for i in range(start, min(end, len(toks))):
        tok = int(toks[i])
        if not is_frets(tok):
            continue
        mask = tok - FRETS_OFFSET + 1
        open_bit, lanes = mask & 32, mask & 31
        new = 0
        for lane in range(5):
            if lanes >> lane & 1:
                tgt = lane + shift
                if not 0 <= tgt <= max_fret:
                    return None
                new |= 1 << tgt
        toks[i] = frets_token(new | open_bit)
    return toks


def save_tokens(output_dir, song_id, arrays):
    song_dir = Path(output_dir) / 'tokens' / song_id
    song_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(song_dir / 'tokens.npz', **arrays)


def vocab_description():
    return {
        'PAD': PAD,
        'BOS': BOS,
        'EOS': EOS,
        'BEAT': BEAT,
        'POS_k': (
            f'{POS_OFFSET}..{POS_OFFSET + SUBDIV - 1} (k = token - {POS_OFFSET}, 1/{SUBDIV} beat)'
        ),
        'FRETS_m': (
            f'{FRETS_OFFSET}..{VOCAB_SIZE - 1} (m = token - {FRETS_OFFSET - 1}; '
            'bits 0-4 G R Y B O, bit 5 open)'
        ),
        'vocab_size': VOCAB_SIZE,
        'sustain_classes_beats': SUSTAIN_CLASSES.tolist(),
        'subdiv': SUBDIV,
    }


def write_vocab_config(out_dir):
    path = Path(out_dir) / 'tokenizer_config.json'
    vocab = vocab_description()
    if path.exists():
        if json.loads(path.read_text()) != vocab:
            raise ValueError(
                f'{path} describes a different token vocabulary; write to a new --out folder'
            )
        return path
    path.write_text(json.dumps(vocab, indent=2) + '\n')
    return path


def times_from_beat_positions(positions, beat_times_s) -> np.ndarray:
    p = np.asarray(positions, dtype=np.float64)
    b = np.asarray(beat_times_s, dtype=np.float64)
    idx = np.arange(len(b), dtype=np.float64)
    t = np.interp(p, idx, b)
    first, last = b[1] - b[0], b[-1] - b[-2]
    t = np.where(p < 0, b[0] + p * first, t)
    t = np.where(p > len(b) - 1, b[-1] + (p - (len(b) - 1)) * last, t)
    return t


def subdivision_times(beat_times_s, subdiv=SUBDIV) -> np.ndarray:
    n = (len(beat_times_s) - 1) * subdiv + 1
    return times_from_beat_positions(np.arange(n) / subdiv, beat_times_s)
