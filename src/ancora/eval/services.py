import itertools

import numpy as np

ONSET_TOLERANCE_S = 0.1
MAX_OFFSET_S = 0.2
OFFSET_STEP_S = 0.005


def match_onsets(ref_times, pred_times, tol=ONSET_TOLERANCE_S):
    # greedy one-to-one, closest pairs first; returns sorted (ref_index, pred_index)
    ref, pred = np.asarray(ref_times, dtype=np.float64), np.asarray(pred_times, dtype=np.float64)
    pred_order = np.argsort(pred, kind='stable')
    pred_sorted = pred[pred_order]
    lo = np.searchsorted(pred_sorted, ref - tol, side='left')
    hi = np.searchsorted(pred_sorted, ref + tol, side='right')
    pairs = sorted(
        (abs(ref[r] - pred_sorted[p]), r, int(pred_order[p]))
        for r in range(len(ref))
        for p in range(lo[r], hi[r])
    )
    used_ref, used_pred, matches = set(), set(), []
    for _, r, p in pairs:
        if r not in used_ref and p not in used_pred:
            used_ref.add(r)
            used_pred.add(p)
            matches.append((r, p))
    return sorted(matches)


def _anchor_lane(frets):
    lanes = frets & 0b11111
    return (lanes & -lanes).bit_length() - 1 if lanes else -1


def score_chart(ref, pred, tol=ONSET_TOLERANCE_S, max_offset=MAX_OFFSET_S, step=OFFSET_STEP_S):
    # ref, pred: [(time_s, frets, ...)]; pred is shifted by the global offset that matches the
    # most onsets, since charts often sit a constant offset from the audio
    ref_t = np.array([e[0] for e in ref], dtype=np.float64)
    pred_t = np.array([e[0] for e in pred], dtype=np.float64)
    ref_f = [e[1] for e in ref]
    pred_f = [e[1] for e in pred]

    n_steps = round(max_offset / step)
    best_key, offset, matches = None, 0.0, []
    for k in range(-n_steps, n_steps + 1):
        shifted = pred_t + k * step
        found = match_onsets(ref_t, shifted, tol)
        err = np.mean([abs(ref_t[r] - shifted[p]) for r, p in found]) if found else 0.0
        key = (len(found), -err, -abs(k))
        if best_key is None or key > best_key:
            best_key, offset, matches = key, k * step, found
    n_match = len(matches)

    precision = n_match / len(pred) if pred else 0.0
    recall = n_match / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lane = np.mean([ref_f[r] == pred_f[p] for r, p in matches]) if matches else 0.0
    steps = [
        (
            np.sign(_anchor_lane(ref_f[r2]) - _anchor_lane(ref_f[r1])),
            np.sign(_anchor_lane(pred_f[p2]) - _anchor_lane(pred_f[p1])),
        )
        for (r1, p1), (r2, p2) in itertools.pairwise(matches)
    ]
    contour = np.mean([a == b for a, b in steps]) if steps else 0.0
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'offset_s': offset,
        'lane_accuracy': float(lane),
        'contour_accuracy': float(contour),
        'count_ratio': len(pred) / len(ref) if ref else 0.0,
    }
