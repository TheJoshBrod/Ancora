import numpy as np
import pytest

from ancora.eval.services import match_onsets, score_chart

G, R, Y, B, OPEN = 1, 2, 4, 8, 32


def _chart(times, frets):
    return [(float(t), f, 0.0) for t, f in zip(times, frets, strict=True)]


def test_match_onsets_prefers_closest_pairs():
    assert match_onsets([1.0, 1.15], [1.08, 1.12]) == [(0, 0), (1, 1)]
    assert match_onsets([1.0], [1.05, 0.97]) == [(0, 1)]
    assert match_onsets([1.0], [1.2]) == []
    assert match_onsets([], [1.0]) == [] and match_onsets([1.0], []) == []


def test_identical_charts_score_perfectly():
    ref = _chart([1, 1.5, 2, 2.5], [G, R, Y | B, OPEN])
    s = score_chart(ref, ref)
    assert s['f1'] == 1 and s['lane_accuracy'] == 1 and s['contour_accuracy'] == 1
    assert s['offset_s'] == 0 and s['count_ratio'] == 1


def test_constant_offset_is_found():
    ref = _chart(np.arange(20) * 0.25 + 1, [G] * 20)
    s = score_chart(ref, _chart(np.arange(20) * 0.25 + 1.13, [G] * 20))
    assert s['f1'] == 1 and s['offset_s'] == pytest.approx(-0.13)


def test_missing_extra_and_wrong_lanes():
    ref = _chart([1, 2, 3, 4], [G, R, Y, B])
    pred = _chart([1.02, 2.0, 3.0, 5.0, 6.0], [G, Y, Y, B, B])
    s = score_chart(ref, pred, max_offset=0)
    assert s['precision'] == pytest.approx(3 / 5) and s['recall'] == pytest.approx(3 / 4)
    assert s['f1'] == pytest.approx(2 / 3)
    assert s['lane_accuracy'] == pytest.approx(2 / 3)
    assert s['count_ratio'] == pytest.approx(5 / 4)


def test_contour_rewards_same_motion_on_other_lanes():
    ref = _chart([1, 2, 3, 4], [G, R, Y, R])
    shifted = _chart([1, 2, 3, 4], [R, Y, B, Y])
    s = score_chart(ref, shifted)
    assert s['lane_accuracy'] == 0 and s['contour_accuracy'] == 1
    flat = score_chart(ref, _chart([1, 2, 3, 4], [G, G, G, G]))
    assert flat['contour_accuracy'] == 0


def test_empty_prediction():
    s = score_chart(_chart([1, 2], [G, R]), [])
    assert s['f1'] == 0 and s['recall'] == 0 and s['count_ratio'] == 0
