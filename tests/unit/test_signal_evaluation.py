# evaluate.py のシグナル評価（クラス別閾値）ユニットテスト
# 方針: max 確率閾値（NEUTRAL が独占）の代わりに UP/DOWN 個別確率に閾値をかけ、
#       運用点は validation set で選んでから test set に適用する

import math

import numpy as np
import pytest

from evaluate import select_operating_points, signal_metrics, signal_sweep


def _make_probs(p_up: list[float], p_down: list[float]) -> np.ndarray:
    """UP/DOWN 確率を指定し、残りを NEUTRAL に割り当てた確率配列を作る。"""
    up = np.asarray(p_up, dtype=np.float64)
    down = np.asarray(p_down, dtype=np.float64)
    neutral = 1.0 - up - down
    return np.stack([up, down, neutral], axis=1)


# ============================================================
# signal_sweep
# ============================================================
def test_signal_sweep_counts_and_precision():
    """
    閾値ごとのシグナル件数と的中率が正しく計算されること

    Arrange: UP 確率 [0.7, 0.7, 0.4, 0.1]、正解 [UP, NEUTRAL, UP, DOWN]
    Act: thresholds=(0.5,) でスイープ
    Assert: UP は 2 件中 1 件的中（precision=0.5、coverage=0.5）
    """
    y_prob = _make_probs([0.7, 0.7, 0.4, 0.1], [0.1, 0.1, 0.1, 0.2])
    y_true = np.array([0, 2, 0, 1])

    sweep = signal_sweep(y_true, y_prob, thresholds=(0.5,))

    up_row = sweep[sweep["class"] == "UP"].iloc[0]
    assert up_row["n_signals"] == 2
    assert up_row["precision"] == pytest.approx(0.5)
    assert up_row["coverage"] == pytest.approx(0.5)


def test_signal_sweep_zero_signals_gives_nan_precision():
    """
    シグナル 0 件の閾値では precision が NaN になること
    """
    y_prob = _make_probs([0.3, 0.2], [0.1, 0.1])
    y_true = np.array([0, 1])

    sweep = signal_sweep(y_true, y_prob, thresholds=(0.9,))

    assert (sweep["n_signals"] == 0).all()
    assert sweep["precision"].isna().all()


def test_signal_sweep_excludes_neutral():
    """
    スイープ対象が UP / DOWN のみで NEUTRAL を含まないこと
    （NEUTRAL の高確信度はエントリーシグナルにならないため）
    """
    y_prob = _make_probs([0.5, 0.5], [0.2, 0.2])
    y_true = np.array([0, 1])

    sweep = signal_sweep(y_true, y_prob, thresholds=(0.4,))

    assert set(sweep["class"]) == {"UP", "DOWN"}


# ============================================================
# select_operating_points
# ============================================================
def test_select_operating_points_picks_lowest_qualifying_threshold():
    """
    target_precision と min_signals を満たす閾値のうち件数最大
    （= 最も低い閾値）が選ばれること

    Arrange: UP 確率 0.55 のサンプル 10 件（全部正解）
    Act: target_precision=0.6, min_signals=5, thresholds=(0.4, 0.5, 0.6)
    Assert: t=0.6 は 0 件で不適、0.4 と 0.5 は 10 件で適格 → 件数最大は同数なので
            idxmax で先頭の 0.4 が選ばれる
    """
    y_prob = _make_probs([0.55] * 10, [0.1] * 10)
    y_true = np.zeros(10, dtype=np.int64)

    points = select_operating_points(
        y_true, y_prob, target_precision=0.6, min_signals=5,
        thresholds=(0.4, 0.5, 0.6),
    )

    assert points["t_up"] == pytest.approx(0.4)
    assert math.isnan(points["t_down"])  # DOWN は件数 0 で不適格


def test_select_operating_points_respects_min_signals():
    """
    的中率 100% でも件数が min_signals 未満なら運用点に選ばれないこと
    （少数サンプルの的中率はノイズのため）
    """
    y_prob = _make_probs([0.9, 0.9, 0.1], [0.05, 0.05, 0.1])
    y_true = np.array([0, 0, 2])

    points = select_operating_points(
        y_true, y_prob, target_precision=0.6, min_signals=3,
        thresholds=(0.5,),
    )

    assert math.isnan(points["t_up"])


def test_select_operating_points_no_qualifying_returns_nan():
    """
    target_precision を満たす閾値がない場合は NaN が返ること
    """
    # UP 確率は高いが正解は全部 NEUTRAL → precision 0
    y_prob = _make_probs([0.7] * 50, [0.1] * 50)
    y_true = np.full(50, 2, dtype=np.int64)

    points = select_operating_points(
        y_true, y_prob, target_precision=0.6, min_signals=10,
    )

    assert math.isnan(points["t_up"])
    assert math.isnan(points["t_down"])


# ============================================================
# signal_metrics
# ============================================================
def test_signal_metrics_applies_thresholds_per_class():
    """
    クラス別閾値での的中率と件数が test set に正しく適用されること

    Arrange: UP 確率 [0.7, 0.65, 0.3]、DOWN 確率 [0.1, 0.1, 0.6]、
             正解 [UP, NEUTRAL, DOWN]
    Act: t_up=0.6, t_down=0.5
    Assert: UP シグナル 2 件中 1 件的中、DOWN シグナル 1 件中 1 件的中
    """
    y_prob = _make_probs([0.7, 0.65, 0.3], [0.1, 0.1, 0.6])
    y_true = np.array([0, 2, 1])

    metrics = signal_metrics(y_true, y_prob, t_up=0.6, t_down=0.5)

    assert metrics["n_signals_up"] == 2
    assert metrics["precision_up_signal"] == pytest.approx(0.5)
    assert metrics["n_signals_down"] == 1
    assert metrics["precision_down_signal"] == pytest.approx(1.0)


def test_signal_metrics_nan_threshold_means_no_signals():
    """
    運用点が NaN（val で適格閾値なし）の場合、シグナル 0 件・precision NaN になること
    """
    y_prob = _make_probs([0.7, 0.7], [0.2, 0.2])
    y_true = np.array([0, 0])

    metrics = signal_metrics(y_true, y_prob, t_up=math.nan, t_down=math.nan)

    assert metrics["n_signals_up"] == 0
    assert metrics["n_signals_down"] == 0
    assert math.isnan(metrics["precision_up_signal"])
    assert math.isnan(metrics["precision_down_signal"])
