"""Phase B: OCOブレイクアウト戦略の決済ルール比較バックテスト。

Work Plan: docs/plans/work_plan_oco_strategy_improvement.md § Phase B

ベースライン（次の5分足終値で決済）に対して以下を比較する:
  - hold N: N本（5分足）保有して終値決済（N=1がベースライン）
  - TP/SL: トリガー価格から +TP銭 利確 / −SL銭 損切り（M1パスで判定、期限は hold 上限）
  - trailing: M1ごとに有利方向の極値−T銭 を追従するストップ
  - break-even: +BE銭 乗ったらストップを建値へ移動（SL/TP併用）

手順（リークなし）:
  - 運用点 t_move=0.94, δ=2.5銭 は前段の val 選択を固定流用
  - 決済ルールとパラメータは val で選択（3シード平均）、test には選んだ1点のみ適用
  - 往復バー（同一M1で両側トリガー到達）は 50/50 ランダムで約定側を決定

実行例:
    uv run python scripts/backtest_oco_exits.py \\
        --model-path /tmp/eval-models-v2/best_lstm_focal.pt
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models.configs import MODEL_CONFIGS  # noqa: E402
COLS = ["datetime", "open", "high", "low", "close", "volume"]

T_MOVE = 0.94
DELTA = 0.025
SPREAD = 0.002
SLIP = 0.003
THETA = 0.032


@dataclass(frozen=True)
class ExitRule:
    """決済ルール。値は円（0.01 = 1銭）。None は無効。"""

    name: str
    hold_bars: int = 1  # 最大保有本数（5分足）
    tp: float | None = None  # トリガー価格からの利確幅
    sl: float | None = None  # トリガー価格からの損切り幅
    trail: float | None = None  # トレーリング幅
    break_even: float | None = None  # この幅だけ乗ったらSLを建値へ


RULES: tuple[ExitRule, ...] = (
    ExitRule("baseline_hold1"),
    ExitRule("hold3", hold_bars=3),
    ExitRule("hold6", hold_bars=6),
    ExitRule("tp3_sl1.5", hold_bars=6, tp=0.030, sl=0.015),
    ExitRule("tp5_sl1.5", hold_bars=6, tp=0.050, sl=0.015),
    ExitRule("tp5_sl2.5", hold_bars=6, tp=0.050, sl=0.025),
    ExitRule("tp8_sl2.5", hold_bars=6, tp=0.080, sl=0.025),
    ExitRule("sl1.5_only", hold_bars=6, sl=0.015),
    ExitRule("sl2.5_only", hold_bars=6, sl=0.025),
    ExitRule("trail1.5", hold_bars=6, trail=0.015),
    ExitRule("trail2.5", hold_bars=6, trail=0.025),
    ExitRule("be1_sl2.5", hold_bars=6, sl=0.025, break_even=0.010),
    ExitRule("be1_sl2.5_tp5", hold_bars=6, tp=0.050, sl=0.025, break_even=0.010),
)


def load_market():
    m5 = pd.read_csv(REPO / "data/processed/USDJPY_M5.csv", header=None, names=COLS)
    m5["datetime"] = pd.to_datetime(m5["datetime"])
    m1 = pd.read_csv(REPO / "data/raw/USDJPY_M1.csv", header=None, names=COLS)
    m1["datetime"] = pd.to_datetime(m1["datetime"])
    return m5, m1


def find_offsets(close: np.ndarray, y_test: np.ndarray, y_val: np.ndarray):
    diff = np.empty(len(close))
    diff[:-1] = close[1:] - close[:-1]
    diff[-1] = np.nan
    labels = np.full(len(close), 2, dtype=np.int64)
    labels[diff > THETA] = 0
    labels[diff < -THETA] = 1
    pattern = y_test[:500]
    a_test = -1
    for start in range(len(labels) - len(y_test)):
        if np.array_equal(labels[start : start + 500], pattern):
            a_test = start
            break
    assert a_test >= 0, "testオフセット特定失敗"
    a_val = a_test - len(y_val)
    mismatch = (labels[a_val : a_val + len(y_val)] != y_val).mean()
    assert mismatch < 0.01, f"valオフセット不一致率: {mismatch:.4f}"
    return a_val, a_test


def predict_p_move(model_path: Path, X: np.ndarray) -> np.ndarray:
    cfg = MODEL_CONFIGS["lstm_focal"]
    model = cfg.build_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            p = model(torch.from_numpy(X[i : i + 4096].astype(np.float32))).numpy()
            out.append(p[:, 0] + p[:, 1])
    return np.concatenate(out)


def simulate(
    rule: ExitRule,
    p_move: np.ndarray,
    a_offset: int,
    close: np.ndarray,
    m5_dt: np.ndarray,
    m1_dt: np.ndarray,
    m1_high: np.ndarray,
    m1_low: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """決済ルール1つ分のトレード履歴 (time, pnl) を返す。"""
    sig_idx = np.where(p_move >= T_MOVE)[0]
    recs = []
    last_exit_time = np.datetime64("1970-01-01")

    for k in sig_idx:
        i = a_offset + k
        if i + rule.hold_bars + 1 >= len(close):
            continue
        bar_start = m5_dt[i + 1]
        if bar_start < last_exit_time:  # 建玉保有中のシグナルはスキップ
            continue

        p0 = close[i]
        buy_trig, sell_trig = p0 + DELTA, p0 - DELTA
        deadline = bar_start + np.timedelta64(5 * rule.hold_bars, "m")
        lo = np.searchsorted(m1_dt, bar_start, "left")
        hi_entry = np.searchsorted(m1_dt, bar_start + np.timedelta64(5, "m"), "left")
        hi_all = np.searchsorted(m1_dt, deadline, "left")
        if hi_entry <= lo:
            continue

        # --- エントリー判定（最初の5分以内、OCO） ---
        side = 0
        j_entry = -1
        for j in range(lo, hi_entry):
            hb = m1_high[j] >= buy_trig
            hs = m1_low[j] <= sell_trig
            if hb and hs:
                side = 1 if rng.random() < 0.5 else -1
                j_entry = j
                break
            if hb:
                side = 1
                j_entry = j
                break
            if hs:
                side = -1
                j_entry = j
                break
        if side == 0:
            continue

        entry = buy_trig if side == 1 else sell_trig

        # --- 決済シミュレーション（M1パス、エントリーバー含む。バー内は悲観順） ---
        tp_px = entry + side * rule.tp if rule.tp is not None else None
        sl_px = entry - side * rule.sl if rule.sl is not None else None
        peak = entry
        exit_px = None
        for j in range(j_entry, hi_all):
            hi_j, lo_j = m1_high[j], m1_low[j]
            fav = hi_j if side == 1 else lo_j  # 有利方向の極値
            adv = lo_j if side == 1 else hi_j  # 不利方向の極値

            # 悲観順: 先に損切り側を判定（同一バー内でTP/SL両方届く場合はSL扱い）
            if sl_px is not None and (adv - sl_px) * side <= 0:
                exit_px = sl_px
                break
            if rule.trail is not None:
                trail_px = peak - side * rule.trail
                if (adv - trail_px) * side <= 0 and (trail_px - entry) * side != 0:
                    exit_px = trail_px
                    break
            if tp_px is not None and (fav - tp_px) * side >= 0:
                exit_px = tp_px
                break

            # バー通過後に有利方向の極値を更新（トレーリング・建値移動用）
            if (fav - peak) * side > 0:
                peak = fav
            if (
                rule.break_even is not None
                and sl_px is not None
                and (peak - entry) * side >= rule.break_even
                and (sl_px - entry) * side < 0
            ):
                sl_px = entry  # 建値ストップへ移動

        if exit_px is None:
            exit_px = close[i + rule.hold_bars]  # 期限到来: 保有最終バーの終値
            exit_time = deadline
        else:
            exit_time = m1_dt[min(j, hi_all - 1)] + np.timedelta64(1, "m")

        pnl = (exit_px - entry) * side - SPREAD - SLIP
        recs.append((bar_start, pnl))
        last_exit_time = exit_time

    return pd.DataFrame(recs, columns=["time", "pnl"])


def main() -> None:
    parser = argparse.ArgumentParser(description="決済ルール比較バックテスト")
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    m5, m1 = load_market()
    close = m5["close"].values
    m5_dt = m5["datetime"].values
    m1_dt = m1["datetime"].values
    m1_high = m1["high"].values
    m1_low = m1["low"].values

    y_test = np.load(REPO / "data/npy/y_test.npy")
    y_val = np.load(REPO / "data/npy/y_val.npy")
    a_val, a_test = find_offsets(close, y_test, y_val)
    print(f"val:  {m5['datetime'].iloc[a_val]} 〜 {m5['datetime'].iloc[a_test]}")
    print(f"test: {m5['datetime'].iloc[a_test]} 〜")

    p_move_test_path = Path("/tmp/p_move_test.npy")
    if p_move_test_path.exists():
        pm = np.load(p_move_test_path)
        p_move_test = pm[:, 0] + pm[:, 1] if pm.ndim == 2 else pm
    else:
        p_move_test = predict_p_move(args.model_path, np.load(REPO / "data/npy/X_test.npy"))
    p_move_val = predict_p_move(args.model_path, np.load(REPO / "data/npy/X_val.npy"))

    market = (close, m5_dt, m1_dt, m1_high, m1_low)

    # ---------- val でルール選択（3シード平均） ----------
    print(f"\n=== val でのルール比較（t_move={T_MOVE}, δ={DELTA*100}銭, "
          f"spread {SPREAD*100}銭 + slip {SLIP*100}銭, 3シード平均） ===")
    print(f"{'rule':>16s} {'trades':>7s} {'平均(銭)':>9s} {'合計(円/ドル)':>11s}")
    results = {}
    for rule in RULES:
        avgs, totals, n_trades = [], [], []
        for seed in range(3):
            df = simulate(rule, p_move_val, a_val, *market,
                          rng=np.random.default_rng(seed))
            avgs.append(df["pnl"].mean() * 100)
            totals.append(df["pnl"].sum())
            n_trades.append(len(df))
        results[rule.name] = (rule, float(np.mean(avgs)))
        print(f"{rule.name:>16s} {int(np.mean(n_trades)):7d} {np.mean(avgs):9.3f} "
              f"{np.mean(totals):11.3f}")

    best_rule, best_avg = max(results.values(), key=lambda rv: rv[1])
    print(f"\n選択ルール: {best_rule.name}（val平均 {best_avg:.3f}銭/トレード）")

    # ---------- test に適用（5シード） ----------
    print(f"\n=== test 適用: {best_rule.name} vs baseline_hold1（5シード） ===")
    for rule in (RULES[0], best_rule):
        avgs, totals, dds = [], [], []
        monthly_last = None
        for seed in range(5):
            df = simulate(rule, p_move_test, a_test, *market,
                          rng=np.random.default_rng(seed))
            cum = df["pnl"].cumsum().values
            dd = float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0.0
            avgs.append(df["pnl"].mean() * 100)
            totals.append(df["pnl"].sum())
            dds.append(dd)
            monthly_last = df
        print(f"\n{rule.name}: 平均 {np.mean(avgs):.3f}銭/トレード "
              f"(シード範囲 {min(avgs):.2f}〜{max(avgs):.2f}) | "
              f"合計 {np.mean(totals):.1f}円/ドル | 最悪DD {max(dds)*100:.0f}銭")
        assert monthly_last is not None
        monthly_last["month"] = pd.to_datetime(monthly_last["time"]).dt.to_period("M")
        recent = monthly_last[monthly_last["month"] >= pd.Period("2025-05")]
        print(f"  直近12ヶ月（2025-05〜）: {len(recent)}トレード "
              f"合計 {recent['pnl'].sum()*100*1000:,.0f}円/標準ロット")


if __name__ == "__main__":
    main()
