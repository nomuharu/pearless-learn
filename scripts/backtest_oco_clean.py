"""Phase A-4: p_clean ゲート統合バックテスト。

Work Plan: docs/plans/work_plan_oco_strategy_improvement.md § Phase A

シグナル条件: p_move >= t_move AND p_clean >= t_clean
  - p_move: lstm_focal の p_up + p_down（move 検知）
  - p_clean: lstm_clean の P(クリーン) を往復/クリーン2クラスに正規化した値

手順（リークなし）:
  - (t_move, t_clean) のグリッドを val で評価して1点選択 → test に適用
  - 往復バーの期待値は理論値（−δ−spread−slip）で算入（決定的・シード不要）
  - 比較対象: p_clean ゲートなし（t_clean=0）の従来運用点

実行例:
    uv run python scripts/backtest_oco_clean.py \\
        --move-model /tmp/eval-models-v2/best_lstm_focal.pt \\
        --clean-model /tmp/eval-models-v2/best_lstm_clean.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models.configs import MODEL_CONFIGS  # noqa: E402

COLS = ["datetime", "open", "high", "low", "close", "volume"]
DELTA = 0.025
SPREAD = 0.002
SLIP = 0.003
THETA = 0.032


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
    assert a_test >= 0
    a_val = a_test - len(y_val)
    mismatch = (labels[a_val : a_val + len(y_val)] != y_val).mean()
    assert mismatch < 0.01
    return a_val, a_test


def predict(model_name: str, model_path: Path, X: np.ndarray) -> np.ndarray:
    cfg = MODEL_CONFIGS[model_name]
    model = cfg.build_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            out.append(model(torch.from_numpy(X[i : i + 4096].astype(np.float32))).numpy())
    return np.concatenate(out)


def run(
    sig_mask: np.ndarray,
    a_offset: int,
    close: np.ndarray,
    m5_dt: np.ndarray,
    m1_dt: np.ndarray,
    m1_high: np.ndarray,
    m1_low: np.ndarray,
) -> pd.DataFrame:
    """シグナルマスクに対する OCO トレード履歴 (time, pnl) を返す。

    往復（同一M1で両側到達）は理論期待値 −δ−spread−slip で算入する。
    """
    recs = []
    for k in np.where(sig_mask)[0]:
        i = a_offset + k
        if i + 1 >= len(close):
            continue
        p0 = close[i]
        bt, st = p0 + DELTA, p0 - DELTA
        exit_px = close[i + 1]
        t0 = m5_dt[i + 1]
        t1 = t0 + np.timedelta64(5, "m")
        lo = np.searchsorted(m1_dt, t0, "left")
        hi = np.searchsorted(m1_dt, t1, "left")
        if hi <= lo:
            continue
        side = 0
        pnl = None
        for j in range(lo, hi):
            hb = m1_high[j] >= bt
            hs = m1_low[j] <= st
            if hb and hs:
                pnl = -DELTA - SPREAD - SLIP
                break
            if hb:
                side = 1
                break
            if hs:
                side = -1
                break
        if pnl is None:
            if side == 0:
                continue
            pnl = ((exit_px - bt) if side == 1 else (st - exit_px)) - SPREAD - SLIP
        recs.append((t0, pnl))
    return pd.DataFrame(recs, columns=["time", "pnl"])


def main() -> None:
    parser = argparse.ArgumentParser(description="p_cleanゲート統合バックテスト")
    parser.add_argument("--move-model", type=Path, required=True)
    parser.add_argument("--clean-model", type=Path, required=True)
    args = parser.parse_args()

    m5 = pd.read_csv(REPO / "data/processed/USDJPY_M5.csv", header=None, names=COLS)
    m5["datetime"] = pd.to_datetime(m5["datetime"])
    m1 = pd.read_csv(REPO / "data/raw/USDJPY_M1.csv", header=None, names=COLS)
    m1["datetime"] = pd.to_datetime(m1["datetime"])
    close = m5["close"].values
    market = (close, m5["datetime"].values, m1["datetime"].values,
              m1["high"].values, m1["low"].values)

    y_test = np.load(REPO / "data/npy/y_test.npy")
    y_val = np.load(REPO / "data/npy/y_val.npy")
    a_val, a_test = find_offsets(close, y_test, y_val)

    X_val = np.load(REPO / "data/npy/X_val.npy")
    X_test = np.load(REPO / "data/npy/X_test.npy")
    pm_val = predict("lstm_focal", args.move_model, X_val)
    pm_test = predict("lstm_focal", args.move_model, X_test)
    p_move_val = pm_val[:, 0] + pm_val[:, 1]
    p_move_test = pm_test[:, 0] + pm_test[:, 1]
    pc_val = predict("lstm_clean", args.clean_model, X_val)
    pc_test = predict("lstm_clean", args.clean_model, X_test)
    # クラス0=往復, 1=クリーン。2クラスに正規化した P(クリーン)
    p_clean_val = pc_val[:, 1] / (pc_val[:, 0] + pc_val[:, 1])
    p_clean_test = pc_test[:, 1] / (pc_test[:, 0] + pc_test[:, 1])

    # ---------- 診断: p_clean は往復を判別できているか（test、ラベル参照） ----------
    y_clean_test = np.load(REPO / "data/labels/y_clean_test.npy")
    touched = y_clean_test != 2
    sig94 = p_move_test >= 0.94
    print("=== 診断: p_move>=0.94 シグナルバーでの往復率（p_clean 分位別、test） ===")
    m = sig94 & touched
    qs = np.quantile(p_clean_test[m], [0.25, 0.5, 0.75])
    bins = [-np.inf, *qs, np.inf]
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mm = m & (p_clean_test > b0) & (p_clean_test <= b1)
        if mm.sum() == 0:
            continue
        whip = (y_clean_test[mm] == 0).mean()
        print(f"  p_clean ({b0:7.3f}, {b1:7.3f}]: n={mm.sum():5d} 往復率={whip:.1%}")

    # ---------- val でグリッド選択 ----------
    print("\n=== val グリッド（δ=2.5銭, spread 0.2銭 + slip 0.3銭） ===")
    print(f"{'t_move':>6s} {'t_clean':>7s} {'trades':>7s} {'平均(銭)':>9s} {'合計(円/ドル)':>11s}")
    best = None
    for t_move in [0.88, 0.90, 0.92, 0.94]:
        for t_clean in [0.0, 0.5, 0.6, 0.7, 0.8]:
            mask = (p_move_val >= t_move) & (p_clean_val >= t_clean)
            df = run(mask, a_val, *market)
            if len(df) < 200:
                continue
            avg = df["pnl"].mean() * 100
            print(f"{t_move:6.2f} {t_clean:7.2f} {len(df):7d} {avg:9.3f} "
                  f"{df['pnl'].sum():11.3f}")
            if best is None or avg > best[2]:
                best = (t_move, t_clean, avg, len(df))

    assert best is not None
    t_m, t_c, avg_v, n_v = best
    print(f"\n選択: t_move={t_m}, t_clean={t_c}（val平均 {avg_v:.3f}銭, {n_v}件）")

    # ---------- test 適用 ----------
    print("\n=== test 適用 ===")
    candidates = [("baseline (t_move=0.94, gateなし)", 0.94, 0.0)]
    if (t_m, t_c) != (0.94, 0.0):
        candidates.append((f"選択点 (t_move={t_m}, t_clean={t_c})", t_m, t_c))
    for name, tm, tc in candidates:
        mask = (p_move_test >= tm) & (p_clean_test >= tc)
        df = run(mask, a_test, *market)
        df["month"] = pd.to_datetime(df["time"]).dt.to_period("M")
        recent = df[df["month"] >= pd.Period("2025-05")]
        avg = df["pnl"].mean() * 100
        print(f"{name}: {len(df)}件 平均 {avg:.3f}銭 合計 {df['pnl'].sum():.1f}円/ドル | "
              f"直近12ヶ月 {len(recent)}件 {recent['pnl'].sum()*100*1000:+,.0f}円/標準ロット")


if __name__ == "__main__":
    main()
