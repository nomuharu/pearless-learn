"""OCOブレイクアウト戦略の再検証（リークなし手順）。

修正点:
  1. t_move と δ は validation 期間のみで選択し、test には選んだ1点だけを適用
  2. 往復バー（同一M1で両側トリガー到達）は理論期待値 −δ−spread として算入
  3. スリッページ感度（0 / 0.3 / 0.5銭）を test 結果に併記
"""

import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/nomu/claude_code/pearless")

from models.configs import MODEL_CONFIGS  # noqa: E402

M5_CSV = "/home/nomu/claude_code/pearless/data/processed/USDJPY_M5.csv"
M1_CSV = "/home/nomu/claude_code/pearless/data/raw/USDJPY_M1.csv"
COLS = ["datetime", "open", "high", "low", "close", "volume"]
DATA = "/home/nomu/claude_code/pearless/data/npy"

m5 = pd.read_csv(M5_CSV, header=None, names=COLS)
m5["datetime"] = pd.to_datetime(m5["datetime"])
close = m5["close"].values
m5_dt = m5["datetime"].values

y_test = np.load(f"{DATA}/y_test.npy")
y_val = np.load(f"{DATA}/y_val.npy")

# ---------- オフセット特定（test と val 両方を検証） ----------
diff = np.empty(len(close))
diff[:-1] = close[1:] - close[:-1]
diff[-1] = np.nan
THETA = 0.032
labels = np.full(len(close), 2, dtype=np.int64)
labels[diff > THETA] = 0
labels[diff < -THETA] = 1

pattern = y_test[:500]
A_test = -1
for start in range(len(labels) - len(y_test)):
    if np.array_equal(labels[start : start + 500], pattern):
        A_test = start
        break
assert A_test >= 0
assert np.array_equal(labels[A_test : A_test + len(y_test)], y_test)
A_val = A_test - len(y_val)
# theta=0.032 ちょうどの浮動小数点境界ケース(0.12%)があるため完全一致は要求しない
mismatch = (labels[A_val : A_val + len(y_val)] != y_val).mean()
assert mismatch < 0.01, f"valオフセット不一致率が高すぎる: {mismatch:.4f}"
print(f"val:  {m5['datetime'].iloc[A_val]} 〜 {m5['datetime'].iloc[A_test]}")
print(f"test: {m5['datetime'].iloc[A_test]} 〜 {m5['datetime'].iloc[A_test + len(y_test)]}")

# ---------- p_move を val / test 両方で計算 ----------
cfg = MODEL_CONFIGS["lstm_focal"]
model = cfg.build_model()
model.load_state_dict(
    torch.load("/tmp/eval-models-v2/best_lstm_focal.pt", map_location="cpu", weights_only=True)
)
model.eval()


def predict_move(X: np.ndarray) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            p = model(torch.from_numpy(X[i : i + 4096].astype(np.float32))).numpy()
            out.append(p[:, 0] + p[:, 1])
    return np.concatenate(out)


p_move_test = np.load("/tmp/p_move_test.npy")
p_move_test = p_move_test[:, 0] + p_move_test[:, 1]
p_move_val = predict_move(np.load(f"{DATA}/X_val.npy"))

m1 = pd.read_csv(M1_CSV, header=None, names=COLS)
m1["datetime"] = pd.to_datetime(m1["datetime"])
m1_dt = m1["datetime"].values
m1_high = m1["high"].values
m1_low = m1["low"].values


def run(p_move: np.ndarray, A: int, t_move: float, delta: float, spread: float,
        slippage: float = 0.0):
    """往復バーは期待値 −δ−spread−slippage で算入した推定値を返す。"""
    sig_idx = np.where(p_move >= t_move)[0]
    n_clean = n_amb = n_win = 0
    pnl_sum = 0.0
    for k in sig_idx:
        i = A + k
        if i + 1 >= len(close):
            continue
        p0 = close[i]
        buy_trig, sell_trig = p0 + delta, p0 - delta
        exit_px = close[i + 1]
        t0 = m5_dt[i + 1]
        t1 = t0 + np.timedelta64(5, "m")
        lo = np.searchsorted(m1_dt, t0, "left")
        hi = np.searchsorted(m1_dt, t1, "left")
        if hi <= lo:
            continue
        side = 0
        ambiguous = False
        for j in range(lo, hi):
            hb = m1_high[j] >= buy_trig
            hs = m1_low[j] <= sell_trig
            if hb and hs:
                ambiguous = True
                break
            if hb:
                side = 1
                break
            if hs:
                side = -1
                break
        if ambiguous:
            n_amb += 1
            pnl_sum += -delta - spread - slippage
            continue
        if side == 0:
            continue
        pnl = (exit_px - buy_trig) if side == 1 else (sell_trig - exit_px)
        pnl -= spread + slippage
        n_clean += 1
        n_win += pnl > 0
        pnl_sum += pnl
    n_total = n_clean + n_amb
    return {
        "trades": n_total,
        "clean": n_clean,
        "ambiguous": n_amb,
        "win_rate_clean": n_win / max(n_clean, 1),
        "avg_pnl_sen": pnl_sum / max(n_total, 1) * 100,
        "total_pnl_yen": pnl_sum,
    }


SPREAD = 0.002

# ---------- 1. val でグリッドを評価して運用点を選ぶ ----------
print("\n=== val でのグリッド評価（spread 0.2銭・スリッページなし） ===")
print(f"{'t_move':>6s} {'δ銭':>5s} {'trades':>7s} {'amb':>6s} {'平均(銭)':>9s}")
best = None
for t_move in [0.85, 0.88, 0.90, 0.92, 0.93, 0.94]:
    for delta in [0.010, 0.015, 0.020, 0.025, 0.030]:
        r = run(p_move_val, A_val, t_move, delta, SPREAD)
        if r["trades"] < 200:  # 統計的に薄い運用点は除外
            continue
        print(f"{t_move:6.2f} {delta*100:5.1f} {r['trades']:7d} {r['ambiguous']:6d} "
              f"{r['avg_pnl_sen']:9.3f}")
        if best is None or r["avg_pnl_sen"] > best[2]["avg_pnl_sen"]:
            best = (t_move, delta, r)

assert best is not None
t_sel, d_sel, r_val = best
print(f"\n選択された運用点: t_move={t_sel}, δ={d_sel*100:.1f}銭 "
      f"(val平均 {r_val['avg_pnl_sen']:.3f}銭/トレード, {r_val['trades']}件)")

# ---------- 2. test に適用（スリッページ感度付き） ----------
print("\n=== test 適用（val選択の1点のみ・スリッページ感度） ===")
print(f"{'slip銭':>7s} {'trades':>7s} {'clean':>6s} {'amb':>6s} {'clean勝率':>8s} "
      f"{'平均(銭)':>9s} {'合計(円/ドル)':>12s}")
for slip in [0.0, 0.003, 0.005]:
    r = run(p_move_test, A_test, t_sel, d_sel, SPREAD, slippage=slip)
    print(f"{slip*100:7.1f} {r['trades']:7d} {r['clean']:6d} {r['ambiguous']:6d} "
          f"{r['win_rate_clean']:8.1%} {r['avg_pnl_sen']:9.3f} {r['total_pnl_yen']:12.3f}")
print("\n参考: 標準ロット(10万通貨)の場合、平均(銭)×1000 = 円/トレード")
