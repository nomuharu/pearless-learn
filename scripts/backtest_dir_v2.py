"""lstm_dir_v2 の方向予測を使った成行エントリーのシンプルバックテスト。

2段構成:
  Stage 1: lstm_focal（p_move）でボラティリティが高いバーを選別
  Stage 2: lstm_dir_v2（方向）で UP/DOWN を予測し成行エントリー

エントリー・エグジット:
  シグナル足の終値でエントリー（次足始値 ≒ 終値と近似、成行）
  1足後（5分後）の終値でエグジット

実行前提:
  - data/npy_mtf/ に MTF npy が存在すること（pipeline_mtf.py 生成済み）
  - production モデル（lstm_focal）のチェックポイントが存在すること
  - best_lstm_dir_v2.pt がカレントディレクトリ or --dir-v2-ckpt で指定

使用例:
    uv run python scripts/backtest_dir_v2.py \\
        --dir-v2-ckpt /path/to/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.configs import MODEL_CONFIGS

M5_CSV = "data/processed/USDJPY_M5.csv"
MTF_NPY_DIR = Path("data/npy_mtf")
M5_NPY_DIR = Path("data/npy")

FOCAL_CKPT = (
    "production/strategies/oco-breakout-wf/checkpoints/lstm_focal_20260611.pt"
)
SPREAD = 0.002  # 0.2 銭


def _load_model(config_name: str, ckpt_path: str) -> torch.nn.Module:
    cfg = MODEL_CONFIGS[config_name]
    model = cfg.build_model()
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def _predict_batch(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            p = model(torch.from_numpy(X[i : i + 4096].astype(np.float32))).numpy()
            out.append(p)
    return np.concatenate(out)


def _find_offset(
    close: np.ndarray, y_ref: np.ndarray, theta: float = 0.032
) -> int:
    """y_test/y_val の先頭500件でオフセットを特定する（backtest_oco と同じ方法）。"""
    diff = np.empty(len(close))
    diff[:-1] = close[1:] - close[:-1]
    diff[-1] = np.nan
    labels = np.full(len(close), 2, dtype=np.int64)
    labels[diff > theta] = 0
    labels[diff < -theta] = 1
    pattern = y_ref[:500]
    for start in range(len(labels) - len(y_ref)):
        if np.array_equal(labels[start : start + 500], pattern):
            return start
    raise RuntimeError("オフセット特定失敗。theta か npy の不一致を確認してください")


def backtest(
    p_move: np.ndarray,
    p_dir: np.ndarray,
    close: np.ndarray,
    offset: int,
    t_move: float,
    t_dir: float,
    spread: float,
) -> dict[str, float | int]:
    """成行エントリーのバックテストを実行する。

    Args:
        p_move: p_up + p_down（Stage 1 フィルタ）。
        p_dir : p_up（UP 確率、Stage 2 方向）。
        close : M5 終値系列（全期間）。
        offset: y_test の先頭が close 何番目に対応するか。
        t_move: p_move のシグナル閾値。
        t_dir : p_dir のシグナル閾値（>= t_dir → UP, <= 1-t_dir → DOWN）。
        spread: スプレッドコスト（円/pip 単位）。

    Returns:
        バックテスト結果の dict。
    """
    sig_move = p_move >= t_move
    sig_up   = p_dir >= t_dir
    sig_down = p_dir <= (1.0 - t_dir)

    n_up = n_down = n_win = 0
    pnl_sum = 0.0

    for k in range(len(p_move)):
        if not sig_move[k]:
            continue
        i = offset + k
        if i + 1 >= len(close):
            continue

        entry_px = close[i]
        exit_px  = close[i + 1]

        if sig_up[k]:
            pnl = exit_px - entry_px - spread
            n_up += 1
        elif sig_down[k]:
            pnl = entry_px - exit_px - spread
            n_down += 1
        else:
            continue

        pnl_sum += pnl
        n_win += pnl > 0

    n_total = n_up + n_down
    return {
        "trades": n_total,
        "n_up": n_up,
        "n_down": n_down,
        "win_rate": n_win / max(n_total, 1),
        "avg_pnl_sen": pnl_sum / max(n_total, 1) * 100,
        "total_pnl_yen": pnl_sum,
    }


def main(dir_v2_ckpt: str) -> None:
    # --- データ読み込み ---
    m5 = pd.read_csv(
        M5_CSV, header=None, names=["datetime", "open", "high", "low", "close", "volume"]
    )
    m5["datetime"] = pd.to_datetime(m5["datetime"])
    close = m5["close"].values

    y_val  = np.load(MTF_NPY_DIR / "y_val.npy")
    y_test = np.load(MTF_NPY_DIR / "y_test.npy")
    X_val_mtf  = np.load(MTF_NPY_DIR / "X_val_mtf.npy")
    X_test_mtf = np.load(MTF_NPY_DIR / "X_test_mtf.npy")
    X_val_m5   = np.load(M5_NPY_DIR / "X_val.npy")
    X_test_m5  = np.load(M5_NPY_DIR / "X_test.npy")

    # --- オフセット特定 ---
    A_test = _find_offset(close, y_test)
    A_val  = A_test - len(y_val)
    print(f"val :  {m5['datetime'].iloc[A_val]}  〜  {m5['datetime'].iloc[A_test]}")
    print(f"test:  {m5['datetime'].iloc[A_test]} 〜  {m5['datetime'].iloc[A_test + len(y_test)]}")

    # --- Stage 1: lstm_focal（M5 16列）で p_move を計算 ---
    focal_model = _load_model("lstm_focal", FOCAL_CKPT)
    p_focal_val  = _predict_batch(focal_model, X_val_m5)
    p_focal_test = _predict_batch(focal_model, X_test_m5)
    p_move_val  = p_focal_val[:, 0] + p_focal_val[:, 1]
    p_move_test = p_focal_test[:, 0] + p_focal_test[:, 1]

    # --- Stage 2: lstm_dir_v2（MTF 24列）で方向を計算 ---
    dir_model = _load_model("lstm_dir_v2", dir_v2_ckpt)
    p_dir_val  = _predict_batch(dir_model, X_val_mtf)
    p_dir_test = _predict_batch(dir_model, X_test_mtf)
    p_up_val  = p_dir_val[:, 0]
    p_up_test = p_dir_test[:, 0]

    # --- val でグリッドサーチ ---
    print("\n=== val でのグリッド評価 ===")
    print(f"{'t_move':>6s} {'t_dir':>6s} {'trades':>7s} {'UP':>5s} {'DOWN':>5s} "
          f"{'勝率':>7s} {'平均(銭)':>9s}")

    best: tuple[float, float, dict[str, float | int]] | None = None
    for t_move in [0.80, 0.85, 0.88, 0.90, 0.92]:
        for t_dir in [0.52, 0.54, 0.56, 0.58, 0.60]:
            r = backtest(p_move_val, p_up_val, close, A_val, t_move, t_dir, SPREAD)
            if r["trades"] < 100:
                continue
            print(
                f"{t_move:6.2f} {t_dir:6.2f} {r['trades']:7d} {r['n_up']:5d} {r['n_down']:5d} "
                f"{r['win_rate']:7.1%} {r['avg_pnl_sen']:9.3f}"
            )
            if best is None or r["avg_pnl_sen"] > best[2]["avg_pnl_sen"]:
                best = (t_move, t_dir, r)

    if best is None:
        print("シグナルなし（閾値条件を満たすトレードが 100 件以下）")
        return

    t_sel, td_sel, r_val = best
    print(
        f"\n選択運用点: t_move={t_sel}, t_dir={td_sel} "
        f"(val 平均 {r_val['avg_pnl_sen']:.3f}銭, {r_val['trades']}件)"
    )

    # --- test に適用 ---
    print("\n=== test 適用（val 選択の 1 点） ===")
    r_test = backtest(p_move_test, p_up_test, close, A_test, t_sel, td_sel, SPREAD)
    print(f"trades    : {r_test['trades']}")
    print(f"UP/DOWN   : {r_test['n_up']} / {r_test['n_down']}")
    print(f"勝率      : {r_test['win_rate']:.1%}")
    print(f"平均(銭)  : {r_test['avg_pnl_sen']:.3f}")
    print(f"合計(円)  : {r_test['total_pnl_yen']:.3f}")
    print()
    print("参考: 標準ロット(10万通貨)の場合、平均(銭)×1000 = 円/トレード")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="lstm_dir_v2 成行バックテスト")
    parser.add_argument(
        "--dir-v2-ckpt",
        required=True,
        help="lstm_dir_v2 チェックポイント .pt のパス",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.dir_v2_ckpt)
