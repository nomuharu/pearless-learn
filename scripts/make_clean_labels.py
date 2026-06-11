"""Phase A-1: 「クリーンブレイク vs 往復」ラベルの生成。

Work Plan: docs/plans/work_plan_oco_strategy_improvement.md § Phase A

各決定バー i（既存ウィンドウ k に対応）について、次の5分足の高値/安値が
P0 ± δ（P0 = close[i]、δ = 2.5銭）に到達したかで 3 値ラベルを作る:

    0 = 往復   （両側に到達。OCOの期待値は −δ−spread で構造的に負け）
    1 = クリーン（片側のみ到達。ブレイクに乗れる）
    2 = 不到達 （どちらにも届かず。OCOは約定しない＝学習から除外する側）

既存の y_{split}.npy と同じ並び・同じ長さで y_clean_{split}.npy を出力する。
クラス 2 を除外して 0/1 の2クラス学習を行う点は train_lstm_dir.ipynb と同型。

出力先: data/labels/（pearless-aux-labels dataset としてアップロードする。
2.66GB の本体 dataset を更新せずに済ませるための分離）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

COLS = ["datetime", "open", "high", "low", "close", "volume"]
DELTA = 0.025
THETA = 0.032


def find_test_offset(close: np.ndarray, y_test: np.ndarray) -> int:
    """既存ラベルとの照合で test 先頭の決定バー CSV index を特定する。"""
    diff = np.empty(len(close))
    diff[:-1] = close[1:] - close[:-1]
    diff[-1] = np.nan
    labels = np.full(len(close), 2, dtype=np.int64)
    labels[diff > THETA] = 0
    labels[diff < -THETA] = 1
    pattern = y_test[:500]
    for start in range(len(labels) - len(y_test)):
        if np.array_equal(labels[start : start + 500], pattern):
            return start
    raise RuntimeError("test オフセット特定失敗")


def main() -> None:
    m5 = pd.read_csv(REPO / "data/processed/USDJPY_M5.csv", header=None, names=COLS)
    close = m5["close"].values
    high = m5["high"].values
    low = m5["low"].values

    y = {s: np.load(REPO / f"data/npy/y_{s}.npy") for s in ["train", "val", "test"]}
    a_test = find_test_offset(close, y["test"])
    offsets = {
        "test": a_test,
        "val": a_test - len(y["val"]),
        "train": a_test - len(y["val"]) - len(y["train"]),
    }

    out_dir = REPO / "data/labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, a in offsets.items():
        n = len(y[split])
        idx = np.arange(a, a + n)  # 決定バー
        nxt = idx + 1  # 値動きが起きる次のバー
        assert nxt[-1] < len(close)
        touched_up = high[nxt] >= close[idx] + DELTA
        touched_dn = low[nxt] <= close[idx] - DELTA

        y_clean = np.full(n, 2, dtype=np.int64)  # 2 = 不到達
        y_clean[touched_up ^ touched_dn] = 1  # 片側のみ = クリーン
        y_clean[touched_up & touched_dn] = 0  # 両側 = 往復

        counts = np.bincount(y_clean, minlength=3)
        print(
            f"{split:5s}: n={n}  往復={counts[0]} ({counts[0]/n:.1%})  "
            f"クリーン={counts[1]} ({counts[1]/n:.1%})  不到達={counts[2]} ({counts[2]/n:.1%})"
        )
        np.save(out_dir / f"y_clean_{split}.npy", y_clean)

    meta = out_dir / "dataset-metadata.json"
    if not meta.exists():
        meta.write_text(
            '{\n  "title": "Pearless Aux Labels",\n'
            '  "id": "nomuhosokawa/pearless-aux-labels",\n'
            '  "licenses": [{"name": "CC0-1.0"}]\n}\n'
        )
    print(f"\n出力: {out_dir}/y_clean_{{train,val,test}}.npy")


if __name__ == "__main__":
    main()
