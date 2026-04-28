# Task 3-1: evaluate.py 実装（AC-015/016/021）

## タスク概要

全モデルを同一テストデータで評価し、メトリクス（Accuracy / F1 / Precision / AUC-ROC / 高信頼度的中率）を計算して比較 CSV を出力する CLI スクリプトを実装する。

## 対象ファイル

- `evaluate.py` (新規)

## 調査対象

- `/home/nomu/claude_code/pearless/models/base.py` (BaseModel、クラス定数)
- `/home/nomu/claude_code/pearless/models/patchtst.py` (PatchTST)
- `/home/nomu/claude_code/pearless/models/itransformer.py` (iTransformer)
- `/home/nomu/claude_code/pearless/models/cnn.py` (CNNModel)
- `/home/nomu/claude_code/pearless/docs/design/fx-prediction-design.md` (§ 評価メトリクス定義、§ AC-015/016/021)
- `/home/nomu/claude_code/pearless/docs/plans/work_plan.md` (§ Task 3-1)

## 実装手順

### Step 1: メトリクス計算関数の設計

以下のメトリクスを計算する:

| メトリクス | 説明 | AC |
|---|---|---|
| Accuracy | 全クラス正解率 | AC-015 |
| F1(UP) | UP(0) クラスの F1 スコア | AC-015 |
| F1(DOWN) | DOWN(1) クラスの F1 スコア | AC-015 |
| Precision(UP) | UP(0) クラスの Precision | AC-015 |
| Precision(DOWN) | DOWN(1) クラスの Precision | AC-015 |
| AUC-ROC | 多クラス AUC-ROC (OvR) | AC-015 |
| 高信頼度的中率 | prob > threshold のサンプルの正解率 | AC-016 |

### Step 2: evaluate.py 実装

```python
"""evaluate.py - 全モデル評価スクリプト。

Usage:
    python evaluate.py --model patchtst --model-path data/best_patchtst.pt \\
                       --test-data data/ --output-dir logs/
    python evaluate.py --model all --model-path-dir data/ --test-data data/ \\
                       --output-dir logs/ --threshold 0.8
"""
import argparse
import csv
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, roc_auc_score
)

from models.base import BaseModel


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.8,
) -> dict[str, float]:
    """メトリクスを計算して辞書で返す。"""
    # 高信頼度サンプルのマスク
    max_prob = y_prob.max(axis=1)
    high_conf_mask = max_prob >= threshold
    high_conf_accuracy = (
        float((y_pred[high_conf_mask] == y_true[high_conf_mask]).mean())
        if high_conf_mask.sum() > 0
        else float("nan")
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_up": float(f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "f1_down": float(f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "precision_up": float(precision_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "precision_down": float(precision_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob, multi_class="ovr")),
        f"precision_at_{threshold}": high_conf_accuracy,
        "n_high_conf": int(high_conf_mask.sum()),
    }
```

### Step 3: 全モデル比較 CSV 出力（AC-021）

- `--model all` 指定時に PatchTST / iTransformer / CNN の全結果を1つの CSV にまとめる
- CNN 列が欠損なく存在することを保証

```csv
model,accuracy,f1_up,f1_down,precision_up,precision_down,auc_roc,precision_at_0.8,n_high_conf
patchtst,...
itransformer,...
cnn,...
```

### Step 4: CLI 引数設計

```python
parser.add_argument("--model", choices=["patchtst", "itransformer", "cnn", "all"])
parser.add_argument("--threshold", type=float, default=0.8)  # AC-016
parser.add_argument("--test-data", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, default=Path("logs/"))
```

## 品質保証メカニズム

このタスクには特定のファイル対象の品質メカニズムはないが、project-wide の uv sync が適用される。

## 動作確認方法

```bash
# 単一モデル評価（--dry-run 相当の動作確認）
uv run python evaluate.py --model patchtst \
    --model-path data/best_patchtst.pt \
    --test-data data/ \
    --output-dir /tmp/eval/

# --threshold オプション確認
uv run python evaluate.py --model all \
    --model-path-dir data/ \
    --test-data data/ \
    --output-dir /tmp/eval/ \
    --threshold 0.8

ls /tmp/eval/
# 期待: evaluation_results_*.csv が生成される
```

**成功基準**:
- `python evaluate.py --model patchtst` が CSV ファイルを出力する (AC-015)
- `--threshold` オプションが高信頼度的中率の計算に反映される (AC-016)
- 全モデルの比較 CSV に CNN 列が欠損なく存在する (AC-021)

**検証レベル**: L1（機能動作検証）

## 完了条件

- [x] Implementation: `python evaluate.py --model patchtst` が CSV ファイルを出力する (AC-015)
- [x] Quality: `--threshold` オプションが高信頼度的中率の計算に反映される (AC-016)
- [x] Integration: 全モデルの比較 CSV に CNN 列が欠損なく存在する (AC-021)
