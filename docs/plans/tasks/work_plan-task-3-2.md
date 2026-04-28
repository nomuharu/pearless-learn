# Task 3-2: 全モデル比較評価実行と PRD 成功基準確認

## タスク概要

Kaggle で学習済みの PatchTST / iTransformer / CNN を同一 `X_test.npy` / `y_test.npy` で評価し、PRD 成功基準（UP/DOWN F1 +5pt、Precision@0.8 ≥ 70%）の達成状況を確認する。評価結果を `logs/` に保存する。

## 対象ファイル

- `logs/evaluation_results_{timestamp}.csv` (生成物)

## 調査対象

- `/home/nomu/claude_code/pearless/evaluate.py` (Task 3-1 の実装)
- `/home/nomu/claude_code/pearless/docs/prd.md` (PRD 定量的成功基準)
- `/home/nomu/claude_code/pearless/docs/design/fx-prediction-design.md` (§ Acceptance Criteria AC-015/016)
- `data/X_test.npy`, `data/y_test.npy` (Task 1-1 で生成されたテストデータ)

## 実行手順

### Step 1: 学習済みモデルの準備

Kaggle から各モデルの `best_model.pt` をダウンロードし、`data/` に配置する:
- `data/best_patchtst.pt`
- `data/best_itransformer.pt`
- `data/best_cnn.pt`

### Step 2: 全モデル評価実行

```bash
uv run python evaluate.py \
    --model all \
    --model-path-dir data/ \
    --test-data data/ \
    --output-dir logs/ \
    --threshold 0.8
```

### Step 3: PRD 成功基準の確認

生成された CSV を確認して以下を検証する:

| 基準 | 閾値 | 確認方法 |
|---|---|---|
| UP/DOWN F1 が CNN ベースラインより +5pt 以上 | CNN F1 + 0.05 | CSV の f1_up / f1_down 列を比較 |
| Precision@0.8 ≥ 70% | 0.70 | CSV の precision_at_0.8 列 |
| 推論時間 < 50ms | 50ms | AC-017 確認済み（Task 2-7） |

### Step 4: 評価結果の記録

評価結果サマリーを確認して最良モデルを選定する。

## 品質保証メカニズム

このタスクは評価実行タスクのため、コード品質メカニズムの対象外。PRD 成功基準の達成確認が主目的。

## 動作確認方法

```bash
# 評価実行
uv run python evaluate.py --model all --model-path-dir data/ --test-data data/ --output-dir logs/

# CSV 確認
python -c "
import csv
with open('logs/evaluation_results_latest.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f\"{row['model']}: F1_UP={row['f1_up']}, Precision@0.8={row['precision_at_0.8']}\")
"
```

**成功基準**:
- 評価 CSV が生成され、3モデルの比較が可能な状態 (AC-015)
- PRD 成功基準の達成状況がレポートに記載される

**検証レベル**: L1（機能動作検証）

## 完了条件

- [x] Implementation: `evaluate.py --model all` が 3 モデル比較 CSV を出力する (AC-015)
- [x] Quality: 評価 CSV に PatchTST / iTransformer / CNN の全行が含まれる (AC-021)
- [x] Integration: PRD 定量的成功基準（F1 +5pt、Precision@0.8 ≥ 70%）の達成状況が確認され `logs/` に保存されている
