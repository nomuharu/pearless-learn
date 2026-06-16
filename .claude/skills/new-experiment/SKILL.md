---
invocation_mode: ask
---

# new-experiment: 新手法実験セットアップ

新しいモデル手法・特徴量を試すときに実行する。
実験ディレクトリの作成からKaggleアップロード準備まで統一手順で進める。

## 使い方

```
/new-experiment [実験名] [suffix]
```

- `実験名`: モデル設定名（例: `lstm_dir_v2`）
- `suffix`: npy保存ディレクトリのサフィックス（例: `mtf`）

引数を省略した場合はユーザーに確認する。

## 実行手順

以下のステップを順番に実施する。

### Step 1: 入力確認

ユーザーに以下を確認する（省略時のみ）:
- モデル名（MODEL_NAME, 例: `lstm_dir_v3`）
- データサフィックス（SUFFIX, 例: `mtf2`）
- 実験の概要（前回との変更点）

### Step 2: 制約チェック

以下を絶対に触らないことを確認する:
- `production/` 配下
- `mt4/PearlessBreakout.mq4`
- 既存モデルの重みファイル（`*.pt`）

### Step 3: .gitignore への追記

`.gitignore` に以下のパターンを追加する（既存の `data/npy/*` ブロックの直後）:

```
data/npy_<SUFFIX>/*
!data/npy_<SUFFIX>/.gitkeep
!data/npy_<SUFFIX>/dataset-metadata.json
```

**重要**: 除外→許可の順（`data/npy_<SUFFIX>/*` の直後に `!` 行）にすること。順序が逆だと許可が効かない。

### Step 4: モデル設定追加

`models/configs.py` に新エントリを追加する:

```python
"<MODEL_NAME>": ModelConfig(
    name="<MODEL_NAME>",
    model_cls=LSTMModel,
    features=ALL_FEATURES,  # または適切な features 定数
    model_kwargs={"hidden_size": 128, "n_layers": 2},
    train=TrainConfig(early_stop_metric="val_f1_updown"),
),
```

既存エントリに `features=` が明示されていない場合は `features=M5_FEATURES` を追記して後方互換を保つ。

### Step 5: パイプライン作成

`pipeline_<SUFFIX>.py` を新規作成する:
- 出力: `data/npy_<SUFFIX>/X_train_<SUFFIX>.npy`, `X_val_<SUFFIX>.npy`, `X_test_<SUFFIX>.npy`
- ラベル（`y_*.npy`）は変更がなければ既存 `data/npy/` から再利用可
- 上位足特徴量は `pd.merge_asof(direction='backward')` でルックアヘッドを防ぐ
- 末尾に `assert list(df.columns) == list(ALL_FEATURES)` でカラム順を検証する

### Step 6: データ生成ディレクトリ準備

```bash
mkdir -p data/npy_<SUFFIX>
touch data/npy_<SUFFIX>/.gitkeep
```

`data/npy_<SUFFIX>/dataset-metadata.json` を作成する:

```json
{
  "title": "Pearless USDJPY M5 <SUFFIX上文字>",
  "id": "nomuhosokawa/pearless-usdjpy-m5-<SUFFIX>",
  "licenses": [{"name": "CC0-1.0"}]
}
```

### Step 7: スモークテスト

```bash
uv run python pipeline_<SUFFIX>.py --rows 5000
```

- shape が `(N, 60, <n_features>)` であることを確認
- NaN が 0 であることを確認

### Step 8: 学習ノートブック作成

`notebooks/train_<MODEL_NAME>.ipynb` を作成する。
既存の `notebooks/train_lstm_dir_v2.ipynb` を参考に以下を変更:
- データセット名（`pearless-usdjpy-m5-<SUFFIX>`）
- npy ファイル名（`X_train_<SUFFIX>.npy` など）
- MODEL_NAME 定数
- shape アサーション（`X_train.shape[2] == <n_features>`）

### Step 9: バックテストスクリプト作成

`scripts/backtest_<MODEL_NAME>.py` を作成する。
`scripts/backtest_dir_v2.py` を参考に Stage 2 モデル名と npy パスを変更する。

### Step 10: 実験記録作成

`docs/plans/experiment_<MODEL_NAME>.md` を作成する（以下の構成で）:

```markdown
# 実験記録: <MODEL_NAME>

作成日: <YYYY-MM-DD>

## 背景・動機
（前回実験との差分、なぜこの変更か）

## 仮説

## 追加特徴量
| 特徴量名 | 計算元 | 意味 |
|----------|--------|------|

## 成功基準
| 指標 | 目標値 | 前回 |
|------|--------|------|
| val_f1_updown | > 0.54 | ~0.50 |
| val_accuracy  | > 54%  | 50.3% |
| test 平均(銭) | > 0.1銭 | — |
| test 取引件数 | > 100件 | — |

## 制約
- production/ 配下は変更しない
- 既存モデルの重みは上書きしない
```

### Step 11: Kaggleアップロード手順を案内

以下のコマンドをユーザーに提示して終了する:

```bash
# 1. データ npy を生成
uv run python pipeline_<SUFFIX>.py

# 2. Kaggle Dataset 新規作成（初回のみ）
kaggle datasets create -p data/npy_<SUFFIX>

# 3. ソースコードを pearless-src に反映
cp -r models data/src/models
cp -r inference data/src/inference
kaggle datasets version -p data/src/ -m "add <MODEL_NAME>" --dir-mode zip

# 4. Kaggle でノートブックを実行 → best_model.pt をダウンロード

# 5. バックテスト
uv run python scripts/backtest_<MODEL_NAME>.py --ckpt /path/to/best_model.pt
```
