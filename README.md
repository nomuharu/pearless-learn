# Pearless - USDJPY 5分足方向予測 ML システム

USDJPY の 5 分足データから次の 5 分足の方向（UP / DOWN / NEUTRAL）を予測する ML システムです。
PatchTST / iTransformer / CNN ベースラインの 3 モデルを比較評価し、自動売買エントリーシグナルの生成を目的としています。

---

## セットアップ手順

### 1. 依存関係のインストール

```bash
uv sync
```

### 2. Kaggle 認証設定

https://www.kaggle.com/settings/api で「Create New Token」をクリックし、`~/.kaggle/kaggle.json` に配置してください。

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

> **注意**: `kaggle.json` の `key` フィールドには通常の API キーではなく、`~/.kaggle/` 配下の access token を使用してください。通常の API キーでは書き込み系操作（データセット作成・ノートブック push）が 401 エラーになる場合があります。

---

## データパイプライン実行手順

### データディレクトリ構成

```
data/
├── raw/          # 生データ（M1 CSV）← ここに配置
├── processed/    # リサンプリング後（M5 CSV）
└── npy/          # pipeline.py 出力（numpy 配列）
```

各ディレクトリは git 追跡されますが、ファイルは `.gitignore` で除外されています。

### 1. 1分足 → 5分足 リサンプリング

MT4 等からエクスポートした 1 分足 CSV（ヘッダーなし、列: `datetime, open, high, low, close, volume`）を `data/raw/` に配置し、リサンプリングします。

```bash
uv run python scripts/resample_m1_to_m5.py
# デフォルト: data/raw/USDJPY_M1.csv → data/processed/USDJPY_M5.csv

# パスを指定する場合:
uv run python scripts/resample_m1_to_m5.py \
    --input data/raw/USDJPY_M1.csv \
    --output data/processed/USDJPY_M5.csv
```

### 2. 特徴量計算・分割・正規化

5 分足 CSV から 16 特徴量計算・ラベル生成・時系列分割・正規化・numpy 配列保存を行います。

```bash
uv run python pipeline.py --csv-path data/processed/USDJPY_M5.csv --output-dir data/npy/
```

成功すると `data/npy/` ディレクトリに以下のファイルが生成されます。

```
data/npy/
├── X_train.npy   (N_train, 60, 16)
├── y_train.npy   (N_train,)
├── X_val.npy     (N_val, 60, 16)
├── y_val.npy     (N_val,)
├── X_test.npy    (N_test, 60, 16)
├── y_test.npy    (N_test,)
└── scaler.pkl    (StandardScaler)
```

---

## Kaggle 学習手順

### 1. Kaggle Dataset アップロード

パイプライン出力（numpy 配列）を Kaggle Dataset に公開します。

```bash
uv run --env-file .env python scripts/upload_dataset.py --data-dir data/npy/ --dataset-name pearless-usdjpy-m5
```

確認のみ（ドライラン）:

```bash
uv run --env-file .env python scripts/upload_dataset.py --data-dir data/npy/ --dataset-name pearless-usdjpy-m5 --dry-run
```

### 2. ノートブック実行方法

`notebooks/` ディレクトリの Jupyter ノートブックを Kaggle にアップロードし、GPU（T4）を有効化して「Save & Run All」（commit mode）で実行してください。

| ノートブック | モデル |
|---|---|
| `notebooks/train_patchtst.ipynb` | PatchTST |
| `notebooks/train_itransformer.ipynb` | iTransformer |
| `notebooks/train_cnn.ipynb` | CNN ベースライン |
| `notebooks/compare_models.ipynb` | モデル比較 |

学習完了後、`/kaggle/working/best_model.pt` をローカルの `data/` ディレクトリにダウンロードしてください。

---

## 評価手順

ダウンロードした `best_model.pt` と `data/` の numpy 配列を使って全モデルを比較評価します。

```bash
# 単一モデルの評価
python evaluate.py --model patchtst \
                   --model-path data/best_patchtst.pt \
                   --test-data data/ \
                   --output-dir logs/

# 全モデル一括比較
python evaluate.py --model all \
                   --model-path-dir data/ \
                   --test-data data/ \
                   --output-dir logs/ \
                   --threshold 0.8
```

評価結果は `logs/` ディレクトリに CSV ファイルとして出力されます。
`--threshold` オプションで高信頼度的中率（デフォルト 0.8）の計算閾値を変更できます。

---

## 推論手順

### InferenceEngine の使い方（Named Pipe スタブ経由）

```python
import torch
from models.patchtst import PatchTST
from inference.engine import InferenceEngine
from inference.pipe_stub import NamedPipeStub

# モデルのロード
model = PatchTST()
model.load_state_dict(torch.load("data/best_patchtst.pt", map_location="cpu"))

# スタブ（MT4 なしでのダミーデータ生成）を使用した推論
stub = NamedPipeStub()
engine = InferenceEngine(model=model, scaler_path="data/scaler.pkl", data_source=stub)

result = engine.predict()
print(result)
# {
#     "signal": "UP",
#     "probabilities": {"UP": 0.72, "DOWN": 0.18, "NEUTRAL": 0.10},
#     "inference_ms": 12.3,
# }
```

実際の MT4 との連携は `DataSourceInterface` を継承したクラスを実装し、`data_source` に渡すだけで切り替えられます（インターフェース変更なし）。

---

## アーキテクチャ概要

```
data/raw/USDJPY_M1.csv
    └─ scripts/resample_m1_to_m5.py
           └─ data/processed/USDJPY_M5.csv
                  └─ pipeline.py（16特徴量計算・分割・正規化）
                         └─ data/npy/（X_*.npy, y_*.npy, scaler.pkl）
                                ├─ scripts/upload_dataset.py → Kaggle Dataset
                                │          └─ notebooks/*.ipynb → best_model.pt
                                ├─ inference/engine.py（CPU推論）
                                │     ├─ inference/pipe_stub.py（MT4スタブ）
                                │     └─ models/（PatchTST / iTransformer / CNN）
                                └─ evaluate.py（メトリクス比較CSV出力）→ logs/
```

---

## 実装の設計書対応関係

各モジュールのコードコメントに設計書・ADR・AC 番号との対応を明示しています。

| ファイル | 対応する設計書セクション |
|---|---|
| `pipeline.py` | Design Doc § Data Flow, § Contract Definitions / ADR-0002 |
| `models/patchtst.py` | Design Doc § PatchTST / ADR-0001 |
| `models/itransformer.py` | Design Doc § iTransformer |
| `models/cnn.py` | Design Doc § CNN ベースライン |
| `models/training.py` | Design Doc § 学習ループ共通設定 |
| `inference/engine.py` | Design Doc § InferenceEngine |
| `inference/interface.py` | Design Doc § DataSourceInterface |
| `inference/pipe_stub.py` | Design Doc § NamedPipeStub |
| `evaluate.py` | Design Doc § 評価メトリクス定義 |
| `scripts/upload_dataset.py` | Design Doc § Kaggle Dataset 連携 |

---

## 注意事項

- wandb は使用しません（スコープ外）。実験管理は `logs/` の CSV ログのみで行います。
- Kaggle API トークンはコードにハードコードせず、`.env` ファイルまたは環境変数で管理してください。
- ローカル実行は CPU（WSL2）を前提としています。Kaggle 学習は GPU T4 で行います。
