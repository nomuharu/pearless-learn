# FX 5分足方向予測 — モデル設計書 v3

## 1. プロジェクト概要

USDJPY 5分足データから、**次の5分足で価格が上昇するか下降するかを予測**し、自動売買のエントリーシグナルを出すシステム。

以前のCNNモデル（1分足ベース、振れ幅分類、精度約64%〜78%）を発展させ、最新のDLアーキテクチャで精度向上を目指す。

### 予測タスク定義

| 項目 | 内容 |
|---|---|
| ベース時間足 | **5分足** |
| 入力 | 直近60本の5分足 × 16特徴量（= 過去5時間分） |
| 出力 | 3クラス分類: `UP` / `DOWN` / `NEUTRAL` |
| UPの定義 | 次の5分足の終値 − 現在の終値 ≥ +θ |
| DOWNの定義 | 次の5分足の終値 − 現在の終値 ≤ −θ |
| NEUTRALの定義 | 上記どちらにも該当しない |
| 予測ホライズン | **1本先**（= 5分後） |

> **注**: 閾値θは過去データの分布（例: 変動幅の上位/下位25%など）から決定する。以前のシステムでは1分足で0.007を使用していたが、5分足では値幅が大きくなるため再調整が必要。

### 以前のシステム（1分足）との違い

| 項目 | 以前（1分足） | 今回（5分足） |
|---|---|---|
| ベース時間足 | M1 | **M5** |
| 入力長 | 25本（25分間） | **60本（5時間）** |
| 予測対象 | 振れ幅（HIGH/LOW） | **方向（UP/DOWN/NEUTRAL）** |
| モデル構成 | HIGH用 + LOW用の2モデル | **1モデルで3クラス分類** |
| アーキテクチャ | CNN | **PatchTST / iTransformer** |
| 推論頻度 | 毎分 | **5分ごと** |

### 比較モデル

| モデル | 特長 | 選定理由 |
|---|---|---|
| **PatchTST** | パッチ化 + Transformer | 時間方向の局所・広域パターン両立 |
| **iTransformer** | 変数軸Attention | テクニカル指標間の動的関係をモデル化 |
| **既存CNN** | ベースライン | 以前のモデルとの比較用（5分足に再構築） |

> **Bi-Mamba（SSM）は候補から除外。** 理由: 公式 mamba-ssm パッケージが Linux + CUDA 必須であり、GPUがほぼない環境では速度メリットを活かせない。将来GPU環境が整ったら再検討可能。

---

## 2. 開発環境

### 2.1 ローカル環境: WSL2 + Ubuntu + uv

Anaconda ではなく **WSL2 + uv** を推奨する。

| 比較項目 | Anaconda + Jupyter | WSL2 + uv |
|---|---|---|
| 環境再現性 | conda-lock必要 | uv.lock で完全再現 |
| パッケージ速度 | 遅い（conda resolve） | 極めて高速 |
| PyTorch+TA-Lib競合 | 起きやすい | pip依存で安定 |
| Kaggle/Colabとの互換 | condaの再現が面倒 | requirements.txt出力で即再現 |
| ディスク使用量 | 大（数GB〜） | 最小限 |

**セットアップ手順:**

```bash
# 1. WSL2 Ubuntu のインストール（PowerShellで実行）
wsl --install -d Ubuntu

# 2. Ubuntu 起動後、uv をインストール
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 3. プロジェクト作成
cd ~
uv init fx-predictor
cd fx-predictor

# 4. Python バージョン指定
uv python install 3.11
uv python pin 3.11

# 5. 依存パッケージ追加
uv add torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv add pandas numpy scikit-learn matplotlib
uv add pandas-ta        # テクニカル指標（TA-Libの代替、C依存なし）
uv add wandb             # 実験管理（任意）

# 6. Jupyter を使いたい場合
uv add jupyterlab
uv run jupyter lab
```

**ローカルでやること:**
- データパイプライン構築（5分足集約、特徴量計算、ラベル生成、分割）
- 小規模データでのデバッグ・動作確認
- 推論（本番運用、CPU で十分）

### 2.2 クラウド学習環境

GPUがほぼない環境では、学習はクラウドで実行する。

#### 【推奨】Kaggle Notebooks

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA T4 × 2（合計30GB VRAM） |
| 無料枠 | **週30時間**（毎週土曜UTC 0時リセット） |
| セッション持続時間 | 最大12時間 |
| セッション切断対策 | **commit mode（バックグラウンド実行）** でブラウザを閉じても学習継続 |
| 永続ストレージ | 20GB（セッション間で保持） |
| チェックポイント保存 | `/kaggle/working/` に保存 → commit の output として残る |

**Kaggle での学習ワークフロー:**

```
1. ローカル（WSL2）でデータパイプラインを実行
   → X_train.npy, y_train.npy, ... を生成（5分足ベース）

2. numpy 配列を Kaggle Dataset としてアップロード

3. Kaggle Notebook を作成
   - GPU 有効化（Settings → Accelerator → GPU T4 x2）
   - Datasetを追加 → モデル定義 + 学習コードを記述

4. 「Save Version」→「Save & Run All (Commit)」
   → バックグラウンドで学習実行（ブラウザ閉じてOK、PCシャットダウンもOK）

5. 完了後、/kaggle/working/ のモデル(.pt)をダウンロード
   → ローカルに持ってきて推論用に使う
```

#### 【次点】Lightning AI Studio

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA T4 |
| 無料枠 | 月15クレジット（≒ 月15時間GPU） |
| 環境の永続性 | auto-sleep 後もファイル・パッケージ完全保持 |
| 永続ストレージ | 100GB |
| インターフェース | VS Code（ブラウザ版） |
| おすすめ用途 | デバッグ・プロトタイピング（GPU時間が少ないため本番学習は Kaggle で） |

---

## 3. 共通データパイプライン（全モデル共通）

### 3.1 データフロー

```
USDJPY_M5.csv（5分足OHLCV）
  ※ M1データしかない場合は5分足に集約してから使用
  ↓
Step 1: テクニカル指標の計算（16特徴量）
  ↓
Step 2: ラベル生成（次の足の方向分類）
  ↓
Step 3: ウィンドウ化（直近60本を1サンプルとする）
  ↓
Step 4: 時系列分割（時間順でtrain 70% / val 15% / test 15%）
  ↓
Step 5: 正規化（trainデータの統計量でval/testも正規化）
  ↓
→ numpy配列として保存 → Kaggle にアップロード → 各モデルで学習
```

### 3.2 1分足から5分足への集約（M1データしかない場合）

```python
def aggregate_to_m5(df_m1):
    """1分足OHLCVを5分足に集約"""
    df_m1['datetime'] = pd.to_datetime(df_m1['datetime'])
    df_m1 = df_m1.set_index('datetime')

    df_m5 = df_m1.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return df_m5
```

### 3.3 入力特徴量（既存11個 + 追加候補 = 16個）

**既存（以前のモデルから引き継ぎ、5分足で再計算）:**

| # | 特徴量 | 説明 |
|---|---|---|
| 1 | MA60乖離率 | 終値とMA60の乖離度（= 5時間移動平均） |
| 2 | 天井度 | 過去60本の最高値との距離（= 5時間の天井） |
| 3 | MA20 | 20本移動平均（= 100分） |
| 4 | MA10 | 10本移動平均（= 50分） |
| 5 | 前足比 | 終値の変化率（前の5分足との比較） |
| 6 | 曜日 | 0=月〜4=金 |
| 7 | HLO | 高値 − 安値（5分足のレンジ） |
| 8 | diff_HLO_and_Average | HLOと平均レンジの差 |
| 9 | CCI(20) | 商品チャネル指数（20本 = 100分） |
| 10 | RSI(9) | 相対力指数（9本 = 45分） |
| 11 | 振れ幅 | 5分足の高値−始値 or 始値−安値 |

**追加候補（5分足予測向け）:**

| # | 特徴量 | 説明 | 追加理由 |
|---|---|---|---|
| 12 | VWAP乖離率 | 出来高加重平均価格との乖離 | 短期的な「適正位置」の指標 |
| 13 | ボリンジャーバンド%B | BB(20)における現在価格の位置 | 逆張り/順張りの判断材料 |
| 14 | MACD histogram | MACDのヒストグラム値 | モメンタムの加速/減速 |
| 15 | ATR(14) | 平均真の範囲（14本 = 70分） | ボラティリティの動的変化 |
| 16 | 時間帯 | sin/cos変換（周期=288 = 1日の5分足本数） | 東京/ロンドン/NY市場の時間効果 |

> **注**: 特徴量は最終的にアブレーションスタディで取捨選択する。5分足ではMA・RSI等の期間パラメータの意味が1分足時と異なる点に注意（MA60 = 5時間移動平均）。

### 3.4 ラベル生成

```python
def create_label(df, horizon=1, threshold=None):
    """
    次の5分足（horizon=1本先）の方向を分類
    threshold は過去データから自動決定も可能
    """
    future_close = df['close'].shift(-horizon)
    diff = future_close - df['close']

    if threshold is None:
        # 変動幅の上位25%をUP/DOWN閾値とする例
        threshold = diff.abs().quantile(0.75)

    label = np.where(diff >= threshold, 0,       # UP
            np.where(diff <= -threshold, 1,      # DOWN
                     2))                          # NEUTRAL
    return label
```

> **ポイント**: 5分足では1本先（horizon=1）が「5分後」に相当する。1分足時代の horizon=5 と同義。

### 3.5 データ分割（時系列分割）

```
|<--- Train (70%) --->|<-- Val (15%) -->|<-- Test (15%) -->|
|      2015-2021      |    2022-2023    |    2024-2025     |
```

- **ランダム分割は使わない**（未来のデータがtrainに混入するリークを防ぐ）
- Walk-forward検証も最終評価で実施推奨

### 3.6 データ量の目安（5分足）

| 期間 | 5分足の概算本数 | 備考 |
|---|---|---|
| 1年 | 約75,000本 | 24h × 60/5 × 260営業日 |
| 10年（2015-2025） | 約750,000本 | Train + Val + Test 合計 |
| Train（7年） | 約525,000本 | |
| Val（2年） | 約150,000本 | |
| Test（2年） | 約150,000本 | |

---

## 4. モデル別設計

### 4.1 モデルA: PatchTST（分類版）

**コンセプト:** 5分足系列をパッチ（小ウィンドウ）に分割してTransformerに入力。局所パターンとグローバル依存関係を両立。

```
入力: (batch, 60, 16)  ← 直近60本の5分足 × 16特徴量（= 過去5時間）
  ↓
RevIN（可逆インスタンス正規化）
  ↓
パッチ分割: patch_len=6, stride=6 → (batch, 10パッチ, 6×16=96)
  ※ 1パッチ = 30分間の5分足6本
  ↓
線形埋め込み: 96 → d_model=128
  ↓
位置エンコーディング（学習可能）
  ↓
Transformer Encoder × 3層
  - Multi-Head Attention (n_heads=8)
  - Feed-Forward (dim_ff=256)
  - Dropout=0.2
  ↓
Global Average Pooling over パッチ次元
  ↓
分類ヘッド: Dense(128) → ReLU → Dropout → Dense(3, softmax)
  ↓
出力: [UP確率, DOWN確率, NEUTRAL確率]
```

**ハイパーパラメータ:**

| パラメータ | 値 | 備考 |
|---|---|---|
| input_len | 60 | 直近60本の5分足（= 5時間） |
| patch_len | 6 | 6本 = 30分単位のパターン抽出 |
| stride | 6 | 非オーバーラップ |
| d_model | 128 | 埋め込み次元 |
| n_heads | 8 | Attention ヘッド数 |
| n_layers | 3 | Encoder層数 |
| dim_ff | 256 | FFNの中間次元 |
| dropout | 0.2 | |

> **パッチ長の設計意図**: 5分足6本 = 30分。東京/ロンドン/NY市場のセッション内での短期パターンを1パッチで捉え、パッチ間のAttentionで市場セッション間の関係を学習する。

**学習設定:**

| 項目 | 値 |
|---|---|
| 損失関数 | CrossEntropyLoss（クラス重み付き） |
| 最適化 | AdamW (lr=1e-4, weight_decay=1e-4) |
| スケジューラ | CosineAnnealingWarmRestarts (T_0=10) |
| バッチサイズ | 256 |
| 最大エポック | 100 |
| 早期停止 | Val Loss 15エポック非改善で停止 |
| クラス重み | UP/DOWNの比率に応じて自動調整（NEUTRALは軽く） |

**既存CNNとの違い:**
- CNN: 局所的なパターンのみ抽出 → PatchTST: パッチ内の局所性 + パッチ間のAttentionで広域パターンも捕捉
- CNN: 2モデル（HIGH/LOW別） → PatchTST: 1モデルで3クラス分類（シンプル）
- CNN: 入力25本の1分足（25分） → PatchTST: 入力60本の5分足（5時間）でより広い文脈を利用

---

### 4.2 モデルB: iTransformer（分類版）

**コンセプト:** 通常のTransformerが時間軸にAttentionをかけるのに対し、iTransformerは**特徴量（変数）軸にAttention**をかける。「RSIとCCIが同時に極端な値 → シグナル」のような特徴量間の相互作用を直接モデル化。

```
入力: (batch, 60, 16)  ← 直近60本の5分足 × 16特徴量
  ↓
転置: (batch, 16, 60)  ← 各特徴量を1つの「トークン」として扱う
  ↓
線形埋め込み: 60 → d_model=128（各特徴量の5時間分の時系列を圧縮）
  ↓
Transformer Encoder × 3層
  - Multi-Head Attention (n_heads=8)  ← 特徴量間のAttention
  - Feed-Forward (dim_ff=256)
  - Dropout=0.2
  ↓
全特徴量トークンを集約（Mean Pooling）
  ↓
分類ヘッド: Dense(128) → ReLU → Dropout → Dense(3, softmax)
  ↓
出力: [UP確率, DOWN確率, NEUTRAL確率]
```

**ハイパーパラメータ:** PatchTSTと同一（d_model=128, n_heads=8, n_layers=3, dim_ff=256, dropout=0.2）

**学習設定:** PatchTSTと同一（損失関数・最適化・スケジューラ共通）

**PatchTSTとの違い:**
- PatchTST: 「時間方向のパターン」に強い（トレンド、モメンタム）
- iTransformer: 「特徴量間の動的な関係性」に強い（指標のコンビネーション）
- どちらが勝つかはデータ依存 → 両方試して比較

---

## 5. 学習フェーズの全体像

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      共通部分（ローカル WSL2 で1回だけ実行）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ① データ読み込み（M5 or M1→M5集約）
  ② テクニカル指標計算（5分足ベースで16特徴量）
  ③ ラベル生成（1本先の方向分類）
  ④ ウィンドウ化（直近60本のスライディングウィンドウ）
  ⑤ 時系列 train/val/test 分割
  ⑥ 正規化（StandardScaler, trainの統計量で固定）
  ⑦ numpy配列として保存 + scaler保存（推論時に使う）
     → X_train.npy, y_train.npy, X_val.npy, ...
  ⑧ Kaggle Dataset としてアップロード

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    モデル別部分（Kaggle GPU でモデルごとに実行）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⑨ PyTorch Dataset/DataLoader作成
     ※ 両モデルとも入力は同じ (batch, 60, 16) の numpy 配列

  ⑩ モデル定義・初期化

  ⑪ 学習ループ（GPU使用）
     - チェックポイントを毎エポック /kaggle/working/ に保存
     - commit mode でバックグラウンド実行

  ⑫ 評価（同じテストデータで公平比較）

  ⑬ ベストモデル .pt をダウンロード → ローカルへ
```

### まとめ: 何が同じで何が違うか

| フェーズ | 共通？ | 実行場所 |
|---|---|---|
| データ読み込み・5分足集約 | ✅ 共通 | ローカル (WSL2) |
| 特徴量計算（5分足ベース） | ✅ 共通 | ローカル |
| ラベル生成（1本先の方向） | ✅ 共通 | ローカル |
| ウィンドウ化（60本） | ✅ 共通 | ローカル |
| train/val/test分割 | ✅ 共通 | ローカル |
| 正規化 | ✅ 共通 | ローカル |
| 入力テンソル形状 | ✅ 共通 | (batch, 60, 16) |
| モデルアーキテクチャ | ❌ モデル別 | Kaggle GPU |
| 損失関数・最適化・早期停止 | ✅ 共通 | Kaggle GPU |
| 評価メトリクス | ✅ 共通 | Kaggle GPU |
| 推論（本番） | — | ローカル CPU |

---

## 6. 評価計画

### 6.1 メトリクス

| メトリクス | 説明 | 重視度 |
|---|---|---|
| Accuracy | 全体の正解率 | ◯ |
| F1 (UP/DOWN) | UP/DOWNクラスのF1スコア | ◎（最重視） |
| Precision (UP/DOWN) | エントリー判断の精度 | ◎（実トレード直結） |
| AUC-ROC | クラス分離能力 | ◯ |
| 高信頼度の的中率 | 確率>0.8でのPrecision | ◎（以前のシステムと同じ思想） |
| 推論時間 | 1サンプルあたりの推論ms | ◯ |

### 6.2 ベースライン

- **既存CNN**（5分足データで再構築）を同条件で比較
- **ランダム分類器**（各クラスの出現率に応じた確率的予測）

### 6.3 最終的なトレードシミュレーション

モデル選定後、以下のバックテストを実施:

- 高信頼度（確率>閾値）のシグナルのみでエントリー
- スプレッド・スリッページを考慮
- シャープレシオ、最大ドローダウン、勝率を算出

---

## 7. 推論フロー（本番運用 — ローカルCPU）

```
MT4 → (新しい5分足確定時にデータ送信 via Named Pipe)
  ↓
Python (WSL2, CPU): 直近60本の5分足バッファを更新
  ↓
テクニカル指標計算（5分足ベースで16特徴量）
  ↓
正規化（学習時のscalerを使用）
  ↓
選定モデルで推論（PatchTST or iTransformer, CPU推論）
  ↓
UP確率 / DOWN確率を比較
  ↓
閾値を超えた方のシグナルを送信
  ↓
MT4 → "UP" / "DOWN" / "NOENTRY" を受け取る
```

- 推論頻度: **5分ごと**（新しい5分足が確定するたび）
- 推論時間: CPU使用時 < 50ms（モデルのパラメータ数が小さいため十分高速）
- 以前のシステムより推論頻度が1/5に減るため、CPU負荷も大幅に軽減

---

## 8. 実装スケジュール（案）

| Phase | 内容 | 実行場所 | 目安 |
|---|---|---|---|
| Phase 0 | WSL2 + uv 環境構築 | ローカル | 半日 |
| Phase 1 | 共通パイプライン（5分足集約・特徴量・ラベル・分割） | ローカル | 1-2日 |
| Phase 1.5 | numpy配列を Kaggle Dataset にアップ | ローカル→Kaggle | 30分 |
| Phase 2 | 2モデル実装 + 学習 | Kaggle GPU | 2-3日 |
| Phase 3 | 評価・比較・モデル選定 | Kaggle GPU | 1日 |
| Phase 4 | ベストモデルDL → MT4連携・リアルタイム推論整備 | ローカル | 2-3日 |
| Phase 5 | バックテスト・閾値最適化 | ローカル | 2-3日 |

---

## 9. 技術スタック

| 項目 | ツール | 備考 |
|---|---|---|
| ローカルOS | WSL2 Ubuntu | Windows上のLinux環境 |
| パッケージ管理 | uv | pip互換、高速、lockfile対応 |
| 言語 | Python 3.11 | |
| DLフレームワーク | PyTorch 2.x | ローカルはCPU版、KaggleはGPU版 |
| テクニカル指標 | pandas-ta | TA-Libの代替（C依存なし） |
| 学習環境 | Kaggle Notebooks (GPU T4×2) | commit modeでバックグラウンド学習 |
| 実験管理 | Weights & Biases (wandb) | 任意だが推奨 |
| MT4連携 | Named Pipe | 既存実装を流用 |

---

## 付録A: Kaggle 学習の詳細Tips

### チェックポイント保存コード

```python
import torch
import os

CHECKPOINT_DIR = "/kaggle/working/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']

# 使用例
for epoch in range(start_epoch, max_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)

    save_checkpoint(model, optimizer, epoch, val_loss,
                    f"{CHECKPOINT_DIR}/model_epoch{epoch}.pt")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, val_loss,
                        f"{CHECKPOINT_DIR}/best_model.pt")
```

### GPU 時間の節約術

- 前処理（5分足集約、特徴量計算、ウィンドウ化）はローカルで完了させてからアップロード。KaggleではGPUを使った学習のみ行う。
- mixed precision (fp16) を有効化して学習速度を約2倍にする:
  ```python
  scaler = torch.cuda.amp.GradScaler()
  with torch.cuda.amp.autocast():
      output = model(X)
      loss = criterion(output, y)
  ```
- 最初は直近2年分でデバッグ → 本番は全期間で commit 実行

### Kaggle Dataset のアップロード

```bash
# Kaggle CLI（WSL2から）
uv add kaggle
uv run kaggle datasets init -p ./data
# → dataset-metadata.json を編集
uv run kaggle datasets create -p ./data
```

または Kaggle Web UI から「New Dataset」→ ファイルをドラッグ&ドロップ。

---

## 付録B: Lightning AI Studio の利用（補助環境）

Kaggle の commit mode は実行途中の確認ができないため、デバッグには Lightning AI Studio が便利。

- 無料プラン: 月15クレジット（≒15時間GPU）、100GB永続ストレージ
- auto-sleep でファイル・パッケージすべて保持（次回起動で即再開可能）
- VS Code インターフェースで本格的な開発
- 4時間操作なしで auto-sleep（データは消えない、実行中プロセスは中断）
- **本番学習はGPU時間が多い Kaggle、デバッグは環境が消えない Lightning AI** という使い分けが効率的
