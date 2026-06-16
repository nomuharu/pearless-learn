# 実験記録: lstm_dir_v2（マルチタイムフレーム方向予測）

作成日: 2026-06-16

## 背景・動機

前回実験（cade311, lstm_dir）で USDJPY M5 16特徴量のみによる方向予測を試みたが、
val 方向正答率 **50.3%（コイントス）** に終わった。

- 問題の本質: M5 OHLCV 由来の 16 特徴量だけでは「今どちらに動いているか」の文脈が不足
- エッジが存在するのはボラティリティ側（lstm_focal の p_move）のみと確認済み

方向予測に再挑戦するために、前回と異なる変数を 1 つ変える。
選択: **マルチタイムフレーム（MTF）特徴量の追加**

## 仮説

M15・H1 の上位足トレンド方向（MA20乖離率・RSI）を加えることで、
5 分足の短期ノイズに埋もれていた慣性成分を捉えられる可能性がある。

## 実装内容

### 追加特徴量（8列）

| 特徴量名           | 計算元 | 意味                    |
|--------------------|--------|-------------------------|
| m15_ma20_deviation | M15    | 中期トレンド方向        |
| m15_rsi            | M15    | 中期モメンタム（RSI9）  |
| m15_hlo_ratio      | M15    | 中期ボラティリティ水準  |
| m15_atr_ratio      | M15    | 中期ボラティリティスケール |
| h1_ma20_deviation  | H1     | 長期トレンド方向        |
| h1_rsi             | H1     | 長期モメンタム（RSI14） |
| h1_hlo_ratio       | H1     | 長期ボラティリティ水準  |
| h1_bb_pband        | H1     | 長期価格帯位置（BB%B）  |

ルックアヘッド対策: `pd.merge_asof(direction='backward')` で
各 M5 バーに「その時刻より前の最新確定バー」の値を付与。

### 変更ファイル

| ファイル                        | 変更内容                                              |
|---------------------------------|-------------------------------------------------------|
| `models/configs.py`             | `ALL_FEATURES` を 16→24 列に拡張。`M5_FEATURES` 定数追加。既存モデルに `features=M5_FEATURES` を明示。`lstm_dir_v2` エントリ追加 |
| `pipeline_mtf.py`               | 新規作成。MTF 24列 npy を生成するパイプライン         |
| `notebooks/train_lstm_dir_v2.ipynb` | 新規作成。Kaggle 用学習ノートブック               |
| `scripts/backtest_dir_v2.py`    | 新規作成。成行エントリーのバックテストスクリプト      |

### データ

| ファイル              | 用途                              |
|-----------------------|-----------------------------------|
| `data/npy_mtf/X_train_mtf.npy` | shape=(744065, 60, 24)   |
| `data/npy_mtf/X_val_mtf.npy`   | shape=(159442, 60, 24)   |
| `data/npy_mtf/X_test_mtf.npy`  | shape=(159444, 60, 24)   |
| `data/npy_mtf/scaler_mtf.pkl`  | StandardScaler（MTF版）  |

## モデル構成

- アーキテクチャ: 双方向 LSTM（hidden=128, n_layers=2）
- 入力: 24特徴量（M5 16 + M15/H1 8）
- 学習データ: UP/DOWN のみ（NEUTRAL 除外）
- 損失: weighted_ce
- 最適化基準: val_f1_updown

## 推論パイプライン（想定）

```
M5バー → lstm_focal（p_move >= t_move）→ MTF特徴量計算 → lstm_dir_v2（方向予測）→ 成行エントリー
```

## 次のステップ

1. `data/npy_mtf/` を Kaggle Dataset `pearless-usdjpy-m5-mtf` としてアップロード
   ```bash
   kaggle datasets create -p data/npy_mtf
   ```
2. `notebooks/train_lstm_dir_v2.ipynb` を Kaggle で実行（GPU P100）
3. 結果確認:
   - val_f1_updown が lstm_dir（~0.50）を有意に上回るか？
   - val_accuracy が 54% 以上になるか？
4. チェックポイントをダウンロードして `scripts/backtest_dir_v2.py` で test 期待値を確認
5. test 平均が +0.1 銭/トレード 以上 かつ 100 件以上であれば production 昇格を検討

## 成功基準

| 指標                | 目標値          | 前回（lstm_dir）|
|---------------------|-----------------|-----------------|
| val_f1_updown       | > 0.54          | ~0.50           |
| val_accuracy        | > 54%           | 50.3%           |
| test 平均(銭)       | > 0.1 銭        | 未計測          |
| test 取引件数       | > 100 件        | —               |

## 制約・注意事項

- production/ 配下は変更しない
- lstm_focal の重みは上書きしない
- mt4/PearlessBreakout.mq4 は変更しない
- 失敗した場合でも experiment_lstm_dir_v2.md を更新して記録を残す
