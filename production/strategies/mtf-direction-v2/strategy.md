# mtf-direction-v2

最終更新: 2026-06-16
状態: forward-testing
MT4 MagicNumber: （未割り当て。MT4 組み込み時に決定する）

## 構成サマリ

USDJPY M5 の2段構成スキャルピング戦略。
Stage 1 で lstm_focal がボラティリティ高バーを絞り込み、
Stage 2 で lstm_dir_v2（MTF 24特徴量）が方向を予測して成行エントリー。

| 項目 | 値 | 根拠 |
|---|---|---|
| Stage 1 モデル | lstm_focal（oco-breakout-wf 共用） | p_move = p_up + p_down でボラ選別 |
| Stage 2 モデル | lstm_dir_v2（M5 16列 + M15/H1 8列 = 24特徴量） | val_f1_updown ~0.727 |
| 閾値 | t_move=0.88, t_dir=0.60 | val グリッドサーチで選択 |
| エントリー | シグナル足終値で成行（p_up >= 0.60 → BUY、p_up <= 0.40 → SELL） | — |
| 決済 | 1足後（5分後）の終値で成行クローズ | backtest_dir_v2.py の設定と同じ |
| TP/SL | なし（時間エグジットのみ） | — |
| 保有時間 | 5分（1足固定） | — |

## バックテスト期待値

- test 期間 2024-03〜2026-04（spread 0.2銭込み）
- **6,280トレード、平均 +2.902銭/トレード、勝率 65.8%**
- 頻度: 約 8.7件/日

## 採用チェックポイント

| ファイル | 内容 | 学習データ |
|---|---|---|
| checkpoints/lstm_dir_v2_20260616.pt | 方向予測 Stage 2（MTF 24特徴量, val_f1_updown ~0.727） | train期間 2012-01〜2022-01（70/15/15分割） |

Stage 1 の lstm_focal は `oco-breakout-wf/checkpoints/lstm_focal_20260611.pt` を共用。

## 採用履歴

| 日付 | 変更 | 根拠 |
|---|---|---|
| 2026-06-16 | 初版採用: lstm_focal + lstm_dir_v2(MTF) + 成行5分エグジット | docs/plans/experiment_lstm_dir_v2.md。val_f1 ~0.727、test 平均 +2.902銭、6280件 |
