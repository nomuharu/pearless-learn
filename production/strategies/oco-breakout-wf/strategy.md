# oco-breakout-wf

最終更新: 2026-06-12
状態: forward-testing
MT4 MagicNumber: 20260611

## 構成サマリ

USDJPY M5 の OCO ブレイクアウト戦略 + walk-forward 再学習。

| 項目 | 値 | 根拠 |
|---|---|---|
| moveモデル | lstm_focal（focal loss, val_f1_updown 基準） | 6モデル比較で AUC 0.685 トップ |
| 運用方式 | walk-forward（月次 fine-tune + 閾値再選択） | 直近12ヶ月 −81,300→+48,400円/Lot に反転 |
| シグナル | p_move = p_up + p_down >= t（tは直近分布の99.726%分位） | シグナル頻度をバックテストと整合 |
| エントリー | 確定値 P0 ± **δ=2.5銭** の OCO 逆指値（片方約定で他方取消） | val グリッド選択 |
| 決済 | **次の5分足の終値**（TP/SL/トレーリングは使わない） | Phase B で全ストップ系を棄却 |
| スプレッドガード | 現在スプレッド > **0.5銭** なら発注見送り | 指標時の拡大スプレッド対策 |
| 実行 | mt4/PearlessBreakout.mq4 + scripts/mt4_signal_writer.py | レイテンシ実測 約16ms |

## バックテスト期待値（フォワードテストの比較基準）

- test 期間 2024-03〜2026-04（M1パス再構成、spread 0.2銭 + slip 0.3銭）
- walk-forward: **818トレード、平均 +1.25銭/トレード**、直近12ヶ月 +48,400円/Lot
- 参考（固定モデル運用時）: 481トレード、平均 +2.80銭、直近12ヶ月は赤字
- 既知のリスク: レジーム依存（2026-04 は −11万円/Lot）。1ヶ月単位の赤字は想定内

## 採用チェックポイント

| ファイル | 内容 | 学習データ |
|---|---|---|
| checkpoints/lstm_focal_20260611.pt | move検知（ウォームスタート元） | train期間 2012-01〜2022-01（定常化16特徴量） |

※ 実運用の重みは月次 fine-tune で更新する（README の月次メンテナンス手順）。

## 採用履歴

| 日付 | 変更 | 根拠 |
|---|---|---|
| 2026-06-12 | 初版採用: lstm_focal + OCO(δ2.5銭) + walk-forward | docs/plans/work_plan_oco_strategy_improvement.md（Phase B/A 棄却、E 採用） |
