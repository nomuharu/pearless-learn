# フォワードテスト記録: {YYYY-MM}_{識別子}

戦略: {strategy-id}（このファイルは production/strategies/{strategy-id}/forward_tests/ 配下に置く）

## 設定

| 項目 | 値 |
|---|---|
| 期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| 口座 | （ブローカー名・デモ/リアル） |
| ロット | |
| 使用チェックポイント | ../checkpoints/{model}_{YYYYMMDD}.pt |
| 運用閾値 | （再選択した値があれば） |
| EA設定の変更点 | （デフォルトとの差分のみ。MagicNumberは strategy.md と一致させる） |

## 結果サマリ

`uv run python scripts/summarize_forward_test.py <このdir>/trades.csv` の出力を貼る。
期待値は同じ戦略の expectations.json から自動で比較される。

```
（ここに貼り付け）
```

## 期待値との乖離の評価

- 平均損益の乖離が −1銭 を超える場合は原因を特定する（候補: 逆指値の滑り、
  実スプレッド、シグナル遅延、レジーム）
- 判定（継続 / 設定変更 / 状態変更）:

## 所見

- 約定品質（逆指値の滑り）:
- スプレッドガードの発動回数・タイミング:
- シグナル遅延（足確定→発注の実測）:
- その他気づき:

## trades.csv フォーマット

```csv
time,side,entry_price,exit_price,lots,pnl_yen
2026-06-12 14:35,buy,154.025,154.061,0.10,360
```

- time: エントリーした5分足の開始時刻
- side: buy / sell
- pnl_yen: スプレッド・手数料込みの実現損益（円）
- 約定しなかったシグナル（OCO不成立）は行を作らない
