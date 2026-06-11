# production/ — 採用済み運用構成（複数戦略対応）

このディレクトリは**採用が確定した運用構成**とその検証記録を置く場所。
実験・検証中のものは従来どおり `scripts/` / `notebooks/` / `docs/plans/` で行い、
採用が決まったらここに「昇格」させる（昇格・変更・引退は必ず commit で記録）。

複数の手法を同時に採用できる。戦略は `strategies/<strategy-id>/` 単位で独立しており、
それぞれが自分のモデル重み・期待値・フォワードテスト記録を持つ。

```
production/
├── README.md                  ← このファイル（運用ルール）
├── TEMPLATE_forward_test.md   ← フォワードテスト記録の共通テンプレート
└── strategies/
    └── <strategy-id>/         ← 戦略ごとに独立（kebab-case の ID）
        ├── strategy.md        ← 定義・状態・パラメータ・採用履歴
        ├── expectations.json  ← バックテスト期待値（サマリスクリプトが参照）
        ├── checkpoints/       ← この戦略の採用重み（git 追跡、{model}_{YYYYMMDD}.pt）
        └── forward_tests/     ← この戦略のデモ/リアルのテスト記録
            └── YYYY-MM_xxx/
                ├── session.md（TEMPLATE_forward_test.md をコピー）
                ├── trades.csv
                └── (MT4レポート等の原本)
```

## 戦略一覧

| ID | 状態 | 概要 | 採用日 |
|---|---|---|---|
| [oco-breakout-wf](strategies/oco-breakout-wf/strategy.md) | forward-testing | move検知 + OCO逆指値ブレイクアウト + walk-forward | 2026-06-12 |

状態の定義: `forward-testing`（デモ検証中）→ `live`（実弾運用）→ `retired`（引退。
ディレクトリは消さず strategy.md に引退理由を記録）

## 運用ルール

1. **新戦略の採用**: `strategies/<新ID>/` を作成し、strategy.md・expectations.json・
   重みを置き、この README の戦略一覧に行を追加して commit
2. **フォワードテスト**: TEMPLATE_forward_test.md をコピーしてセッション開始。
   終了時に trades.csv を作成し、
   `uv run python scripts/summarize_forward_test.py strategies/<id>/forward_tests/<dir>/trades.csv`
   でサマリを出して session.md に貼る（期待値は同じ戦略の expectations.json から自動参照）
3. **同一口座で複数EA**: MT4 の MagicNumber を戦略ごとに一意にする
   （strategy.md に記録。oco-breakout-wf = 20260611）
4. **判定**: フォワード実績と expectations.json の乖離が大きい場合は原因を
   session.md に記録し、必要なら状態を変更する

## 月次メンテナンス（walk-forward 系戦略）

- 直近1年分でモデルを fine-tune（notebooks/train_walkforward.ipynb の1チャンク分と同手順）
- 直近15,000本の予測分布から閾値を再選択
- 新しい重みを当該戦略の `checkpoints/` に日付付きで追加し、strategy.md を更新して commit
