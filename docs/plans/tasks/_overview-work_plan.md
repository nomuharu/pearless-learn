# 全体設計ドキュメント: Pearless ML システム実装

生成日時: 2026-04-21
対象プランドキュメント: work_plan.md

---

## プロジェクト概要

### 目的・ゴール

USDJPY 5分足 OHLCV データから次の5分足の方向（UP/DOWN/NEUTRAL）を予測するMLシステムを新規構築する。
Kaggle GPU 無料枠を活用した費用ゼロの学習パイプラインと、Named Pipe スタブによるMT4将来連携インターフェースを確立する。

### 背景・コンテキスト

- 1分足 CNN の精度限界（Accuracy 64〜78%）解消が動機
- 3クラス分類（UP=0/DOWN=1/NEUTRAL=2）に変更
- PatchTST / iTransformer / CNN 3モデルを比較評価して最良を選定
- ローカル WSL2 CPU + Kaggle T4 GPU の2環境で動作

---

## タスク分割設計

### 分割方針

Phase 0〜1: Horizontal Slice（基盤優先）
- 環境・インターフェース定義 → データパイプライン → テスト の順で基盤を固める

Phase 2: Vertical Slice（モデル別）
- BaseModel → PatchTST / iTransformer / CNN → 推論エンジン → テスト の順
- 各モデルは独立してコミット可能

Phase 3〜Final: 評価・品質保証
- 全モデル評価 → E2E テスト → 静的解析

### タスク間関係マップ

```
Task 0-1: pyproject.toml / uv 環境構築
  ↓
Task 0-2: DataSourceInterface 定義 (inference/interface.py)
  ↓
Task 0-3: テストスケルトン fixture 実装 (Red 状態)
  ↓
Task 1-1: pipeline.py 実装（feature_engineering / create_label / split / normalize）
  ↓
Task 1-2: パイプライン Integration Test 実装・実行 (AC-001〜AC-007)
  ↓ (並行)
Task 1-3: upload_dataset.py 実装 (AC-026)

Task 2-1: models/base.py (BaseModel 抽象クラス)
  ↓ (並行)
Task 2-2: PatchTST + training.py + Kaggle ノートブック (AC-009/010/022/023)
Task 2-3: iTransformer + Kaggle ノートブック (AC-011/012)
Task 2-4: CNN + Kaggle ノートブック (AC-021)
  ↓ (全モデル実装完了後)
Task 2-5: 推論エンジン + Named Pipe スタブ (AC-017〜020)
Task 2-6: モデル Integration Test (test_models.int.py: AC-009〜012)
Task 2-7: 推論エンジン Integration Test (test_inference_engine.int.py: AC-017〜020)
  ↓
Task 3-1: evaluate.py 実装 (AC-015/016/021)
  ↓
Task 3-2: 全モデル比較評価実行

Task QA-1: 全 AC 達成確認
Task QA-2: E2E テスト実装・実行
Task QA-3: 静的解析・lint・型チェック・カバレッジ
Task QA-4: ドキュメント最終確認
```

### インターフェース変更影響分析

| 既存インターフェース | 新インターフェース | 変換要否 | 対応タスク |
|---|---|---|---|
| なし（新規）| DataSourceInterface.fetch_latest_ohlcv(n_bars: int) | — | Task 0-2 |
| なし（新規）| run_pipeline(csv_path, output_dir) | — | Task 1-1 |
| なし（新規）| BaseModel.forward(x: Tensor) | — | Task 2-1 |
| なし（新規）| InferenceEngine.predict() | — | Task 2-5 |

### 共通処理ポイント

- `models/training.py`: 3モデル共通学習ループ（Task 2-2 で先行実装し 2-3/2-4 が流用）
- `DEVICE = torch.device(...)` パターン: 全モデルファイルで統一
- `class_weights` CrossEntropyLoss: 全モデルで共通（クラス不均衡対策）
- テスト fixture `synthetic_ohlcv_df`, `synthetic_ohlcv_csv`: test_pipeline と e2e で同一パターン

---

## 実装上の注意事項

### 全フェーズ共通で守る原則

1. wandb 使用禁止（コードに一切含めない）
2. any 型使用禁止（Python 型ヒントで具体型を使用）
3. 変数再代入を避け immutable に組む
4. pandas-ta のみ使用（TA-Lib 禁止）
5. クラス番号固定: UP=0, DOWN=1, NEUTRAL=2
6. パラメータ数 ≤ 10M（CPU 推論 50ms 制約）
7. Kaggle API トークンは環境変数から読み込む（ハードコード禁止）

### リスクと対策

- **リスク**: NEUTRAL クラス過剰予測
  - 対策: CrossEntropyLoss に `weight=class_weights` を適用
- **リスク**: NaN 処理ミスによるデータリーク
  - 対策: 各ステップ後に shape・NaN 数・クラス分布のアサーション
- **リスク**: scaler 不整合
  - 対策: `scaler.pkl` を `data/` に保存し推論エンジンが必ずロードする構造を強制

### 影響スコープ管理

- 変更許容スコープ: プロジェクト全体（新規）
- 保護領域: `tests/` 内のスケルトンコメント・fixture アノテーション（変更禁止）
