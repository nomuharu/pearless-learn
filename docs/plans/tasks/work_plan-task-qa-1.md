# Task QA-1: 全 Design Doc Acceptance Criteria 達成確認

## タスク概要

AC-001〜AC-026 の全項目を一括チェックリストで確認する。未達成 AC がある場合は対応タスクを追加する。

## 対象ファイル

（実装ファイルなし。チェックリスト確認タスク。）

## 調査対象

- `/home/nomu/claude_code/pearless/docs/design/fx-prediction-design.md` (§ Acceptance Criteria 全項目)
- `/home/nomu/claude_code/pearless/docs/plans/tasks/` (各フェーズ完了チェックリスト)

## 確認手順

### AC チェックリスト

| AC | 内容 | タスク | 確認方法 |
|---|---|---|---|
| AC-001 | 7ファイル生成 | Task 1-2 | `pytest test_pipeline.int.py::test_pipeline_generates_all_required_output_files` |
| AC-002 | X shape (N,60,16) | Task 1-2 | `pytest test_pipeline.int.py::test_pipeline_output_x_arrays_have_correct_shape` |
| AC-003 | 時系列分割順序 | Task 1-2 | `pytest test_pipeline.int.py::test_time_series_split_preserves_temporal_order_and_ratios` |
| AC-004 | Scaler fit=train のみ | Task 1-2 | `pytest test_pipeline.int.py::test_normalizer_fits_only_on_train_data_not_val_or_test` |
| AC-005 | 16特徴量 NaN ゼロ | Task 1-2 | `pytest test_pipeline.int.py::test_feature_engineering_produces_16_features_with_no_nan` |
| AC-007 | quantile 0.75 ラベル | Task 1-2 | `pytest test_pipeline.int.py::test_label_generation_uses_quantile_075_threshold_when_not_specified` |
| AC-008 | SHA-256 冪等性 | Task 1-2 | `pytest test_pipeline.int.py::test_pipeline_is_idempotent_sha256` |
| AC-009 | PatchTST shape (B,3) | Task 2-6 | `pytest test_models.int.py::test_patchtst_forward_output_shape_and_softmax` |
| AC-010 | RevIN 存在 | Task 2-6 | `pytest test_models.int.py::test_patchtst_forward_output_shape_and_softmax` |
| AC-011 | iTransformer shape (B,3) | Task 2-6 | `pytest test_models.int.py::test_itransformer_forward_output_shape_and_transpose` |
| AC-012 | 転置操作 | Task 2-6 | `pytest test_models.int.py::test_itransformer_forward_output_shape_and_transpose` |
| AC-013 | Kaggle PatchTST commit | Task 2-2 | ノートブック実行完了確認 |
| AC-014 | best_model.pt 生成 | Task 2-2 | ノートブック出力確認 |
| AC-015 | 評価 CSV 生成 | Task 3-2 | `ls logs/evaluation_results_*.csv` |
| AC-016 | --threshold 反映 | Task 3-1 | `evaluate.py --threshold` 動作確認 |
| AC-017 | 推論 50ms 未満 | Task 2-7 | `pytest test_inference_engine.int.py::test_inference_engine_completes_within_50ms_average_over_100_runs` |
| AC-018 | predict() 戻り値 | Task 2-7 | `pytest test_inference_engine.int.py::test_inference_engine_returns_signal_and_three_probabilities` |
| AC-019 | 100回推論エラーゼロ | Task 2-7 | `pytest test_inference_engine.int.py::test_stub_100_consecutive_predictions_produce_no_errors` |
| AC-020 | DataSource DI | Task 2-7 | `pytest test_inference_engine.int.py::test_inference_engine_accepts_any_data_source_interface_subclass` |
| AC-021 | CNN 列が CSV に存在 | Task 3-2 | evaluate.py 出力 CSV 確認 |
| AC-022 | 学習ログ CSV 出力 | Task 2-2 | `ls logs/training_log_*.csv` |
| AC-023 | ログ列名一致 | Task 2-2 | CSV ヘッダー確認 |
| AC-024 | uv sync 成功 | Task 0-1 | `uv sync` 実行 |
| AC-025 | uv export 成功 | Task 0-1 | `uv export --format requirements-txt` 実行 |
| AC-026 | upload_dataset.py | Task 1-3 | `--dry-run` 実行確認 |

### 全テスト一括実行

```bash
uv run pytest tests/integration/ -v
```

- [x] AC-001〜AC-026 全項目達成確認

## 動作確認方法

```bash
uv run pytest tests/ -v --tb=short
```

**成功基準**:
- 未達成 AC がゼロ
- 未達成 AC が見つかった場合は追加タスクを作成

**検証レベル**: L2（テスト動作検証）

## 完了条件

- [x] AC-001〜AC-026 全項目のチェックリスト確認完了
- [x] 未達成 AC がある場合は追加タスクを作成して対応済み
