# Task 1-2: パイプライン Integration Test 実装・実行（AC-001〜AC-007）

## タスク概要

`tests/integration/test_pipeline.int.py` の6つのテスト関数本体（現在 `pass`）を実装し、全テストを Green にする。SHA-256 冪等性テストも追加する。

## 対象ファイル

- `tests/integration/test_pipeline.int.py` (テスト本体実装)

## 調査対象

- `/home/nomu/claude_code/pearless/tests/integration/test_pipeline.int.py` (スケルトンのアサーション仕様を必ず熟読)
- `/home/nomu/claude_code/pearless/pipeline.py` (Task 1-1 の実装、関数シグネチャ確認)
- `/home/nomu/claude_code/pearless/docs/plans/work_plan.md` (§ Task 1-2、§ Phase 1 完了基準)

## 実装手順

### Step 1: 各テスト関数の実装

スケルトンのコメントに記載されているアサーション仕様に従って実装する。変更してはいけないのはコメント・fixture・関数名。

**test_pipeline_generates_all_required_output_files (AC-001)**:
```python
from pipeline import run_pipeline

run_pipeline(str(synthetic_ohlcv_csv), str(tmp_path))
expected_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy",
                  "X_test.npy", "y_test.npy", "scaler.pkl"]
for fname in expected_files:
    assert (tmp_path / fname).exists(), f"{fname} が生成されていない"
```

**test_pipeline_output_x_arrays_have_correct_shape (AC-002)**:
- X_train.npy を np.load でロードし shape[1]==60, shape[2]==16, dtype==float32 を確認

**test_time_series_split_preserves_temporal_order_and_ratios (AC-003)**:
- N=1000 の合成 (X, y)、インデックスを埋め込んで順序確認
- `max(X_train[-1]) < min(X_val[0])` の順序確認

**test_normalizer_fits_only_on_train_data_not_val_or_test (AC-004)**:
- X_val の値域を X_train から意図的にシフト
- 正規化後の X_train mean ≈ 0 (±0.05), std ≈ 1 (±0.05)
- 正規化後の X_val mean が 0 から外れていることを確認

**test_feature_engineering_produces_16_features_with_no_nan (AC-005)**:
- `feature_engineering()` に合成 DataFrame を渡し、shape[1]==16, isna().sum().sum()==0 を確認

**test_label_generation_uses_quantile_075_threshold_when_not_specified (AC-007)**:
- `create_label(df, horizon=1, threshold=None)`
- NEUTRAL(2) 比率 ≈ 0.75 (±0.05)

### Step 2: SHA-256 冪等性テスト追加

```python
def test_pipeline_is_idempotent_sha256(synthetic_ohlcv_csv, tmp_path):
    """同一 CSV 入力から同一 numpy 配列が生成される（SHA-256 一致）。"""
    import hashlib
    output_dir_1 = tmp_path / "run1"
    output_dir_2 = tmp_path / "run2"
    output_dir_1.mkdir()
    output_dir_2.mkdir()
    run_pipeline(str(synthetic_ohlcv_csv), str(output_dir_1))
    run_pipeline(str(synthetic_ohlcv_csv), str(output_dir_2))
    for fname in ["X_train.npy", "y_train.npy"]:
        h1 = hashlib.sha256((output_dir_1 / fname).read_bytes()).hexdigest()
        h2 = hashlib.sha256((output_dir_2 / fname).read_bytes()).hexdigest()
        assert h1 == h2, f"{fname} のハッシュが一致しない"
```

## 品質保証メカニズム

| メカニズム | 確認内容 | 適用箇所 |
|---|---|---|
| パイプライン冪等性（SHA-256 検証）| 同一 CSV 入力 → 同一 numpy 出力 | `test_pipeline_is_idempotent_sha256` |
| パイプライン各ステップ後アサーション | shape・NaN 数・クラス分布 | AC-002、AC-005 テスト |

## 動作確認方法

```bash
# 全テスト実行
uv run pytest tests/integration/test_pipeline.int.py -v

# 期待出力:
# test_pipeline_generates_all_required_output_files PASSED
# test_pipeline_output_x_arrays_have_correct_shape PASSED
# test_time_series_split_preserves_temporal_order_and_ratios PASSED
# test_normalizer_fits_only_on_train_data_not_val_or_test PASSED
# test_feature_engineering_produces_16_features_with_no_nan PASSED
# test_label_generation_uses_quantile_075_threshold_when_not_specified PASSED
# test_pipeline_is_idempotent_sha256 PASSED
# 7 passed
```

**成功基準**:
- `pytest tests/integration/test_pipeline.int.py` が全 pass（7/7）
- AC-001〜AC-007 が一括確認済み

**検証レベル**: L2（テスト動作検証）

## 完了条件

- [x] Implementation: 6テスト関数 + SHA-256 冪等性テスト、合計7テスト関数が実装済み
- [x] Quality: `pytest tests/integration/test_pipeline.int.py` が全 pass（8/8）
- [x] Integration: AC-001〜AC-007 および SHA-256 冪等性が確認済み
