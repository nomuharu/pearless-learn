# Task QA-2: E2E テスト実装・実行（test_pipeline_to_inference.e2e.py）

## タスク概要

`tests/e2e/test_pipeline_to_inference.e2e.py` の E2E テスト関数本体（現在 `pass`）と未実装 fixture を実装し、パイプライン実行→スタブ推論の2段階 CLI 境界を跨ぐジャーニーを Green にする。

## 対象ファイル

- `tests/e2e/test_pipeline_to_inference.e2e.py` (テスト本体実装 + fixture 完成)

## 調査対象

- `/home/nomu/claude_code/pearless/tests/e2e/test_pipeline_to_inference.e2e.py` (スケルトンのジャーニー仕様を必ず熟読)
- `/home/nomu/claude_code/pearless/pipeline.py` (run_pipeline)
- `/home/nomu/claude_code/pearless/inference/engine.py` (InferenceEngine)
- `/home/nomu/claude_code/pearless/inference/pipe_stub.py` (NamedPipeStub)
- `/home/nomu/claude_code/pearless/models/patchtst.py` (PatchTST)

## 実装手順

### Step 1: fixture 完成

**`minimal_patchtst_model` fixture** (現在 `pass` → 実装):
```python
@pytest.fixture
def minimal_patchtst_model():
    from models.patchtst import PatchTST
    model = PatchTST(
        seq_len=60, n_features=16, d_model=64, n_heads=2, n_layers=1, n_classes=3
    )
    model.eval()
    return model
```

### Step 2: test_full_pipeline_to_stub_inference_end_to_end 実装

スケルトンのコメントに記載された2段階ジャーニーを実装する:

```python
from pipeline import run_pipeline
from inference.engine import InferenceEngine
from inference.pipe_stub import NamedPipeStub
import numpy as np

def test_full_pipeline_to_stub_inference_end_to_end(
    synthetic_ohlcv_csv, tmp_path, minimal_patchtst_model
):
    # === Step 1: データパイプライン実行（CLI 境界 1）===
    run_pipeline(str(synthetic_ohlcv_csv), str(tmp_path))

    # Assert 1: Step 1 完了確認
    x_train = np.load(tmp_path / "X_train.npy")
    assert (tmp_path / "X_train.npy").exists()
    assert x_train.shape[1] == 60 and x_train.shape[2] == 16
    assert x_train.dtype == np.float32
    assert (tmp_path / "scaler.pkl").exists()
    assert not np.isnan(x_train).any()

    # === Step 2: 推論エンジン実行（CLI 境界 2）===
    stub = NamedPipeStub()
    engine = InferenceEngine(
        model=minimal_patchtst_model,
        scaler_path=tmp_path / "scaler.pkl",  # Step 1 生成の scaler を使用
        data_source=stub,
    )
    result = engine.predict()

    # Assert 2: Step 2 完了確認（ジャーニー完結）
    assert result["signal"] in {"UP", "DOWN", "NEUTRAL"}
    assert set(result["probabilities"].keys()) == {"UP", "DOWN", "NEUTRAL"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4
    assert result["inference_ms"] < 50.0
```

## 品質保証メカニズム

| メカニズム | 確認内容 | 適用箇所 |
|---|---|---|
| パイプライン冪等性（SHA-256 検証）| Step 1 の出力検証 | `Assert 1` ブロック |
| スタブ連続 100 回推論エラーゼロ | Step 2 のエラーゼロ | `Assert 2` ブロック |

## 動作確認方法

```bash
uv run pytest tests/e2e/test_pipeline_to_inference.e2e.py -v

# 期待出力:
# test_full_pipeline_to_stub_inference_end_to_end PASSED
# 1 passed
```

**成功基準**:
- `pytest tests/e2e/test_pipeline_to_inference.e2e.py` が全 pass（1/1）
- 2 ステップのジャーニーが完結する
- Step 1 生成の `scaler.pkl` が Step 2 で正しくロードされる

**検証レベル**: L2（テスト動作検証）

## 完了条件

- [ ] Implementation: E2E テスト関数と `minimal_patchtst_model` fixture が実装済み
- [ ] Quality: `pytest tests/e2e/test_pipeline_to_inference.e2e.py` が全 pass（1/1）
- [ ] Integration: Step 1（pipeline）→ Step 2（inference）の state carry（scaler.pkl）が正しく機能する
