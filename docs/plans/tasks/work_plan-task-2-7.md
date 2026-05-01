# Task 2-7: 推論エンジン Integration Test 実装・実行（test_inference_engine.int.py: AC-017〜020）

## タスク概要

`tests/integration/test_inference_engine.int.py` の4つのテスト関数本体（現在 `pass`）と未実装 fixture を実装し、全テストを Green にする。

## 対象ファイル

- `tests/integration/test_inference_engine.int.py` (テスト本体実装 + fixture 完成)

## 調査対象

- `/home/nomu/claude_code/pearless/tests/integration/test_inference_engine.int.py` (スケルトンのアサーション仕様を必ず熟読)
- `/home/nomu/claude_code/pearless/inference/engine.py` (Task 2-5 の InferenceEngine)
- `/home/nomu/claude_code/pearless/inference/pipe_stub.py` (Task 2-5 の NamedPipeStub)
- `/home/nomu/claude_code/pearless/models/patchtst.py` (Task 2-2 の PatchTST)
- `/home/nomu/claude_code/pearless/inference/interface.py` (DataSourceInterface)

## 実装手順

### Step 1: import 追加

```python
import pickle
import time
import torch
import torch.nn as nn
from inference.engine import InferenceEngine
from inference.pipe_stub import NamedPipeStub
from inference.interface import DataSourceInterface
from models.patchtst import PatchTST
from sklearn.preprocessing import StandardScaler
```

### Step 2: fixture 完成

**`saved_scaler_path` fixture** (現在 `pass` → 実装):
```python
@pytest.fixture
def saved_scaler_path(tmp_path):
    rng = np.random.default_rng(seed=42)
    X_train = rng.random((100, 60, 16)).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, 16))
    scaler_path = tmp_path / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    return scaler_path
```

**`stub_model` fixture** (現在 `pass` → 実装):
```python
@pytest.fixture
def stub_model():
    class StubModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            return torch.tensor([[0.7, 0.2, 0.1]]).repeat(batch_size, 1)
    return StubModel()
```

**`minimal_patchtst_model` fixture** (`pytest.skip` → 実際の PatchTST):
```python
@pytest.fixture
def minimal_patchtst_model():
    model = PatchTST(
        seq_len=60, n_features=16, d_model=64, n_heads=2, n_layers=1, n_classes=3
    )
    model.eval()
    return model
```

### Step 3: 各テスト関数の実装

**test_inference_engine_returns_signal_and_three_probabilities (AC-018)**:
```python
engine = InferenceEngine(
    model=stub_model, scaler_path=saved_scaler_path, data_source=mock_data_source
)
result = engine.predict()
assert result["signal"] in {"UP", "DOWN", "NEUTRAL"}
assert set(result["probabilities"].keys()) == {"UP", "DOWN", "NEUTRAL"}
assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4
assert isinstance(result["inference_ms"], float)
```

**test_inference_engine_completes_within_50ms_average_over_100_runs (AC-017)**:
```python
engine = InferenceEngine(
    model=minimal_patchtst_model, scaler_path=saved_scaler_path, data_source=mock_data_source
)
elapsed_times = []
for _ in range(100):
    start = time.perf_counter()
    engine.predict()
    elapsed_times.append((time.perf_counter() - start) * 1000.0)
avg_ms = sum(elapsed_times) / 100
# CI 環境では 100ms を下限とする（スケルトンのコメント参照）
assert avg_ms < 50.0
```

**test_stub_100_consecutive_predictions_produce_no_errors (AC-019)**:
```python
engine = InferenceEngine(
    model=minimal_patchtst_model,
    scaler_path=saved_scaler_path,
    data_source=NamedPipeStub(),
)
errors: list[Exception] = []
results: list[dict] = []
for _ in range(100):
    try:
        result = engine.predict()
        results.append(result)
    except Exception as e:
        errors.append(e)
assert len(errors) == 0, f"エラー件数: {len(errors)}, 最初のエラー: {errors[0] if errors else None}"
assert len(results) == 100
```

**test_inference_engine_accepts_any_data_source_interface_subclass (AC-020)**:
```python
class CustomDataSource(DataSourceInterface):
    def fetch_latest_ohlcv(self, n_bars: int = 60) -> pd.DataFrame:
        rng = np.random.default_rng(seed=0)
        close = 150.0 + rng.normal(0, 0.5, n_bars).cumsum()
        return pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n_bars, freq="5min"),
            "open": close, "high": close + 0.3, "low": close - 0.3,
            "close": close, "volume": np.ones(n_bars) * 500,
        })

engine_a = InferenceEngine(stub_model, saved_scaler_path, NamedPipeStub())
engine_b = InferenceEngine(stub_model, saved_scaler_path, CustomDataSource())
result_a = engine_a.predict()
result_b = engine_b.predict()
assert result_a["signal"] in {"UP", "DOWN", "NEUTRAL"}
assert result_b["signal"] in {"UP", "DOWN", "NEUTRAL"}
```

## 品質保証メカニズム

| メカニズム | 確認内容 | 適用箇所 |
|---|---|---|
| スタブ連続 100 回推論エラーゼロ | 推論エンジン安定性 (AC-019) | `test_stub_100_consecutive_predictions_produce_no_errors` |

## 動作確認方法

```bash
uv run pytest tests/integration/test_inference_engine.int.py -v

# 期待出力:
# test_inference_engine_returns_signal_and_three_probabilities PASSED
# test_inference_engine_completes_within_50ms_average_over_100_runs PASSED
# test_stub_100_consecutive_predictions_produce_no_errors PASSED
# test_inference_engine_accepts_any_data_source_interface_subclass PASSED
# 4 passed
```

**成功基準**:
- `pytest tests/integration/test_inference_engine.int.py` が全 pass（4/4）
- AC-017〜AC-020 達成確認

**検証レベル**: L2（テスト動作検証）

## 完了条件

- [ ] Implementation: 4つのテスト関数と3つの fixture が全て実装済み
- [ ] Quality: `pytest tests/integration/test_inference_engine.int.py` が全 pass（4/4）
- [ ] Integration: AC-017〜AC-020 が一括確認済み
