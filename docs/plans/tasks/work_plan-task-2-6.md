# Task 2-6: モデル Integration Test 実装・実行（test_models.int.py: AC-009〜012）

## タスク概要

`tests/integration/test_models.int.py` の5つのテスト関数本体（現在 `pass`）を実装し、全テストを Green にする。PatchTST / iTransformer の shape・RevIN・転置・パラメータ数・BaseModel インターフェースを検証する。

## 対象ファイル

- `tests/integration/test_models.int.py` (テスト本体実装)

## 調査対象

- `/home/nomu/claude_code/pearless/tests/integration/test_models.int.py` (スケルトンのアサーション仕様を必ず熟読)
- `/home/nomu/claude_code/pearless/models/patchtst.py` (Task 2-2 の実装)
- `/home/nomu/claude_code/pearless/models/itransformer.py` (Task 2-3 の実装)
- `/home/nomu/claude_code/pearless/models/base.py` (Task 2-1 の BaseModel)

## 実装手順

### Step 1: import 追加

```python
import torch
from models.patchtst import PatchTST
from models.itransformer import iTransformer
from models.base import BaseModel
```

### Step 2: 各テスト関数の実装

**test_patchtst_forward_output_shape_and_softmax (AC-009/010)**:
```python
model = PatchTST(seq_len=60, n_features=16, patch_len=6, stride=6,
                 d_model=128, n_heads=8, n_layers=3, dim_ff=256, dropout=0.0, n_classes=3)
model.eval()
x = torch.randn(4, 60, 16)
output = model.forward(x)
assert output.shape == (4, 3)
assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-4)
assert output.min() >= 0.0
assert hasattr(model, 'revin')  # AC-010
```

**test_patchtst_parameter_count_under_10_million**:
```python
model = PatchTST()
total_params = sum(p.numel() for p in model.parameters())
assert total_params <= 10_000_000
```

**test_itransformer_forward_output_shape_and_transpose (AC-011/012)**:
```python
model = iTransformer(seq_len=60, n_features=16, d_model=128, n_heads=8,
                     n_layers=3, dim_ff=256, dropout=0.0, n_classes=3)
model.eval()
x = torch.randn(4, 60, 16)
output = model.forward(x)
assert output.shape == (4, 3)
assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-4)
assert output.min() >= 0.0
```

**test_itransformer_has_no_revin_module**:
```python
model = iTransformer()
assert not any('revin' in name for name, _ in model.named_modules())
```

**test_all_models_implement_base_model_interface**:
- parametrize fixture `"patchtst_model"` / `"itransformer_model"` の request.getfixturevalue を使用
- fixture を `conftest.py` に追加するか、テストファイル内に定義

### Step 3: fixture の追加

`test_all_models_implement_base_model_interface` が `parametrize` を使用しているため、対応 fixture を追加:

```python
@pytest.fixture
def patchtst_model():
    return PatchTST()

@pytest.fixture
def itransformer_model():
    return iTransformer()
```

## 品質保証メカニズム

| メカニズム | 確認内容 | 適用箇所 |
|---|---|---|
| モデルパラメータ数チェック（10M 以下）| CPU 推論コスト制約 | `test_patchtst_parameter_count_under_10_million` |

## 動作確認方法

```bash
uv run pytest tests/integration/test_models.int.py -v

# 期待出力:
# test_patchtst_forward_output_shape_and_softmax PASSED
# test_patchtst_parameter_count_under_10_million PASSED
# test_itransformer_forward_output_shape_and_transpose PASSED
# test_itransformer_has_no_revin_module PASSED
# test_all_models_implement_base_model_interface[patchtst_model] PASSED
# test_all_models_implement_base_model_interface[itransformer_model] PASSED
# 6 passed
```

**成功基準**:
- `pytest tests/integration/test_models.int.py` が全 pass（5〜6テスト）
- AC-009〜AC-012 達成確認

**検証レベル**: L2（テスト動作検証）

## 完了条件

- [ ] Implementation: 5つのテスト関数が全て実装済み（`pass` を除去）
- [ ] Quality: `pytest tests/integration/test_models.int.py` が全 pass（テスト解決: 5/5 以上）
- [ ] Integration: AC-009〜AC-012 が一括確認済み
