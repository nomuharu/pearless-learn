# Task 0-3: Integration テストスケルトン Red 実装（fixture 実装）

## タスク概要

既存テストスケルトンの fixture 関数（`pass` のまま）を実装し、`pytest --collect-only` が通る Red 状態を確立する。テスト本体は `pass` のままにし、fixture のみ実装する。

## 対象ファイル

- `tests/integration/test_pipeline.int.py` (fixture 実装のみ)
- `tests/integration/test_inference_engine.int.py` (fixture 実装のみ)
- `tests/__init__.py` (確認・必要に応じて修正)

## 調査対象

- `/home/nomu/claude_code/pearless/tests/integration/test_pipeline.int.py` (スケルトンの fixture コメント確認)
- `/home/nomu/claude_code/pearless/tests/integration/test_inference_engine.int.py` (スケルトンの fixture コメント確認)
- `/home/nomu/claude_code/pearless/docs/plans/work_plan.md` (§ Task 0-3、§ テストスケルトンファイル一覧)

## 実装手順

### Step 1: test_pipeline.int.py の fixture 確認

スケルトンを確認すると `synthetic_ohlcv_df` と `synthetic_ohlcv_csv` は既に実装済み。
現時点で変更不要。

### Step 2: test_inference_engine.int.py の fixture 実装

以下の fixture が `pass` のままになっているため実装する:

**`saved_scaler_path` fixture**:
```python
@pytest.fixture
def saved_scaler_path(tmp_path):
    """StandardScaler を (100, 60, 16) の合成データで fit し tmp_path/scaler.pkl として保存してパスを返す。"""
    import pickle
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed=42)
    X_train = rng.random((100, 60, 16)).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, 16))
    scaler_path = tmp_path / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    return scaler_path
```

**`stub_model` fixture**:
```python
@pytest.fixture
def stub_model():
    """BaseModel のスタブ。forward(x) は常に (batch, 3) の softmax 済み Tensor を返す。"""
    import torch
    import torch.nn as nn

    class StubModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            probs = torch.tensor([[0.7, 0.2, 0.1]]).repeat(batch_size, 1)
            return probs

    return StubModel()
```

**`minimal_patchtst_model` fixture**:
```python
@pytest.fixture
def minimal_patchtst_model():
    """PatchTST の最小構成（実 PyTorch モデル）。CPU デバイスで推論。"""
    # Phase 2 Task 2-2 で PatchTST が実装されるまで pass のままにする
    # pytest.skip を使って Red 状態を保つ
    pytest.skip("PatchTST 未実装（Task 2-2 で実装予定）")
```

### Step 3: pytest 収集確認

```bash
uv run pytest tests/ --collect-only
```

- エラーなく収集されることを確認
- テスト関数は `pass` なので `PASSED` になる（Red 状態は `stub_model`/`minimal_patchtst_model` が `pass` の段階で確立）

注: スケルトンの `test_models.int.py` は torch をインポートするが、Phase 2 実装前でも `--collect-only` が通る必要がある。import エラーが出る場合は `pytest.importorskip("torch")` を追加する。

## 品質保証メカニズム

このタスクではテスト収集が通ることのみを確認する。品質チェックはQA-3 で実施。

## 動作確認方法

```bash
# テスト収集確認（エラーなし）
uv run pytest tests/ --collect-only

# テスト実行（全 pass または skip が許容される Red 状態）
uv run pytest tests/integration/test_pipeline.int.py -v
uv run pytest tests/integration/test_inference_engine.int.py -v
```

**成功基準**:
- `pytest tests/ --collect-only` でテスト関数が収集される（エラーなし）
- import エラーが一切ない
- テスト本体は `pass` のまま（実装は後続タスクで行う）

**検証レベル**: L2（テスト動作検証）

## 完了条件

- [x] Implementation: `saved_scaler_path`、`stub_model`、`minimal_patchtst_model` fixture が実装済み
- [x] Quality: `pytest tests/ --collect-only` でテスト関数が収集されエラーがない
- [x] Integration: テスト実行時に import エラーが発生しない（Red 状態が維持されている）
