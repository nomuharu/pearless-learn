# Task 2-3: iTransformer 実装 + Kaggle ノートブック作成（AC-011/012）

## タスク概要

iTransformer（特徴量軸 Attention）を実装し、Kaggle commit mode 対応ノートブックを作成する。ADR-0001 に従い RevIN は適用しない。

## 対象ファイル

- `models/itransformer.py` (新規)
- `notebooks/train_itransformer.ipynb` (新規)

## 調査対象

- `/home/nomu/claude_code/pearless/models/base.py` (Task 2-1 の BaseModel)
- `/home/nomu/claude_code/pearless/models/training.py` (Task 2-2 の共通学習ループ)
- `/home/nomu/claude_code/pearless/docs/design/fx-prediction-design.md` (§ iTransformer 設計仕様)
- `/home/nomu/claude_code/pearless/docs/adr/ADR-0001-model-architecture-selection.md` (iTransformer 実装ガイドライン)
- `/home/nomu/claude_code/pearless/tests/integration/test_models.int.py` (AC-011/012 アサーション仕様)

## 実装手順

### Step 1: iTransformer 実装

**核心的設計**: 入力を `(B, 60, 16)` → `(B, 16, 60)` に転置し、特徴量軸（16次元）で Self-Attention を計算する。

```python
class iTransformer(BaseModel):
    """iTransformer — 特徴量軸 Attention の時系列分類モデル。

    設計上の特徴:
        - 入力 (B, T, C) を (B, C, T) に転置し、C 軸（特徴量）で Attention を計算する。
        - RevIN は適用しない（ADR-0001 Implementation Guidance 準拠）。
    """

    def __init__(
        self,
        seq_len: int = 60,
        n_features: int = 16,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
        n_classes: int = 3,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 転置: (B, T, C) → (B, C, T) = (B, 16, 60)
        x_transposed = x.permute(0, 2, 1)
        # 2. 特徴量軸で Transformer Encoder
        # 3. Global Average Pooling → Linear → softmax
        ...
```

### Step 2: RevIN 非存在の確認

`models/itransformer.py` には `RevIN` に関するコード・import を一切含めない。

`test_itransformer_has_no_revin_module` テストを Green にするため:
```python
# iTransformer の forward 後に以下が true であることを確認
assert not any('revin' in name for name, _ in model.named_modules())
```

### Step 3: notebooks/train_itransformer.ipynb 作成

- Task 2-2 の `train_patchtst.ipynb` と同等の構造
- `PatchTST` を `iTransformer` に差し替えるのみ

## 品質保証メカニズム

| メカニズム | 確認内容 | 適用箇所 |
|---|---|---|
| モデルパラメータ数チェック（10M 以下）| CPU 推論コスト制約 | test_models.int.py |

## 動作確認方法

```python
import torch
from models.itransformer import iTransformer

model = iTransformer()
model.eval()
x = torch.randn(4, 60, 16)
output = model.forward(x)

# AC-011: output shape
assert output.shape == (4, 3), f"Expected (4, 3), got {output.shape}"
assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-4)
assert output.min() >= 0.0
print("OK: shape OK, softmax OK")

# AC-012: RevIN 非存在
assert not any('revin' in name for name, _ in model.named_modules()), "RevIN が存在する（ADR-0001 違反）"
print("OK: RevIN なし（ADR-0001 準拠）")

# パラメータ数
total_params = sum(p.numel() for p in model.parameters())
assert total_params <= 10_000_000
print(f"OK: パラメータ数 {total_params:,}")
```

**成功基準**:
- `iTransformer.forward(x).shape == (4, 3)` かつ softmax 合計≈1.0 (AC-011)
- forward 内で `(B, 60, 16)` → `(B, 16, 60)` の転置操作が行われる (AC-012)
- RevIN モジュールが存在しない（ADR-0001 準拠）
- Kaggle ノートブックが commit mode で実行完了

**検証レベル**: L1（機能動作検証）

## 完了条件

- [x] Implementation: `iTransformer.forward(x).shape == (4, 3)` かつ転置操作が forward 内で行われる (AC-011/012)
- [x] Quality: パラメータ数 ≤ 10M、RevIN モジュールが存在しない（ADR-0001 準拠）
- [x] Integration: Kaggle ノートブックが commit mode で実行完了
