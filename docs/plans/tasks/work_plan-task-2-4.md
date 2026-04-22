# Task 2-4: CNN ベースライン実装 + Kaggle ノートブック作成（AC-021）

## タスク概要

5分足対応 CNN ベースラインモデルを実装し、Kaggle commit mode 対応ノートブックを作成する。PatchTST / iTransformer との比較評価（evaluate.py）で CNN 列が存在することを保証する（AC-021）。

## 対象ファイル

- `models/cnn.py` (新規)
- `notebooks/train_cnn.ipynb` (新規)

## 調査対象

- `/home/nomu/claude_code/pearless/models/base.py` (Task 2-1 の BaseModel)
- `/home/nomu/claude_code/pearless/models/training.py` (Task 2-2 の共通学習ループ)
- `/home/nomu/claude_code/pearless/docs/design/fx-prediction-design.md` (§ CNN ベースライン設計仕様)

## 実装手順

### Step 1: CNN ベースライン実装

```python
class CNNModel(BaseModel):
    """5分足 OHLCV 時系列分類の CNN ベースライン。

    アーキテクチャ:
        - Conv1D × 複数層（チャネル数は設計書に従う）
        - BatchNorm + ReLU + MaxPool
        - Flatten → Linear → softmax

    目的: PatchTST / iTransformer との比較基準。
    """

    def __init__(
        self,
        seq_len: int = 60,
        n_features: int = 16,
        n_classes: int = 3,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) = (B, 60, 16)
        # Conv1D は (B, C, T) を期待するため permute
        x_t = x.permute(0, 2, 1)  # (B, 16, 60)
        ...
        return torch.softmax(logits, dim=-1)
```

- shape 契約: input `(B, 60, 16)` → output `(B, 3)`
- `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

### Step 2: notebooks/train_cnn.ipynb 作成

- Task 2-2 の `train_patchtst.ipynb` と同等の構造
- `PatchTST` を `CNNModel` に差し替え

## 品質保証メカニズム

| メカニズム | 確認内容 | 適用箇所 |
|---|---|---|
| モデルパラメータ数チェック（10M 以下）| CPU 推論コスト制約 | test_models.int.py (AC-017 前提) |

## 動作確認方法

```python
import torch
from models.cnn import CNNModel

model = CNNModel()
model.eval()
x = torch.randn(4, 60, 16)
output = model.forward(x)

assert output.shape == (4, 3), f"Expected (4, 3), got {output.shape}"
assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-4)
print(f"OK: shape={output.shape}, softmax OK")

total_params = sum(p.numel() for p in model.parameters())
print(f"パラメータ数: {total_params:,}")
```

**成功基準**:
- `CNNModel.forward(x).shape == (4, 3)` かつ softmax 合計≈1.0
- Kaggle ノートブックが commit mode で実行完了

**検証レベル**: L1（機能動作検証）

## 完了条件

- [x] Implementation: `CNNModel(batch=4, seq=60, feat=16).forward(x).shape == (4, 3)` (AC-021)
- [x] Quality: 学習ループで CSV ログが出力される（training.py 流用）
- [x] Integration: Kaggle ノートブックが commit mode で実行完了
