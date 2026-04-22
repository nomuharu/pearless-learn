# Task 2-2: PatchTST 実装 + training.py + Kaggle ノートブック作成（AC-009/010/013/014/022/023）

## タスク概要

PatchTST モデル（RevIN + パッチ化 Transformer）を実装し、共通学習ループ `training.py` を整備し、Kaggle commit mode 対応ノートブックを作成する。

## 対象ファイル

- `models/patchtst.py` (新規)
- `models/training.py` (新規)
- `notebooks/train_patchtst.ipynb` (新規)

## 調査対象

- `/home/nomu/claude_code/pearless/models/base.py` (Task 2-1 の BaseModel、クラス定数確認)
- `/home/nomu/claude_code/pearless/docs/design/fx-prediction-design.md` (§ PatchTST 設計仕様、RevIN、ハイパーパラメータ)
- `/home/nomu/claude_code/pearless/docs/adr/ADR-0001-model-architecture-selection.md` (PatchTST 実装ガイドライン)
- `/home/nomu/claude_code/pearless/tests/integration/test_models.int.py` (AC-009/010 アサーション仕様)

## 実装手順

### Step 1: RevIN モジュール実装

RevIN（Reversible Instance Normalization）を `models/patchtst.py` 内に実装:

```python
class RevIN(nn.Module):
    """Reversible Instance Normalization（Kim et al., 2022）。"""
    def __init__(self, n_features: int, eps: float = 1e-5) -> None: ...
    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        # mode='norm': 正規化（forward pass）
        # mode='denorm': 逆正規化（output 後）
        ...
```

### Step 2: PatchTST 実装

ハイパーパラメータ（デフォルト値）:
- `seq_len=60`, `n_features=16`, `patch_len=6`, `stride=6`
- `d_model=128`, `n_heads=8`, `n_layers=3`, `dim_ff=256`, `dropout=0.1`, `n_classes=3`

```python
class PatchTST(BaseModel):
    """パッチ化 Transformer（PatchTST）による時系列分類モデル。"""

    def __init__(
        self,
        seq_len: int = 60,
        n_features: int = 16,
        patch_len: int = 6,
        stride: int = 6,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
        n_classes: int = 3,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. RevIN normalize
        # 2. Patch 化: (B, T, C) → (B*C, n_patches, patch_len)
        # 3. Transformer Encoder
        # 4. Global Average Pooling → Linear → softmax
        # 5. RevIN denorm は分類では不要（任意）
        ...
```

- `forward` の出力: `F.softmax(logits, dim=-1)` で shape `(B, 3)`

### Step 3: models/training.py 実装（共通学習ループ）

```python
def train(
    model: BaseModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    model_name: str,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    checkpoint_dir: str = "/kaggle/working/",
) -> None:
```

- `CrossEntropyLoss(weight=class_weights)` — class_weights は y_train から計算
- `AdamW` optimizer
- `CosineAnnealingWarmRestarts` scheduler
- Early stopping（val loss が patience エポック改善しない場合）
- CSV ログ: `logs/training_log_{model_name}_{timestamp}.csv`（AC-022/AC-023）
- wandb 禁止（コードに一切含めない）
- GPU/CPU 切り替え: `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- チェックポイント: `best_model.pt` を `checkpoint_dir` に保存

### Step 4: notebooks/train_patchtst.ipynb 作成

- Kaggle commit mode 対応（セル実行が完走する）
- `/kaggle/input/pearless-usdjpy-m5/` からデータをロード
- `/kaggle/working/best_model.pt` にチェックポイント保存

## 品質保証メカニズム

| メカニズム | 確認内容 | 適用箇所 |
|---|---|---|
| モデルパラメータ数チェック（10M 以下）| CPU 推論コスト制約 | 手動確認 + test_models.int.py |
| スタブ連続 100 回推論エラーゼロ | 推論エンジン安定性 | Task 2-7 で検証 |

## 動作確認方法

```python
import torch
from models.patchtst import PatchTST

# AC-009: forward output shape 確認
model = PatchTST()
model.eval()
x = torch.randn(4, 60, 16)
output = model.forward(x)
assert output.shape == (4, 3), f"Expected (4, 3), got {output.shape}"
assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-4), "softmax 合計が 1 でない"
assert output.min() >= 0.0

# AC-010: RevIN モジュール存在確認
assert hasattr(model, 'revin'), "RevIN モジュールが存在しない"
print("OK: shape OK, softmax OK, RevIN OK")

# パラメータ数確認（≤ 10M）
total_params = sum(p.numel() for p in model.parameters())
assert total_params <= 10_000_000, f"パラメータ数が 10M 超: {total_params:,}"
print(f"OK: パラメータ数 {total_params:,} ≤ 10M")
```

**成功基準**:
- `PatchTST(batch=4, seq=60, feat=16).forward(x).shape == (4, 3)` かつ softmax 合計≈1.0 (AC-009)
- RevIN モジュールが存在する (AC-010)
- パラメータ数 ≤ 10M
- `notebooks/train_patchtst.ipynb` が commit mode で実行完了し `best_model.pt` が生成される (AC-013/014)

**検証レベル**: L1（機能動作検証）

## 完了条件

- [x] Implementation: `PatchTST.forward(x).shape == (4, 3)` かつ softmax 合計≈1.0 (AC-009)
- [x] Quality: パラメータ数 ≤ 10M、RevIN モジュールが存在する (AC-010)、CSV ログが出力される (AC-022/023)
- [x] Integration: `notebooks/train_patchtst.ipynb` が Kaggle commit mode で実行完了し `best_model.pt` が生成される (AC-013/014)
