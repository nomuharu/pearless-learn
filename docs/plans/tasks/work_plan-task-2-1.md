# Task 2-1: models/base.py（BaseModel 抽象クラス）実装

## タスク概要

PatchTST / iTransformer / CNN 共通の抽象基底クラス `BaseModel` を定義する。入出力 shape 契約（B, 60, 16）→（B, 3）をクラス定数として文書化し、全具象モデルの DI 基盤を確立する。

## 対象ファイル

- `models/__init__.py` (新規)
- `models/base.py` (新規)

## 調査対象

- `/home/nomu/claude_code/pearless/docs/design/fx-prediction-design.md` (§ Contract Definitions、§ Integration Points List の BaseModel 節)
- `/home/nomu/claude_code/pearless/docs/adr/ADR-0001-model-architecture-selection.md` (モデルインターフェース設計方針)
- `/home/nomu/claude_code/pearless/docs/plans/work_plan.md` (§ Task 2-1)

## 実装手順

### Step 1: models/ パッケージ作成

```bash
mkdir -p models
touch models/__init__.py
```

### Step 2: BaseModel 実装

```python
"""models/base.py - 全モデル共通の抽象基底クラス。"""
from abc import abstractmethod

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """ML モデルの抽象基底クラス。

    入出力 shape 契約:
        Input:  (B, SEQ_LEN, N_FEATURES) = (B, 60, 16)
        Output: (B, N_CLASSES) = (B, 3)   softmax 済み確率

    クラス定数:
        SEQ_LEN:     時系列ウィンドウサイズ（60 本）
        N_FEATURES:  特徴量数（16）
        N_CLASSES:   クラス数（3: UP=0, DOWN=1, NEUTRAL=2）
    """

    SEQ_LEN: int = 60
    N_FEATURES: int = 16
    N_CLASSES: int = 3

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """推論を実行する。

        Args:
            x: shape (B, SEQ_LEN, N_FEATURES) の入力テンソル。

        Returns:
            shape (B, N_CLASSES) の softmax 済み確率テンソル。
        """
        ...
```

- `nn.Module` と Python `ABC` の両立: `nn.Module` は独自の `__abstractmethods__` 機構を持つため `abstractmethod` で `NotImplementedError` を引き起こす

### Step 3: 動作確認

```python
from models.base import BaseModel
import torch

# 未実装サブクラスのテスト
class IncompleteModel(BaseModel):
    pass

model = IncompleteModel()
x = torch.randn(4, 60, 16)
try:
    model.forward(x)
    assert False, "NotImplementedError が発生するはず"
except (NotImplementedError, TypeError):
    print("OK: 未実装で NotImplementedError")

# クラス定数確認
assert BaseModel.SEQ_LEN == 60
assert BaseModel.N_FEATURES == 16
assert BaseModel.N_CLASSES == 3
print("OK: クラス定数確認済み")
```

## 品質保証メカニズム

このタスクには project-wide の品質メカニズムが適用される:
- `uv sync 完全再現`: torch が正しくインストールされていること

## 動作確認方法

```bash
uv run python -c "
from models.base import BaseModel
import torch

class ConcreteModel(BaseModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(torch.randn(x.shape[0], self.N_CLASSES), dim=-1)

model = ConcreteModel()
x = torch.randn(4, 60, 16)
out = model.forward(x)
assert out.shape == (4, 3), f'Expected (4, 3), got {out.shape}'
print(f'OK: output shape={out.shape}')
"
```

**成功基準**:
- `BaseModel` が正しく定義され、具象クラスが `forward` を実装しないと `NotImplementedError` が発生する
- クラス定数 `SEQ_LEN=60`, `N_FEATURES=16`, `N_CLASSES=3` が定義されている

**検証レベル**: L3（ビルド成功検証）

## 完了条件

- [x] Implementation: `BaseModel` が `nn.Module` を継承し、`forward` が `@abstractmethod` で定義されている
- [x] Quality: 未実装サブクラスの `forward` 呼び出しで `NotImplementedError` が発生する
- [x] Integration: クラス定数 `SEQ_LEN`, `N_FEATURES`, `N_CLASSES` が定義され、具象クラスが正常に動作する
