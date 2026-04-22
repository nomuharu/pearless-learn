# models/base.py の BaseModel ユニットテスト
# Design Doc: fx-prediction-design.md § BaseModel

import pytest
import torch
import torch.nn as nn


# ============================================================
# BaseModel クラス定数の検証
# ============================================================
def test_base_model_class_constants():
    """
    BaseModel にクラス定数 SEQ_LEN=60, N_FEATURES=16, N_CLASSES=3 が定義されていること

    Arrange: BaseModel をインポート
    Act: クラス定数を参照
    Assert: 各定数が期待値と一致する
    """
    from models.base import BaseModel

    assert BaseModel.SEQ_LEN == 60
    assert BaseModel.N_FEATURES == 16
    assert BaseModel.N_CLASSES == 3


# ============================================================
# BaseModel が nn.Module を継承しているか検証
# ============================================================
def test_base_model_inherits_nn_module():
    """
    BaseModel が torch.nn.Module を継承していること

    Arrange: BaseModel をインポート
    Assert: issubclass(BaseModel, nn.Module) が True
    """
    from models.base import BaseModel

    assert issubclass(BaseModel, nn.Module)


# ============================================================
# 未実装サブクラスの forward 呼び出しで NotImplementedError が発生するか検証
# ============================================================
def test_incomplete_subclass_forward_raises_error():
    """
    forward を実装しないサブクラスの forward 呼び出しで NotImplementedError が発生すること。
    nn.Module と abstractmethod の組み合わせでは、インスタンス化時に TypeError が発生しない場合がある。
    forward メソッドの ... (Ellipsis) は呼び出し時に NotImplementedError を発生させる。

    Arrange: forward を実装しないサブクラスを定義してインスタンス化
    Act: forward(x) を呼び出す
    Assert: NotImplementedError が発生する
    """
    from models.base import BaseModel

    class IncompleteModel(BaseModel):
        pass

    model = IncompleteModel()
    x = torch.randn(4, 60, 16)
    with pytest.raises((NotImplementedError, TypeError)):
        model.forward(x)


# ============================================================
# 具象クラスが正常に動作するか検証
# ============================================================
def test_concrete_subclass_forward_works():
    """
    forward を実装した具象サブクラスが正常に動作すること

    Arrange: forward を実装した ConcreteModel を定義
    Act: x = torch.randn(4, 60, 16) を forward に渡す
    Assert: output.shape == (4, 3)
    """
    from models.base import BaseModel

    class ConcreteModel(BaseModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.softmax(torch.randn(x.shape[0], self.N_CLASSES), dim=-1)

    model = ConcreteModel()
    x = torch.randn(4, 60, 16)
    output = model.forward(x)

    assert output.shape == (4, 3)


# ============================================================
# 具象クラスが BaseModel の isinstance であることを検証
# ============================================================
def test_concrete_subclass_is_instance_of_base_model():
    """
    具象サブクラスのインスタンスが BaseModel の isinstance であること

    Arrange: ConcreteModel を定義してインスタンス化
    Assert: isinstance(model, BaseModel) が True
    """
    from models.base import BaseModel

    class ConcreteModel(BaseModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.softmax(torch.randn(x.shape[0], self.N_CLASSES), dim=-1)

    model = ConcreteModel()
    assert isinstance(model, BaseModel)
    assert isinstance(model, nn.Module)
