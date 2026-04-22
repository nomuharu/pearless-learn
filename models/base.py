"""models/base.py - 全モデル共通の抽象基底クラス。

Design Doc: fx-prediction-design.md § BaseModel
Contract Definitions: forward(x: Tensor[B, T, F]) -> Tensor[B, 3]
"""

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
               B はバッチサイズ、SEQ_LEN=60、N_FEATURES=16。

        Returns:
            shape (B, N_CLASSES) の softmax 済み確率テンソル。
            N_CLASSES=3: UP=0, DOWN=1, NEUTRAL=2。
        """
        raise NotImplementedError
