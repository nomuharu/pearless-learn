"""models/lstm.py - LSTM（RNN系）時系列分類モデル。

旧システム（RNN ベース）との比較対象として追加。
Contract Definitions: forward(x: Tensor[B, T, F]) -> Tensor[B, 3] softmax 済み確率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel


class LSTMModel(BaseModel):
    """双方向 LSTM による時系列分類モデル。

    アーキテクチャ:
        Input: (B, seq_len, n_features)
        LSTM(hidden_size, n_layers, bidirectional) batch_first
        最終時刻の出力（直近足の表現、双方向は順方向・逆方向を連結）
        Classification head: Linear → ReLU → Dropout(0.3) → Linear(n_classes)
        Output: (B, n_classes) softmax 確率

    主要ハイパーパラメータ:
        seq_len=60, n_features=16, hidden_size=128, n_layers=2,
        dropout=0.2, bidirectional=True, n_classes=3
    """

    def __init__(
        self,
        seq_len: int = 60,
        n_features: int = 16,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """推論を実行する。

        Args:
            x: shape (B, seq_len, n_features) の入力テンソル。

        Returns:
            shape (B, n_classes) の softmax 済み確率テンソル。
        """
        # (B, T, F) → (B, T, out_dim)
        out, _ = self.lstm(x)
        # 最終時刻（直近足）の表現を使用。双方向の場合は逆方向側も
        # 同時刻の出力に連結されている
        last = out[:, -1, :]
        logits = self.classifier(last)
        return F.softmax(logits, dim=-1)
