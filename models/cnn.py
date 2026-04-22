"""models/cnn.py - CNN ベースラインモデル。

Design Doc: fx-prediction-design.md § CNN ベースライン
AC-021: 全メトリクスが PatchTST / iTransformer と同一 CSV レポートに出力され、CNN 列が欠損なく存在すること
Contract Definitions: forward(x: Tensor[B, T, F]) -> Tensor[B, 3]
"""

import torch
import torch.nn as nn

from models.base import BaseModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNModel(BaseModel):
    """5分足 OHLCV 時系列分類の CNN ベースライン。

    アーキテクチャ:
        Input: (B, 60, 16) → permute → (B, 16, 60) for Conv1d
        Conv1d(16, 32, kernel_size=3, padding=1) → BatchNorm → ReLU
        Conv1d(32, 64, kernel_size=3, padding=1) → BatchNorm → ReLU
        Conv1d(64, 128, kernel_size=3, padding=1) → BatchNorm → ReLU
        AdaptiveAvgPool1d(1) → Flatten
        Linear(128, 3)
        Output: (B, 3) softmax 済み確率

    目的: PatchTST / iTransformer との比較基準（AC-021）。
    パラメータ数目安: 約 40K（10M 制約を大幅に下回る）。
    """

    def __init__(
        self,
        seq_len: int = 60,
        n_features: int = 16,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self._seq_len = seq_len
        self._n_features = n_features
        self._n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """推論を実行する。

        Args:
            x: shape (B, SEQ_LEN, N_FEATURES) の入力テンソル。
               B はバッチサイズ、SEQ_LEN=60、N_FEATURES=16。

        Returns:
            shape (B, N_CLASSES) の softmax 済み確率テンソル。
            N_CLASSES=3: UP=0, DOWN=1, NEUTRAL=2。
        """
        # x: (B, T, C) = (B, 60, 16)
        # Conv1d は (B, C, T) を期待するため permute
        x_t = x.permute(0, 2, 1)  # (B, 16, 60)
        out = self.conv_block1(x_t)  # (B, 32, 60)
        out = self.conv_block2(out)  # (B, 64, 60)
        out = self.conv_block3(out)  # (B, 128, 60)
        out = self.pool(out)  # (B, 128, 1)
        out = out.flatten(start_dim=1)  # (B, 128)
        logits = self.classifier(out)  # (B, 3)
        return torch.softmax(logits, dim=-1)
