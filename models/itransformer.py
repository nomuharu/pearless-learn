"""models/itransformer.py - iTransformer モデル実装。

Design Doc: fx-prediction-design.md § iTransformer
ADR-0001: RevIN は適用しない。直接転置操作 (B,60,16) → (B,16,60) を行う。
Contract Definitions: forward(x: Tensor[B, 60, 16]) -> Tensor[B, 3] softmax 済み確率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel


class iTransformer(BaseModel):
    """iTransformer — 特徴量軸 Attention の時系列分類モデル。

    設計上の特徴:
        - 入力 (B, T, C) を (B, C, T) に転置し、C 軸（特徴量）で Attention を計算する。
        - RevIN は適用しない（ADR-0001 Implementation Guidance 準拠）。

    アーキテクチャ:
        Input: (B, seq_len, n_features) = (B, 60, 16)
        転置: (B, 16, 60)  ← RevIN なし
        Linear embedding: 60 → d_model=128（各特徴量を時間軸でembedding）
        Learnable positional encoding（feature次元に対して）
        TransformerEncoder × n_layers 層（特徴量軸で Self-Attention）
        Global Average Pooling over feature 次元
        Classification head: Linear(d_model, d_model) → ReLU → Dropout(0.3) → Linear(d_model, n_classes)
        Output: (B, n_classes) softmax 確率

    Design Doc: fx-prediction-design.md § iTransformer
    主要ハイパーパラメータ:
        seq_len=60, n_features=16, d_model=128, n_heads=8, n_layers=3, dim_ff=256, dropout=0.2, n_classes=3
    """

    def __init__(
        self,
        seq_len: int = 60,
        n_features: int = 16,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.2,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model
        self.n_classes = n_classes

        # 特徴量軸 embedding: 各特徴量の時系列（長さ seq_len）を d_model 次元へ射影
        # 転置後の shape (B, n_features, seq_len) に対して、seq_len → d_model の Linear を適用
        self.feature_embedding = nn.Linear(seq_len, d_model)

        # 学習可能な位置エンコーディング（特徴量次元に対して）
        self.positional_encoding = nn.Parameter(torch.zeros(1, n_features, d_model))

        # Transformer Encoder（特徴量軸で Self-Attention を計算）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )

        # 分類ヘッド: Linear(d_model, d_model) → ReLU → Dropout(0.3) → Linear(d_model, n_classes)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """推論を実行する。

        Args:
            x: shape (B, seq_len, n_features) の入力テンソル。
               B はバッチサイズ、seq_len=60、n_features=16。

        Returns:
            shape (B, n_classes) の softmax 済み確率テンソル。
            n_classes=3: UP=0, DOWN=1, NEUTRAL=2。
        """
        # 1. 転置: (B, T, C) → (B, C, T) = (B, 16, 60)
        #    RevIN は適用しない（ADR-0001 準拠）
        x_transposed = x.permute(0, 2, 1)

        # 2. 特徴量軸 embedding: (B, n_features, seq_len) → (B, n_features, d_model)
        x_embedded = self.feature_embedding(x_transposed)

        # 3. 位置エンコーディングを加算
        x_embedded = x_embedded + self.positional_encoding

        # 4. Transformer Encoder（特徴量軸で Attention を計算）:
        #    (B, n_features, d_model) → (B, n_features, d_model)
        x_encoded = self.transformer_encoder(x_embedded)

        # 5. Global Average Pooling over feature 次元:
        #    (B, n_features, d_model) → (B, d_model)
        x_pooled = x_encoded.mean(dim=1)

        # 6. 分類ヘッド: (B, d_model) → (B, n_classes)
        logits = self.classifier(x_pooled)

        # 7. Softmax（推論時は明示的に適用）
        return F.softmax(logits, dim=-1)
