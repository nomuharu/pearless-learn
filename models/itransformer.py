"""models/itransformer.py - iTransformer モデル実装。

Design Doc: fx-prediction-design.md § iTransformer
ADR-0001: RevIN は適用しない。直接転置操作 (B,60,16) → (B,16,60) を行う。
Contract Definitions: forward(x: Tensor[B, 60, 16]) -> Tensor[B, 3] softmax 済み確率
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel


class _SelfAttention(nn.Module):
    """Scaled Dot-Product Self-Attention（手動実装、CUDA互換性確保）。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, HD = self.n_heads, self.head_dim
        # (B, S, D) → (B, H, S, HD)
        q = self.q(x).view(B, S, H, HD).transpose(1, 2)
        k = self.k(x).view(B, S, H, HD).transpose(1, 2)
        v = self.v(x).view(B, S, H, HD).transpose(1, 2)
        # Attention weights
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, S, S)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        # (B, H, S, HD) → (B, S, D)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class _TransformerBlock(nn.Module):
    """Transformer Block（Pre-Norm, 手動実装）。"""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = _SelfAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


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
        self.feature_embedding = nn.Linear(seq_len, d_model)

        # 学習可能な位置エンコーディング（特徴量次元に対して）
        self.positional_encoding = nn.Parameter(torch.zeros(1, n_features, d_model))

        # Transformer Encoder（完全手動実装でCUDA互換性を確保）
        self.encoder_blocks = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])

        # 分類ヘッド
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

        Returns:
            shape (B, n_classes) の softmax 済み確率テンソル。
        """
        # 1. 転置: (B, T, C) → (B, C, T) = (B, 16, 60)
        x_transposed = x.permute(0, 2, 1)

        # 2. 特徴量軸 embedding: (B, n_features, seq_len) → (B, n_features, d_model)
        x_embedded = self.feature_embedding(x_transposed)

        # 3. 位置エンコーディングを加算
        x_embedded = x_embedded + self.positional_encoding

        # 4. Transformer Encoder blocks
        x_encoded = x_embedded
        for block in self.encoder_blocks:
            x_encoded = block(x_encoded)

        # 5. Global Average Pooling: (B, n_features, d_model) → (B, d_model)
        x_pooled = x_encoded.mean(dim=1)

        # 6. 分類ヘッド
        logits = self.classifier(x_pooled)

        return F.softmax(logits, dim=-1)
