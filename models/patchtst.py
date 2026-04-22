"""models/patchtst.py - PatchTST モデル実装。

Design Doc: fx-prediction-design.md § PatchTST
ADR-0001: RevIN は PatchTST のみに必須（iTransformer には適用しない）
Contract Definitions: forward(x: Tensor[B, 60, 16]) -> Tensor[B, 3] softmax 済み確率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel


class RevIN(nn.Module):
    """Reversible Instance Normalization（Kim et al., 2022）。

    時系列の分布シフトを吸収するための正規化モジュール。
    mode='norm' で正規化、mode='denorm' で逆正規化を行う。
    分類タスクでは norm のみ使用する。
    """

    def __init__(self, n_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        # 学習可能なアフィンパラメータ（特徴量次元ごと）
        self.affine_weight = nn.Parameter(torch.ones(n_features))
        self.affine_bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """正規化または逆正規化を実行する。

        Args:
            x: shape (B, T, C) の入力テンソル。
            mode: 'norm' で正規化、'denorm' で逆正規化。

        Returns:
            正規化または逆正規化済みのテンソル（shape は入力と同一）。
        """
        if mode == "norm":
            # 時間軸（dim=1）に沿って平均・分散を計算
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + self.eps
            x_norm = (x - mean) / std
            # アフィン変換（ブロードキャスト: (B, 1, C) → (B, T, C)）
            return x_norm * self.affine_weight + self.affine_bias
        elif mode == "denorm":
            # denorm は分類タスクでは通常不要（予測値の逆変換に使用）
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + self.eps
            x_denorm = (x - self.affine_bias) / (self.affine_weight + self.eps)
            return x_denorm * std + mean
        else:
            raise ValueError(
                f"mode は 'norm' または 'denorm' を指定してください。got: {mode!r}"
            )


class PatchTST(BaseModel):
    """パッチ化 Transformer（PatchTST）による時系列分類モデル。

    アーキテクチャ:
        Input: (B, seq_len, n_features)
        RevIN → patch_len=6, stride=6 → n_patches 個のパッチ
        各パッチ: patch_len × n_features 次元 → Linear embedding: → d_model
        Learnable positional encoding
        TransformerEncoder × n_layers 層
        Global Average Pooling over patch 次元
        Classification head: Linear(d_model, d_model) → ReLU → Dropout → Linear(d_model, n_classes)
        Output: (B, n_classes) softmax 確率

    Design Doc: fx-prediction-design.md § PatchTST
    主要ハイパーパラメータ:
        seq_len=60, n_features=16, patch_len=6, stride=6
        d_model=128, n_heads=8, n_layers=3, dim_ff=256, dropout=0.2, n_classes=3
    """

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
        dropout: float = 0.2,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_classes = n_classes

        # パッチ数を計算: floor((seq_len - patch_len) / stride) + 1
        self.n_patches = (seq_len - patch_len) // stride + 1
        # 1パッチあたりの入力次元: patch_len × n_features
        self.patch_dim = patch_len * n_features

        # RevIN モジュール（ADR-0001: PatchTST のみに必須）
        self.revin = RevIN(n_features=n_features)

        # パッチ埋め込み: patch_dim → d_model
        self.patch_embedding = nn.Linear(self.patch_dim, d_model)

        # 学習可能な位置エンコーディング（パッチ数 × d_model）
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.n_patches, d_model))

        # Transformer Encoder
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

        Returns:
            shape (B, n_classes) の softmax 済み確率テンソル。
        """
        batch_size = x.size(0)

        # 1. RevIN 正規化: (B, T, C) → (B, T, C)
        x = self.revin(x, mode="norm")

        # 2. パッチ化: (B, T, C) → (B, n_patches, patch_len * C)
        # unfold で時間軸をパッチに分割
        # x.unfold(dimension, size, step) → (B, C, n_patches, patch_len)
        x_unfolded = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # x_unfolded shape: (B, n_patches, n_features, patch_len)
        # → (B, n_patches, patch_len * n_features)
        x_patches = x_unfolded.contiguous().view(
            batch_size, self.n_patches, self.patch_dim
        )

        # 3. パッチ埋め込み: (B, n_patches, patch_dim) → (B, n_patches, d_model)
        x_embed = self.patch_embedding(x_patches)

        # 4. 位置エンコーディングを加算
        x_embed = x_embed + self.positional_encoding

        # 5. Transformer Encoder: (B, n_patches, d_model) → (B, n_patches, d_model)
        x_encoded = self.transformer_encoder(x_embed)

        # 6. Global Average Pooling: (B, n_patches, d_model) → (B, d_model)
        x_pooled = x_encoded.mean(dim=1)

        # 7. 分類ヘッド: (B, d_model) → (B, n_classes)
        logits = self.classifier(x_pooled)

        # 8. Softmax（CrossEntropyLoss は内部でsoftmaxを含むが、推論時は明示的に適用）
        return F.softmax(logits, dim=-1)
