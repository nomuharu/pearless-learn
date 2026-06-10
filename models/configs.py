"""models/configs.py - モデル別設定（使用特徴量・アーキテクチャ・学習ハイパラ）の一元管理。

方針: 問題（train/val/test 分割・ラベル）は全モデル共通、特徴量とハイパラはモデル別。
npy はフル特徴量（ALL_FEATURES 順、16列）で 1 セットだけ生成し、
学習・評価時に ModelConfig.select_features() で必要な列だけに絞る。
これによりモデルごとに Kaggle Dataset を作り直す必要がない。

新モデルの追加手順:
    1. models/xxx.py に BaseModel 継承クラスを実装する
    2. MODEL_CONFIGS にエントリを 1 つ追加する（features / model_kwargs / train を指定）
    evaluate.py の CLI choices や比較 CSV は MODEL_CONFIGS から自動生成される。

注意: このモジュールは Kaggle 配布（pearless-src dataset）に含まれるため、
pipeline.py など models/ 外のモジュールに依存しないこと。
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models.base import BaseModel
from models.cnn import CNNModel
from models.itransformer import iTransformer
from models.patchtst import PatchTST

# 全特徴量の正準順序（ADR-0002）。npy の特徴量軸（最終軸）と一致する。
# pipeline.FEATURE_NAMES はこの定義を参照する（単一情報源）。
ALL_FEATURES: tuple[str, ...] = (
    "ma60_deviation",  # MA60乖離率: (close - MA60) / MA60
    "ceiling_degree",  # 天井度: close.rolling(60).max()
    "ma20",  # MA20 単純移動平均
    "ma10",  # MA10 単純移動平均
    "prev_ratio",  # 前足比: close.pct_change()
    "hlo",  # HLO: high - low
    "diff_hlo_and_average",  # HLO - HLO の14期間移動平均
    "cci",  # CCI(20): ta.trend.CCIIndicator
    "rsi",  # RSI(9): ta.momentum.RSIIndicator
    "swing",  # 振れ幅: abs(high - open)
    "vwap_deviation",  # VWAP乖離率: (close - VWAP) / VWAP
    "bb_pband",  # BB%B: ta.volatility.BollingerBands.bollinger_pband()
    "macd_hist",  # MACDヒストグラム: ta.trend.MACD.macd_diff()
    "atr",  # ATR(14): ta.volatility.AverageTrueRange
    "time_sin",  # 時間帯sin: sin(2π * time_index / 288)
    "time_cos",  # 時間帯cos: cos(2π * time_index / 288)
)


@dataclass(frozen=True)
class TrainConfig:
    """学習ハイパーパラメータ。models.training.train() の引数に対応する。"""

    n_epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 15
    scheduler_t0: int = 10


@dataclass(frozen=True)
class ModelConfig:
    """1 モデル分の設定（使用特徴量・アーキテクチャ・学習ハイパラ）。

    Attributes:
        name: モデル識別名。チェックポイント（best_{name}.pt）や
            学習ログ（training_log_{name}_*.csv）のファイル名に使用する。
        model_cls: BaseModel を継承したモデルクラス。
        features: 使用する特徴量名のタプル。ALL_FEATURES の部分集合であること。
            順序は ALL_FEATURES に合わせる必要はない（指定順に列が並ぶ）。
        model_kwargs: model_cls のコンストラクタに渡す追加引数
            （n_features は features から自動算出されるため含めないこと）。
        train: 学習ハイパーパラメータ。
    """

    name: str
    model_cls: type[BaseModel]
    features: tuple[str, ...] = ALL_FEATURES
    model_kwargs: dict[str, int | float] = field(default_factory=dict)
    train: TrainConfig = TrainConfig()

    def __post_init__(self) -> None:
        unknown = [f for f in self.features if f not in ALL_FEATURES]
        if unknown:
            raise ValueError(
                f"ALL_FEATURES に存在しない特徴量: {unknown}（model={self.name!r}）"
            )
        if len(set(self.features)) != len(self.features):
            raise ValueError(f"特徴量名が重複しています（model={self.name!r}）")
        if "n_features" in self.model_kwargs:
            raise ValueError(
                f"n_features は features から自動算出されるため "
                f"model_kwargs に含めないでください（model={self.name!r}）"
            )

    @property
    def n_features(self) -> int:
        """使用する特徴量数。"""
        return len(self.features)

    def feature_indices(self) -> tuple[int, ...]:
        """ALL_FEATURES 基準の列インデックス（フル特徴量 npy から列を選ぶ用）。"""
        return tuple(ALL_FEATURES.index(f) for f in self.features)

    def select_features(
        self, X: np.ndarray[Any, np.dtype[np.float32]]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """フル特徴量配列 (N, T, len(ALL_FEATURES)) から使用列だけを抽出する。

        Args:
            X: 最終軸が ALL_FEATURES 順のフル特徴量配列。

        Returns:
            shape (N, T, n_features) の配列。全特徴量使用時は X をそのまま返す。
        """
        if X.shape[-1] != len(ALL_FEATURES):
            raise ValueError(
                f"X の特徴量軸は {len(ALL_FEATURES)} を期待: got {X.shape[-1]}"
            )
        if self.features == ALL_FEATURES:
            return X
        return X[:, :, list(self.feature_indices())]

    def build_model(self) -> BaseModel:
        """設定に従いモデルインスタンスを生成する。"""
        return self.model_cls(n_features=self.n_features, **self.model_kwargs)


# モデル名 → 設定のレジストリ。新モデルはここに 1 エントリ追加するだけでよい。
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "patchtst": ModelConfig(
        name="patchtst",
        model_cls=PatchTST,
        # アーキテクチャ既定値: seq_len=60, patch_len=6, stride=6,
        # d_model=128, n_heads=8, n_layers=3, dim_ff=256, dropout=0.2
    ),
    "itransformer": ModelConfig(
        name="itransformer",
        model_cls=iTransformer,
        # アーキテクチャ既定値: seq_len=60, d_model=128, n_heads=8,
        # n_layers=3, dim_ff=256, dropout=0.2
    ),
    "cnn": ModelConfig(
        name="cnn",
        model_cls=CNNModel,
        # アーキテクチャ既定値: seq_len=60（Conv チャネルは 32→64→128 固定）
    ),
}


def get_config(name: str) -> ModelConfig:
    """モデル名から設定を取得する。

    Raises:
        ValueError: 未登録のモデル名が指定された場合。
    """
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"未知のモデル名: {name!r}。有効な値: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[name]
