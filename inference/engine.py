"""推論エンジン。

Design Doc: fx-prediction-design.md § InferenceEngine
AC-017: CPU 推論 50ms 未満
AC-018: predict() 戻り値契約
AC-020: DataSourceInterface 差し替え可能性
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import torch

from inference.interface import DataSourceInterface
from models.base import BaseModel
from pipeline import feature_engineering

# クラスマッピング: UP=0, DOWN=1, NEUTRAL=2 (Design Doc § Glossary)
_SIGNAL_MAP: dict[int, str] = {0: "UP", 1: "DOWN", 2: "NEUTRAL"}

# feature_engineering の NaN ウォームアップ行数（MA60 = 59 行 + 安全マージン）
_WARMUP_BARS: int = 60


class InferenceEngine:
    """scaler.pkl ロード・特徴量エンジニアリング・モデル推論を統合する推論エンジン。

    コンストラクタでスケーラーをロードし、モデルを eval モードに設定する。

    predict() 戻り値仕様:
        {
            "signal": "UP" | "DOWN" | "NEUTRAL",
            "probabilities": {"UP": float, "DOWN": float, "NEUTRAL": float},
            "inference_ms": float,
        }
    """

    def __init__(
        self,
        model: BaseModel,
        scaler_path: str | Path,
        data_source: DataSourceInterface,
    ) -> None:
        """推論エンジンを初期化する。

        Args:
            model: forward(x) を持つ BaseModel サブクラス。
            scaler_path: pickle 形式で保存された StandardScaler のパス。
            data_source: DataSourceInterface 実装（NamedPipeStub 等）。
        """
        self._model = model
        self._data_source = data_source
        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)
        self._model.eval()

    def predict(self) -> dict[str, object]:
        """推論を実行しシグナルと確率を返す。

        feature_engineering の NaN ウォームアップのため、
        SEQ_LEN + _WARMUP_BARS 本の OHLCV を取得し、
        feature_engineering 適用後の末尾 SEQ_LEN 行を推論に使用する。

        Returns:
            {
                "signal": "UP" | "DOWN" | "NEUTRAL",
                "probabilities": {"UP": float, "DOWN": float, "NEUTRAL": float},
                "inference_ms": float,
            }
        """
        start = time.perf_counter()

        n_fetch = BaseModel.SEQ_LEN + _WARMUP_BARS
        df_raw = self._data_source.fetch_latest_ohlcv(n_bars=n_fetch)

        df_features = feature_engineering(df_raw)

        # 末尾 SEQ_LEN 行を推論ウィンドウとして使用
        df_window = df_features.iloc[-BaseModel.SEQ_LEN :]

        x_raw = df_window.values.reshape(
            1, BaseModel.SEQ_LEN, BaseModel.N_FEATURES
        ).astype(np.float32)

        x_scaled = self._scaler.transform(
            x_raw.reshape(-1, BaseModel.N_FEATURES)
        ).reshape(1, BaseModel.SEQ_LEN, BaseModel.N_FEATURES)

        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            probs_tensor = self._model(x_tensor)  # (1, 3)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        probs: list[float] = probs_tensor[0].tolist()
        signal_idx = int(np.argmax(probs))

        return {
            "signal": _SIGNAL_MAP[signal_idx],
            "probabilities": {
                "UP": probs[0],
                "DOWN": probs[1],
                "NEUTRAL": probs[2],
            },
            "inference_ms": elapsed_ms,
        }
