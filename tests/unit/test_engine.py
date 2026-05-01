"""inference/engine.py の単体テスト。

Design Doc: fx-prediction-design.md § InferenceEngine
AC-017, AC-018, AC-020: 推論エンジンの戻り値・性能・差し替え可能性
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from inference.engine import InferenceEngine
from inference.interface import DataSourceInterface
from inference.pipe_stub import NamedPipeStub
from models.base import BaseModel


# ============================================================
# テスト用フィクスチャ
# ============================================================


@pytest.fixture()
def saved_scaler_path(tmp_path: Path) -> Path:
    """StandardScaler を合成データで fit して tmp_path に保存。"""
    rng = np.random.default_rng(seed=42)
    x_train = rng.random((100, 60, 16)).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, 16))
    scaler_path = tmp_path / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    return scaler_path


@pytest.fixture()
def stub_model() -> BaseModel:
    """常に (batch, 3) softmax 済みテンソルを返すスタブモデル。"""

    class _StubModel(BaseModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            return torch.tensor([[0.7, 0.2, 0.1]]).repeat(batch_size, 1)

    return _StubModel()


@pytest.fixture()
def ohlcv_with_datetime() -> pd.DataFrame:
    """feature_engineering 互換の datetime 列付き (60,6) DataFrame。"""
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=120, freq="5min"),
            "open": 150.0 + rng.normal(0, 0.1, 120),
            "high": 150.3 + rng.normal(0, 0.1, 120),
            "low": 149.7 + rng.normal(0, 0.1, 120),
            "close": 150.0 + rng.normal(0, 0.1, 120),
            "volume": rng.integers(100, 1000, 120).astype(float),
        }
    )


@pytest.fixture()
def mock_data_source(ohlcv_with_datetime: pd.DataFrame) -> DataSourceInterface:
    """DataSourceInterface モック: (120,6) DataFrame を返す。"""
    from unittest.mock import MagicMock

    source = MagicMock(spec=DataSourceInterface)
    source.fetch_latest_ohlcv.return_value = ohlcv_with_datetime
    return source


# ============================================================
# InferenceEngine 戻り値仕様（AC-018）
# ============================================================


class TestInferenceEngineReturnContract:
    """predict() の戻り値が契約を満たすことを検証する。"""

    def test_predict_returns_dict_with_signal_key(
        self,
        stub_model: BaseModel,
        saved_scaler_path: Path,
        mock_data_source: DataSourceInterface,
    ) -> None:
        """predict() が "signal" キーを持つ dict を返すこと。"""
        engine = InferenceEngine(
            model=stub_model,
            scaler_path=saved_scaler_path,
            data_source=mock_data_source,
        )
        result = engine.predict()
        assert "signal" in result

    def test_predict_signal_is_valid_label(
        self,
        stub_model: BaseModel,
        saved_scaler_path: Path,
        mock_data_source: DataSourceInterface,
    ) -> None:
        """signal は "UP"/"DOWN"/"NEUTRAL" のいずれかであること。"""
        engine = InferenceEngine(
            model=stub_model,
            scaler_path=saved_scaler_path,
            data_source=mock_data_source,
        )
        result = engine.predict()
        assert result["signal"] in {"UP", "DOWN", "NEUTRAL"}

    def test_predict_returns_three_probabilities(
        self,
        stub_model: BaseModel,
        saved_scaler_path: Path,
        mock_data_source: DataSourceInterface,
    ) -> None:
        """probabilities が UP/DOWN/NEUTRAL の3キーを持つこと。"""
        engine = InferenceEngine(
            model=stub_model,
            scaler_path=saved_scaler_path,
            data_source=mock_data_source,
        )
        result = engine.predict()
        assert set(result["probabilities"].keys()) == {"UP", "DOWN", "NEUTRAL"}

    def test_predict_probabilities_sum_to_one(
        self,
        stub_model: BaseModel,
        saved_scaler_path: Path,
        mock_data_source: DataSourceInterface,
    ) -> None:
        """確率の合計が 1.0 に近いこと（±1e-4）。"""
        engine = InferenceEngine(
            model=stub_model,
            scaler_path=saved_scaler_path,
            data_source=mock_data_source,
        )
        result = engine.predict()
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-4

    def test_predict_returns_inference_ms_as_float(
        self,
        stub_model: BaseModel,
        saved_scaler_path: Path,
        mock_data_source: DataSourceInterface,
    ) -> None:
        """inference_ms が float 型であること。"""
        engine = InferenceEngine(
            model=stub_model,
            scaler_path=saved_scaler_path,
            data_source=mock_data_source,
        )
        result = engine.predict()
        assert isinstance(result["inference_ms"], float)

    def test_predict_inference_ms_is_non_negative(
        self,
        stub_model: BaseModel,
        saved_scaler_path: Path,
        mock_data_source: DataSourceInterface,
    ) -> None:
        """inference_ms が非負であること。"""
        engine = InferenceEngine(
            model=stub_model,
            scaler_path=saved_scaler_path,
            data_source=mock_data_source,
        )
        result = engine.predict()
        assert result["inference_ms"] >= 0.0

    def test_predict_signal_matches_argmax_of_probabilities(
        self,
        stub_model: BaseModel,
        saved_scaler_path: Path,
        mock_data_source: DataSourceInterface,
    ) -> None:
        """stub_model が [0.7, 0.2, 0.1] を返す場合、signal は "UP" であること。"""
        engine = InferenceEngine(
            model=stub_model,
            scaler_path=saved_scaler_path,
            data_source=mock_data_source,
        )
        result = engine.predict()
        # stub_model は [0.7, 0.2, 0.1] を返す → index 0 → UP
        assert result["signal"] == "UP"


# ============================================================
# DataSourceInterface 差し替え可能性（AC-020）
# ============================================================


class TestInferenceEngineDataSourceSubstitution:
    """DataSourceInterface の差し替えが正常動作することを検証する。"""

    def test_accepts_named_pipe_stub_as_data_source(
        self,
        stub_model: BaseModel,
        saved_scaler_path: Path,
    ) -> None:
        """NamedPipeStub を data_source に注入して predict() が成功すること。"""
        engine = InferenceEngine(
            model=stub_model,
            scaler_path=saved_scaler_path,
            data_source=NamedPipeStub(seed=42),
        )
        result = engine.predict()
        assert result["signal"] in {"UP", "DOWN", "NEUTRAL"}

    def test_accepts_custom_data_source_subclass(
        self,
        stub_model: BaseModel,
        saved_scaler_path: Path,
    ) -> None:
        """任意の DataSourceInterface サブクラスを注入して predict() が成功すること。"""

        class _CustomDataSource(DataSourceInterface):
            def fetch_latest_ohlcv(self, n_bars: int = 60) -> pd.DataFrame:
                rng = np.random.default_rng(seed=7)
                close = 120.0 + rng.normal(0, 0.3, 120).cumsum()
                spread = rng.uniform(0.0, 0.2, 120)
                return pd.DataFrame(
                    {
                        "datetime": pd.date_range(
                            "2024-06-01", periods=120, freq="5min"
                        ),
                        "open": close + rng.uniform(-0.05, 0.05, 120),
                        "high": close + spread,
                        "low": close - spread,
                        "close": close,
                        "volume": rng.integers(50, 500, 120).astype(float),
                    }
                )

        engine = InferenceEngine(
            model=stub_model,
            scaler_path=saved_scaler_path,
            data_source=_CustomDataSource(),
        )
        result = engine.predict()
        assert result["signal"] in {"UP", "DOWN", "NEUTRAL"}
