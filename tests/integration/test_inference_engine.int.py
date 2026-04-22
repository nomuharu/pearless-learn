# 推論エンジン Integration Test - Design Doc: fx-prediction-design.md
# Generated: 2026-04-21 | Budget Used: 3/3 integration

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from inference.engine import InferenceEngine
from inference.interface import DataSourceInterface
from inference.pipe_stub import NamedPipeStub

_VALID_SIGNALS = {"UP", "DOWN", "NEUTRAL"}


# ============================================================
# AC-018: 推論シグナル + 確率出力
# ROI: 109 (BV:10 × Freq:10 + Legal:0 + Defect:9)
# Behavior: InferenceEngine.predict() 実行 →
#           "UP"/"DOWN"/"NEUTRAL" シグナル + 各確率3値 が返される
# @category: core-functionality
# @dependency: InferenceEngine, BaseModel, DataSourceInterface, scaler.pkl
# @real-dependency: pytorch
# @complexity: high
# ============================================================
def test_inference_engine_returns_signal_and_three_probabilities(
    mock_data_source, stub_model, saved_scaler_path
):
    """
    AC-018: InferenceEngine.predict()がシグナルと確率3値を返すこと

    Arrange:
      - DataSourceInterface をモック: fetch_latest_ohlcv() が (120, 6) DataFrame を返す
      - BaseModel をモック（または最小構成の実モデル）:
        forward(x) が (1, 3) softmax済みTensorを返す
        例: torch.tensor([[0.7, 0.2, 0.1]])
      - scaler.pkl を合成データで作成しtmp_pathに保存
      - InferenceEngine(model=stub_model, scaler_path=saved_scaler_path,
                        data_source=mock_data_source) を構築
    Act:
      - result = engine.predict()
    Assert:
      - result["signal"] in {"UP", "DOWN", "NEUTRAL"}
      - set(result["probabilities"].keys()) == {"UP", "DOWN", "NEUTRAL"}
      - abs(sum(result["probabilities"].values()) - 1.0) < 1e-4
      - result["inference_ms"] が float 型であること
    Pass criteria:
      - 全フィールド存在 + 確率合計≈1.0 → Pass
      - フィールド欠損 または 確率合計≠1.0 → Fail
    Verification items:
      - signal は "UP", "DOWN", "NEUTRAL" の3択
      - probabilities の合計 ≈ 1.0
      - inference_ms は float (非負)
    """
    engine = InferenceEngine(stub_model, saved_scaler_path, mock_data_source)

    result = engine.predict()

    assert result["signal"] in _VALID_SIGNALS
    assert set(result["probabilities"].keys()) == _VALID_SIGNALS
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4
    assert isinstance(result["inference_ms"], float)


# ============================================================
# AC-017: 推論時間 50ms 制約
# ROI: 100 (BV:10 × Freq:9 + Legal:0 + Defect:10)
# Behavior: predict() を100回実行 → 平均推論時間が50ms未満
# @category: core-functionality
# @dependency: InferenceEngine, BaseModel, DataSourceInterface, scaler.pkl
# @real-dependency: pytorch (CPU)
# @complexity: high
# ============================================================
@pytest.mark.skip(reason="PatchTST実装後（Task 2-2完了後）に有効化")
def test_inference_engine_completes_within_50ms_average_over_100_runs(
    mock_data_source, saved_scaler_path
):
    """
    AC-017: 100回連続推論の平均時間が50ms未満であること（WSL2 CPU環境）

    Arrange:
      - DataSourceInterface モック: (120, 6) DataFrame を返す
      - パラメータ数が10M以下のPatchTSTまたはiTransformerの最小構成（または実モデル）
        を CPU デバイスで初期化
      - InferenceEngine を構築
    Act:
      - import time.perf_counter を使用して100回 predict() を計測
      - avg_ms = sum(elapsed_times) / 100
    Assert:
      - avg_ms < 50.0
    Pass criteria:
      - 平均推論時間 < 50ms → Pass
      - 50ms以上 → Fail（警告: CI環境の負荷によりフレーキーになりうる。
                         CI時は100ms以下を下限とする代替基準を設けること）
    Verification items:
      - 100回の全推論でエラーなし
      - avg_ms < 50.0
    """
    # PatchTST 実装後にモデルを差し替える
    # minimal_patchtst_model を引数に追加し、stub_model → minimal_patchtst_model へ変更
    pass


# ============================================================
# AC-019: スタブ 100回推論エラーゼロ
# ROI: 82 (BV:9 × Freq:8 + Legal:0 + Defect:10)
# Behavior: NamedPipeStub 経由で100回 predict() 実行 → エラーゼロ
# @category: core-functionality
# @dependency: InferenceEngine, NamedPipeStub, BaseModel, scaler.pkl
# @real-dependency: pytorch
# @complexity: medium
# ============================================================
def test_stub_100_consecutive_predictions_produce_no_errors(
    stub_model, saved_scaler_path
):
    """
    AC-019: NamedPipeStubを使用した100回連続推論でエラーが0件であること

    Arrange:
      - NamedPipeStub() をインスタンス化
      - InferenceEngine(model=stub_model,
                        scaler_path=saved_scaler_path,
                        data_source=NamedPipeStub()) を構築
    Act:
      - predict() を100回実行、例外をすべてキャッチして記録
    Assert:
      - エラー件数 == 0
      - 100回すべてで result["signal"] が {"UP","DOWN","NEUTRAL"} に含まれること
    Pass criteria:
      - 100回中エラー0件 → Pass
      - 1件でも例外発生 → Fail
    Verification items:
      - errors == []
      - len(results) == 100
      - すべての result["signal"] が有効値
    """
    engine = InferenceEngine(stub_model, saved_scaler_path, NamedPipeStub())

    errors: list[Exception] = []
    results: list[dict] = []

    for _ in range(100):
        try:
            result = engine.predict()
            results.append(result)
        except Exception as exc:
            errors.append(exc)

    assert errors == [], (
        f"推論中に {len(errors)} 件のエラーが発生しました: {errors[:3]}"
    )
    assert len(results) == 100
    assert all(r["signal"] in _VALID_SIGNALS for r in results)


# ============================================================
# AC-020: DataSourceInterface 差替え可能性
# ROI: 65 (BV:8 × Freq:7 + Legal:0 + Defect:9)
# Behavior: DataSourceInterface のサブクラスを差し替えても
#           InferenceEngine は同一インターフェースで動作する
# @category: integration
# @dependency: InferenceEngine, DataSourceInterface
# @complexity: medium
# ============================================================
def test_inference_engine_accepts_any_data_source_interface_subclass(
    stub_model, saved_scaler_path
):
    """
    AC-020: DataSourceInterface を継承した任意のサブクラスを
            InferenceEngine に注入できること

    Arrange:
      - DataSourceInterface を継承した CustomDataSource クラスを定義
        (fetch_latest_ohlcv が合成(120, 6) DataFrameを返す)
    Act:
      - engine_a = InferenceEngine(stub_model, saved_scaler_path, NamedPipeStub())
        result_a = engine_a.predict()
      - engine_b = InferenceEngine(stub_model, saved_scaler_path, CustomDataSource())
        result_b = engine_b.predict()
    Assert:
      - result_a["signal"] in {"UP", "DOWN", "NEUTRAL"}
      - result_b["signal"] in {"UP", "DOWN", "NEUTRAL"}
      - 両エンジンがエラーなく実行されること
    Pass criteria:
      - 2つの異なるDataSource実装が同じインターフェースで動作 → Pass
    """

    class CustomDataSource(DataSourceInterface):
        """テスト専用の DataSourceInterface 実装。"""

        def fetch_latest_ohlcv(self, n_bars: int = 60) -> pd.DataFrame:
            rng = np.random.default_rng(seed=99)
            close = 120.0 + rng.normal(0, 0.3, n_bars).cumsum()
            spread = rng.uniform(0.0, 0.2, n_bars)
            return pd.DataFrame(
                {
                    "datetime": pd.date_range(
                        "2024-06-01", periods=n_bars, freq="5min"
                    ),
                    "open": close + rng.uniform(-0.1, 0.1, n_bars),
                    "high": close + spread,
                    "low": close - spread,
                    "close": close,
                    "volume": rng.integers(200, 800, n_bars).astype(float),
                }
            )

    engine_a = InferenceEngine(stub_model, saved_scaler_path, NamedPipeStub())
    result_a = engine_a.predict()

    engine_b = InferenceEngine(stub_model, saved_scaler_path, CustomDataSource())
    result_b = engine_b.predict()

    assert result_a["signal"] in _VALID_SIGNALS
    assert result_b["signal"] in _VALID_SIGNALS


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def saved_scaler_path(tmp_path):
    """
    StandardScalerを(100, 60, 16)の合成データでfitし、
    tmp_path/scaler.pklとして保存してパスを返す。
    """
    import pickle
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed=42)
    X_train = rng.random((100, 60, 16)).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, 16))
    scaler_path = tmp_path / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    return scaler_path


@pytest.fixture
def mock_data_source():
    """
    DataSourceInterface をモック。
    fetch_latest_ohlcv() は (120, 6) の合成OHLCVデータ（datetime列付き）を返す。
    engine.py は SEQ_LEN(60) + WARMUP(60) = 120 行を要求する。
    """
    n_bars = 120
    rng = np.random.default_rng(seed=42)
    ohlcv_data = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=n_bars, freq="5min"),
            "open": 150.0 + rng.normal(0, 0.1, n_bars),
            "high": 150.3 + rng.normal(0, 0.1, n_bars),
            "low": 149.7 + rng.normal(0, 0.1, n_bars),
            "close": 150.0 + rng.normal(0, 0.1, n_bars),
            "volume": rng.integers(100, 1000, n_bars).astype(float),
        }
    )
    source = MagicMock()
    source.fetch_latest_ohlcv.return_value = ohlcv_data
    return source


@pytest.fixture
def stub_model():
    """
    BaseModel のスタブ。
    forward(x) は常に (batch, 3) のsoftmax済みTensorを返す。
    実装時: torch.tensor([[0.7, 0.2, 0.1]]) を返す最小モデル。
    """
    import torch
    import torch.nn as nn

    class StubModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            probs = torch.tensor([[0.7, 0.2, 0.1]]).repeat(batch_size, 1)
            return probs

    return StubModel()


@pytest.fixture
def minimal_patchtst_model():
    """
    PatchTST の最小構成（実PyTorchモデル）。
    パラメータ: seq_len=60, n_features=16, d_model=64, n_heads=2, n_layers=1
    CPUデバイスで推論。
    @real-dependency: pytorch
    """
    # Phase 2 Task 2-2 で PatchTST が実装されるまで pass のままにする
    # pytest.skip を使って Red 状態を保つ
    pytest.skip("PatchTST 未実装（Task 2-2 で実装予定）")
