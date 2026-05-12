# パイプライン → 推論 E2E Test - Design Doc: fx-prediction-design.md
# Generated: 2026-04-21 | Budget Used: 1/2 E2E
# Test Type: End-to-End Test
# Implementation Timing: 全フェーズ実装完了後に実行

import pytest
import numpy as np
import pandas as pd


# ============================================================
# User Journey: データ準備 → 推論（パイプライン実行 → スタブ経由推論）
# ROI: 100 (BV:10 × Freq:9 + Legal:0 + Defect:10) | reserved slot: multi-step journey
# Journey Steps:
#   Step 1 (CLI境界): run_pipeline(csv_path, output_dir)
#           → X_*.npy, y_*.npy, scaler.pkl を生成
#   Step 2 (CLI境界): InferenceEngine(model, scaler_path, NamedPipeStub())
#           → predict() でシグナル生成
# State carries: scaler.pkl (Step1 → Step2), numpy配列の形状検証
# Completion point: predict() が "UP"/"DOWN"/"NEUTRAL" シグナルを返す
# @category: e2e
# @dependency: full-system
# @complexity: high
# ============================================================
def test_full_pipeline_to_stub_inference_end_to_end(
    synthetic_ohlcv_csv, tmp_path, minimal_patchtst_model
):
    """
    User Journey: CSVデータ準備 → パイプライン実行 → Named Pipe スタブ経由推論

    Arrange:
      - 合成OHLCVデータCSVを準備（300本以上）
      - tmp_path を output_dir として使用
      - CPU用 PatchTST 最小構成モデルを準備

    === Step 1: データパイプライン実行（CLI境界1）===
    Act 1:
      - run_pipeline(csv_path=synthetic_ohlcv_csv, output_dir=tmp_path)
    Assert 1 (Step1完了確認):
      - X_train.npy が存在する
      - X_train の shape == (N, 60, 16) かつ dtype == float32
      - scaler.pkl が tmp_path に存在する
      - NaN が存在しない

    === Step 2: 推論エンジン実行（CLI境界2） ===
    Act 2:
      - NamedPipeStub() をインスタンス化
      - InferenceEngine(
            model=minimal_patchtst_model,
            scaler_path=tmp_path / "scaler.pkl",
            data_source=NamedPipeStub()
        ) を構築
      - result = engine.predict()
    Assert 2 (Step2完了確認・ジャーニー完結):
      - result["signal"] in {"UP", "DOWN", "NEUTRAL"}
      - set(result["probabilities"].keys()) == {"UP", "DOWN", "NEUTRAL"}
      - abs(sum(result["probabilities"].values()) - 1.0) < 1e-4
      - result["inference_ms"] < 500.0  # E2E は初回ウォームアップ込みのため500ms以内

    Pass criteria:
      - Step1のnumpy配列 → Step2のscaler.pklが正しく引き継がれてシグナルが返る → Pass
      - いずれかのステップで例外 → Fail

    Verification items:
      - Step1: scaler.pkl が生成されること (AC-004)
      - Step1→Step2 state carry: scaler.pklがInferenceEngineに正しくロードされること
      - Step2: シグナル + 確率3値が返ること (AC-018)
      - Step2: 推論時間 < 500ms (E2Eはウォームアップ込み。AC-017の100回平均50ms確認はintegrationテストで実施)
      - Step2: スタブエラーゼロ (AC-019)
            ※ 1回の呼び出しでスタブエラーゼロを確認（100回連続はintegrationテストで検証済み）
    """
    from pipeline import run_pipeline
    from inference.engine import InferenceEngine
    from inference.pipe_stub import NamedPipeStub

    # === Step 1: データパイプライン実行（CLI 境界 1）===
    run_pipeline(str(synthetic_ohlcv_csv), str(tmp_path))

    # Assert 1: Step 1 完了確認
    assert (tmp_path / "X_train.npy").exists()
    assert (tmp_path / "y_train.npy").exists()
    assert (tmp_path / "X_val.npy").exists()
    assert (tmp_path / "y_val.npy").exists()
    assert (tmp_path / "X_test.npy").exists()
    assert (tmp_path / "y_test.npy").exists()
    x_train = np.load(tmp_path / "X_train.npy")
    assert x_train.shape[1] == 60 and x_train.shape[2] == 16
    assert x_train.dtype == np.float32
    assert (tmp_path / "scaler.pkl").exists()
    assert not np.isnan(x_train).any()

    # === Step 2: 推論エンジン実行（CLI 境界 2）===
    stub = NamedPipeStub()
    engine = InferenceEngine(
        model=minimal_patchtst_model,
        scaler_path=tmp_path / "scaler.pkl",
        data_source=stub,
    )
    result = engine.predict()

    # Assert 2: Step 2 完了確認（ジャーニー完結）
    assert result["signal"] in {"UP", "DOWN", "NEUTRAL"}
    assert set(result["probabilities"].keys()) == {"UP", "DOWN", "NEUTRAL"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4
    # E2Eは初回ウォームアップ（PyTorchのJITコンパイル等）を含むため500ms以内を確認。
    # AC-017（100回平均50ms未満）の厳密な検証はintegrationテストで実施。
    assert result["inference_ms"] < 500.0


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def synthetic_ohlcv_df():
    """
    ATR(14)・BB(20)・MA60等の計算に十分な行数（300本）の合成OHLCVデータ。
    シード42で再現性を保証。
    """
    rng = np.random.default_rng(seed=42)
    n = 300
    close = 150.0 + rng.normal(0, 0.5, n).cumsum()
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 00:00", periods=n, freq="5min"),
            "open": close + rng.uniform(-0.1, 0.1, n),
            "high": close + rng.uniform(0.0, 0.3, n),
            "low": close - rng.uniform(0.0, 0.3, n),
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
        }
    )


@pytest.fixture
def synthetic_ohlcv_csv(synthetic_ohlcv_df, tmp_path):
    """synthetic_ohlcv_df をCSVとして保存しパスを返す。"""
    csv_path = tmp_path / "USDJPY_M5_e2e.csv"
    synthetic_ohlcv_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def minimal_patchtst_model():
    """
    PatchTST の最小構成（実PyTorchモデル）。CPU推論用。
    パラメータ: seq_len=60, n_features=16, d_model=64, n_heads=2, n_layers=1
    @real-dependency: pytorch
    """
    from models.patchtst import PatchTST

    model = PatchTST(
        seq_len=60, n_features=16, d_model=64, n_heads=2, n_layers=1, n_classes=3
    )
    model.eval()
    return model
