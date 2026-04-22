# データパイプライン Integration Test - Design Doc: fx-prediction-design.md
# Generated: 2026-04-21 | Budget Used: 3/3 integration

import pytest
import numpy as np
import pandas as pd


# ============================================================
# AC-001: パイプライン出力ファイル生成
# ROI: 110 (BV:10 × Freq:10 + Legal:0 + Defect:10)
# Behavior: run_pipeline(csv_path, output_dir) 実行 →
#           X_train.npy, y_train.npy, X_val.npy, y_val.npy,
#           X_test.npy, y_test.npy, scaler.pkl が output_dir に生成される
# @category: core-functionality
# @dependency: pipeline.run_pipeline, filesystem
# @real-dependency: filesystem
# @complexity: high
# ============================================================
def test_pipeline_generates_all_required_output_files(synthetic_ohlcv_csv, tmp_path):
    """
    AC-001: パイプライン実行時、7つの出力ファイルが生成されること

    Arrange:
      - 60本以上の合成OHLCVデータをCSVとして準備
    Act:
      - run_pipeline(csv_path, output_dir=tmp_path) を実行
    Assert:
      - X_train.npy, y_train.npy, X_val.npy, y_val.npy,
        X_test.npy, y_test.npy, scaler.pkl が tmp_path に存在すること
    Pass criteria:
      - 7ファイルすべてが存在 → Pass
      - 1ファイルでも欠損 → Fail
    """
    pass


# ============================================================
# AC-002: numpy配列の入力形状 (N, 60, 16)
# ROI: 109 (BV:10 × Freq:10 + Legal:0 + Defect:9)
# Behavior: run_pipeline 実行 → X_*.npy の shape が (N, 60, 16) であること
# @category: core-functionality
# @dependency: pipeline.run_pipeline, numpy, filesystem
# @real-dependency: filesystem, pytorch
# @complexity: medium
# ============================================================
def test_pipeline_output_x_arrays_have_correct_shape(synthetic_ohlcv_csv, tmp_path):
    """
    AC-002: 生成されるX配列の形状が (N, 60, 16) であること

    Arrange:
      - 合成OHLCVデータCSVを準備（十分なサンプル数確保のため300本以上）
    Act:
      - run_pipeline(csv_path, output_dir=tmp_path) を実行
      - X_train.npy, X_val.npy, X_test.npy を np.load でロード
    Assert:
      - x_train.shape == (N_train, 60, 16)
      - x_val.shape   == (N_val,   60, 16)
      - x_test.shape  == (N_test,  60, 16)
      - dtype == float32
    Pass criteria:
      - 全3配列がshape[1]==60, shape[2]==16 を満たす → Pass
      - いずれかが異なる → Fail
    Verification items:
      - shape[1] == 60 (ウィンドウ長)
      - shape[2] == 16 (特徴量数)
      - dtype == np.float32
    """
    pass


# ============================================================
# AC-003: 時系列分割順序・比率（データリーク防止）
# ROI: 110 (BV:10 × Freq:10 + Legal:0 + Defect:10)
# Behavior: split_time_series(X, y) 実行 →
#           train 70% / val 15% / test 15% が時系列順序で分割される
# @category: core-functionality
# @dependency: pipeline.split_time_series
# @complexity: high
# ============================================================
def test_time_series_split_preserves_temporal_order_and_ratios():
    """
    AC-003: 時系列分割がtrain 70% / val 15% / test 15% の順序分割で行われ、
            val/testがtrainより時系列的に後であること

    Arrange:
      - サンプル数 N=1000 の合成 (X, y) を用意
      - X は各行にインデックス番号を埋め込んだ配列（順序追跡用）
    Act:
      - split_time_series(X, y, ratios=[0.70, 0.15, 0.15]) を実行
    Assert:
      - len(X_train) + len(X_val) + len(X_test) == N
      - X_train の最大インデックス < X_val の最小インデックス
      - X_val の最大インデックス  < X_test の最小インデックス
      - 比率誤差が ±1サンプル以内（整数切り捨て許容）
    Pass criteria:
      - 順序不変条件 + 合計サンプル数一致 → Pass
      - シャッフルが発生した場合（インデックス逆転） → Fail
    Verification items:
      - train比率 ≈ 70% (±1サンプル許容)
      - val比率   ≈ 15% (±1サンプル許容)
      - test比率  ≈ 15% (±1サンプル許容)
      - 時系列順序: max(train_idx) < min(val_idx) < min(test_idx)
    """
    pass


# ============================================================
# AC-004: StandardScaler fit=trainデータのみ（データリーク防止）
# ROI: 100 (BV:10 × Freq:9 + Legal:0 + Defect:10)
# Behavior: normalize(X_train, X_val, X_test) 実行 →
#           Scalerはtrainのみでfit、val/testはtransformのみ適用
# @category: core-functionality
# @dependency: pipeline.normalize, sklearn.StandardScaler
# @real-dependency: filesystem (scaler.pkl保存)
# @complexity: high
# ============================================================
def test_normalizer_fits_only_on_train_data_not_val_or_test(tmp_path):
    """
    AC-004: StandardScalerのfitがtrainデータのみに実行され、
            val/testにはtransformのみが適用されること

    Arrange:
      - X_train: (100, 60, 16)、X_val: (20, 60, 16)、X_test: (20, 60, 16) の合成配列
      - X_val, X_test の値域を意図的に X_train の範囲外にシフト
    Act:
      - normalize(X_train, X_val, X_test, scaler_path=tmp_path/"scaler.pkl") を実行
    Assert:
      - 正規化済み X_train の各特徴量の mean ≈ 0.0 (±0.05)
      - 正規化済み X_train の各特徴量の std  ≈ 1.0 (±0.05)
      - 正規化済み X_val の mean が 0 から外れていること（trainでfitされたscalerを使用）
      - scaler.pkl が tmp_path に存在すること
    Pass criteria:
      - X_train の mean/std が基準内 → データリークなし → Pass
      - X_val が standardize後も平均≠0 → trainのみfitの証明
    Verification items:
      - X_train_normalized.reshape(-1, 16).mean(axis=0) ≈ 0 (各特徴量)
      - X_train_normalized.reshape(-1, 16).std(axis=0) ≈ 1 (各特徴量)
      - scaler.pkl が存在する
    """
    pass


# ============================================================
# AC-005: 16特徴量NaNゼロ
# ROI: 100 (BV:9 × Freq:10 + Legal:0 + Defect:10)
# Behavior: feature_engineering(df) 実行 →
#           16特徴量が全てNaNなしで出力される（先頭NaN行はドロップ済み）
# @category: core-functionality
# @dependency: pipeline.feature_engineering, pandas-ta
# @complexity: medium
# ============================================================
def test_feature_engineering_produces_16_features_with_no_nan():
    """
    AC-005: feature_engineering() が16特徴量をNaNなしで返すこと

    Arrange:
      - datetime, open, high, low, close, volume の6カラムを持つ
        合成OHLCVデータ DataFrame（ATR(14)計算のため最低100本以上）
    Act:
      - df_features = feature_engineering(df_ohlcv) を実行
    Assert:
      - df_features.shape[1] == 16
      - df_features.isna().sum().sum() == 0
    Pass criteria:
      - NaN数ゼロ かつ 特徴量数16 → Pass
      - NaN残存 または 特徴量数不一致 → Fail
    Verification items:
      - shape[1] == 16
      - isna().sum().sum() == 0
      - 先頭NaN行がドロップされていること（元のdf行数 > df_features行数）
    """
    pass


# ============================================================
# AC-007: ラベル自動閾値（quantile 0.75）
# ROI: 90 (BV:9 × Freq:9 + Legal:0 + Defect:9)
# Behavior: When threshold が指定されない場合 →
#           θ = diff.abs().quantile(0.75) で自動決定、NEUTRAL ≈ 75%
# @category: core-functionality
# @dependency: pipeline.create_label
# @complexity: medium
# ============================================================
def test_label_generation_uses_quantile_075_threshold_when_not_specified():
    """
    AC-007: threshold未指定時、閾値がdiff.abs().quantile(0.75)で自動決定され
            NEUTRAL(2)が約75%、UP(0)+DOWN(1)が約25%となること

    Arrange:
      - 500行の合成OHLCVデータ（十分なクラス分布確認のため）
    Act:
      - labels = create_label(df, horizon=1, threshold=None) を実行
    Assert:
      - neutral_ratio = (labels == 2).mean() ≈ 0.75 (±0.05)
      - up_down_ratio = (labels != 2).mean() ≈ 0.25 (±0.05)
      - labels.dtype == np.int64
      - set(labels) ⊆ {0, 1, 2}
    Pass criteria:
      - NEUTRAL比率が0.70〜0.80の範囲内 → Pass
    Verification items:
      - クラス番号: UP=0, DOWN=1, NEUTRAL=2 のみ
      - NEUTRAL比率 ≈ 75% (±5%)
      - dtype == int64
    """
    pass


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def synthetic_ohlcv_df():
    """
    ATR(14)・BB(20)・MA60などの計算に十分な行数（300本）の合成OHLCVデータ。
    ランダムシード固定で再現性を保証する。
    """
    rng = np.random.default_rng(seed=42)
    n = 300
    close = 150.0 + rng.normal(0, 0.5, n).cumsum()
    data = {
        "datetime": pd.date_range("2024-01-01 00:00", periods=n, freq="5min"),
        "open": close + rng.uniform(-0.1, 0.1, n),
        "high": close + rng.uniform(0.0, 0.3, n),
        "low": close - rng.uniform(0.0, 0.3, n),
        "close": close,
        "volume": rng.integers(100, 1000, n).astype(float),
    }
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_ohlcv_csv(synthetic_ohlcv_df, tmp_path):
    """synthetic_ohlcv_df をCSVとしてtmp_pathに保存し、パスを返す。"""
    csv_path = tmp_path / "USDJPY_M5_test.csv"
    synthetic_ohlcv_df.to_csv(csv_path, index=False)
    return csv_path
