# データパイプライン Integration Test - Design Doc: fx-prediction-design.md
# Generated: 2026-04-21 | Budget Used: 3/3 integration

import pytest
import numpy as np
import pandas as pd

from pipeline import (
    feature_engineering,
    create_label,
    split_time_series,
    normalize,
    run_pipeline,
)


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
    output_dir = str(tmp_path / "output")
    run_pipeline(csv_path=str(synthetic_ohlcv_csv), output_dir=output_dir)

    expected_files = [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test.npy",
        "y_test.npy",
        "scaler.pkl",
    ]
    for fname in expected_files:
        fpath = tmp_path / "output" / fname
        assert fpath.exists(), f"出力ファイルが存在しません: {fname}"


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
    output_dir = str(tmp_path / "output")
    run_pipeline(csv_path=str(synthetic_ohlcv_csv), output_dir=output_dir)

    x_train = np.load(str(tmp_path / "output" / "X_train.npy"))
    x_val = np.load(str(tmp_path / "output" / "X_val.npy"))
    x_test = np.load(str(tmp_path / "output" / "X_test.npy"))

    for name, arr in [("X_train", x_train), ("X_val", x_val), ("X_test", x_test)]:
        assert arr.shape[1] == 60, (
            f"{name}.shape[1] が 60 ではありません: {arr.shape[1]}"
        )
        assert arr.shape[2] == 16, (
            f"{name}.shape[2] が 16 ではありません: {arr.shape[2]}"
        )
        assert arr.dtype == np.float32, (
            f"{name}.dtype が float32 ではありません: {arr.dtype}"
        )


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
    n = 1000
    # X の各サンプルにインデックス番号を埋め込む（順序追跡用）
    X = (
        np.arange(n, dtype=np.float32)
        .reshape(n, 1, 1)
        .repeat(16, axis=2)
        .repeat(60, axis=1)
    )
    y = np.zeros(n, dtype=np.int64)

    X_train, y_train, X_val, y_val, X_test, y_test = split_time_series(
        X, y, ratios=[0.70, 0.15, 0.15]
    )

    # 合計サンプル数一致
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == n, f"分割後合計 {total} が N={n} と不一致"

    # 時系列順序確認（インデックス番号で追跡）
    train_max_idx = X_train[:, 0, 0].max()
    val_min_idx = X_val[:, 0, 0].min()
    val_max_idx = X_val[:, 0, 0].max()
    test_min_idx = X_test[:, 0, 0].min()

    assert train_max_idx < val_min_idx, (
        "X_train の末尾が X_val の先頭より後になっています（シャッフル検出）"
    )
    assert val_max_idx < test_min_idx, (
        "X_val の末尾が X_test の先頭より後になっています（シャッフル検出）"
    )

    # 比率確認（±1 サンプル許容）
    assert abs(len(X_train) - int(n * 0.70)) <= 1, f"train 比率が不正: {len(X_train)}"
    assert abs(len(X_val) - int(n * 0.15)) <= 1, f"val 比率が不正: {len(X_val)}"
    assert abs(len(X_test) - int(n * 0.15)) <= 1, f"test 比率が不正: {len(X_test)}"


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
    rng = np.random.default_rng(seed=0)
    X_train = rng.standard_normal((100, 60, 16)).astype(np.float32) * 2.0
    # X_val は train より 100 大きい値域にシフト（意図的に範囲外）
    X_val = rng.standard_normal((20, 60, 16)).astype(np.float32) + 100.0
    X_test = rng.standard_normal((20, 60, 16)).astype(np.float32)

    scaler_path = str(tmp_path / "scaler.pkl")
    X_train_norm, X_val_norm, X_test_norm = normalize(
        X_train, X_val, X_test, scaler_path
    )

    # X_train の mean ≈ 0, std ≈ 1 を確認
    train_mean = X_train_norm.reshape(-1, 16).mean(axis=0)
    train_std = X_train_norm.reshape(-1, 16).std(axis=0)
    assert np.all(np.abs(train_mean) < 0.05), (
        f"X_train mean が 0 から外れています: {train_mean}"
    )
    assert np.all(np.abs(train_std - 1.0) < 0.05), (
        f"X_train std が 1 から外れています: {train_std}"
    )

    # X_val の mean は 0 から大きく外れているはず（train only fit の証明）
    val_mean = X_val_norm.reshape(-1, 16).mean()
    assert val_mean > 1.0, (
        f"X_val mean が 0 に近すぎます（train/val で fit された可能性）: {val_mean:.4f}"
    )

    # scaler.pkl が存在すること
    assert (tmp_path / "scaler.pkl").exists(), "scaler.pkl が存在しません"


# ============================================================
# AC-005: 16特徴量NaNゼロ
# ROI: 100 (BV:9 × Freq:10 + Legal:0 + Defect:10)
# Behavior: feature_engineering(df) 実行 →
#           16特徴量が全てNaNなしで出力される（先頭NaN行はドロップ済み）
# @category: core-functionality
# @dependency: pipeline.feature_engineering, pandas-ta
# @complexity: medium
# ============================================================
def test_feature_engineering_produces_16_features_with_no_nan(synthetic_ohlcv_df):
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
    df_features = feature_engineering(synthetic_ohlcv_df)

    assert df_features.shape[1] == 16, (
        f"特徴量数が 16 ではありません: {df_features.shape[1]}"
    )
    assert df_features.isna().sum().sum() == 0, "NaN が残存しています"
    assert len(df_features) < len(synthetic_ohlcv_df), (
        "先頭 NaN 行がドロップされていません"
    )


# ============================================================
# AC-007: ラベル自動閾値（quantile 0.75）
# ROI: 90 (BV:9 × Freq:9 + Legal:0 + Defect:9)
# Behavior: When threshold が指定されない場合 →
#           θ = diff.abs().quantile(0.75) で自動決定、NEUTRAL ≈ 75%
# @category: core-functionality
# @dependency: pipeline.create_label
# @complexity: medium
# ============================================================
def test_label_generation_uses_quantile_075_threshold_when_not_specified(
    synthetic_ohlcv_df,
):
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
    # create_label には close カラムが必要
    rng = np.random.default_rng(seed=42)
    n = 500
    close_values = 150.0 + rng.normal(0, 0.5, n).cumsum()
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 00:00", periods=n, freq="5min"),
            "open": close_values + rng.uniform(-0.1, 0.1, n),
            "high": close_values + rng.uniform(0.0, 0.3, n),
            "low": close_values - rng.uniform(0.0, 0.3, n),
            "close": close_values,
            "volume": rng.integers(100, 1000, n).astype(float),
        }
    )

    labels = create_label(df, horizon=1, threshold=None)

    neutral_ratio = (labels == 2).mean()
    assert 0.70 <= neutral_ratio <= 0.80, (
        f"NEUTRAL 比率が期待範囲外: {neutral_ratio:.3f} (期待: 0.70〜0.80)"
    )
    assert labels.dtype == np.int64, f"dtype が int64 ではありません: {labels.dtype}"
    assert set(np.unique(labels)) <= {0, 1, 2}, (
        f"不正なクラス番号: {set(np.unique(labels))}"
    )


# ============================================================
# AC-006: 時間帯特徴量 sin/cos エンコーディング
# ROI: 90 (BV:9 × Freq:9 + Legal:0 + Defect:9)
# Behavior: feature_engineering(df) 実行 →
#           time_sin, time_cos カラムが [-1, 1] に収まり、
#           time_index=0 で sin≈0.0, cos≈1.0 となること
# @category: core-functionality
# @dependency: pipeline.feature_engineering
# @complexity: low
# ============================================================
def test_feature_engineering_time_sin_cos_encoding():
    """
    AC-006: 時間帯特徴量の sin/cos エンコーディングが正しい値域と期待値を持つこと

    Arrange:
      - time_index=0（00:00）となる行を先頭に持つ合成OHLCVデータを準備
        （hour=0, minute=0 → time_index = 0 * 12 + 0 // 5 = 0）
    Act:
      - df_features = feature_engineering(df) を実行
    Assert:
      - time_sin カラムが全て [-1, 1] の範囲内
      - time_cos カラムが全て [-1, 1] の範囲内
      - time_index=0 の行で time_sin ≈ 0.0 (sin(0) = 0)
      - time_index=0 の行で time_cos ≈ 1.0 (cos(0) = 1)
    Pass criteria:
      - 全サンプルの値域 [-1, 1] かつ 特定時刻の期待値一致 → Pass
    Verification items:
      - time_sin.min() >= -1.0, time_sin.max() <= 1.0
      - time_cos.min() >= -1.0, time_cos.max() <= 1.0
      - time_index=0 → sin≈0.0, cos≈1.0
    """
    rng = np.random.default_rng(seed=42)
    n = 300
    # 先頭を 00:00 から始めることで time_index=0 の行を確保する
    datetimes = pd.date_range("2024-01-01 00:00", periods=n, freq="5min")
    close = 150.0 + rng.normal(0, 0.5, n).cumsum()
    df = pd.DataFrame(
        {
            "datetime": datetimes,
            "open": close + rng.uniform(-0.1, 0.1, n),
            "high": close + rng.uniform(0.0, 0.3, n),
            "low": close - rng.uniform(0.0, 0.3, n),
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
        }
    )

    df_features = feature_engineering(df)

    # 値域 [-1, 1] の検証
    assert df_features["time_sin"].min() >= -1.0, (
        "time_sin の最小値が -1 を下回っています"
    )
    assert df_features["time_sin"].max() <= 1.0, "time_sin の最大値が 1 を超えています"
    assert df_features["time_cos"].min() >= -1.0, (
        "time_cos の最小値が -1 を下回っています"
    )
    assert df_features["time_cos"].max() <= 1.0, "time_cos の最大値が 1 を超えています"

    # time_index=0（00:00）の行: sin(0)=0.0, cos(0)=1.0 を検証
    # feature_engineering は先頭 NaN 行をドロップするため、
    # datetime カラムを元に 00:00 の行を特定する
    midnight_mask = df_features.index.isin(
        df[df["datetime"].dt.hour * 12 + df["datetime"].dt.minute // 5 == 0].index
    )
    midnight_rows = df_features[midnight_mask]
    assert len(midnight_rows) > 0, "time_index=0 の行が df_features に存在しません"

    expected_sin = 0.0  # sin(2π * 0 / 288) = sin(0) = 0
    expected_cos = 1.0  # cos(2π * 0 / 288) = cos(0) = 1
    actual_sin = midnight_rows["time_sin"].iloc[0]
    actual_cos = midnight_rows["time_cos"].iloc[0]

    assert abs(actual_sin - expected_sin) < 1e-6, (
        f"time_index=0 の time_sin が期待値 {expected_sin} と異なります: {actual_sin}"
    )
    assert abs(actual_cos - expected_cos) < 1e-6, (
        f"time_index=0 の time_cos が期待値 {expected_cos} と異なります: {actual_cos}"
    )


# ============================================================
# SHA-256 冪等性テスト
# Behavior: 同一CSV入力から同一numpy配列が生成される（SHA-256一致）
# @category: core-functionality
# @dependency: pipeline.run_pipeline, hashlib, filesystem
# ============================================================
def test_pipeline_is_idempotent_sha256(synthetic_ohlcv_csv, tmp_path):
    """同一 CSV 入力から同一 numpy 配列が生成される（SHA-256 一致）。"""
    import hashlib

    output_dir_1 = tmp_path / "run1"
    output_dir_2 = tmp_path / "run2"
    output_dir_1.mkdir()
    output_dir_2.mkdir()
    run_pipeline(str(synthetic_ohlcv_csv), str(output_dir_1))
    run_pipeline(str(synthetic_ohlcv_csv), str(output_dir_2))
    for fname in ["X_train.npy", "y_train.npy"]:
        h1 = hashlib.sha256((output_dir_1 / fname).read_bytes()).hexdigest()
        h2 = hashlib.sha256((output_dir_2 / fname).read_bytes()).hexdigest()
        assert h1 == h2, f"{fname} のハッシュが一致しない"


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
