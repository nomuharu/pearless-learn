"""データパイプライン: 16特徴量計算・ラベル生成・ウィンドウ化・時系列分割・正規化・numpy保存。

Design Doc: fx-prediction-design.md § Data Flow, § Contract Definitions
ADR-0002: taライブラリによる16特徴量API定義

AC-001: X_train.npy / y_train.npy / X_val.npy / y_val.npy / X_test.npy / y_test.npy / scaler.pkl 生成
AC-002: X_*.npy の shape が (N, 60, 16)
AC-003: train 70% / val 15% / test 15% 時系列順序分割
AC-004: StandardScaler は train データのみで fit
AC-005: 16特徴量 NaN ゼロ
AC-006: 時間帯特徴量は sin/cos エンコーディング（周期 288）
AC-007: threshold=None 時は diff.abs().quantile(0.75) で自動決定
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ADR-0002: 16特徴量の列名（順序固定）
# Design Doc: § Data Flow Step 1
FEATURE_NAMES: list[str] = [
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
]

# クラス番号定義（Design Doc § Glossary, AC-007）
CLASS_UP: int = 0
CLASS_DOWN: int = 1
CLASS_NEUTRAL: int = 2


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame から16特徴量を計算する。

    ADR-0002に定義された16特徴量を taライブラリ および pandas/numpy で計算する。
    先頭の NaN 行はドロップして返す。

    Args:
        df: datetime, open, high, low, close, volume を含む DataFrame。

    Returns:
        FEATURE_NAMES 列を持つ 16列の特徴量 DataFrame（先頭 NaN 行ドロップ済み）。

    Raises:
        ValueError: 必須カラムが存在しない場合。

    Asserts:
        - shape[1] == 16
        - isna().sum().sum() == 0
    """
    required_columns = {"open", "high", "low", "close", "volume", "datetime"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"必須カラムが存在しません: {missing}")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    datetime_col = pd.to_datetime(df["datetime"])

    # MA60乖離率: (close - MA60) / MA60
    ma60 = close.rolling(60).mean()
    ma60_deviation = (close - ma60) / ma60

    # 天井度: close.rolling(60).max()（ADR-0002 #2）
    ceiling_degree = close.rolling(60).max()

    # MA20 / MA10
    ma20 = close.rolling(20).mean()
    ma10 = close.rolling(10).mean()

    # 前足比: pct_change()
    prev_ratio = close.pct_change()

    # HLO: high - low
    hlo = high - low

    # diff_HLO_and_Average: HLO - HLO.rolling(14).mean()
    diff_hlo_and_average = hlo - hlo.rolling(14).mean()

    # CCI(20): ADR-0002 #9, ta.trend.CCIIndicator
    cci = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20).cci()

    # RSI(9): ADR-0002 #10, ta.momentum.RSIIndicator
    rsi = ta.momentum.RSIIndicator(close=close, window=9).rsi()

    # 振れ幅: abs(high - open)
    swing = (high - df["open"]).abs()

    # VWAP乖離率: (close - VWAP) / VWAP, ta.volume.VolumeWeightedAveragePrice
    vwap = ta.volume.VolumeWeightedAveragePrice(
        high=high, low=low, close=close, volume=volume, window=14
    ).volume_weighted_average_price()
    vwap_deviation = (close - vwap) / vwap

    # BB%B: ta.volatility.BollingerBands.bollinger_pband()
    bb_indicator = ta.volatility.BollingerBands(close=close, window=20)
    bb_pband = bb_indicator.bollinger_pband()

    # MACDヒストグラム: ta.trend.MACD.macd_diff()
    macd_indicator = ta.trend.MACD(close=close)
    macd_hist = macd_indicator.macd_diff()

    # ATR(14): ta.volatility.AverageTrueRange
    atr = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    # 時間帯 sin/cos エンコーディング（AC-006, 周期288 = 1日の5分足本数）
    time_index = datetime_col.dt.hour * 12 + datetime_col.dt.minute // 5
    time_sin = np.sin(2 * np.pi * time_index / 288)
    time_cos = np.cos(2 * np.pi * time_index / 288)

    df_features = pd.DataFrame(
        {
            "ma60_deviation": ma60_deviation.values,
            "ceiling_degree": ceiling_degree.values,
            "ma20": ma20.values,
            "ma10": ma10.values,
            "prev_ratio": prev_ratio.values,
            "hlo": hlo.values,
            "diff_hlo_and_average": diff_hlo_and_average.values,
            "cci": cci.values,
            "rsi": rsi.values,
            "swing": swing.values,
            "vwap_deviation": vwap_deviation.values,
            "bb_pband": bb_pband.values,
            "macd_hist": macd_hist.values,
            "atr": atr.values,
            "time_sin": time_sin,
            "time_cos": time_cos,
        },
        index=df.index,
    )

    # 先頭 NaN 行をドロップ（ADR-0002 Implementation Guidance）
    df_out = df_features.dropna()

    # ステップ後アサーション（Design Doc § Quality Assurance Mechanisms）
    assert df_out.shape[1] == 16, f"特徴量数が 16 ではありません: {df_out.shape[1]}"
    assert df_out.isna().sum().sum() == 0, "特徴量に NaN が残存しています"

    logger.info(
        "feature_engineering 完了: shape=%s, NaN=%d, ドロップ行数=%d",
        df_out.shape,
        df_out.isna().sum().sum(),
        len(df) - len(df_out),
    )
    return df_out


def create_label(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold: float | None = None,
) -> np.ndarray[Any, np.dtype[np.int64]]:
    """3クラスラベルを生成する。UP=0, DOWN=1, NEUTRAL=2。

    Design Doc: § Data Flow Step 2, AC-007, AC-008
    クラス番号: UP=0, DOWN=1, NEUTRAL=2

    Args:
        df: close カラムを含む DataFrame。feature_engineering() の出力を想定。
        horizon: 先読み足数。デフォルト 1。
        threshold: 閾値 θ。None の場合は diff.abs().quantile(0.75) で自動決定（AC-007）。

    Returns:
        shape (len(df),) の int64 ndarray。値は {0, 1, 2}。
        最後の horizon 行は NaN となるため、呼び出し元でウィンドウ化前に切り捨てること。

    Asserts:
        - set(np.unique(labels)) <= {0, 1, 2}
    """
    diff = df["close"].shift(-horizon) - df["close"]

    # AC-007: threshold=None の場合は自動決定
    if threshold is None:
        theta = diff.abs().quantile(0.75)
    else:
        # AC-008: 外部から指定された閾値を使用
        theta = float(threshold)

    labels = np.full(len(df), CLASS_NEUTRAL, dtype=np.int64)
    labels[diff > theta] = CLASS_UP
    labels[diff < -theta] = CLASS_DOWN

    # ステップ後アサーション
    unique_classes = set(np.unique(labels))
    assert unique_classes <= {0, 1, 2}, (
        f"不正なクラス番号が存在します: {unique_classes}"
    )

    neutral_ratio = (labels == CLASS_NEUTRAL).mean()
    logger.info(
        "create_label 完了: theta=%.6f, UP=%d, DOWN=%d, NEUTRAL=%d (%.1f%%)",
        theta,
        (labels == CLASS_UP).sum(),
        (labels == CLASS_DOWN).sum(),
        (labels == CLASS_NEUTRAL).sum(),
        neutral_ratio * 100,
    )
    return labels


def create_windows(
    features: np.ndarray[Any, np.dtype[Any]],
    labels: np.ndarray[Any, np.dtype[Any]],
    window_size: int = 60,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.int64]]]:
    """スライディングウィンドウで時系列サンプルを生成する。

    Design Doc: § Data Flow Step 3, AC-002

    Args:
        features: shape (N, n_features) の float32 ndarray。
        labels: shape (N,) の int64 ndarray。
        window_size: ウィンドウサイズ。デフォルト 60。

    Returns:
        X: shape (N_win, window_size, n_features), dtype=float32
        y: shape (N_win,), dtype=int64
    """
    n_samples = len(features)
    n_features = features.shape[1]
    n_windows = n_samples - window_size

    if n_windows <= 0:
        raise ValueError(
            f"サンプル数 {n_samples} がウィンドウサイズ {window_size} 以下のため、ウィンドウを生成できません"
        )

    x_windows = np.empty((n_windows, window_size, n_features), dtype=np.float32)
    y_windows = np.empty(n_windows, dtype=np.int64)

    for i in range(n_windows):
        x_windows[i] = features[i : i + window_size]
        y_windows[i] = labels[i + window_size]

    logger.info(
        "create_windows 完了: X.shape=%s, y.shape=%s",
        x_windows.shape,
        y_windows.shape,
    )
    return x_windows, y_windows


def split_time_series(
    X: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
    ratios: list[float] | None = None,
) -> tuple[
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
]:
    """時系列順序を保ったまま train/val/test に分割する。シャッフル禁止。

    Design Doc: § Data Flow Step 4, AC-003

    Args:
        X: shape (N, window_size, n_features) の ndarray。
        y: shape (N,) の ndarray。
        ratios: [train比率, val比率, test比率]。デフォルト [0.70, 0.15, 0.15]。

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test) のタプル。

    Asserts:
        - 全サンプル数の合計が N と一致
    """
    if ratios is None:
        ratios = [0.70, 0.15, 0.15]

    n = len(X)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    x_train = X[:n_train]
    y_train = y[:n_train]
    x_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]
    x_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]

    # ステップ後アサーション（AC-003）
    total = len(x_train) + len(x_val) + len(x_test)
    assert total == n, f"分割後のサンプル合計 {total} が元の N={n} と一致しません"

    logger.info(
        "split_time_series 完了: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(x_train),
        len(x_train) / n * 100,
        len(x_val),
        len(x_val) / n * 100,
        len(x_test),
        len(x_test) / n * 100,
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def normalize(
    X_train: np.ndarray[Any, np.dtype[Any]],
    X_val: np.ndarray[Any, np.dtype[Any]],
    X_test: np.ndarray[Any, np.dtype[Any]],
    scaler_path: str,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float32]],
    np.ndarray[Any, np.dtype[np.float32]],
    np.ndarray[Any, np.dtype[np.float32]],
]:
    """StandardScaler を train のみで fit し、val/test には transform のみ適用する。

    Design Doc: § Data Flow Step 5, AC-004
    scaler.pkl を scaler_path に保存する。

    Args:
        X_train: shape (N_train, window_size, n_features) の ndarray。
        X_val: shape (N_val, window_size, n_features) の ndarray。
        X_test: shape (N_test, window_size, n_features) の ndarray。
        scaler_path: scaler.pkl の保存先パス。

    Returns:
        (X_train_norm, X_val_norm, X_test_norm) の float32 ndarray タプル。
    """
    n_features = X_train.shape[2]

    # 2D に reshape して fit（AC-004: train のみ）
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    # transform を適用して元の shape に戻す
    x_train_norm = (
        scaler.transform(X_train.reshape(-1, n_features))
        .reshape(X_train.shape)
        .astype(np.float32)
    )
    x_val_norm = (
        scaler.transform(X_val.reshape(-1, n_features))
        .reshape(X_val.shape)
        .astype(np.float32)
    )
    x_test_norm = (
        scaler.transform(X_test.reshape(-1, n_features))
        .reshape(X_test.shape)
        .astype(np.float32)
    )

    # scaler.pkl を保存（AC-004）
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    logger.info(
        "normalize 完了: scaler保存先=%s, train mean≈0確認=%.4f",
        scaler_path,
        x_train_norm.reshape(-1, n_features).mean(),
    )
    return x_train_norm, x_val_norm, x_test_norm


def run_pipeline(
    csv_path: str,
    output_dir: str,
    window_size: int = 60,
    horizon: int = 1,
    threshold: float | None = None,
    ratios: list[float] | None = None,
) -> None:
    """end-to-end パイプライン実行。SHA-256 冪等性確認付き。

    Design Doc: § Contract Definitions, AC-001〜AC-008

    Args:
        csv_path: USDJPY_M5.csv のパス。
        output_dir: numpy 配列 / scaler.pkl の出力ディレクトリ。
        window_size: ウィンドウサイズ。デフォルト 60。
        horizon: ラベル生成の先読み足数。デフォルト 1。
        threshold: ラベル生成の閾値。None の場合は自動決定。
        ratios: train/val/test 分割比率。デフォルト [0.70, 0.15, 0.15]。

    Raises:
        FileNotFoundError: csv_path が存在しない場合。
        ValueError: CSV の必須カラムが欠損している場合。
    """
    if ratios is None:
        ratios = [0.70, 0.15, 0.15]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV ファイルが見つかりません: {csv_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # SHA-256 で入力 CSV を記録（冪等性保証: Design Doc § Invariants）
    raw_df = pd.read_csv(csv_path)
    input_hash = hashlib.sha256(raw_df.to_csv().encode()).hexdigest()
    logger.info("入力CSV SHA-256: %s", input_hash)

    logger.info("Step 1: feature_engineering 開始")
    df_features = feature_engineering(raw_df)
    logger.info("Step 1 完了: shape=%s", df_features.shape)

    # create_label には close カラムが必要なため raw_df の対応行を使用
    # feature_engineering はドロップ後の index を保持する
    df_raw_aligned = raw_df.loc[df_features.index].copy()
    df_features_with_close = df_features.copy()
    df_features_with_close["close"] = df_raw_aligned["close"].values

    logger.info("Step 2: create_label 開始")
    labels_full = create_label(
        df_features_with_close, horizon=horizon, threshold=threshold
    )
    logger.info("Step 2 完了: labels.shape=%s", labels_full.shape)

    # 有効なラベル（最後の horizon 行は NaN 予定のため除外）
    features_array = df_features.to_numpy(dtype=np.float32)
    valid_n = len(features_array) - horizon
    features_valid = features_array[:valid_n]
    labels_valid = labels_full[:valid_n]

    logger.info("Step 3: create_windows 開始")
    X, y = create_windows(features_valid, labels_valid, window_size=window_size)
    logger.info("Step 3 完了: X.shape=%s, y.shape=%s", X.shape, y.shape)

    logger.info("Step 4: split_time_series 開始")
    X_train, y_train, X_val, y_val, X_test, y_test = split_time_series(
        X, y, ratios=ratios
    )
    logger.info(
        "Step 4 完了: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test)
    )

    scaler_path = str(output_path / "scaler.pkl")
    logger.info("Step 5: normalize 開始")
    X_train_norm, X_val_norm, X_test_norm = normalize(
        X_train, X_val, X_test, scaler_path
    )
    logger.info("Step 5 完了")

    # numpy 配列を保存（AC-001）
    np.save(str(output_path / "X_train.npy"), X_train_norm)
    np.save(str(output_path / "y_train.npy"), y_train)
    np.save(str(output_path / "X_val.npy"), X_val_norm)
    np.save(str(output_path / "y_val.npy"), y_val)
    np.save(str(output_path / "X_test.npy"), X_test_norm)
    np.save(str(output_path / "y_test.npy"), y_test)

    logger.info(
        "パイプライン完了: 出力ディレクトリ=%s, 入力SHA-256=%s",
        output_dir,
        input_hash,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="USDJPY 5分足 データパイプライン（16特徴量 → numpy 保存）"
    )
    parser.add_argument("--csv-path", required=True, help="入力 CSV ファイルパス")
    parser.add_argument(
        "--output-dir", default="data/", help="出力ディレクトリ（デフォルト: data/）"
    )
    parser.add_argument(
        "--window-size", type=int, default=60, help="ウィンドウサイズ（デフォルト: 60）"
    )
    parser.add_argument(
        "--horizon", type=int, default=1, help="先読み足数（デフォルト: 1）"
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="ラベル閾値（省略時は自動決定）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        window_size=args.window_size,
        horizon=args.horizon,
        threshold=args.threshold,
    )
