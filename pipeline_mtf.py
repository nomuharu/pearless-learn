"""マルチタイムフレーム（MTF）データパイプライン。

pipeline.py の 16特徴量に加え、M15・H1 の上位足コンテキスト特徴量（8列）を付加した
24列の npy を生成する。lstm_dir_v2 （方向予測 v2）専用。

生成ファイル:
    X_train_mtf.npy / X_val_mtf.npy / X_test_mtf.npy  shape=(N, 60, 24)
    y_train.npy / y_val.npy / y_test.npy               M5 と共通（再生成）
    scaler_mtf.pkl                                       MTF 版スケーラー

上位足特徴量の align 方針:
    各 M5 バーに対して「その時刻以前の最新 M15/H1 確定バー」の値を left join で付与
    （前方補完なし: join_asof direction='backward'）。
    M5 バーが M15/H1 の未確定バー（同一バー内）を参照しないことでルックアヘッドを防ぐ。

使用方法:
    uv run python pipeline_mtf.py --csv-path data/processed/USDJPY_M5.csv \\
                                  --output-dir data/npy_mtf
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import StandardScaler

from models.configs import ALL_FEATURES, M5_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLASS_UP: int = 0
CLASS_DOWN: int = 1
CLASS_NEUTRAL: int = 2

# M15 / H1 特徴量名（ALL_FEATURES の末尾 8 列と一致する順序）
MTF_FEATURES: tuple[str, ...] = ALL_FEATURES[len(M5_FEATURES):]


def _compute_m5_features(df_m5: pd.DataFrame) -> pd.DataFrame:
    """M5 OHLCV から 16 特徴量を計算する（pipeline.feature_engineering と同等）。"""
    close = df_m5["close"]
    high = df_m5["high"]
    low = df_m5["low"]
    volume = df_m5["volume"]
    time_index = df_m5.index.hour * 12 + df_m5.index.minute // 5

    ma60 = close.rolling(60).mean()
    ma60_deviation = (close - ma60) / ma60
    ceiling_distance = (close.rolling(60).max() - close) / close
    ma20 = close.rolling(20).mean()
    ma20_deviation = (close - ma20) / ma20
    ma10 = close.rolling(10).mean()
    ma10_deviation = (close - ma10) / ma10
    prev_ratio = close.pct_change()
    hlo_ratio = (high - low) / close
    diff_hlo_and_average = hlo_ratio - hlo_ratio.rolling(14).mean()
    cci = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20).cci()
    rsi = ta.momentum.RSIIndicator(close=close, window=9).rsi()
    swing_ratio = (high - df_m5["open"]).abs() / close
    vwap = ta.volume.VolumeWeightedAveragePrice(
        high=high, low=low, close=close, volume=volume, window=14
    ).volume_weighted_average_price()
    vwap_deviation = (close - vwap) / vwap
    bb_pband = ta.volatility.BollingerBands(close=close, window=20).bollinger_pband()
    macd_hist = ta.trend.MACD(close=close).macd_diff()
    atr_ratio = (
        ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()
        / close
    )
    time_sin = np.sin(2 * np.pi * time_index / 288)
    time_cos = np.cos(2 * np.pi * time_index / 288)

    return pd.DataFrame(
        {
            "ma60_deviation": ma60_deviation,
            "ceiling_distance": ceiling_distance,
            "ma20_deviation": ma20_deviation,
            "ma10_deviation": ma10_deviation,
            "prev_ratio": prev_ratio,
            "hlo_ratio": hlo_ratio,
            "diff_hlo_and_average": diff_hlo_and_average,
            "cci": cci,
            "rsi": rsi,
            "swing_ratio": swing_ratio,
            "vwap_deviation": vwap_deviation,
            "bb_pband": bb_pband,
            "macd_hist": macd_hist,
            "atr_ratio": atr_ratio,
            "time_sin": time_sin,
            "time_cos": time_cos,
        },
        index=df_m5.index,
    )


def _compute_upper_tf_features(
    df_upper: pd.DataFrame, rsi_window: int, label_prefix: str
) -> pd.DataFrame:
    """上位足 OHLCV から MTF コンテキスト特徴量を計算する。

    Args:
        df_upper: datetime インデックス付き OHLCV DataFrame（M15 または H1）。
        rsi_window: RSI のウィンドウ幅（M15=9, H1=14）。
        label_prefix: 列名プレフィックス（"m15" または "h1"）。

    Returns:
        MTF 特徴量 DataFrame（インデックスは上位足の時刻）。
    """
    close = df_upper["close"]
    high = df_upper["high"]
    low = df_upper["low"]

    ma20 = close.rolling(20).mean()
    ma20_deviation = (close - ma20) / ma20
    rsi = ta.momentum.RSIIndicator(close=close, window=rsi_window).rsi()
    hlo_ratio = (high - low) / close
    atr_ratio = (
        ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()
        / close
    )
    bb_pband = ta.volatility.BollingerBands(close=close, window=20).bollinger_pband()

    cols = {
        f"{label_prefix}_ma20_deviation": ma20_deviation,
        f"{label_prefix}_rsi": rsi,
        f"{label_prefix}_hlo_ratio": hlo_ratio,
    }
    if label_prefix == "m15":
        cols[f"{label_prefix}_atr_ratio"] = atr_ratio
    else:
        cols[f"{label_prefix}_bb_pband"] = bb_pband

    return pd.DataFrame(cols, index=df_upper.index)


def _align_upper_tf(df_m5: pd.DataFrame, df_upper_feat: pd.DataFrame) -> pd.DataFrame:
    """上位足特徴量を M5 バーに align する（ルックアヘッドなし）。

    M5 バーの時刻に対して「その時刻よりも前の最新確定バー」の値を付与する。
    merge_asof の direction='backward' を使用する。

    Args:
        df_m5: M5 の datetime インデックス付き DataFrame（インデックスがキー）。
        df_upper_feat: 上位足特徴量 DataFrame（インデックスがバー時刻）。

    Returns:
        M5 インデックスに align された上位足特徴量 DataFrame。
    """
    m5_times = df_m5.index.to_series().reset_index(drop=True)
    upper_reset = df_upper_feat.reset_index().rename(columns={"datetime": "time"})
    m5_df = pd.DataFrame({"time": m5_times})

    merged = pd.merge_asof(
        m5_df.sort_values("time"),
        upper_reset.sort_values("time"),
        on="time",
        direction="backward",
    ).set_index("time")
    merged.index.name = None
    merged.index = pd.DatetimeIndex(merged.index)
    return merged.reindex(df_m5.index)


def feature_engineering_mtf(df_m5_raw: pd.DataFrame) -> pd.DataFrame:
    """M5 OHLCV から 24 特徴量（M5 16列 + M15/H1 8列）を計算する。

    Args:
        df_m5_raw: datetime, open, high, low, close, volume を含む M5 DataFrame。

    Returns:
        ALL_FEATURES 列を持つ 24列 DataFrame（先頭 NaN 行ドロップ済み）。
    """
    required = {"open", "high", "low", "close", "volume", "datetime"}
    missing = required - set(df_m5_raw.columns)
    if missing:
        raise ValueError(f"必須カラムが存在しません: {missing}")

    df_m5 = df_m5_raw.copy()
    df_m5["datetime"] = pd.to_datetime(df_m5["datetime"])
    df_m5 = df_m5.set_index("datetime")

    # M15 / H1 へリサンプル
    df_m15 = df_m5.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    df_h1 = df_m5.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    df_m15.index.name = "datetime"
    df_h1.index.name = "datetime"

    logger.info("M15 行数: %d | H1 行数: %d", len(df_m15), len(df_h1))

    # 各タイムフレームの特徴量計算
    feat_m5 = _compute_m5_features(df_m5)
    feat_m15 = _compute_upper_tf_features(df_m15, rsi_window=9, label_prefix="m15")
    feat_h1 = _compute_upper_tf_features(df_h1, rsi_window=14, label_prefix="h1")

    # 上位足を M5 に align
    aligned_m15 = _align_upper_tf(df_m5, feat_m15)
    aligned_h1 = _align_upper_tf(df_m5, feat_h1)

    df_all = pd.concat([feat_m5, aligned_m15, aligned_h1], axis=1)
    df_out = df_all.dropna()

    assert df_out.shape[1] == len(ALL_FEATURES), (
        f"特徴量数不整合: {df_out.shape[1]} != {len(ALL_FEATURES)}"
    )
    assert df_out.isna().sum().sum() == 0, "MTF 特徴量に NaN が残存"

    logger.info(
        "feature_engineering_mtf 完了: shape=%s, ドロップ行数=%d",
        df_out.shape,
        len(df_m5) - len(df_out),
    )
    return df_out


def create_label(
    close: pd.Series,
    horizon: int = 1,
    threshold: float | None = None,
) -> np.ndarray[Any, np.dtype[np.int64]]:
    """3クラスラベルを生成する（pipeline.create_label と同等）。"""
    diff = close.shift(-horizon) - close
    theta = diff.abs().quantile(0.75) if threshold is None else float(threshold)
    labels = np.full(len(close), CLASS_NEUTRAL, dtype=np.int64)
    labels[diff > theta] = CLASS_UP
    labels[diff < -theta] = CLASS_DOWN
    logger.info(
        "ラベル生成: theta=%.6f, UP=%d, DOWN=%d, NEUTRAL=%d",
        theta,
        (labels == CLASS_UP).sum(),
        (labels == CLASS_DOWN).sum(),
        (labels == CLASS_NEUTRAL).sum(),
    )
    return labels


def create_windows(
    features: np.ndarray[Any, np.dtype[Any]],
    labels: np.ndarray[Any, np.dtype[Any]],
    window_size: int = 60,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.int64]]]:
    """スライディングウィンドウでサンプルを生成する。"""
    n_samples = len(features)
    n_features = features.shape[1]
    n_windows = n_samples - window_size
    if n_windows <= 0:
        raise ValueError(f"サンプル数 {n_samples} がウィンドウサイズ {window_size} 以下")

    x_windows = np.empty((n_windows, window_size, n_features), dtype=np.float32)
    y_windows = np.empty(n_windows, dtype=np.int64)
    for i in range(n_windows):
        x_windows[i] = features[i : i + window_size]
        y_windows[i] = labels[i + window_size]
    return x_windows, y_windows


def run_pipeline_mtf(
    csv_path: str,
    output_dir: str,
    window_size: int = 60,
    horizon: int = 1,
    threshold: float | None = None,
    ratios: list[float] | None = None,
) -> None:
    """MTF パイプラインのエンドツーエンド実行。

    Args:
        csv_path: USDJPY_M5.csv のパス。
        output_dir: npy / scaler の出力ディレクトリ。
        window_size: ウィンドウサイズ（デフォルト 60）。
        horizon: 先読み足数（デフォルト 1）。
        threshold: ラベル閾値（None で自動決定）。
        ratios: train/val/test 比率（デフォルト [0.70, 0.15, 0.15]）。
    """
    if ratios is None:
        ratios = [0.70, 0.15, 0.15]

    raw_df = pd.read_csv(
        csv_path,
        header=None,
        names=["datetime", "open", "high", "low", "close", "volume"],
    )
    logger.info("M5 CSV 読み込み: %d 行", len(raw_df))

    df_feat = feature_engineering_mtf(raw_df)

    # ラベル生成用に close を df_feat のインデックスに align
    raw_df_indexed = raw_df.copy()
    raw_df_indexed["datetime"] = pd.to_datetime(raw_df_indexed["datetime"])
    raw_df_indexed = raw_df_indexed.set_index("datetime")
    close_aligned = raw_df_indexed.loc[df_feat.index, "close"]

    labels_full = create_label(close_aligned, horizon=horizon, threshold=threshold)

    valid_n = len(df_feat) - horizon
    features_array = df_feat.to_numpy(dtype=np.float32)[:valid_n]
    labels_valid = labels_full[:valid_n]

    X, y = create_windows(features_array, labels_valid, window_size=window_size)
    logger.info("ウィンドウ生成: X.shape=%s", X.shape)

    n = len(X)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]
    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]
    logger.info("分割: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test))

    # 正規化（train のみで fit）
    n_features = X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    X_train_n = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape).astype(np.float32)
    X_val_n = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape).astype(np.float32)
    X_test_n = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape).astype(np.float32)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(str(out / "X_train_mtf.npy"), X_train_n)
    np.save(str(out / "X_val_mtf.npy"), X_val_n)
    np.save(str(out / "X_test_mtf.npy"), X_test_n)
    np.save(str(out / "y_train.npy"), y_train)
    np.save(str(out / "y_val.npy"), y_val)
    np.save(str(out / "y_test.npy"), y_test)

    scaler_path = out / "scaler_mtf.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    logger.info("MTF パイプライン完了: 出力先=%s", output_dir)
    logger.info("X_train_mtf: %s | X_val_mtf: %s | X_test_mtf: %s",
                X_train_n.shape, X_val_n.shape, X_test_n.shape)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="USDJPY M5 → 24特徴量 MTF パイプライン"
    )
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-dir", default="data/npy_mtf")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline_mtf(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        window_size=args.window_size,
        horizon=args.horizon,
        threshold=args.threshold,
    )
