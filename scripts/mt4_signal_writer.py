"""MT4連携用シグナル生成スクリプト。

PearlessBreakout.mq4（EA）と対になる Python 側プロセス。

動作:
  1. MT4 の Files ディレクトリにある pearless_bars.csv（EAが5分足確定ごとに
     出力する直近バーのOHLCV）を監視する
  2. 更新を検知したら 16特徴量を計算 → 学習時の scaler で正規化 →
     lstm_focal で推論し p_move = p_up + p_down を得る
  3. pearless_signal.csv に「バー時刻,p_move」を書き込む（EAが読んで発注判断）

Usage:
    uv run python scripts/mt4_signal_writer.py \\
        --mt4-files-dir "/path/to/MT4/MQL4/Files" \\
        --model-path data/best_lstm_focal.pt \\
        --scaler-path data/npy/scaler.pkl

WSL から Windows 側 MT4 と連携する場合:
    MT4 の「ファイル → データフォルダを開く」で出るパスを /mnt/c 経由で指定する。
    例: --mt4-files-dir "/mnt/c/Users/<name>/AppData/Roaming/MetaQuotes/Terminal/<hash>/MQL4/Files"
    9P 越しのファイルI/Oは数十ms の追加遅延があるが、5分足運用では無視できる。
    EA 書き込み中のロック衝突に備え、読み失敗は次のポーリングで自動再試行する。
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.configs import MODEL_CONFIGS
from pipeline import feature_engineering

WINDOW = 60
POLL_SEC = 0.2


def compute_p_move(
    bars: pd.DataFrame,
    model: torch.nn.Module,
    scaler: object,
) -> float:
    """直近バー群から p_move（= p_up + p_down）を計算する。"""
    features = feature_engineering(bars)
    if len(features) < WINDOW:
        raise ValueError(f"特徴量行数不足: {len(features)} < {WINDOW}")
    window = features.values[-WINDOW:].astype(np.float64)
    # 学習時と同一の scaler（train で fit 済み StandardScaler）で正規化
    flat = scaler.transform(window)  # type: ignore[attr-defined]
    x = torch.from_numpy(flat.astype(np.float32)).unsqueeze(0)  # (1, 60, 16)
    with torch.no_grad():
        prob = model(x)[0]
    return float(prob[0] + prob[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="MT4向けシグナル生成ループ")
    parser.add_argument("--mt4-files-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--scaler-path", type=Path, required=True)
    parser.add_argument("--model-name", default="lstm_focal")
    args = parser.parse_args()

    bars_path = args.mt4_files_dir / "pearless_bars.csv"
    signal_path = args.mt4_files_dir / "pearless_signal.csv"

    config = MODEL_CONFIGS[args.model_name]
    model = config.build_model()
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print(f"監視開始: {bars_path}")
    last_mtime = 0.0
    while True:
        try:
            mtime = bars_path.stat().st_mtime
        except FileNotFoundError:
            time.sleep(POLL_SEC)
            continue
        if mtime <= last_mtime:
            time.sleep(POLL_SEC)
            continue

        t0 = time.perf_counter()
        try:
            bars = pd.read_csv(
                bars_path,
                header=None,
                names=["datetime", "open", "high", "low", "close", "volume"],
            )
        except (PermissionError, OSError, pd.errors.ParserError):
            # EA が書き込み中（Windowsファイルロック）や書きかけの場合は
            # mtime を更新せず次のポーリングで再試行する
            time.sleep(POLL_SEC)
            continue
        last_mtime = mtime

        p_move = compute_p_move(bars, model, scaler)
        bar_time = bars["datetime"].iloc[-1]  # 最新確定バーの時刻（EA側と一致させる）
        # EA が読んでいる最中に書きかけ内容を見せないよう、一時ファイル経由の
        # rename で原子的に置き換える（WSL→Windows の 9P 越しでも有効）
        tmp_path = signal_path.with_suffix(".tmp")
        tmp_path.write_text(f"{bar_time},{p_move:.4f}\n")
        tmp_path.replace(signal_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"{bar_time}: p_move={p_move:.4f} ({elapsed_ms:.0f}ms)")


if __name__ == "__main__":
    main()
