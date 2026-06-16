"""MT4連携用シグナル生成スクリプト（mtf-direction-v2 戦略）。

PearlessDirection.mq4（EA）と対になる Python 側プロセス。

動作:
  1. MT4 の Files ディレクトリにある pearless_bars.csv（EAが5分足確定ごとに
     出力する直近バーのOHLCV）を監視する
  2. 更新を検知したら 24特徴量（M5 16列 + M15/H1 8列）を計算 →
     scaler_mtf で正規化 → 2段推論:
       Stage 1: lstm_focal で p_move = p_up + p_down を得る
       Stage 2: p_move >= t_move のときのみ lstm_dir_v2 で方向を推論
  3. pearless_dir_signal.csv に「バー時刻,p_move,direction,p_up」を書き込む
     direction: "BUY" / "SELL" / "SKIP"（Stage1 閾値未満）

Usage:
    uv run python scripts/mt4_signal_writer_dir_v2.py \\
        --mt4-files-dir "/path/to/MT4/MQL4/Files" \\
        --focal-model-path checkpoints/best_lstm_focal.pt \\
        --dir-model-path checkpoints/best_lstm_dir_v2.pt \\
        --scaler-m5-path data/npy/scaler.pkl \\
        --scaler-mtf-path data/npy_mtf/scaler_mtf.pkl

WSL から Windows 側 MT4 と連携する場合:
    MT4 の「ファイル → データフォルダを開く」で出るパスを /mnt/c 経由で指定する。
    例: --mt4-files-dir "/mnt/c/Users/<name>/AppData/Roaming/MetaQuotes/Terminal/<hash>/MQL4/Files"
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

# scripts/ から実行した場合でもプロジェクトルートをパスに追加する
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from models.configs import MODEL_CONFIGS
from pipeline import feature_engineering
from pipeline_mtf import feature_engineering_mtf

WINDOW = 60
POLL_SEC = 0.2

T_MOVE_DEFAULT = 0.88
T_DIR_DEFAULT = 0.60


def _load_model(model_name: str, model_path: Path) -> torch.nn.Module:
    config = MODEL_CONFIGS[model_name]
    model = config.build_model()
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def _load_scaler(scaler_path: Path) -> object:
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def compute_signal(
    bars: pd.DataFrame,
    focal_model: torch.nn.Module,
    dir_model: torch.nn.Module,
    scaler_m5: object,
    scaler_mtf: object,
    t_move: float,
    t_dir: float,
) -> tuple[float, str, float]:
    """2段推論でシグナルを計算する。

    Returns:
        (p_move, direction, p_up)
        direction: "BUY" / "SELL" / "SKIP"
        p_up: Stage2 の上昇確率（SKIP時は 0.0）
    """
    # --- Stage 1: lstm_focal（M5 16特徴量） ---
    feat_m5 = feature_engineering(bars)
    if len(feat_m5) < WINDOW:
        raise ValueError(f"M5 特徴量行数不足: {len(feat_m5)} < {WINDOW}")
    window_m5 = feat_m5.values[-WINDOW:].astype(np.float64)
    flat_m5 = scaler_m5.transform(window_m5)  # type: ignore[attr-defined]
    x_m5 = torch.from_numpy(flat_m5.astype(np.float32)).unsqueeze(0)  # (1, 60, 16)
    with torch.no_grad():
        prob_focal = focal_model(x_m5)[0]
    p_move = float(prob_focal[0] + prob_focal[1])

    if p_move < t_move:
        return p_move, "SKIP", 0.0

    # --- Stage 2: lstm_dir_v2（MTF 24特徴量） ---
    feat_mtf = feature_engineering_mtf(bars)
    if len(feat_mtf) < WINDOW:
        raise ValueError(f"MTF 特徴量行数不足: {len(feat_mtf)} < {WINDOW}")
    window_mtf = feat_mtf.values[-WINDOW:].astype(np.float64)
    flat_mtf = scaler_mtf.transform(window_mtf)  # type: ignore[attr-defined]
    x_mtf = torch.from_numpy(flat_mtf.astype(np.float32)).unsqueeze(0)  # (1, 60, 24)
    with torch.no_grad():
        prob_dir = dir_model(x_mtf)[0]
    # モデル出力: [p_up, p_down, p_neutral]（CLASS_UP=0, CLASS_DOWN=1）
    p_up = float(prob_dir[0])
    p_down = float(prob_dir[1])

    if p_up >= t_dir:
        return p_move, "BUY", p_up
    if p_down >= t_dir:
        return p_move, "SELL", p_up

    return p_move, "SKIP", p_up


def main() -> None:
    parser = argparse.ArgumentParser(description="MT4向け2段シグナル生成ループ（mtf-direction-v2）")
    parser.add_argument("--mt4-files-dir", type=Path, required=True)
    parser.add_argument("--focal-model-path", type=Path, required=True,
                        help="lstm_focal の重みファイルパス")
    parser.add_argument("--dir-model-path", type=Path, required=True,
                        help="lstm_dir_v2 の重みファイルパス")
    parser.add_argument("--scaler-m5-path", type=Path, required=True,
                        help="M5 用 scaler（data/npy/scaler.pkl）")
    parser.add_argument("--scaler-mtf-path", type=Path, required=True,
                        help="MTF 用 scaler（data/npy_mtf/scaler_mtf.pkl）")
    parser.add_argument("--t-move", type=float, default=T_MOVE_DEFAULT,
                        help=f"Stage1 閾値（デフォルト {T_MOVE_DEFAULT}）")
    parser.add_argument("--t-dir", type=float, default=T_DIR_DEFAULT,
                        help=f"Stage2 方向閾値（デフォルト {T_DIR_DEFAULT}）")
    args = parser.parse_args()

    bars_path = args.mt4_files_dir / "pearless_bars.csv"
    signal_path = args.mt4_files_dir / "pearless_dir_signal.csv"

    print("モデル読み込み中...")
    focal_model = _load_model("lstm_focal", args.focal_model_path)
    dir_model = _load_model("lstm_dir_v2", args.dir_model_path)
    scaler_m5 = _load_scaler(args.scaler_m5_path)
    scaler_mtf = _load_scaler(args.scaler_mtf_path)

    print(f"監視開始: {bars_path}")
    print(f"閾値: t_move={args.t_move}, t_dir={args.t_dir}")
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
            time.sleep(POLL_SEC)
            continue
        last_mtime = mtime

        try:
            p_move, direction, p_up = compute_signal(
                bars, focal_model, dir_model,
                scaler_m5, scaler_mtf,
                args.t_move, args.t_dir,
            )
        except ValueError as e:
            print(f"推論スキップ: {e}")
            continue

        bar_time = bars["datetime"].iloc[-1]
        tmp_path = signal_path.with_suffix(".tmp")
        tmp_path.write_text(f"{bar_time},{p_move:.4f},{direction},{p_up:.4f}\n")
        tmp_path.replace(signal_path)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if direction == "SKIP":
            print(f"{bar_time}: p_move={p_move:.4f} ({elapsed_ms:.0f}ms)     skip")
        else:
            print(f"{bar_time}: p_move={p_move:.4f} p_up={p_up:.4f} ({elapsed_ms:.0f}ms) >>> {direction}")


if __name__ == "__main__":
    main()
