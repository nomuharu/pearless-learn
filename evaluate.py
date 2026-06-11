"""evaluate.py - 全モデル評価スクリプト。

Design Doc: fx-prediction-design.md § 評価メトリクス定義
AC-015: テストデータに対して全メトリクスが CSV ファイルに出力される
AC-016: --threshold オプションで高信頼度的中率の計算閾値を指定（デフォルト 0.8）
AC-021: 全モデルの比較 CSV に CNN 列が欠損なく存在する
wandb 禁止: コードに一切含めない

シグナル評価（クラス別閾値）:
    NEUTRAL が 80% を占める 3 クラスでは max softmax 確率への閾値（precision_at_0.8）
    は NEUTRAL 予測のみを通すため、エントリーシグナルの評価には使えない。
    代わりに UP / DOWN 個別の確率に閾値（p_up >= t_up でロング等）をかけ、
    閾値は validation set の precision-coverage スイープから選び（運用点）、
    test set での的中率・シグナル件数を報告する。

Usage:
    python evaluate.py --model patchtst --model-path data/best_patchtst.pt \\
                       --test-data data/ --output-dir logs/
    python evaluate.py --model all --model-path-dir data/ --test-data data/ \\
                       --val-data data/ --target-precision 0.6 \\
                       --output-dir logs/ --threshold 0.8

クラス定義: UP=0, DOWN=1, NEUTRAL=2
"""

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

from models.base import BaseModel
from models.configs import MODEL_CONFIGS, get_config

# クラス番号定義（Design Doc § クラス番号固定）
_CLASS_NAMES = ["UP", "DOWN", "NEUTRAL"]

# シグナル評価のデフォルト閾値グリッド（クラス別確率に対する閾値）
_SWEEP_THRESHOLDS: tuple[float, ...] = tuple(
    round(float(t), 3) for t in np.arange(0.30, 0.901, 0.025)
)
# 推論時のバッチサイズ（テストデータ全件を一括でGPU/CPUに載せない）
_BATCH_SIZE = 4096


def compute_metrics(
    y_true: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any],
    y_prob: np.ndarray[Any, Any],
    threshold: float = 0.8,
) -> dict[str, float | int]:
    """評価メトリクスを計算して辞書で返す。

    Args:
        y_true: 正解ラベル配列 shape (N,)。値は {0, 1, 2}。
        y_pred: 予測ラベル配列 shape (N,)。値は {0, 1, 2}。
        y_prob: softmax 確率配列 shape (N, 3)。
        threshold: 高信頼度フィルタの閾値（デフォルト 0.8）。AC-016 参照。

    Returns:
        accuracy, f1_up, f1_down, precision_up, precision_down,
        auc_roc, precision_at_{threshold}, n_high_conf を含む dict。
    """
    max_prob = y_prob.max(axis=1)
    high_conf_mask = max_prob >= threshold
    n_high_conf = int(high_conf_mask.sum())

    if n_high_conf > 0:
        high_conf_accuracy: float = float(
            (y_pred[high_conf_mask] == y_true[high_conf_mask]).mean()
        )
    else:
        high_conf_accuracy = math.nan

    precision_key = f"precision_at_{threshold}"

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_up": float(
            f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)
        ),
        "f1_down": float(
            f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)
        ),
        "precision_up": float(
            precision_score(
                y_true, y_pred, labels=[0], average="macro", zero_division=0
            )
        ),
        "precision_down": float(
            precision_score(
                y_true, y_pred, labels=[1], average="macro", zero_division=0
            )
        ),
        "auc_roc": float(roc_auc_score(y_true, y_prob, multi_class="ovr")),
        precision_key: high_conf_accuracy,
        "n_high_conf": n_high_conf,
    }


def signal_sweep(
    y_true: np.ndarray[Any, Any],
    y_prob: np.ndarray[Any, Any],
    thresholds: tuple[float, ...] = _SWEEP_THRESHOLDS,
) -> pd.DataFrame:
    """UP / DOWN のクラス別確率に対する閾値スイープを行う。

    各閾値 t について「p_class >= t のサンプルだけにシグナルを出した場合」の
    的中率（precision）とシグナル件数を計算する。NEUTRAL は対象外。

    Args:
        y_true: 正解ラベル配列 shape (N,)。値は {0, 1, 2}。
        y_prob: softmax 確率配列 shape (N, 3)。
        thresholds: スイープする閾値のタプル。

    Returns:
        columns = [class, threshold, n_signals, precision, coverage] の DataFrame。
        coverage は全サンプルに対するシグナル発生率。
    """
    rows = []
    for class_idx, class_name in enumerate(_CLASS_NAMES[:2]):  # UP / DOWN のみ
        p_class = y_prob[:, class_idx]
        for t in thresholds:
            mask = p_class >= t
            n_signals = int(mask.sum())
            precision = (
                float((y_true[mask] == class_idx).mean())
                if n_signals > 0
                else math.nan
            )
            rows.append(
                {
                    "class": class_name,
                    "threshold": t,
                    "n_signals": n_signals,
                    "precision": precision,
                    "coverage": n_signals / max(len(y_true), 1),
                }
            )
    return pd.DataFrame(rows)


def select_operating_points(
    y_true: np.ndarray[Any, Any],
    y_prob: np.ndarray[Any, Any],
    target_precision: float = 0.6,
    min_signals: int = 30,
    thresholds: tuple[float, ...] = _SWEEP_THRESHOLDS,
) -> dict[str, float]:
    """validation set 上で UP / DOWN それぞれの運用点（閾値）を選ぶ。

    選択規則: precision >= target_precision かつ n_signals >= min_signals を
    満たす閾値のうち、シグナル件数が最大のもの（= 最も低い閾値）。
    条件を満たす閾値がない場合は NaN を返す（そのクラスはシグナルを出さない）。

    閾値の絶対値はモデル・クラス比・class weight に依存するため固定せず、
    必ず validation set で選んでから test set に適用すること（過適合防止）。

    Args:
        y_true: validation の正解ラベル shape (N,)。
        y_prob: validation の softmax 確率 shape (N, 3)。
        target_precision: 運用点に要求する最低的中率。
        min_signals: 統計的に意味を持たせる最低シグナル件数。
        thresholds: 探索する閾値のタプル。

    Returns:
        {"t_up": float, "t_down": float}（条件を満たさない場合は NaN）。
    """
    sweep = signal_sweep(y_true, y_prob, thresholds)
    points: dict[str, float] = {}
    for class_name, key in [("UP", "t_up"), ("DOWN", "t_down")]:
        candidates = sweep[
            (sweep["class"] == class_name)
            & (sweep["precision"] >= target_precision)
            & (sweep["n_signals"] >= min_signals)
        ]
        if len(candidates) == 0:
            points[key] = math.nan
        else:
            best_idx = candidates["n_signals"].idxmax()
            points[key] = float(candidates["threshold"].loc[best_idx])
    return points


def signal_metrics(
    y_true: np.ndarray[Any, Any],
    y_prob: np.ndarray[Any, Any],
    t_up: float,
    t_down: float,
) -> dict[str, float | int]:
    """選択済みの運用点（クラス別閾値）でのシグナル評価メトリクスを計算する。

    Args:
        y_true: 正解ラベル shape (N,)。
        y_prob: softmax 確率 shape (N, 3)。
        t_up: UP シグナルの閾値（NaN の場合は UP シグナルなし）。
        t_down: DOWN シグナルの閾値（NaN の場合は DOWN シグナルなし）。

    Returns:
        t_up, t_down, precision_up_signal, n_signals_up,
        precision_down_signal, n_signals_down を含む dict。
    """

    def _one_class(threshold: float, class_idx: int) -> tuple[float, int]:
        if math.isnan(threshold):
            return math.nan, 0
        mask = y_prob[:, class_idx] >= threshold
        n = int(mask.sum())
        precision = float((y_true[mask] == class_idx).mean()) if n > 0 else math.nan
        return precision, n

    precision_up, n_up = _one_class(t_up, 0)
    precision_down, n_down = _one_class(t_down, 1)
    return {
        "t_up": t_up,
        "t_down": t_down,
        "precision_up_signal": precision_up,
        "n_signals_up": n_up,
        "precision_down_signal": precision_down,
        "n_signals_down": n_down,
    }


def evaluate_model(
    model: BaseModel,
    X_test: np.ndarray[Any, Any],
    y_test: np.ndarray[Any, Any],
    device: torch.device,
) -> dict[str, object]:
    """モデルを eval モードで推論し評価メトリクスを返す。

    追加コンテキスト仕様（AC-015/016/021）に準拠:
        - モデルを eval() モードに切り替えて推論する
        - torch.no_grad() を使用する

    Args:
        model: BaseModel の実装クラスインスタンス。
        X_test: テスト入力 ndarray shape (N, 60, 16)。
        y_test: テスト正解ラベル ndarray shape (N,)。
        device: 推論デバイス。

    Returns:
        accuracy, f1_macro, f1_per_class, confusion_matrix を含む dict。
    """
    model.eval()
    model.to(device)

    x_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)

    with torch.no_grad():
        y_prob_tensor = model(x_tensor)

    y_prob = y_prob_tensor.cpu().numpy()
    y_pred = y_prob.argmax(axis=1).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_per_class": f1_score(
            y_test, y_pred, average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def compare_models(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """複数モデルの評価結果を比較する DataFrame を生成する。

    AC-021: 全モデル（patchtst / itransformer / cnn）の比較 CSV に
    CNN 列が欠損なく存在することを保証する。

    Args:
        results: model_name → metrics dict のマッピング。

    Returns:
        model 列を含む比較 DataFrame。
    """
    rows = [{"model": model_name, **metrics} for model_name, metrics in results.items()]
    return pd.DataFrame(rows)


def _load_split(
    data_dir: Path,
    split: str = "test",
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """指定 split のデータを numpy ファイルから読み込む。

    Args:
        data_dir: X_{split}.npy と y_{split}.npy が格納されたディレクトリ。
        split: "test" または "val"。

    Returns:
        (X, y) のタプル。

    Raises:
        FileNotFoundError: npy ファイルが存在しない場合。
    """
    x_path = data_dir / f"X_{split}.npy"
    y_path = data_dir / f"y_{split}.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"X_{split}.npy が見つかりません: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"y_{split}.npy が見つかりません: {y_path}")

    X: np.ndarray[Any, Any] = np.load(x_path)
    y: np.ndarray[Any, Any] = np.load(y_path)
    return X, y


def _predict_proba(
    model: BaseModel,
    X: np.ndarray[Any, Any],
    device: torch.device,
) -> np.ndarray[Any, Any]:
    """バッチ推論で softmax 確率を計算する。

    Args:
        model: eval モード済みの BaseModel インスタンス。
        X: 入力配列 shape (N, T, F)。特徴量選択済みであること。
        device: 推論デバイス。

    Returns:
        softmax 確率配列 shape (N, 3)。
    """
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), _BATCH_SIZE):
            xb = torch.from_numpy(X[i : i + _BATCH_SIZE].astype(np.float32)).to(device)
            probs.append(model(xb).cpu().numpy())
    return np.concatenate(probs)


def _load_model(
    model_name: str,
    model_path: Path,
    device: torch.device,
) -> BaseModel:
    """モデルを生成してチェックポイントを読み込む。

    モデルは MODEL_CONFIGS の設定（特徴量数・アーキテクチャ）で構築される。

    Args:
        model_name: MODEL_CONFIGS に登録されたモデル名。
        model_path: .pt ファイルのパス。
        device: ロード先デバイス。

    Returns:
        チェックポイントが読み込まれた BaseModel インスタンス。

    Raises:
        ValueError: 未知の model_name が指定された場合。
        FileNotFoundError: モデルファイルが存在しない場合。
    """
    config = get_config(model_name)

    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    model = config.build_model()
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    return model


def _run_single_model_evaluation(
    model_name: str,
    model_path: Path,
    X_test: np.ndarray[Any, Any],
    y_test: np.ndarray[Any, Any],
    device: torch.device,
    threshold: float,
    X_val: np.ndarray[Any, Any] | None = None,
    y_val: np.ndarray[Any, Any] | None = None,
    target_precision: float = 0.6,
    min_signals: int = 30,
    output_dir: Path | None = None,
) -> dict[str, float | int]:
    """単一モデルを評価して compute_metrics + シグナル評価の結果を返す。

    X_test / X_val はフル特徴量（ALL_FEATURES 順）のまま渡すこと。
    モデルごとの特徴量選択（config.features）はこの関数内で行う。

    X_val / y_val が与えられた場合は validation set でクラス別閾値の
    運用点を選び、test set でのシグナル的中率・件数を結果に含める。
    output_dir 指定時は test の閾値スイープを CSV に出力する。

    Args:
        model_name: 評価対象モデル名。
        model_path: モデルチェックポイントのパス。
        X_test: テスト入力 shape (N, 60, len(ALL_FEATURES))。
        y_test: テスト正解ラベル shape (N,)。
        device: 推論デバイス。
        threshold: 高信頼度フィルタ閾値（AC-016）。
        X_val: validation 入力（運用点選択用、省略可）。
        y_val: validation 正解ラベル（運用点選択用、省略可）。
        target_precision: 運用点に要求する最低的中率。
        min_signals: 運用点に要求する最低シグナル件数（val 上）。
        output_dir: スイープ CSV の出力先（省略時は出力しない）。

    Returns:
        compute_metrics の結果に signal_metrics を加えた dict。
    """
    config = get_config(model_name)
    model = _load_model(model_name, model_path, device)
    model.eval()

    y_prob = _predict_proba(model, config.select_features(X_test), device)
    y_pred = y_prob.argmax(axis=1).astype(np.int64)

    metrics = compute_metrics(
        y_true=y_test, y_pred=y_pred, y_prob=y_prob, threshold=threshold
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_path = output_dir / f"signal_sweep_{model_name}_test_{timestamp}.csv"
        signal_sweep(y_test, y_prob).to_csv(sweep_path, index=False)
        print(f"閾値スイープを保存しました: {sweep_path}")

    if X_val is not None and y_val is not None:
        y_prob_val = _predict_proba(model, config.select_features(X_val), device)
        points = select_operating_points(
            y_val,
            y_prob_val,
            target_precision=target_precision,
            min_signals=min_signals,
        )
        metrics = {
            **metrics,
            **signal_metrics(y_test, y_prob, points["t_up"], points["t_down"]),
        }

    return metrics


def _write_csv(
    results: dict[str, dict[str, Any]],
    output_dir: Path,
    model_name: str,
    threshold: float,
) -> Path:
    """評価結果を CSV ファイルに書き込む（AC-015）。

    ファイル名形式: evaluation_results_{model_name}_{timestamp}.csv

    Args:
        results: model_name → metrics dict。
        output_dir: 出力先ディレクトリ。
        model_name: "patchtst", "itransformer", "cnn", "all" のいずれか。
        threshold: 高信頼度フィルタ閾値（ファイル名ではなく内容で使用）。

    Returns:
        書き込んだ CSV ファイルのパス。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"evaluation_results_{model_name}_{timestamp}.csv"

    df = compare_models(results)

    df.to_csv(csv_path, index=False)
    print(f"評価結果を保存しました: {csv_path}")
    return csv_path


def _parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="全モデル評価スクリプト（AC-015/016/021）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_CONFIGS.keys(), "all"],
        required=True,
        help="評価するモデル。'all' で全モデルを比較評価する。",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="単一モデルの .pt ファイルパス（--model が 'all' 以外の場合に必須）。",
    )
    parser.add_argument(
        "--model-path-dir",
        type=Path,
        help=(
            "--model all 指定時のモデルファイル格納ディレクトリ。"
            "MODEL_CONFIGS の各モデルについて best_{name}.pt を探す。"
        ),
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        required=True,
        help="X_test.npy と y_test.npy が格納されたディレクトリ。",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        help=(
            "X_val.npy と y_val.npy が格納されたディレクトリ。"
            "指定時は validation set でクラス別閾値の運用点を選び、"
            "test set でのシグナル的中率・件数を結果に含める。"
        ),
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.6,
        help="運用点選択時に要求する最低的中率（デフォルト: 0.6）。",
    )
    parser.add_argument(
        "--min-signals",
        type=int,
        default=30,
        help="運用点選択時に要求する最低シグナル件数（デフォルト: 30）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/"),
        help="評価結果 CSV の出力先ディレクトリ（デフォルト: logs/）。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="高信頼度的中率の計算閾値（デフォルト: 0.8）。AC-016 参照。",
    )
    return parser.parse_args()


def main() -> None:
    """evaluate.py のエントリポイント。"""
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test, y_test = _load_split(args.test_data, "test")
    if args.val_data is not None:
        X_val, y_val = _load_split(args.val_data, "val")
    else:
        X_val, y_val = None, None

    if args.model == "all":
        if args.model_path_dir is None:
            raise ValueError(
                "--model all 指定時は --model-path-dir を指定してください。"
            )
        model_path_dir: Path = args.model_path_dir
        model_names = list(MODEL_CONFIGS.keys())
        results: dict[str, dict[str, Any]] = {}
        for name in model_names:
            model_path = model_path_dir / f"best_{name}.pt"
            if not model_path.exists():
                print(f"スキップ: {name}（チェックポイントなし: {model_path}）")
                continue
            print(f"評価中: {name} ({model_path})")
            results[name] = _run_single_model_evaluation(
                model_name=name,
                model_path=model_path,
                X_test=X_test,
                y_test=y_test,
                device=device,
                threshold=args.threshold,
                X_val=X_val,
                y_val=y_val,
                target_precision=args.target_precision,
                min_signals=args.min_signals,
                output_dir=args.output_dir,
            )
        _write_csv(results, args.output_dir, "all", args.threshold)
    else:
        if args.model_path is None:
            raise ValueError(
                f"--model {args.model} 指定時は --model-path を指定してください。"
            )
        print(f"評価中: {args.model} ({args.model_path})")
        metrics = _run_single_model_evaluation(
            model_name=args.model,
            model_path=args.model_path,
            X_test=X_test,
            y_test=y_test,
            device=device,
            threshold=args.threshold,
            X_val=X_val,
            y_val=y_val,
            target_precision=args.target_precision,
            min_signals=args.min_signals,
            output_dir=args.output_dir,
        )
        _write_csv({args.model: metrics}, args.output_dir, args.model, args.threshold)


if __name__ == "__main__":
    main()
