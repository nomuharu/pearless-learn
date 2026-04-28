"""evaluate.py - 全モデル評価スクリプト。

Design Doc: fx-prediction-design.md § 評価メトリクス定義
AC-015: テストデータに対して全メトリクスが CSV ファイルに出力される
AC-016: --threshold オプションで高信頼度的中率の計算閾値を指定（デフォルト 0.8）
AC-021: 全モデルの比較 CSV に CNN 列が欠損なく存在する
wandb 禁止: コードに一切含めない

Usage:
    python evaluate.py --model patchtst --model-path data/best_patchtst.pt \\
                       --test-data data/ --output-dir logs/
    python evaluate.py --model all --model-path-dir data/ --test-data data/ \\
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
from models.cnn import CNNModel
from models.itransformer import iTransformer
from models.patchtst import PatchTST


# モデル名 → クラスのマッピング
_MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "patchtst": PatchTST,
    "itransformer": iTransformer,
    "cnn": CNNModel,
}

# クラス番号定義（Design Doc § クラス番号固定）
_CLASS_NAMES = ["UP", "DOWN", "NEUTRAL"]


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


def _load_test_data(
    test_data_dir: Path,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """テストデータを numpy ファイルから読み込む。

    Args:
        test_data_dir: X_test.npy と y_test.npy が格納されたディレクトリ。

    Returns:
        (X_test, y_test) のタプル。

    Raises:
        FileNotFoundError: X_test.npy または y_test.npy が存在しない場合。
    """
    x_path = test_data_dir / "X_test.npy"
    y_path = test_data_dir / "y_test.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"X_test.npy が見つかりません: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"y_test.npy が見つかりません: {y_path}")

    X_test: np.ndarray[Any, Any] = np.load(x_path)
    y_test: np.ndarray[Any, Any] = np.load(y_path)
    return X_test, y_test


def _load_model(
    model_name: str,
    model_path: Path,
    device: torch.device,
) -> BaseModel:
    """モデルを生成してチェックポイントを読み込む。

    Args:
        model_name: "patchtst", "itransformer", "cnn" のいずれか。
        model_path: .pt ファイルのパス。
        device: ロード先デバイス。

    Returns:
        チェックポイントが読み込まれた BaseModel インスタンス。

    Raises:
        ValueError: 未知の model_name が指定された場合。
        FileNotFoundError: モデルファイルが存在しない場合。
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"未知のモデル名: {model_name!r}。有効な値: {list(_MODEL_REGISTRY.keys())}"
        )

    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    model_cls = _MODEL_REGISTRY[model_name]
    model = model_cls()
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
) -> dict[str, float | int]:
    """単一モデルを評価して compute_metrics の結果を返す。

    Args:
        model_name: 評価対象モデル名。
        model_path: モデルチェックポイントのパス。
        X_test: テスト入力 shape (N, 60, 16)。
        y_test: テスト正解ラベル shape (N,)。
        device: 推論デバイス。
        threshold: 高信頼度フィルタ閾値（AC-016）。

    Returns:
        compute_metrics の結果 dict。
    """
    model = _load_model(model_name, model_path, device)
    model.eval()

    x_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)

    with torch.no_grad():
        y_prob_tensor = model(x_tensor)

    y_prob = y_prob_tensor.cpu().numpy()
    y_pred = y_prob.argmax(axis=1).astype(np.int64)

    return compute_metrics(
        y_true=y_test, y_pred=y_pred, y_prob=y_prob, threshold=threshold
    )


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
        choices=["patchtst", "itransformer", "cnn", "all"],
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
            "best_patchtst.pt / best_itransformer.pt / best_cnn.pt を探す。"
        ),
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        required=True,
        help="X_test.npy と y_test.npy が格納されたディレクトリ。",
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

    X_test, y_test = _load_test_data(args.test_data)

    if args.model == "all":
        if args.model_path_dir is None:
            raise ValueError(
                "--model all 指定時は --model-path-dir を指定してください。"
            )
        model_path_dir: Path = args.model_path_dir
        model_names = ["patchtst", "itransformer", "cnn"]
        results: dict[str, dict[str, Any]] = {}
        for name in model_names:
            model_path = model_path_dir / f"best_{name}.pt"
            print(f"評価中: {name} ({model_path})")
            results[name] = _run_single_model_evaluation(
                model_name=name,
                model_path=model_path,
                X_test=X_test,
                y_test=y_test,
                device=device,
                threshold=args.threshold,
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
        )
        _write_csv({args.model: metrics}, args.output_dir, args.model, args.threshold)


if __name__ == "__main__":
    main()
