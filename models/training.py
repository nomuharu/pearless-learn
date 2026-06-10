"""models/training.py - 共通学習ループ。

Design Doc: fx-prediction-design.md § training_loop
Fact: CrossEntropyLoss（クラス重み付き）・AdamW・CosineAnnealingWarmRestarts を全モデル共通で使用
AC-022: logs/training_log_{model_name}_{timestamp}.csv に出力
AC-023: 中断再開時に CSV に追記（行数が増加）
wandb 禁止: コードに一切含めない
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.base import BaseModel
from models.configs import ModelConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_class_weights(
    y_train: np.ndarray[tuple[int], np.dtype[np.intp]],
) -> torch.Tensor:
    """クラス重みを y_train から計算する。

    少数クラス（UP/DOWN）に高い重みを付与することで、
    NEUTRAL クラスの過剰予測を抑制する（NEUTRAL クラス不均衡対策）。

    Args:
        y_train: ラベル配列。値は {0, 1, 2}（UP=0, DOWN=1, NEUTRAL=2）。

    Returns:
        shape (n_classes,) のクラス重みテンソル。
    """
    classes = np.unique(y_train)
    n_samples = len(y_train)
    n_classes = len(classes)
    # class_weight = n_samples / (n_classes * count_per_class)
    counts = np.bincount(y_train, minlength=n_classes)
    weights = n_samples / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def _build_log_path(model_name: str, log_dir: str = "logs") -> Path:
    """学習ログの CSV パスを生成する。

    AC-022: logs/training_log_{model_name}_{timestamp}.csv 形式。
    タイムスタンプは訓練開始時刻（分単位）を使用する。

    Args:
        model_name: モデル識別名（例: "patchtst"）。
        log_dir: ログ保存ディレクトリ（デフォルト "logs"）。

    Returns:
        ログファイルの Path オブジェクト。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    return log_dir_path / f"training_log_{model_name}_{timestamp}.csv"


def _append_log_row(
    log_path: Path,
    row: dict[str, float | int | str],
) -> None:
    """CSV ログに 1 行追記する。

    AC-023: 中断再開時に前回 CSV の後ろに追記できるよう、
    ファイルが既存の場合はヘッダーを書かずにデータ行のみ追記する。

    Args:
        log_path: ログファイルの Path。
        row: 書き込む行データ（dict）。
    """
    fieldnames = ["epoch", "train_loss", "val_loss", "val_accuracy", "elapsed_sec"]
    file_exists = log_path.exists()
    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train(
    model: BaseModel,
    X_train: np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
    y_train: np.ndarray[tuple[int], np.dtype[np.intp]],
    X_val: np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
    y_val: np.ndarray[tuple[int], np.dtype[np.intp]],
    *,
    model_name: str,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    checkpoint_dir: str = "/kaggle/working/",
    log_dir: str = "logs",
    scheduler_t0: int = 10,
) -> None:
    """共通学習ループ。

    CrossEntropyLoss（クラス重み付き）・AdamW・CosineAnnealingWarmRestarts を使用。
    Early stopping、CSV ログ、チェックポイント保存を行う。

    Args:
        model: BaseModel を継承したモデル。
        X_train: 訓練入力データ。shape (N_train, 60, 16)。
        y_train: 訓練ラベル。shape (N_train,)。値 {0, 1, 2}。
        X_val: 検証入力データ。shape (N_val, 60, 16)。
        y_val: 検証ラベル。shape (N_val,)。値 {0, 1, 2}。
        model_name: CSV ログおよびチェックポイントのファイル名に使用する識別名。
        n_epochs: 最大エポック数（デフォルト 100）。
        batch_size: バッチサイズ（デフォルト 256）。
        lr: AdamW 学習率（デフォルト 1e-4）。
        weight_decay: AdamW 重み減衰（デフォルト 1e-4）。
        patience: Early stopping の待機エポック数（デフォルト 15）。
        checkpoint_dir: チェックポイント保存先ディレクトリ（デフォルト /kaggle/working/）。
        log_dir: CSV ログ保存先ディレクトリ（デフォルト "logs"）。
        scheduler_t0: CosineAnnealingWarmRestarts の T_0（デフォルト 10）。
    """
    # デバイスにモデルを転送
    model = model.to(DEVICE)

    # クラス重み計算（NEUTRAL クラス不均衡対策）
    class_weights = _compute_class_weights(y_train).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=scheduler_t0
    )

    # DataLoader 構築
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # チェックポイントディレクトリを作成
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_path / "best_model.pt"

    # CSV ログパスを生成（AC-022）
    log_path = _build_log_path(model_name=model_name, log_dir=log_dir)

    best_val_loss = float("inf")
    patience_counter = 0
    epoch_start_time = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.perf_counter()

        # --- 訓練フェーズ ---
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            # model.forward は softmax 済み確率を返すが、
            # CrossEntropyLoss は logits を期待するため、
            # log_softmax に変換して NLLLoss と等価な計算を行う
            probs = model(X_batch)
            # CrossEntropyLoss に softmax 済み確率を渡すと計算が二重になるため、
            # log(probs) を NLLLoss に通す
            loss = nn.functional.nll_loss(
                torch.log(probs + 1e-8), y_batch, weight=class_weights
            )
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_train_batches += 1

        scheduler.step()

        train_loss = train_loss_sum / max(n_train_batches, 1)

        # --- 検証フェーズ ---
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        n_correct = 0
        n_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                probs = model(X_batch)
                loss = nn.functional.nll_loss(
                    torch.log(probs + 1e-8), y_batch, weight=class_weights
                )
                val_loss_sum += loss.item()
                n_val_batches += 1

                preds = probs.argmax(dim=-1)
                n_correct += (preds == y_batch).sum().item()
                n_total += y_batch.size(0)

        val_loss = val_loss_sum / max(n_val_batches, 1)
        val_accuracy = n_correct / max(n_total, 1)
        elapsed_sec = time.perf_counter() - epoch_start

        # CSV ログ追記（AC-022/AC-023）
        _append_log_row(
            log_path=log_path,
            row={
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_accuracy, 6),
                "elapsed_sec": round(elapsed_sec, 3),
            },
        )

        print(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"elapsed={elapsed_sec:.1f}s"
        )

        # Early stopping とチェックポイント保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved: val_loss={best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    total_elapsed = time.perf_counter() - epoch_start_time
    print(
        f"Training completed in {total_elapsed:.1f}s. Best val_loss={best_val_loss:.4f}"
    )
    print(f"Log saved to: {log_path}")
    print(f"Best model saved to: {best_model_path}")


def train_from_config(
    config: ModelConfig,
    X_train: np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
    y_train: np.ndarray[tuple[int], np.dtype[np.intp]],
    X_val: np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
    y_val: np.ndarray[tuple[int], np.dtype[np.intp]],
    *,
    checkpoint_dir: str = "/kaggle/working/",
    log_dir: str = "logs",
    n_epochs: int | None = None,
    patience: int | None = None,
) -> BaseModel:
    """ModelConfig に従い特徴量選択・モデル構築・学習を一括実行する。

    X_train / X_val はフル特徴量（ALL_FEATURES 順、16列）のまま渡すこと。
    config.features に応じた列選択はこの関数内で行うため、
    呼び出し側で列を絞る必要はない（絞った配列を渡すと shape エラーになる）。

    Args:
        config: モデル設定（MODEL_CONFIGS のエントリ）。
        X_train: 訓練入力データ。shape (N_train, 60, len(ALL_FEATURES))。
        y_train: 訓練ラベル。shape (N_train,)。値 {0, 1, 2}。
        X_val: 検証入力データ。shape (N_val, 60, len(ALL_FEATURES))。
        y_val: 検証ラベル。shape (N_val,)。値 {0, 1, 2}。
        checkpoint_dir: チェックポイント保存先ディレクトリ。
        log_dir: CSV ログ保存先ディレクトリ。
        n_epochs: 指定時は config.train.n_epochs を上書き（スモークテスト用）。
        patience: 指定時は config.train.patience を上書き（スモークテスト用）。

    Returns:
        学習済み（最終エポック時点の）モデルインスタンス。
        ベストモデルは checkpoint_dir/best_model.pt に保存される。
    """
    model = config.build_model()
    t = config.train
    train(
        model=model,
        X_train=config.select_features(X_train),
        y_train=y_train,
        X_val=config.select_features(X_val),
        y_val=y_val,
        model_name=config.name,
        n_epochs=n_epochs if n_epochs is not None else t.n_epochs,
        batch_size=t.batch_size,
        lr=t.lr,
        weight_decay=t.weight_decay,
        patience=patience if patience is not None else t.patience,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        scheduler_t0=t.scheduler_t0,
    )
    return model
