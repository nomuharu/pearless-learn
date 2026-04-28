# evaluate.py のユニットテスト
# Design Doc: fx-prediction-design.md § 評価メトリクス定義
# AC-015: 全メトリクスが CSV ファイルに出力される
# AC-016: --threshold オプションが高信頼度的中率の計算閾値として使用される
# AC-021: 全モデルの比較 CSV に CNN 列が欠損なく存在する

import math

import numpy as np
import pytest


# ============================================================
# compute_metrics: 基本動作の検証
# ============================================================
def test_compute_metrics_returns_required_keys():
    """
    compute_metrics がメトリクス dict に必須キーを含んで返すこと

    Arrange: 完全正解の予測を生成（y_true == y_pred）
    Act: compute_metrics を呼び出す
    Assert: accuracy, f1_up, f1_down, precision_up, precision_down,
            auc_roc, n_high_conf が含まれる
    """
    from evaluate import compute_metrics

    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, 3, size=n)
    y_pred = y_true.copy()
    # 3 クラス softmax 確率（y_pred に対応するクラスの確率を高くする）
    y_prob = np.zeros((n, 3), dtype=np.float32)
    for i, label in enumerate(y_pred):
        y_prob[i, label] = 0.9
        remaining = 0.1 / 2
        for cls in range(3):
            if cls != label:
                y_prob[i, cls] = remaining

    result = compute_metrics(y_true, y_pred, y_prob, threshold=0.8)

    assert "accuracy" in result
    assert "f1_up" in result
    assert "f1_down" in result
    assert "precision_up" in result
    assert "precision_down" in result
    assert "auc_roc" in result
    assert "n_high_conf" in result


def test_compute_metrics_perfect_prediction_accuracy():
    """
    完全正解の予測で accuracy が 1.0 になること（AC-015）

    Arrange: y_true == y_pred となるデータを生成
    Act: compute_metrics を呼び出す
    Assert: accuracy == 1.0
    """
    from evaluate import compute_metrics

    n = 90
    y_true = np.array([0] * 30 + [1] * 30 + [2] * 30, dtype=np.int64)
    y_pred = y_true.copy()
    y_prob = np.zeros((n, 3), dtype=np.float32)
    for i, label in enumerate(y_pred):
        y_prob[i, label] = 0.95
        for cls in range(3):
            if cls != label:
                y_prob[i, cls] = 0.025

    result = compute_metrics(y_true, y_pred, y_prob, threshold=0.8)

    assert result["accuracy"] == pytest.approx(1.0)


# ============================================================
# compute_metrics: 高信頼度的中率の検証（AC-016）
# ============================================================
def test_compute_metrics_high_conf_key_uses_threshold():
    """
    threshold=0.8 指定時に precision_at_0.8 キーが存在すること（AC-016）

    Arrange: threshold=0.8 でデータを生成
    Act: compute_metrics(threshold=0.8) を呼び出す
    Assert: 'precision_at_0.8' キーが存在する
    """
    from evaluate import compute_metrics

    y_true = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    y_prob = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
        ],
        dtype=np.float32,
    )

    result = compute_metrics(y_true, y_pred, y_prob, threshold=0.8)

    assert "precision_at_0.8" in result


def test_compute_metrics_high_conf_key_uses_custom_threshold():
    """
    threshold=0.9 指定時に precision_at_0.9 キーが存在すること（AC-016）

    Arrange: threshold=0.9 でデータを生成
    Act: compute_metrics(threshold=0.9) を呼び出す
    Assert: 'precision_at_0.9' キーが存在する
    """
    from evaluate import compute_metrics

    y_true = np.array([0, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 2], dtype=np.int64)
    y_prob = np.array(
        [
            [0.95, 0.03, 0.02],
            [0.03, 0.95, 0.02],
            [0.02, 0.03, 0.95],
        ],
        dtype=np.float32,
    )

    result = compute_metrics(y_true, y_pred, y_prob, threshold=0.9)

    assert "precision_at_0.9" in result
    assert "precision_at_0.8" not in result


def test_compute_metrics_high_conf_filters_correctly():
    """
    高信頼度サンプル（max_prob >= threshold）のみで正解率を計算すること（AC-016）

    Arrange:
        - サンプル 0: prob=[0.9, ...] → 高信頼度, 正解
        - サンプル 1: prob=[0.6, ...] → 低信頼度（0.8未満）
        - サンプル 2: prob=[0.85, ...] → 高信頼度, 不正解
    Act: compute_metrics(threshold=0.8) を呼び出す
    Assert: precision_at_0.8 == 0.5 (高信頼度2サンプル中1正解)
    """
    from evaluate import compute_metrics

    y_true = np.array([0, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 0], dtype=np.int64)  # サンプル2は不正解
    y_prob = np.array(
        [
            [0.90, 0.05, 0.05],  # 高信頼度、正解
            [0.20, 0.60, 0.20],  # 低信頼度（0.8未満）
            [0.85, 0.08, 0.07],  # 高信頼度、不正解（y_true=2, y_pred=0）
        ],
        dtype=np.float32,
    )

    result = compute_metrics(y_true, y_pred, y_prob, threshold=0.8)

    assert result["precision_at_0.8"] == pytest.approx(0.5)
    assert result["n_high_conf"] == 2


def test_compute_metrics_no_high_conf_samples_returns_nan():
    """
    高信頼度サンプルが0件の場合に precision_at_threshold が nan を返すこと（AC-016）

    Arrange: 全サンプルの max_prob < threshold となるデータ
    Act: compute_metrics(threshold=0.95) を呼び出す
    Assert: precision_at_0.95 が nan, n_high_conf == 0
    """
    from evaluate import compute_metrics

    y_true = np.array([0, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 2], dtype=np.int64)
    y_prob = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5],
        ],
        dtype=np.float32,
    )

    result = compute_metrics(y_true, y_pred, y_prob, threshold=0.95)

    assert math.isnan(result["precision_at_0.95"])
    assert result["n_high_conf"] == 0


# ============================================================
# compare_models: 複数モデルの比較 DataFrame 生成（AC-021）
# ============================================================
def test_compare_models_returns_dataframe_with_all_models():
    """
    compare_models が全モデル（patchtst, itransformer, cnn）を含む DataFrame を返すこと（AC-021）

    Arrange: 3 モデルのダミー評価結果を用意
    Act: compare_models(results) を呼び出す
    Assert: 返り値の DataFrame に 3 行が含まれる
    """
    from evaluate import compare_models

    dummy_metrics = {
        "accuracy": 0.6,
        "f1_up": 0.5,
        "f1_down": 0.5,
        "precision_up": 0.5,
        "precision_down": 0.5,
        "auc_roc": 0.7,
        "precision_at_0.8": 0.65,
        "n_high_conf": 50,
    }
    results = {
        "patchtst": dummy_metrics,
        "itransformer": dummy_metrics,
        "cnn": dummy_metrics,
    }

    df = compare_models(results)

    assert len(df) == 3
    assert set(df["model"].tolist()) == {"patchtst", "itransformer", "cnn"}


def test_compare_models_cnn_row_has_no_missing_values():
    """
    compare_models の CNN 行に欠損値がないこと（AC-021）

    Arrange: cnn を含む 3 モデルの評価結果
    Act: compare_models(results) を呼び出す
    Assert: cnn 行に NaN がない
    """
    from evaluate import compare_models

    dummy_metrics = {
        "accuracy": 0.55,
        "f1_up": 0.48,
        "f1_down": 0.52,
        "precision_up": 0.50,
        "precision_down": 0.53,
        "auc_roc": 0.68,
        "precision_at_0.8": 0.60,
        "n_high_conf": 40,
    }
    results = {
        "patchtst": dummy_metrics,
        "itransformer": dummy_metrics,
        "cnn": dummy_metrics,
    }

    df = compare_models(results)
    cnn_row = df[df["model"] == "cnn"]

    assert len(cnn_row) == 1
    assert not cnn_row.isnull().any().any()


def test_compare_models_contains_required_columns():
    """
    compare_models の返り値に必須カラムが含まれること（AC-015）

    Arrange: シングルモデルの評価結果
    Act: compare_models(results) を呼び出す
    Assert: model, accuracy, f1_up, f1_down, precision_up, precision_down,
            auc_roc カラムが存在する
    """
    from evaluate import compare_models

    dummy_metrics = {
        "accuracy": 0.6,
        "f1_up": 0.5,
        "f1_down": 0.5,
        "precision_up": 0.5,
        "precision_down": 0.5,
        "auc_roc": 0.7,
        "precision_at_0.8": 0.65,
        "n_high_conf": 50,
    }
    results = {"cnn": dummy_metrics}

    df = compare_models(results)

    required_cols = {
        "model",
        "accuracy",
        "f1_up",
        "f1_down",
        "precision_up",
        "precision_down",
        "auc_roc",
    }
    assert required_cols.issubset(set(df.columns))


# ============================================================
# evaluate_model: モデルを eval モードで推論する（追加コンテキスト仕様）
# ============================================================
def test_evaluate_model_returns_required_keys():
    """
    evaluate_model が accuracy, f1_macro, f1_per_class, confusion_matrix を返すこと

    Arrange: ダミーモデルと numpy テストデータ
    Act: evaluate_model を呼び出す
    Assert: 4 つのキーが dict に含まれる
    """
    import torch

    from evaluate import evaluate_model
    from models.base import BaseModel

    class DummyModel(BaseModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            # UP(0) を常に予測する（確率: [0.8, 0.1, 0.1]）
            probs = torch.zeros(batch, 3)
            probs[:, 0] = 0.8
            probs[:, 1] = 0.1
            probs[:, 2] = 0.1
            return probs

    rng = np.random.default_rng(123)
    n = 60
    X_test = rng.standard_normal((n, 60, 16)).astype(np.float32)
    y_test = np.array([0] * 20 + [1] * 20 + [2] * 20, dtype=np.int64)

    model = DummyModel()
    device = torch.device("cpu")

    result = evaluate_model(model, X_test, y_test, device)

    assert "accuracy" in result
    assert "f1_macro" in result
    assert "f1_per_class" in result
    assert "confusion_matrix" in result


def test_evaluate_model_uses_eval_mode():
    """
    evaluate_model 実行後にモデルが eval モードに設定されること

    Arrange: ダミーモデル（training=True の状態から開始）
    Act: evaluate_model を呼び出す
    Assert: モデルが eval モード（model.training == False）
    """
    import torch

    from evaluate import evaluate_model
    from models.base import BaseModel

    class DummyModel(BaseModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            probs = torch.ones(batch, 3) / 3
            return probs

    rng = np.random.default_rng(0)
    X_test = rng.standard_normal((30, 60, 16)).astype(np.float32)
    y_test = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)

    model = DummyModel()
    model.train()  # 明示的に training モードに設定

    device = torch.device("cpu")
    evaluate_model(model, X_test, y_test, device)

    assert not model.training
