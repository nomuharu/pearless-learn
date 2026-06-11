# models/training.py の P4 拡張（focal loss・early stop 指標選択）ユニットテスト

import numpy as np
import pytest
import torch

from models.cnn import CNNModel
from models.configs import ModelConfig, TrainConfig
from models.training import _compute_loss, _f1_per_class, train, train_from_config


# ============================================================
# _compute_loss
# ============================================================
def test_weighted_ce_matches_nll_loss():
    """
    loss_type="weighted_ce" が従来の重み付き NLL と一致すること
    """
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(8, 3), dim=-1)
    y = torch.randint(0, 3, (8,))
    weights = torch.tensor([3.0, 3.0, 0.4])

    expected = torch.nn.functional.nll_loss(
        torch.log(probs + 1e-8), y, weight=weights
    )
    actual = _compute_loss(probs, y, weights, "weighted_ce", focal_gamma=2.0)

    assert torch.allclose(actual, expected)


def test_focal_loss_downweights_easy_examples():
    """
    focal loss が「確信を持って正解している易例」の損失を
    weighted CE より小さくすること（(1-p_t)^γ の効果）
    """
    # 正解クラスに確率 0.95 の易例
    probs = torch.tensor([[0.95, 0.03, 0.02]])
    y = torch.tensor([0])
    weights = torch.ones(3)

    ce = _compute_loss(probs, y, weights, "weighted_ce", focal_gamma=2.0)
    focal = _compute_loss(probs, y, weights, "focal", focal_gamma=2.0)

    assert focal < ce


def test_focal_gamma_zero_equals_weighted_ce():
    """
    γ=0 の focal loss が weighted CE と一致すること（退化ケース）
    """
    torch.manual_seed(1)
    probs = torch.softmax(torch.randn(16, 3), dim=-1)
    y = torch.randint(0, 3, (16,))
    weights = torch.tensor([2.0, 2.0, 0.5])

    ce = _compute_loss(probs, y, weights, "weighted_ce", focal_gamma=2.0)
    focal_g0 = _compute_loss(probs, y, weights, "focal", focal_gamma=0.0)

    assert torch.allclose(ce, focal_g0, atol=1e-6)


def test_unknown_loss_type_raises():
    """
    未知の loss_type で ValueError になること
    """
    probs = torch.softmax(torch.randn(4, 3), dim=-1)
    y = torch.randint(0, 3, (4,))

    with pytest.raises(ValueError, match="loss_type"):
        _compute_loss(probs, y, torch.ones(3), "unknown", focal_gamma=2.0)


# ============================================================
# _f1_per_class
# ============================================================
def test_f1_per_class_known_values():
    """
    F1 = 2TP / (2TP + FP + FN) が既知の値と一致すること

    Arrange: クラス0について TP=2, FP=1, FN=1 → F1 = 4/6
    """
    y_true = torch.tensor([0, 0, 0, 1, 2])
    y_pred = torch.tensor([0, 0, 1, 0, 2])

    assert _f1_per_class(y_true, y_pred, 0) == pytest.approx(4 / 6)


def test_f1_per_class_no_occurrences_returns_zero():
    """
    対象クラスが正解にも予測にも出現しない場合は 0.0 を返すこと
    """
    y_true = torch.tensor([2, 2])
    y_pred = torch.tensor([2, 2])

    assert _f1_per_class(y_true, y_pred, 0) == 0.0


# ============================================================
# train() の early_stop_metric
# ============================================================
def test_train_rejects_unknown_early_stop_metric():
    """
    未知の early_stop_metric で ValueError になること
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 60, 16)).astype(np.float32)
    y = rng.integers(0, 3, 16)

    with pytest.raises(ValueError, match="early_stop_metric"):
        train(
            model=CNNModel(),
            X_train=X,
            y_train=y,
            X_val=X,
            y_val=y,
            model_name="test",
            early_stop_metric="unknown",
        )


def test_train_from_config_passes_extensions(tmp_path):
    """
    train_from_config が focal loss と val_f1_updown 基準で
    エンドツーエンドに動作し、CSV ログに F1 列が含まれること
    """
    config = ModelConfig(
        name="cnn_p4_smoke",
        model_cls=CNNModel,
        train=TrainConfig(
            loss_type="focal",
            focal_gamma=2.0,
            early_stop_metric="val_f1_updown",
        ),
    )
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((64, 60, 16)).astype(np.float32)
    y_tr = rng.integers(0, 3, 64)
    X_va = rng.standard_normal((32, 60, 16)).astype(np.float32)
    y_va = rng.integers(0, 3, 32)

    train_from_config(
        config, X_tr, y_tr, X_va, y_va,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
        n_epochs=2,
        patience=2,
    )

    log_files = list((tmp_path / "logs").glob("training_log_cnn_p4_smoke_*.csv"))
    assert len(log_files) == 1
    header = log_files[0].read_text().splitlines()[0]
    assert "val_f1_up" in header
    assert "val_f1_down" in header
    assert (tmp_path / "ckpt" / "best_model.pt").exists()
