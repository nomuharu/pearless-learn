# models/configs.py の ModelConfig / MODEL_CONFIGS ユニットテスト
# 方針: 問題（ラベル・分割）は全モデル共通、特徴量とハイパラはモデル別

import numpy as np
import pytest
import torch

from models.base import BaseModel
from models.cnn import CNNModel
from models.configs import (
    ALL_FEATURES,
    MODEL_CONFIGS,
    ModelConfig,
    TrainConfig,
    get_config,
)


# ============================================================
# ALL_FEATURES の整合性
# ============================================================
def test_all_features_matches_pipeline_feature_names():
    """
    pipeline.FEATURE_NAMES が ALL_FEATURES と同一順序であること（単一情報源）

    Arrange: 両モジュールをインポート
    Assert: 列名と順序が完全一致する
    """
    from pipeline import FEATURE_NAMES

    assert FEATURE_NAMES == list(ALL_FEATURES)


def test_all_features_matches_base_model_n_features():
    """
    ALL_FEATURES の要素数が BaseModel.N_FEATURES と一致すること

    Assert: len(ALL_FEATURES) == 16
    """
    assert len(ALL_FEATURES) == BaseModel.N_FEATURES


# ============================================================
# MODEL_CONFIGS レジストリ
# ============================================================
def test_registry_contains_three_baseline_models():
    """
    既存3モデル（patchtst / itransformer / cnn）が登録されていること

    Assert: 各キーが存在し name がキーと一致する
    """
    for name in ["patchtst", "itransformer", "cnn"]:
        assert name in MODEL_CONFIGS
        assert MODEL_CONFIGS[name].name == name


def test_registry_build_model_returns_base_model():
    """
    全登録モデルが build_model() で BaseModel インスタンスを生成でき、
    forward の入出力契約 (B, 60, n_features) → (B, 3) を満たすこと

    Arrange: 各 config からモデルを構築
    Act: ダミー入力で forward
    Assert: 出力 shape (B, 3) かつ softmax 合計 ≈ 1
    """
    for config in MODEL_CONFIGS.values():
        model = config.build_model()
        assert isinstance(model, BaseModel)

        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 60, config.n_features)
            out = model(x)
        assert out.shape == (2, 3)
        assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-4)


def test_get_config_unknown_name_raises():
    """
    未登録のモデル名で get_config がValueError を送出すること

    Assert: ValueError にモデル名と有効値が含まれる
    """
    with pytest.raises(ValueError, match="未知のモデル名"):
        get_config("unknown_model")


# ============================================================
# ModelConfig の特徴量選択
# ============================================================
def test_select_features_full_set_returns_same_array():
    """
    全特徴量使用時は select_features が入力をそのまま返すこと（コピーなし）

    Arrange: フル特徴量のダミー配列
    Act: select_features
    Assert: 同一オブジェクトが返る
    """
    config = MODEL_CONFIGS["cnn"]
    X = np.zeros((4, 60, len(ALL_FEATURES)), dtype=np.float32)

    assert config.select_features(X) is X


def test_select_features_subset_picks_correct_columns():
    """
    特徴量サブセット指定時に正しい列が指定順で抽出されること

    Arrange: 列 c に値 c を入れた配列、rsi と cci のみ使う config
    Act: select_features
    Assert: shape (N, T, 2) で値が [rsi列, cci列] のインデックスと一致
    """
    config = ModelConfig(name="test", model_cls=CNNModel, features=("rsi", "cci"))
    X = np.zeros((2, 60, len(ALL_FEATURES)), dtype=np.float32)
    for c in range(len(ALL_FEATURES)):
        X[:, :, c] = c

    selected = config.select_features(X)

    assert selected.shape == (2, 60, 2)
    assert (selected[:, :, 0] == ALL_FEATURES.index("rsi")).all()
    assert (selected[:, :, 1] == ALL_FEATURES.index("cci")).all()


def test_select_features_rejects_wrong_width():
    """
    特徴量軸の幅が ALL_FEATURES と異なる配列を渡すと ValueError になること
    （既に列選択済みの配列を二重に渡す事故の防止）

    Assert: ValueError
    """
    config = ModelConfig(name="test", model_cls=CNNModel, features=("rsi",))
    X_narrow = np.zeros((2, 60, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="特徴量軸"):
        config.select_features(X_narrow)


def test_build_model_uses_subset_n_features():
    """
    特徴量サブセット指定時に build_model が n_features=len(features) で構築すること

    Arrange: 3特徴量の config
    Act: build_model → (B, 60, 3) 入力で forward
    Assert: 出力 shape (B, 3)
    """
    config = ModelConfig(
        name="test", model_cls=CNNModel, features=("rsi", "cci", "atr_ratio")
    )
    model = config.build_model()

    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 60, 3))
    assert out.shape == (2, 3)


# ============================================================
# ModelConfig のバリデーション
# ============================================================
def test_unknown_feature_name_raises():
    """
    ALL_FEATURES に存在しない特徴量名を指定すると ValueError になること
    """
    with pytest.raises(ValueError, match="存在しない特徴量"):
        ModelConfig(name="test", model_cls=CNNModel, features=("no_such_feature",))


def test_duplicate_feature_name_raises():
    """
    特徴量名が重複していると ValueError になること
    """
    with pytest.raises(ValueError, match="重複"):
        ModelConfig(name="test", model_cls=CNNModel, features=("rsi", "rsi"))


def test_n_features_in_model_kwargs_raises():
    """
    model_kwargs に n_features を含めると ValueError になること
    （features からの自動算出と矛盾するため）
    """
    with pytest.raises(ValueError, match="n_features"):
        ModelConfig(name="test", model_cls=CNNModel, model_kwargs={"n_features": 8})


def test_train_config_defaults_match_training_loop():
    """
    TrainConfig のデフォルトが models.training.train() のデフォルトと一致すること

    Assert: 各ハイパラの既定値
    """
    t = TrainConfig()
    assert t.n_epochs == 100
    assert t.batch_size == 256
    assert t.lr == 1e-4
    assert t.weight_decay == 1e-4
    assert t.patience == 15
    assert t.scheduler_t0 == 10
