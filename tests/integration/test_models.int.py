# モデルアーキテクチャ Integration Test - Design Doc: fx-prediction-design.md
# Generated: 2026-04-21 | Budget Used: 3/3 integration

import pytest


# ============================================================
# AC-009 + AC-010: PatchTST forward shape + RevIN組み込み
# ROI: 81 + 63 = 統合81 (BV:9 × Freq:8 + Legal:0 + Defect:9)
# Behavior: PatchTST.forward(x) に (B, 60, 16) を入力 →
#           (B, 3) softmax確率 + RevINが組み込まれていること
# @category: core-functionality
# @dependency: models.patchtst.PatchTST, pytorch
# @real-dependency: pytorch
# @complexity: medium
# ============================================================
def test_patchtst_forward_output_shape_and_softmax():
    """
    AC-009: PatchTSTがinput (B,60,16) → output (B,3) softmax確率を返すこと
    AC-010: RevINモジュールがPatchTSTに組み込まれていること

    Arrange:
      - PatchTST(seq_len=60, n_features=16, patch_len=6, stride=6,
                 d_model=128, n_heads=8, n_layers=3, dim_ff=256,
                 dropout=0.0, n_classes=3) をインスタンス化
      - x = torch.randn(4, 60, 16)  # batch_size=4
    Act:
      - model.eval() に設定
      - output = model.forward(x)
    Assert:
      - output.shape == (4, 3)
      - output.sum(dim=1) ≈ 1.0 (softmax確認, 許容誤差1e-4)
      - output.min() >= 0.0 (確率は非負)
      - hasattr(model, 'revin') == True または RevINモジュールが存在すること
    Pass criteria:
      - shape (4,3) + softmax合計≈1.0 + RevIN存在 → Pass
    Verification items:
      - output.shape == (4, 3)
      - output.sum(dim=-1).allclose(torch.ones(4), atol=1e-4)
      - RevINモジュールの存在確認
    """
    pass


def test_patchtst_parameter_count_under_10_million():
    """
    AC-017の前提条件（CPU推論50ms制約）: PatchTSTのパラメータ数が10M以下であること

    Arrange:
      - PatchTST をデフォルトハイパーパラメータでインスタンス化
    Act:
      - total_params = sum(p.numel() for p in model.parameters())
    Assert:
      - total_params <= 10_000_000
    Pass criteria:
      - パラメータ数 ≤ 10M → Pass
    """
    pass


# ============================================================
# AC-011 + AC-012: iTransformer forward shape + 転置操作
# ROI: 81 + 72 = 統合81 (BV:9 × Freq:8 + Legal:0 + Defect:9)
# Behavior: iTransformer.forward(x) に (B, 60, 16) を入力 →
#           内部で (B, 16, 60) に転置され、(B, 3) softmax確率を返す
# @category: core-functionality
# @dependency: models.itransformer.iTransformer, pytorch
# @real-dependency: pytorch
# @complexity: medium
# ============================================================
def test_itransformer_forward_output_shape_and_transpose():
    """
    AC-011: iTransformerがinput (B,60,16) → output (B,3) softmax確率を返すこと
    AC-012: forward内で (B,60,16) → (B,16,60) の転置操作が行われること

    Arrange:
      - iTransformer(seq_len=60, n_features=16, d_model=128, n_heads=8,
                     n_layers=3, dim_ff=256, dropout=0.0, n_classes=3) をインスタンス化
      - x = torch.randn(4, 60, 16)
    Act:
      - model.eval() に設定
      - output = model.forward(x)
    Assert:
      - output.shape == (4, 3)
      - output.sum(dim=1) ≈ 1.0 (許容誤差1e-4)
      - output.min() >= 0.0
    Pass criteria:
      - shape (4,3) + softmax合計≈1.0 → Pass

    転置操作の間接検証:
      - 転置操作の確認は、forward内のhookまたはモデルコードレビューで実施する。
        テストでは「iTransformerがn_features(16)軸でAttentionをかける」ことを
        forward passの出力形状と正常終了で間接的に確認する。

    注記: iTransformerにRevINは適用しない（ADR-0001 Implementation Guidance準拠）。
    """
    pass


def test_itransformer_has_no_revin_module():
    """
    ADR-0001準拠: iTransformerにRevINが含まれていないこと

    Arrange:
      - iTransformer をデフォルトハイパーパラメータでインスタンス化
    Act:
      - モジュール名一覧を取得
    Assert:
      - 'revin' という名前のサブモジュールが存在しないこと
    Pass criteria:
      - RevIN非存在 → Pass（意図的な設計であることの確認）
    """
    pass


# ============================================================
# AC-009/011共通: BaseModel インターフェース契約
# ROI: 72 (BV:8 × Freq:8 + Legal:0 + Defect:8)
# Behavior: PatchTST と iTransformer が同一の BaseModel インターフェースを実装
# @category: integration
# @dependency: models.base.BaseModel, models.patchtst, models.itransformer
# @real-dependency: pytorch
# @complexity: low
# ============================================================
@pytest.mark.parametrize(
    "model_class_fixture", ["patchtst_model", "itransformer_model"]
)
def test_all_models_implement_base_model_interface(model_class_fixture, request):
    """
    BaseModelインターフェース契約: 全モデルが forward(x: Tensor) -> Tensor を実装し、
    input (B,60,16) → output (B,3) の契約を満たすこと

    Arrange:
      - PatchTST / iTransformer をそれぞれインスタンス化
      - x = torch.zeros(2, 60, 16)
    Act:
      - output = model.forward(x)
    Assert:
      - output.shape == (2, 3)
      - isinstance(model, BaseModel) == True
    Pass criteria:
      - 全モデルが同一の shape 契約を満たす → Pass
    """
    pass
