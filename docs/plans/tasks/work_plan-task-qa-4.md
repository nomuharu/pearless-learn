# Task QA-4: ドキュメント最終確認

## タスク概要

コードコメントに設計書との対応関係が明示されているか、README.md に使用手順が記載されているかを確認する（PRD 定性的成功基準）。

## 対象ファイル

- `README.md` (新規または既存)
- 全 `.py` ファイルのコメント確認

## 調査対象

- `/home/nomu/claude_code/pearless/docs/prd.md` (定性的成功基準)
- `/home/nomu/claude_code/pearless/docs/design/fx-prediction-design.md` (§ 各モジュールの設計仕様)

## 実施手順

### Step 1: README.md 作成・確認

以下の内容が記載されていることを確認（なければ作成）:

1. **プロジェクト概要**: USDJPY 5分足方向予測 ML システムの説明
2. **セットアップ手順**:
   - `uv sync` での依存関係インストール
   - `.env` への Kaggle API トークン設定
3. **データパイプライン実行手順**:
   - `python pipeline.py --csv-path data/USDJPY_M5.csv --output-dir data/`
4. **Kaggle 学習手順**:
   - `scripts/upload_dataset.py` の使い方
   - ノートブック実行方法
5. **評価手順**:
   - `python evaluate.py --model all --test-data data/ --output-dir logs/`
6. **推論手順**:
   - `InferenceEngine` の使い方（スタブ経由）

### Step 2: コードコメント確認

各モジュールに設計書との対応が明示されていることを確認:
- `pipeline.py`: `# Design Doc: fx-prediction-design.md §パイプライン設計` 等
- `models/patchtst.py`: `# ADR-0001: PatchTST 選定根拠` 等
- `inference/engine.py`: `# AC-017/018: 推論エンジン契約` 等

### Step 3: PRD 定性的成功基準の確認

- [ ] コードコメントに設計書との対応関係が明示されている
- [ ] 使用手順が README に記載されている
- [ ] wandb が一切含まれていない
- [ ] Kaggle API トークンがコードにハードコードされていない

## 動作確認方法

```bash
# README の存在確認
ls README.md

# wandb 非存在の最終確認
grep -r "wandb" --include="*.py" --include="*.ipynb" --include="*.md" .
# 出力なし → OK

# ハードコードされた認証情報の確認
grep -r "KAGGLE_KEY\s*=" --include="*.py" . | grep -v "os.environ\|getenv"
# 出力なし → OK
```

**成功基準**:
- README.md に使用手順が記載されている
- コードコメントに設計書との対応関係が明示されている

**検証レベル**: L3（ビルド成功検証）

## 完了条件

- [ ] Implementation: README.md に使用手順が記載されている
- [ ] Quality: コードコメントに設計書との対応関係が明示されている
- [ ] Integration: PRD 定性的成功基準の確認が完了している
