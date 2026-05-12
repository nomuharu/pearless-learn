# Task QA-3: 静的解析・lint・型チェック・カバレッジ

## タスク概要

ruff・mypy・pytest-cov を使った静的品質チェックを実施し、エラーゼロ・カバレッジ 70% 以上を達成する。uv sync の環境再現性を最終確認し、Kaggle 向け requirements.txt を生成する。

## 対象ファイル

（全 `.py` ファイルが対象）

## 調査対象

- `/home/nomu/claude_code/pearless/pyproject.toml` (ruff/mypy 設定確認)
- `/home/nomu/claude_code/pearless/docs/plans/work_plan.md` (§ QA-3)

## 実施手順

### Step 1: ruff lint

```bash
uv run ruff check .
```

- エラーゼロを確認
- 自動修正可能な場合: `uv run ruff check . --fix`

### Step 2: mypy 型チェック

```bash
uv run mypy . --ignore-missing-imports
```

- エラーゼロを確認
- `any` 型使用禁止の確認（CLAUDE.md 要件）

### Step 3: テスト全実行 + カバレッジ測定

```bash
uv run pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=70
```

- 全テスト pass
- カバレッジ 70% 以上

### Step 4: 環境再現性確認（AC-024/025）

```bash
# 依存関係の完全再現確認
uv sync

# Kaggle 向け requirements.txt 生成確認（AC-025）
uv export --format requirements-txt > /tmp/requirements_kaggle.txt
echo "requirements.txt 生成: $(wc -l < /tmp/requirements_kaggle.txt) 行"
```

### Step 5: セキュリティ確認

```bash
# Kaggle API トークンのハードコード確認
grep -r "KAGGLE_KEY\|KAGGLE_USERNAME" --include="*.py" . | grep -v "os.environ\|dotenv\|example"
# 出力なし → OK

# wandb 非存在確認
grep -r "wandb" --include="*.py" --include="*.ipynb" .
# 出力なし → OK
```

## 品質保証メカニズム

| メカニズム | 確認内容 | 確認コマンド |
|---|---|---|
| uv sync 完全再現 | 環境再現性 | `uv sync` |
| ruff エラーゼロ | コードスタイル | `ruff check .` |
| mypy エラーゼロ | 型整合性 | `mypy .` |
| カバレッジ 70% 以上 | テスト網羅率 | `pytest --cov --cov-fail-under=70` |

## 動作確認方法

```bash
# 全品質チェックを一括実行
uv run ruff check . && \
uv run mypy . --ignore-missing-imports && \
uv run pytest tests/ --cov=. --cov-fail-under=70 && \
uv sync && \
uv export --format requirements-txt > /dev/null && \
echo "全品質チェック完了"
```

**成功基準**:
- `ruff check .` — エラーゼロ
- `mypy .` — エラーゼロ（`any` 型使用禁止確認）
- `pytest --cov --cov-fail-under=70` — 全 pass + カバレッジ 70% 以上
- `uv sync` — 正常終了 (AC-024)
- `uv export` — requirements.txt 生成成功 (AC-025)

**検証レベル**: L2（テスト動作検証）+ L3（ビルド成功検証）

## 完了条件

- [x] Implementation: 静的解析エラーが全てゼロ
- [x] Quality: カバレッジ 70% 以上（83.71%）、全テスト pass（61 passed, 1 skipped）
- [x] Integration: `uv sync` による環境再現性確認 (AC-024)、`uv export` で Kaggle 向け requirements.txt 生成 (AC-025、677行)
