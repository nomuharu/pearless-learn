# Task 0-1: uv プロジェクト環境構築と .gitignore 設定

## タスク概要

uvベースのPythonプロジェクト環境を構築する。`pyproject.toml` を作成し依存関係を定義、`.gitignore` と `.env` テンプレートを整備してセキュリティ要件を満たす。

## 対象ファイル

- `pyproject.toml` (新規)
- `.gitignore` (新規)
- `.env.example` (新規)
- `uv.lock` (uv sync により自動生成)

## 調査対象

- `/home/nomu/claude_code/pearless/docs/design/fx-prediction-design.md` (§ Applicable Standards、§ Constraints)
- `/home/nomu/claude_code/pearless/docs/plans/work_plan.md` (§ Task 0-1、§ Phase 0 完了基準)

## 実装手順

### Step 1: pyproject.toml 作成

以下の内容で `pyproject.toml` を作成する:

```toml
[project]
name = "pearless"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2.0",
    "pandas>=2.2.0",
    "pandas-ta>=0.3.14b0",
    "numpy>=1.26.0",
    "scikit-learn>=1.4.0",
    "kaggle>=1.6.0",
    "python-dotenv>=1.0.0",
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", marker = "platform_machine != 'GPU'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.ruff]
line-length = 88

[tool.mypy]
python_version = "3.11"
disallow_any_generics = true
warn_return_any = true
```

- 依存パッケージ: torch (CPU版)、pandas、pandas-ta、numpy、scikit-learn、kaggle、python-dotenv
- ruff・mypy の設定を含める（QA-3 で使用）

### Step 2: .gitignore 作成

以下を除外対象に含める:
- `data/` (OHLCV CSV・npy ファイル)
- `logs/` (学習ログ CSV)
- `.env` (Kaggle API トークン)
- `*.pkl` (scaler.pkl)
- `*.pt` (best_model.pt)
- `__pycache__/`, `*.pyc`
- `.venv/`
- `uv.lock` の扱いを確認（チーム開発なら含める、個人なら除外可）

### Step 3: .env.example 作成

```
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

- `.env` は `.gitignore` で除外済みであることを確認

### Step 4: uv sync 実行確認

```bash
uv sync
uv export --format requirements-txt
```

- `uv sync` が正常終了することを確認
- `uv export` が requirements.txt 形式で出力されることを確認

## 品質保証メカニズム

| メカニズム | 確認内容 |
|---|---|
| uv sync 完全再現 | `uv sync` が正常終了し、依存関係がすべてインストールされること |

## 動作確認方法

```bash
# 依存関係インストール確認
uv sync

# requirements.txt エクスポート確認（Kaggle 向け）
uv export --format requirements-txt > /tmp/requirements.txt
cat /tmp/requirements.txt

# .env が .gitignore に含まれることを確認
grep -e "\.env$" .gitignore

# Python バージョン確認
uv run python --version
```

**成功基準**:
- `uv sync` がエラーなく完了する (AC-024)
- `uv export --format requirements-txt` が実行できる (AC-025)
- `.env` が `.gitignore` で除外されている

**検証レベル**: L3（ビルド成功検証）

## 完了条件

- [ ] Implementation: `uv sync` が正常終了し Python 3.11 環境が構築される (AC-024)
- [ ] Quality: `uv export --format requirements-txt` が実行できる (AC-025)
- [ ] Integration: `.env` が `.gitignore` で除外されており、Kaggle API トークンがコードにハードコードされていない
