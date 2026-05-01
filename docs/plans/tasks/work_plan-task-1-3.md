# Task 1-3: scripts/upload_dataset.py 実装（AC-026）

## タスク概要

`data/` ディレクトリの内容を Kaggle Dataset としてアップロードする CLI スクリプトを実装する。Kaggle API トークンは環境変数から読み込み、ハードコードを禁止する。

## 対象ファイル

- `scripts/__init__.py` (新規)
- `scripts/upload_dataset.py` (新規)

## 調査対象

- `/home/nomu/claude_code/pearless/docs/plans/work_plan.md` (§ Task 1-3、§ Security Considerations)
- `/home/nomu/claude_code/pearless/.env.example` (Task 0-1 で作成した環境変数テンプレート)
- Kaggle API ドキュメント（`kaggle datasets create` CLI コマンド）

## 実装手順

### Step 1: scripts/ パッケージ作成

```bash
mkdir -p scripts
touch scripts/__init__.py
```

### Step 2: upload_dataset.py 実装

```python
"""Kaggle Dataset アップロードスクリプト。

Usage:
    python scripts/upload_dataset.py --data-dir data/ --dataset-name pearless-usdjpy-m5
    python scripts/upload_dataset.py --data-dir data/ --dataset-name pearless-usdjpy-m5 --dry-run
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def _load_kaggle_credentials() -> tuple[str, str]:
    """環境変数から Kaggle 認証情報を読み込む。"""
    load_dotenv()
    username = os.environ.get("KAGGLE_USERNAME")
    api_key = os.environ.get("KAGGLE_KEY")
    if not username or not api_key:
        raise ValueError(
            "KAGGLE_USERNAME および KAGGLE_KEY 環境変数を設定してください。"
            ".env ファイルを確認してください。"
        )
    return username, api_key


def upload_dataset(
    data_dir: Path,
    dataset_name: str,
    *,
    dry_run: bool = False,
) -> None:
    """data_dir の内容を Kaggle Dataset としてアップロードする。"""
    username, _ = _load_kaggle_credentials()
    # dataset-metadata.json の確認・生成
    # kaggle datasets create / version コマンド実行
    # アップロード完了後に Dataset URL を出力
    dataset_url = f"https://www.kaggle.com/datasets/{username}/{dataset_name}"
    if dry_run:
        print(f"[DRY-RUN] アップロードをスキップしました。URL: {dataset_url}")
        return
    # 実際の kaggle CLI 呼び出し
    result = subprocess.run(
        ["kaggle", "datasets", "create", "-p", str(data_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"アップロード完了: {dataset_url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle Dataset アップロードスクリプト")
    parser.add_argument("--data-dir", type=Path, required=True, help="アップロードするデータディレクトリ")
    parser.add_argument("--dataset-name", type=str, required=True, help="Kaggle Dataset 名")
    parser.add_argument("--dry-run", action="store_true", help="API 呼び出しをスキップして URL のみ表示")
    args = parser.parse_args()
    upload_dataset(args.data_dir, args.dataset_name, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

### Step 3: --dry-run の動作確認

```bash
uv run python scripts/upload_dataset.py --data-dir data/ --dataset-name test --dry-run
```

### Step 4: Kaggle API mock テスト追加

`tests/unit/test_upload_dataset.py` を作成し、`subprocess.run` をモックして URL 出力を確認する。

## 品質保証メカニズム

| メカニズム | 確認内容 |
|---|---|
| uv sync 完全再現 | `kaggle` パッケージが pyproject.toml に含まれ `uv sync` 後に利用可能 |

セキュリティ確認:
- Kaggle API トークンが環境変数から読み込まれ、コードにハードコードされていないことを確認

## 動作確認方法

```bash
# 引数なしで usage を表示
uv run python scripts/upload_dataset.py

# --dry-run でAPI呼び出しをスキップ
uv run python scripts/upload_dataset.py --data-dir data/ --dataset-name pearless-test --dry-run

# 期待出力:
# [DRY-RUN] アップロードをスキップしました。URL: https://www.kaggle.com/datasets/{username}/pearless-test

# コードにトークンがハードコードされていないことを確認
grep -r "KAGGLE_KEY\|KAGGLE_USERNAME" scripts/ --include="*.py"
# 出力: os.environ.get("KAGGLE_KEY") のみ（ハードコードなし）
```

**成功基準**:
- `--dry-run` オプションで API 呼び出しをスキップして URL が表示される (AC-026)
- Kaggle API トークンがコードにハードコードされていない

**検証レベル**: L1（機能動作検証）

## 完了条件

- [x] Implementation: スクリプトが引数なしで usage を表示し、`--dry-run` オプションで API 呼び出しをスキップできる
- [x] Quality: Kaggle API トークンが環境変数から読み込まれる（ハードコードなし）
- [x] Integration: Kaggle API mock テストでアップロード完了 URL が出力される (AC-026)
