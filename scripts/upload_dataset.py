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
    """環境変数から Kaggle 認証情報を読み込む。

    Returns:
        (username, api_key) のタプル。

    Raises:
        ValueError: KAGGLE_USERNAME または KAGGLE_KEY が未設定の場合。
    """
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
    """data_dir の内容を Kaggle Dataset としてアップロードする。

    Args:
        data_dir: アップロードするデータディレクトリのパス。
        dataset_name: Kaggle Dataset のスラッグ名。
        dry_run: True の場合、API 呼び出しをスキップして URL のみ表示する。
    """
    username, _ = _load_kaggle_credentials()
    dataset_url = f"https://www.kaggle.com/datasets/{username}/{dataset_name}"

    if dry_run:
        print(f"[DRY-RUN] アップロードをスキップしました。URL: {dataset_url}")
        return

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
    """CLI エントリーポイント。"""
    parser = argparse.ArgumentParser(
        description="data/ ディレクトリの内容を Kaggle Dataset としてアップロードする"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="アップロードするデータディレクトリ（例: data/）",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Kaggle Dataset 名（例: pearless-usdjpy-m5）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="API 呼び出しをスキップして Dataset URL のみ表示する",
    )
    args = parser.parse_args()
    upload_dataset(args.data_dir, args.dataset_name, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
