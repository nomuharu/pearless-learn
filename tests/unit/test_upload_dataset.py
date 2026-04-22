"""scripts/upload_dataset.py のユニットテスト。

AC-026: Kaggle Dataset アップロード完了 URL が出力される。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_dry_run_skips_api_call_and_prints_url(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--dry-run 時に subprocess.run を呼ばず URL を出力する。"""
    from scripts.upload_dataset import upload_dataset

    with patch.dict(
        "os.environ",
        {"KAGGLE_USERNAME": "testuser", "KAGGLE_KEY": "testapikey"},
    ):
        upload_dataset(Path("data/"), "pearless-test", dry_run=True)

    captured = capsys.readouterr()
    assert "[DRY-RUN]" in captured.out
    assert "https://www.kaggle.com/datasets/testuser/pearless-test" in captured.out


def test_upload_calls_kaggle_cli_and_prints_url(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """subprocess.run が呼ばれ、アップロード完了 URL を出力する（AC-026）。"""
    from scripts.upload_dataset import upload_dataset

    mock_result = MagicMock()
    mock_result.returncode = 0

    with (
        patch.dict(
            "os.environ",
            {"KAGGLE_USERNAME": "testuser", "KAGGLE_KEY": "testapikey"},
        ),
        patch("subprocess.run", return_value=mock_result) as mock_run,
    ):
        upload_dataset(tmp_path, "pearless-usdjpy-m5", dry_run=False)
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "kaggle" in call_args
        assert "datasets" in call_args
        assert "create" in call_args

    captured = capsys.readouterr()
    assert "https://www.kaggle.com/datasets/testuser/pearless-usdjpy-m5" in captured.out


def test_upload_exits_on_kaggle_cli_error(tmp_path: Path) -> None:
    """kaggle CLI がエラーを返した場合に sys.exit(1) が呼ばれる。"""
    from scripts.upload_dataset import upload_dataset

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "403 Forbidden"

    with (
        patch.dict(
            "os.environ",
            {"KAGGLE_USERNAME": "testuser", "KAGGLE_KEY": "testapikey"},
        ),
        patch("subprocess.run", return_value=mock_result),
        pytest.raises(SystemExit) as exc_info,
    ):
        upload_dataset(tmp_path, "pearless-test", dry_run=False)

    assert exc_info.value.code == 1


def test_missing_credentials_raises_value_error() -> None:
    """KAGGLE_USERNAME / KAGGLE_KEY が未設定の場合に ValueError が発生する。"""
    from scripts.upload_dataset import upload_dataset

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("scripts.upload_dataset.load_dotenv"),
        pytest.raises(ValueError, match="KAGGLE_USERNAME"),
    ):
        upload_dataset(Path("data/"), "pearless-test", dry_run=True)


def test_no_hardcoded_credentials() -> None:
    """スクリプトのソースコードにトークンがハードコードされていない。"""
    script_path = Path(__file__).parents[2] / "scripts" / "upload_dataset.py"
    source = script_path.read_text()
    # 環境変数の key 名参照は許容、実際の値がないことを確認
    assert "your_kaggle_api_key" not in source
    assert "your_kaggle_username" not in source
