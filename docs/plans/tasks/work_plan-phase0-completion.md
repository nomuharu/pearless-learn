# Phase 0 完了チェックリスト

## 対象タスク

- [ ] Task 0-1: uv プロジェクト環境構築と .gitignore 設定
- [ ] Task 0-2: DataSourceInterface 定義（inference/interface.py）
- [ ] Task 0-3: Integration テストスケルトン Red 実装

## Phase 完了基準確認

### AC-024: uv sync 完全再現

```bash
uv sync
```

- [ ] エラーゼロで完了する
- [ ] Python 3.11 が使用される

### AC-025: uv export 確認

```bash
uv export --format requirements-txt
```

- [ ] requirements.txt 形式で出力される

### DataSourceInterface 定義確認

```python
from inference.interface import DataSourceInterface
import inspect
assert inspect.isabstract(DataSourceInterface)
```

- [ ] `DataSourceInterface` が ABC として正しく定義されている
- [ ] サブクラスが `fetch_latest_ohlcv` を実装しないと `TypeError` が発生する

### セキュリティ確認

```bash
grep -r "KAGGLE_KEY\|KAGGLE_USERNAME" --include="*.py" .
```

- [ ] `.env` が `.gitignore` で除外されている
- [ ] Kaggle API トークンがコードにハードコードされていない

### テストスケルトン収集確認

```bash
uv run pytest tests/ --collect-only
```

- [ ] テスト関数が収集される（エラーなし）

## テストスケルトンファイルパス一覧（Red 状態確認用）

- `/home/nomu/claude_code/pearless/tests/integration/test_pipeline.int.py`
- `/home/nomu/claude_code/pearless/tests/integration/test_inference_engine.int.py`
- `/home/nomu/claude_code/pearless/tests/integration/test_models.int.py`
- `/home/nomu/claude_code/pearless/tests/e2e/test_pipeline_to_inference.e2e.py`
