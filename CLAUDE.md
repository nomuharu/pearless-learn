# CLAUDE.md

## Kaggle CLI 認証

Kaggle認証は`.env`ファイルで管理（`.gitignore`済み）。

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

- `upload_dataset.py`は`load_dotenv()`で自動読み込み済み
- `kaggle`コマンドを直接実行する場合は`.env`を自動読みしないため、以下を使う:

```bash
uv run --env-file .env kaggle datasets list
```
