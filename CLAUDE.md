# CLAUDE.md

## Kaggle CLI 認証

Kaggle認証は`~/.kaggle/kaggle.json`で管理（標準方式）。

```json
{"username": "nomuhosokawa", "key": "...access_token..."}
```

- 通常のAPI Key（`KGAT`で始まらないもの）では書き込み系操作が401になる
- `~/.kaggle/`配下のaccess tokenを使うこと
- `kaggle`コマンドはそのまま実行可能（`--env-file`不要）
