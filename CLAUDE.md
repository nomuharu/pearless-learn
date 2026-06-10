# CLAUDE.md

## Kaggle CLI 認証

Kaggle認証は`~/.kaggle/kaggle.json`で管理（標準方式）。

```json
{"username": "nomuhosokawa", "key": "...access_token..."}
```

- 通常のAPI Key（`KGAT`で始まらないもの）では書き込み系操作が401になる
- `~/.kaggle/`配下のaccess tokenを使うこと
- `kaggle`コマンドはそのまま実行可能（`--env-file`不要）

## Kaggle Datasetへのソースコード配布

Kaggleノートブックからのgit cloneはSecretsが使えないため断念。  
代わりに`models/`と`inference/`を専用の軽量Dataset（`pearless-src`）としてアップロードする方式を採用。
`data/npy/`（pearless-usdjpy-m5、2.66GBのnpy群）と混ぜないこと。

```bash
# models/とinference/をdata/src/にコピーしてからバージョン更新
cp -r models data/src/models
cp -r inference data/src/inference
kaggle datasets version -p data/src/ -m "update source code" --dir-mode zip
```

ノートブック内のパス（REPO_ROOTは両方を試すdual-path構成）:
- ブラウザでDataset追加した場合: `/kaggle/input/pearless-src/`
- CLIのdataset_sources経由の場合: `/kaggle/input/datasets/nomuhosokawa/pearless-src/`
