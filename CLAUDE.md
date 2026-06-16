# CLAUDE.md

## Kaggle CLI 認証

Kaggle認証は`~/.kaggle/kaggle.json`で管理（標準方式）。

```json
{"username": "nomuhosokawa", "key": "...access_token..."}
```

- 通常のAPI Key（`KGAT`で始まらないもの）では書き込み系操作が401になる
- `~/.kaggle/`配下のaccess tokenを使うこと
- `kaggle`コマンドはそのまま実行可能（`--env-file`不要）

## production/ ディレクトリの運用ルール

採用済み戦略は `production/strategies/<strategy-id>/` 配下で管理する。
詳細ルールは `production/README.md` を参照。

- 新戦略採用時: `strategy.md` / `expectations.json` / 重みを置き、`production/README.md` の戦略一覧を更新して commit
- フォワードテスト: `TEMPLATE_forward_test.md` をコピーして `forward_tests/<YYYY-MM_xxx>/session.md` を作成、終了後 `trades.csv` を追加して `summarize_forward_test.py` でサマリ生成
- 月次メンテ（walk-forward系）: fine-tune済みの重みを `checkpoints/` に日付付きで追加し `strategy.md` を更新

## 新実験手順（モデル改善・特徴量追加など）

新しい手法を試すときは `/new-experiment` スキルを使う。

```
/new-experiment [モデル名] [suffix]
```

詳細な手順は `.claude/skills/new-experiment/SKILL.md` を参照。

---

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
