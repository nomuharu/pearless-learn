# Phase 2 完了チェックリスト

## 対象タスク

- [ ] Task 2-1: models/base.py（BaseModel 抽象クラス）実装
- [ ] Task 2-2: PatchTST 実装 + Kaggle ノートブック作成（AC-009/010/013/014/022/023）
- [ ] Task 2-3: iTransformer 実装 + Kaggle ノートブック作成（AC-011/012）
- [ ] Task 2-4: CNN ベースライン実装 + Kaggle ノートブック作成（AC-021）
- [ ] Task 2-5: 推論エンジン + Named Pipe スタブ実装（AC-017〜020）
- [ ] Task 2-6: モデル Integration Test 実装・実行（AC-009〜012）
- [ ] Task 2-7: 推論エンジン Integration Test 実装・実行（AC-017〜020）

## Phase 完了基準確認

### モデル Integration Test 全 pass

```bash
uv run pytest tests/integration/test_models.int.py -v
```

- [ ] 全 5〜6 テスト pass

### 推論エンジン Integration Test 全 pass

```bash
uv run pytest tests/integration/test_inference_engine.int.py -v
```

- [ ] 全 4 テスト pass

### AC-009〜AC-020 達成確認

- [ ] AC-009: PatchTST forward shape `(4, 3)` かつ softmax 合計≈1.0
- [ ] AC-010: RevIN モジュールが PatchTST に存在する
- [ ] AC-011: iTransformer forward shape `(4, 3)` かつ softmax 合計≈1.0
- [ ] AC-012: iTransformer で転置操作が行われる（RevIN なし）
- [ ] AC-013/014: PatchTST Kaggle ノートブックが commit mode 実行完了
- [ ] AC-017: 推論時間 100 回平均 50ms 未満
- [ ] AC-018: predict() が signal + probabilities + inference_ms を返す
- [ ] AC-019: NamedPipeStub 経由 100 回連続推論でエラーゼロ
- [ ] AC-020: DataSourceInterface サブクラスの差し替えが可能

### Kaggle ノートブック実行確認

- [ ] `notebooks/train_patchtst.ipynb` が commit mode で完走し `best_model.pt` が生成された
- [ ] `notebooks/train_itransformer.ipynb` が commit mode で完走した
- [ ] `notebooks/train_cnn.ipynb` が commit mode で完走した

### wandb 非存在確認

```bash
grep -r "wandb" --include="*.py" --include="*.ipynb" .
```

- [ ] wandb に関するコードが一切存在しない

## テストスケルトンファイルパス一覧（確認用）

- `/home/nomu/claude_code/pearless/tests/integration/test_models.int.py`
- `/home/nomu/claude_code/pearless/tests/integration/test_inference_engine.int.py`
