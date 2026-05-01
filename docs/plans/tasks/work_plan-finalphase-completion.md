# Final Phase 完了チェックリスト

## 対象タスク

- [ ] Task QA-1: 全 AC 達成確認
- [ ] Task QA-2: E2E テスト実装・実行
- [ ] Task QA-3: 静的解析・lint・型チェック・カバレッジ
- [ ] Task QA-4: ドキュメント最終確認

## Final Phase 完了基準確認

### E2E テスト全 pass

```bash
uv run pytest tests/e2e/test_pipeline_to_inference.e2e.py -v
```

- [ ] `test_full_pipeline_to_stub_inference_end_to_end` pass

### 全テスト全 pass

```bash
uv run pytest tests/ -v
```

- [ ] 全テスト pass（Integration: 15/15、E2E: 1/1）

### カバレッジ確認

```bash
uv run pytest tests/ --cov=. --cov-fail-under=70
```

- [ ] カバレッジ 70% 以上

### 静的解析エラーゼロ

```bash
uv run ruff check .
uv run mypy . --ignore-missing-imports
```

- [ ] lint エラーゼロ
- [ ] 型チェックエラーゼロ

### AC-001〜AC-026 全項目達成確認

- [ ] AC-001〜AC-008 (パイプライン)
- [ ] AC-009〜AC-014 (モデル)
- [ ] AC-015〜AC-016 (評価)
- [ ] AC-017〜AC-020 (推論エンジン)
- [ ] AC-021 (CNN 比較)
- [ ] AC-022〜AC-023 (学習ログ)
- [ ] AC-024〜AC-025 (uv 環境)
- [ ] AC-026 (Kaggle アップロード)

### PRD 成功基準確認

- [ ] 定量基準: UP/DOWN F1 +5pt、Precision@0.8 ≥ 70%、推論 50ms 未満
- [ ] 定性基準: コードコメント・README 整備

## テストスケルトンファイルパス一覧（確認用）

- `/home/nomu/claude_code/pearless/tests/integration/test_pipeline.int.py`
- `/home/nomu/claude_code/pearless/tests/integration/test_models.int.py`
- `/home/nomu/claude_code/pearless/tests/integration/test_inference_engine.int.py`
- `/home/nomu/claude_code/pearless/tests/e2e/test_pipeline_to_inference.e2e.py`
