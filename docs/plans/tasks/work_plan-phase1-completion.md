# Phase 1 完了チェックリスト

## 対象タスク

- [ ] Task 1-1: pipeline.py 実装
- [ ] Task 1-2: パイプライン Integration Test 作成・実行（AC-001〜AC-007）
- [ ] Task 1-3: upload_dataset.py 実装（AC-026）

## Phase 完了基準確認

### Integration Test 全 pass 確認

```bash
uv run pytest tests/integration/test_pipeline.int.py -v
```

- [ ] 全 7 テスト pass（6 AC テスト + SHA-256 冪等性テスト）

### AC-001〜AC-007 達成確認

- [ ] AC-001: 7 ファイル（X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy, scaler.pkl）が生成される
- [ ] AC-002: X_*.npy の shape が `(N, 60, 16)`, dtype が float32
- [ ] AC-003: 時系列分割が train 70% / val 15% / test 15% の順序分割
- [ ] AC-004: StandardScaler が train データのみで fit
- [ ] AC-005: feature_engineering() が 16 特徴量を NaN なしで返す
- [ ] AC-007: threshold 未指定時、NEUTRAL 比率 ≈ 75% (±5%)

### SHA-256 冪等性確認

```bash
# 同一 CSV から 2 回実行してハッシュが一致することを確認
uv run python pipeline.py --csv-path data/USDJPY_M5.csv --output-dir /tmp/run1
uv run python pipeline.py --csv-path data/USDJPY_M5.csv --output-dir /tmp/run2
python -c "
import hashlib
for f in ['X_train.npy', 'y_train.npy']:
    h1 = hashlib.sha256(open(f'/tmp/run1/{f}', 'rb').read()).hexdigest()
    h2 = hashlib.sha256(open(f'/tmp/run2/{f}', 'rb').read()).hexdigest()
    print(f'{f}: {\"OK\" if h1 == h2 else \"MISMATCH\"}')"
```

- [ ] 同一 CSV から同一 numpy 配列が生成される

### AC-026: upload_dataset.py --dry-run 確認

```bash
uv run python scripts/upload_dataset.py --data-dir data/ --dataset-name pearless-test --dry-run
```

- [ ] `[DRY-RUN]` メッセージと Dataset URL が出力される

## テストスケルトンファイルパス一覧（確認用）

- `/home/nomu/claude_code/pearless/tests/integration/test_pipeline.int.py`
