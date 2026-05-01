# Phase 3 完了チェックリスト

## 対象タスク

- [ ] Task 3-1: evaluate.py 実装（AC-015/016/021）
- [ ] Task 3-2: 全モデル比較評価実行と PRD 成功基準確認

## Phase 完了基準確認

### AC-015/016/021 達成確認

```bash
uv run python evaluate.py --model all --model-path-dir data/ --test-data data/ --output-dir logs/ --threshold 0.8
ls logs/evaluation_results_*.csv
```

- [ ] AC-015: 評価 CSV が生成され、3モデルの比較が可能な状態
- [ ] AC-016: `--threshold` オプションが高信頼度的中率の計算に反映される
- [ ] AC-021: 全モデルの比較 CSV に CNN 列が欠損なく存在する

### PRD 成功基準確認

- [ ] UP/DOWN F1 スコアが CNN ベースラインを +5pt 以上上回っているか確認
- [ ] Precision@0.8 が 70% 以上か確認
- [ ] 評価結果が `logs/` に保存されている

## テストスケルトンファイルパス一覧（確認用）

（Phase 3 のテストスケルトンは存在しない。evaluate.py の動作を直接確認する。）
