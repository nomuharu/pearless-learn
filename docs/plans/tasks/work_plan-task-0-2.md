# Task 0-2: DataSourceInterface 定義（inference/interface.py）

## タスク概要

MT4 将来連携と Named Pipe スタブのためのデータ取得インターフェースを抽象クラスとして定義する。依存性注入（DI）による差し替えを可能にし、推論エンジンとデータソースを疎結合にする。

## 対象ファイル

- `inference/__init__.py` (新規)
- `inference/interface.py` (新規)

## 調査対象

- `/home/nomu/claude_code/pearless/docs/design/fx-prediction-design.md` (§ Contract Definitions、§ Integration Points List)
- `/home/nomu/claude_code/pearless/docs/plans/work_plan.md` (§ Task 0-2、§ Design-to-Plan Traceability)

## 実装手順

### Step 1: inference/ パッケージ作成

```bash
mkdir -p inference
touch inference/__init__.py
```

### Step 2: DataSourceInterface 定義（inference/interface.py）

```python
from abc import ABC, abstractmethod
import pandas as pd


class DataSourceInterface(ABC):
    """データ取得インターフェース。MT4 Named Pipe および将来のデータソースへの DI 抽象。

    契約:
        fetch_latest_ohlcv(n_bars) が呼ばれると、直近 n_bars 本の OHLCV DataFrame を返す。
        返り値の列名: ["open", "high", "low", "close", "volume"]
        返り値の shape: (n_bars, 5)
    """

    @abstractmethod
    def fetch_latest_ohlcv(self, n_bars: int = 60) -> pd.DataFrame:
        """直近 n_bars 本の OHLCV データを取得する。

        Args:
            n_bars: 取得するバー数。デフォルト 60（推論ウィンドウサイズと一致）。

        Returns:
            shape (n_bars, 5) の DataFrame。列: open, high, low, close, volume。
        """
        ...
```

- Python `abc.ABC` を継承
- `@abstractmethod` で `fetch_latest_ohlcv` を定義
- 返り値の DataFrame 仕様をドキュメントコメントに明示
- `any` 型使用禁止

## 品質保証メカニズム

このタスクには project-wide の品質メカニズムが適用される:
- `uv sync 完全再現`: pyproject.toml が正しく設定されていることが前提

## 動作確認方法

```python
# サブクラスが fetch_latest_ohlcv を実装しない場合 TypeError が発生することを確認
from inference.interface import DataSourceInterface

class IncompleteSource(DataSourceInterface):
    pass

try:
    source = IncompleteSource()
    assert False, "TypeError が発生するはず"
except TypeError:
    print("OK: 未実装サブクラスで TypeError が発生した")

# 正しいサブクラスが動作することを確認
import pandas as pd
import numpy as np

class MockSource(DataSourceInterface):
    def fetch_latest_ohlcv(self, n_bars: int = 60) -> pd.DataFrame:
        rng = np.random.default_rng(seed=0)
        return pd.DataFrame({
            "open": rng.normal(150, 1, n_bars),
            "high": rng.normal(150.3, 1, n_bars),
            "low": rng.normal(149.7, 1, n_bars),
            "close": rng.normal(150, 1, n_bars),
            "volume": rng.integers(100, 1000, n_bars).astype(float),
        })

source = MockSource()
df = source.fetch_latest_ohlcv(n_bars=60)
assert df.shape == (60, 5), f"Expected (60, 5), got {df.shape}"
print("OK: DataSourceInterface サブクラスが正常動作")
```

**成功基準**:
- `DataSourceInterface` が ABC として正しく定義されている
- サブクラスが `fetch_latest_ohlcv` を実装しないと `TypeError` が発生する

**検証レベル**: L3（ビルド成功検証）+ L2（テスト動作検証）

## 完了条件

- [x] Implementation: `DataSourceInterface` が `abc.ABC` を継承し、`fetch_latest_ohlcv` が `@abstractmethod` で定義されている
- [x] Quality: 未実装サブクラスのインスタンス化で `TypeError` が発生する
- [x] Integration: 正しく実装したサブクラスが `fetch_latest_ohlcv(n_bars=60)` を呼び出して `(60, 5)` DataFrame を返せる
