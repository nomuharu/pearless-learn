"""Named Pipe スタブ実装。

Design Doc: fx-prediction-design.md § NamedPipeDataSource
AC-018, AC-019, AC-020: DataSourceInterface の MT4 Named Pipe ダミー実装

実際の MT4 Named Pipe 接続はスコープ外。固定シードによる再現性のある
合成 OHLCV データを生成する。
"""

import numpy as np
import pandas as pd

from inference.interface import DataSourceInterface


class NamedPipeStub(DataSourceInterface):
    """DataSourceInterface の Named Pipe スタブ実装。

    MT4 Named Pipe 本番実装前のダミーデータ生成器。
    固定シードで再現性のある OHLCV DataFrame を生成する。

    feature_engineering() の要件に応じ、datetime 列を含む DataFrame を返す。
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed=seed)

    def fetch_latest_ohlcv(self, n_bars: int = 60) -> pd.DataFrame:
        """直近 n_bars 本の合成 OHLCV データを生成する。

        Args:
            n_bars: 生成するバー数。デフォルト 60。

        Returns:
            columns: datetime, open, high, low, close, volume を持つ DataFrame。
            shape: (n_bars, 6)
        """
        close = 150.0 + self._rng.normal(0, 0.5, n_bars).cumsum()
        spread = self._rng.uniform(0.0, 0.3, n_bars)
        high = close + spread
        low = close - spread
        open_ = close + self._rng.uniform(-0.1, 0.1, n_bars)

        return pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=n_bars, freq="5min"),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": self._rng.integers(100, 1000, n_bars).astype(float),
            }
        )
