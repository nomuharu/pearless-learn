"""データ取得インターフェース定義。

Design Doc: fx-prediction-design.md § DataSourceInterface
AC-019, AC-020: DataSourceInterface による抽象化と差し替え可能構造
"""

from abc import ABC, abstractmethod

import pandas as pd


class DataSourceInterface(ABC):
    """データ取得インターフェース。MT4 Named Pipe および将来のデータソースへの DI 抽象。

    契約:
        fetch_latest_ohlcv(n_bars) が呼ばれると、直近 n_bars 本の OHLCV DataFrame を返す。
        返り値の列名: ["datetime", "open", "high", "low", "close", "volume"]
        返り値の shape: (n_bars, 6)
    """

    @abstractmethod
    def fetch_latest_ohlcv(self, n_bars: int = 60) -> pd.DataFrame:
        """直近 n_bars 本の OHLCV データを取得する。

        Args:
            n_bars: 取得するバー数。デフォルト 60（推論ウィンドウサイズと一致）。

        Returns:
            shape (n_bars, 6) の DataFrame。列: datetime, open, high, low, close, volume。
        """
        ...
