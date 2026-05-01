"""inference/pipe_stub.py の単体テスト。

Design Doc: fx-prediction-design.md § NamedPipeDataSource
AC-018, AC-019, AC-020: DataSourceInterface のスタブ実装
"""

import pandas as pd

from inference.pipe_stub import NamedPipeStub


class TestNamedPipeStubInterface:
    """DataSourceInterface 契約の充足を検証する。"""

    def test_fetch_latest_ohlcv_returns_dataframe(self) -> None:
        """fetch_latest_ohlcv() が DataFrame を返すこと。"""
        stub = NamedPipeStub()
        result = stub.fetch_latest_ohlcv()
        assert isinstance(result, pd.DataFrame)

    def test_fetch_latest_ohlcv_default_n_bars_is_60(self) -> None:
        """デフォルト n_bars=60 で 60 行の DataFrame が返ること。"""
        stub = NamedPipeStub()
        result = stub.fetch_latest_ohlcv()
        assert len(result) == 60

    def test_fetch_latest_ohlcv_respects_n_bars_param(self) -> None:
        """指定した n_bars の行数が返ること。"""
        stub = NamedPipeStub()
        result = stub.fetch_latest_ohlcv(n_bars=30)
        assert len(result) == 30

    def test_fetch_latest_ohlcv_has_required_columns(self) -> None:
        """DataSourceInterface 契約の5列 + datetime 列を含むこと。"""
        stub = NamedPipeStub()
        result = stub.fetch_latest_ohlcv()
        required_columns = {"open", "high", "low", "close", "volume", "datetime"}
        assert required_columns.issubset(set(result.columns))

    def test_fetch_latest_ohlcv_no_nan(self) -> None:
        """返された DataFrame に NaN が含まれないこと。"""
        stub = NamedPipeStub()
        result = stub.fetch_latest_ohlcv()
        assert result.isna().sum().sum() == 0

    def test_fetch_latest_ohlcv_numeric_columns_are_positive(self) -> None:
        """数値カラム（volume）が正の値であること。"""
        stub = NamedPipeStub()
        result = stub.fetch_latest_ohlcv()
        assert (result["volume"] > 0).all()

    def test_different_seeds_produce_different_data(self) -> None:
        """異なるシードでは異なるデータが生成されること。"""
        stub_a = NamedPipeStub(seed=42)
        stub_b = NamedPipeStub(seed=99)
        result_a = stub_a.fetch_latest_ohlcv()
        result_b = stub_b.fetch_latest_ohlcv()
        assert not result_a["close"].equals(result_b["close"])

    def test_same_seed_produces_reproducible_data(self) -> None:
        """同じシードでは再現性のあるデータが生成されること。"""
        stub_a = NamedPipeStub(seed=42)
        stub_b = NamedPipeStub(seed=42)
        result_a = stub_a.fetch_latest_ohlcv()
        result_b = stub_b.fetch_latest_ohlcv()
        pd.testing.assert_frame_equal(result_a, result_b)

    def test_high_is_greater_or_equal_to_close(self) -> None:
        """high は close 以上であること（OHLCV の整合性）。"""
        stub = NamedPipeStub()
        result = stub.fetch_latest_ohlcv()
        assert (result["high"] >= result["close"]).all()

    def test_low_is_less_or_equal_to_close(self) -> None:
        """low は close 以下であること（OHLCV の整合性）。"""
        stub = NamedPipeStub()
        result = stub.fetch_latest_ohlcv()
        assert (result["low"] <= result["close"]).all()
