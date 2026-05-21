"""1分足CSVを5分足にリサンプリングするスクリプト。

Usage:
    uv run python scripts/resample_m1_to_m5.py
    uv run python scripts/resample_m1_to_m5.py --input data/raw/USDJPY_M1.csv --output data/processed/USDJPY_M5.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def resample_m1_to_m5(input_path: Path, output_path: Path) -> None:
    """1分足CSVを5分足にリサンプリングして保存する。

    Args:
        input_path: 入力CSV（列: datetime, open, high, low, close, volume、ヘッダーなし）
        output_path: 出力CSV（同形式、ヘッダーなし）
    """
    df = pd.read_csv(
        input_path,
        header=None,
        names=["datetime", "open", "high", "low", "close", "volume"],
        parse_dates=["datetime"],
        index_col="datetime",
    )

    df_m5 = df.resample("5min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_m5.to_csv(output_path, header=False)
    print(f"完了: {len(df)} 行 (M1) → {len(df_m5)} 行 (M5)")
    print(f"出力: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="1分足CSVを5分足にリサンプリングする")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/USDJPY_M1.csv"),
        help="入力M1 CSVパス（デフォルト: data/raw/USDJPY_M1.csv）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/USDJPY_M5.csv"),
        help="出力M5 CSVパス（デフォルト: data/processed/USDJPY_M5.csv）",
    )
    args = parser.parse_args()
    resample_m1_to_m5(args.input, args.output)


if __name__ == "__main__":
    main()
