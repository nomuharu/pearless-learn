"""フォワードテスト結果のサマリ生成。

production/strategies/<id>/forward_tests/<session>/trades.csv を読み、
バックテストと同じ指標（平均損益 銭/トレード、勝率、月次内訳）で集計する。
出力は session.md に貼り付ける想定のプレーンテキスト。

期待値はトレードCSVから親方向に探索して見つかる expectations.json
（戦略ディレクトリ直下に置く）から読む。見つからない場合は比較を省略する。

trades.csv フォーマット（production/TEMPLATE_forward_test.md 参照）:
    time,side,entry_price,exit_price,lots,pnl_yen

Usage:
    uv run python scripts/summarize_forward_test.py \\
        production/strategies/oco-breakout-wf/forward_tests/2026-06_demo/trades.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_expectations(trades_path: Path) -> dict[str, float | str] | None:
    """trades.csv から親ディレクトリを遡って expectations.json を探す。"""
    for parent in trades_path.resolve().parents:
        candidate = parent / "expectations.json"
        if candidate.exists():
            return json.loads(candidate.read_text())
        if parent.name == "production":
            break
    return None


def summarize(trades_path: Path) -> str:
    df = pd.read_csv(trades_path, parse_dates=["time"])
    required = {"time", "side", "entry_price", "exit_price", "lots", "pnl_yen"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"trades.csv に必須列がありません: {missing}")

    exp = load_expectations(trades_path)

    # 値動きベースの損益（銭/ドル）: ロットに依存しない、バックテストと同じ単位
    sign = df["side"].map({"buy": 1, "sell": -1})
    df["pnl_sen"] = (df["exit_price"] - df["entry_price"]) * sign * 100

    n = len(df)
    days = max((df["time"].max() - df["time"].min()).days, 1)
    avg_sen = df["pnl_sen"].mean()

    trades_line = f"トレード数: {n}（週あたり {n / days * 7:.1f}回"
    avg_line = f"平均損益: {avg_sen:+.3f}銭/トレード"
    if exp is not None:
        trades_line += f" / 期待 〜{exp['trades_per_week']:.0f}回）"
        avg_line += (
            f"（期待 {exp['avg_pnl_sen']:+.2f}銭、乖離 "
            f"{avg_sen - float(exp['avg_pnl_sen']):+.3f}銭）"
        )
    else:
        trades_line += "）"
        avg_line += "（expectations.json なし: 期待値比較を省略）"

    lines = [
        f"戦略: {exp['strategy_id']}" if exp else "戦略: (不明)",
        f"期間: {df['time'].min():%Y-%m-%d} 〜 {df['time'].max():%Y-%m-%d} ({days}日)",
        trades_line,
        f"勝率: {(df['pnl_yen'] > 0).mean():.1%}",
        avg_line,
        f"実現損益合計: {df['pnl_yen'].sum():+,.0f}円",
        f"buy/sell 比率: {int((sign == 1).sum())}/{int((sign == -1).sum())}",
        "",
        "月次内訳:",
    ]
    monthly = df.groupby(df["time"].dt.to_period("M")).agg(
        trades=("pnl_yen", "size"),
        pnl_yen=("pnl_yen", "sum"),
        avg_sen=("pnl_sen", "mean"),
    )
    lines.append(monthly.to_string(float_format=lambda v: f"{v:,.2f}"))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="フォワードテストのサマリ生成")
    parser.add_argument("trades_csv", type=Path)
    args = parser.parse_args()
    print(summarize(args.trades_csv))


if __name__ == "__main__":
    main()
