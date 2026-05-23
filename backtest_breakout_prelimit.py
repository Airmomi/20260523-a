#!/usr/bin/env python3
"""
基于开源免费数据源 AkShare 的“涨停前一天”形态回测示例。

策略思想（T日选股、T+1入场）：
1) T日不含“最后一天大阳线”信息，只用到T日收盘前数据。
2) 识别长期横盘后临近突破、均线多头初成、温和放量的个股。
3) T+1开盘买入，持有N天或触发止损/止盈退出。

依赖:
    pip install akshare pandas numpy

用法示例:
    python backtest_breakout_prelimit.py \
        --symbols sh600000,sz000001,sh605500 \
        --start 2022-01-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class Trade:
    symbol: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    hold_days: int
    ret: float
    exit_reason: str


def fetch_daily_akshare(symbol: str, start: str, end: str) -> pd.DataFrame:
    """从 AkShare 拉取 A 股前复权日线数据。"""
    import akshare as ak

    # symbol 形如 sh600000 / sz000001
    market = "sh" if symbol.startswith("sh") else "sz"
    code = symbol[2:]

    df = ak.stock_zh_a_hist(
        symbol=f"{market}{code}",
        period="daily",
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        adjust="qfq",
    )
    if df.empty:
        return df

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_chg",
    }
    df = df.rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for c in ["open", "close", "high", "low", "volume", "amount", "pct_chg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["open", "close", "high", "low", "volume"])


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """构建“涨停前一天”候选信号。"""
    x = df.copy()

    x["ma5"] = x["close"].rolling(5).mean()
    x["ma10"] = x["close"].rolling(10).mean()
    x["ma20"] = x["close"].rolling(20).mean()
    x["ma30"] = x["close"].rolling(30).mean()
    x["vma5"] = x["volume"].rolling(5).mean()

    x["hh20"] = x["high"].rolling(20).max()
    x["std20"] = x["close"].rolling(20).std()
    x["cv20"] = x["std20"] / x["ma20"]

    cond_trend = (x["close"] > x["ma20"]) & (x["close"] > x["ma30"]) & (x["ma5"] > x["ma10"])
    cond_near_high = ((x["hh20"] - x["close"]) / x["hh20"]).between(0, 0.03)
    cond_volume = (x["volume"] > x["vma5"] * 1.2) & (x["volume"] < x["vma5"] * 3.0)
    cond_not_overheat = x["pct_chg"] < 7.0
    cond_compact = x["cv20"] < 0.08

    x["signal"] = cond_trend & cond_near_high & cond_volume & cond_not_overheat & cond_compact
    return x


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    hold_days: int = 5,
    stop_loss: float = -0.05,
    take_profit: float = 0.12,
) -> List[Trade]:
    """T日出信号，T+1开盘入场。"""
    trades: List[Trade] = []
    i = 0
    n = len(df)

    while i < n - 2:
        if not bool(df.iloc[i]["signal"]):
            i += 1
            continue

        entry_i = i + 1
        if entry_i >= n:
            break

        entry_price = float(df.iloc[entry_i]["open"])
        entry_date = df.iloc[entry_i]["date"]
        signal_date = df.iloc[i]["date"]

        max_exit_i = min(entry_i + hold_days, n - 1)
        exit_i = max_exit_i
        exit_reason = "time_exit"

        for j in range(entry_i, max_exit_i + 1):
            day_low = float(df.iloc[j]["low"])
            day_high = float(df.iloc[j]["high"])

            if day_low / entry_price - 1 <= stop_loss:
                exit_i = j
                exit_reason = "stop_loss"
                break
            if day_high / entry_price - 1 >= take_profit:
                exit_i = j
                exit_reason = "take_profit"
                break

        exit_price = float(df.iloc[exit_i]["close"])
        exit_date = df.iloc[exit_i]["date"]
        ret = exit_price / entry_price - 1

        trades.append(
            Trade(
                symbol=symbol,
                signal_date=signal_date,
                entry_date=entry_date,
                exit_date=exit_date,
                entry_price=entry_price,
                exit_price=exit_price,
                hold_days=exit_i - entry_i + 1,
                ret=ret,
                exit_reason=exit_reason,
            )
        )

        i = exit_i + 1

    return trades


def summarize(trades: List[Trade]) -> Dict[str, float]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_ret": 0.0,
            "cum_ret": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

    rets = np.array([t.ret for t in trades], dtype=float)
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1

    pos = rets[rets > 0].sum()
    neg = -rets[rets < 0].sum()

    return {
        "trades": float(len(trades)),
        "win_rate": float((rets > 0).mean()),
        "avg_ret": float(rets.mean()),
        "cum_ret": float(eq[-1] - 1),
        "profit_factor": float(pos / neg) if neg > 0 else np.inf,
        "max_drawdown": float(dd.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, required=True, help="逗号分隔，如 sh600000,sz000001")
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2026-01-01")
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--stop-loss", type=float, default=-0.05)
    parser.add_argument("--take-profit", type=float, default=0.12)
    args = parser.parse_args()

    all_trades: List[Trade] = []

    for symbol in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        try:
            raw = fetch_daily_akshare(symbol, args.start, args.end)
            if raw.empty or len(raw) < 60:
                print(f"[skip] {symbol}: 数据不足")
                continue
            sig = build_signals(raw)
            trades = run_backtest(
                sig,
                symbol=symbol,
                hold_days=args.hold_days,
                stop_loss=args.stop_loss,
                take_profit=args.take_profit,
            )
            all_trades.extend(trades)
            print(f"[done] {symbol}: trades={len(trades)}")
        except Exception as e:
            print(f"[error] {symbol}: {e}")

    stats = summarize(all_trades)
    print("\n===== 回测结果 =====")
    for k, v in stats.items():
        if k in {"win_rate", "avg_ret", "cum_ret", "max_drawdown"}:
            print(f"{k:>12}: {v:.2%}")
        else:
            print(f"{k:>12}: {v}")

    if all_trades:
        out = pd.DataFrame([t.__dict__ for t in all_trades])
        out.to_csv("backtest_trades.csv", index=False, encoding="utf-8-sig")
        print("\n已导出交易明细: backtest_trades.csv")


if __name__ == "__main__":
    main()
