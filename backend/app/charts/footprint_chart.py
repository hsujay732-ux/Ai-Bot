# backend/app/charts/footprint_chart.py
"""
Bid x Ask footprint chart (ATAS / GoCharting style) with delta coloring.

Aggregates trade prints into price-by-time cells showing "bid x ask" volume,
colors each cell green/red by that cell's delta (ask volume - bid volume),
and renders a delta row + volume row beneath the footprint grid.

Usage:
    python -m backend.app.charts.footprint_chart --symbol BTCUSDT --interval 1min --tick-size 1
    python -m backend.app.charts.footprint_chart --demo   # offline synthetic data, no network
"""
import argparse

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import requests
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

BYBIT_REST_BASE = "https://api.bybit.com"

ASK_DELTA_COLOR = "#1b7a3d"   # buy aggressor / positive delta
BID_DELTA_COLOR = "#b5322f"   # sell aggressor / negative delta
NEUTRAL_CELL = "#eeeeee"
VOLUME_HEAT = "#4472c4"


def fetch_recent_trades(symbol: str, limit: int = 1000, category: str = "linear") -> pd.DataFrame:
    """Pull recent public trades (with aggressor side) from Bybit's v5 REST API."""
    resp = requests.get(
        f"{BYBIT_REST_BASE}/v5/market/recent-trade",
        params={"category": category, "symbol": symbol, "limit": min(limit, 1000)},
        timeout=10,
    )
    resp.raise_for_status()
    rows = resp.json()["result"]["list"]
    df = pd.DataFrame(rows)
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    df["timestamp"] = pd.to_datetime(df["time"].astype(np.int64), unit="ms", utc=True)
    return df[["timestamp", "price", "size", "side"]].sort_values("timestamp").reset_index(drop=True)


def synthetic_trades(n: int = 4000, start_price: float = 60000.0, tick_size: float = 1.0, seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic tick tape for offline demoing / testing."""
    rng = np.random.default_rng(seed)
    steps = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
    prices = start_price + np.cumsum(steps) * tick_size
    sides = np.where(rng.random(n) < 0.5, "Buy", "Sell")
    sizes = rng.lognormal(mean=2.0, sigma=1.0, size=n).round(0).clip(1)
    timestamps = pd.date_range("2026-07-13 09:00:00", periods=n, freq="2s", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "price": prices, "size": sizes, "side": sides})


def build_footprint(trades: pd.DataFrame, bar_interval: str = "1min", tick_size: float = 1.0):
    """
    Aggregate raw trades into footprint bars.

    Returns:
        footprint: {bar_timestamp: {price_level: {"bid": vol, "ask": vol}}}
        summary: per-bar DataFrame with bid/ask/delta/volume/cum_delta
    """
    df = trades.copy()
    df["level"] = (df["price"] / tick_size).round() * tick_size
    df["bar"] = df["timestamp"].dt.floor(bar_interval)
    is_buy = df["side"].str.lower() == "buy"
    df["ask_vol"] = np.where(is_buy, df["size"], 0.0)
    df["bid_vol"] = np.where(~is_buy, df["size"], 0.0)

    grouped = df.groupby(["bar", "level"]).agg(bid=("bid_vol", "sum"), ask=("ask_vol", "sum")).reset_index()

    footprint, summary = {}, []
    for bar, sub in grouped.groupby("bar"):
        footprint[bar] = {row.level: {"bid": row.bid, "ask": row.ask} for row in sub.itertuples()}
        total_bid, total_ask = sub["bid"].sum(), sub["ask"].sum()
        summary.append({"bar": bar, "bid": total_bid, "ask": total_ask,
                         "delta": total_ask - total_bid, "volume": total_bid + total_ask})

    summary_df = pd.DataFrame(summary).sort_values("bar").reset_index(drop=True)
    summary_df["cum_delta"] = summary_df["delta"].cumsum()
    return footprint, summary_df


def limit_bars(footprint, summary_df, max_bars: int):
    """Keep only the most recent `max_bars` bars (charts get unreadable past a few dozen columns)."""
    if len(summary_df) <= max_bars:
        return footprint, summary_df
    kept = summary_df.tail(max_bars).reset_index(drop=True)
    kept_bars = set(kept["bar"])
    return {bar: levels for bar, levels in footprint.items() if bar in kept_bars}, kept


def plot_footprint(footprint, summary_df, title: str = "Footprint Chart", out_path: str = "footprint.png"):
    bars = summary_df["bar"].tolist()
    all_levels = sorted({lvl for levels in footprint.values() for lvl in levels}, reverse=True)
    if not bars or not all_levels:
        raise ValueError("No trade data to plot")

    n_bars, n_levels = len(bars), len(all_levels)
    max_cell_vol = max(max(v["bid"], v["ask"]) for levels in footprint.values() for v in levels.values()) or 1.0
    max_bar_vol = summary_df["volume"].max() or 1.0

    fig_w = max(8.0, n_bars * 1.5)
    fig_h = max(6.0, n_levels * 0.3 + 3)
    fig, (ax_fp, ax_delta, ax_vol) = plt.subplots(
        3, 1, figsize=(fig_w, fig_h), gridspec_kw={"height_ratios": [n_levels, 2, 1.5]}
    )

    for bi, bar in enumerate(bars):
        levels = footprint[bar]
        for li, level in enumerate(all_levels):
            cell = levels.get(level, {"bid": 0.0, "ask": 0.0})
            bid, ask = cell["bid"], cell["ask"]
            delta = ask - bid
            intensity = min(abs(delta) / max_cell_vol, 1.0)
            if delta > 0:
                color = mcolors.to_rgba(ASK_DELTA_COLOR, alpha=0.12 + 0.7 * intensity)
            elif delta < 0:
                color = mcolors.to_rgba(BID_DELTA_COLOR, alpha=0.12 + 0.7 * intensity)
            else:
                color = NEUTRAL_CELL
            ax_fp.add_patch(Rectangle((bi, li), 1, 1, facecolor=color, edgecolor="white", linewidth=0.6))
            if bid or ask:
                ax_fp.text(bi + 0.5, li + 0.5, f"{int(bid)} x {int(ask)}", ha="center", va="center", fontsize=6)

    ax_fp.set_xlim(0, n_bars)
    ax_fp.set_ylim(0, n_levels)
    ax_fp.set_yticks([i + 0.5 for i in range(n_levels)])
    ax_fp.set_yticklabels([f"{lvl:g}" for lvl in all_levels], fontsize=7)
    ax_fp.set_xticks([])
    ax_fp.set_title(title)
    ax_fp.set_ylabel("Price")

    for bi, row in summary_df.iterrows():
        color = ASK_DELTA_COLOR if row["delta"] >= 0 else BID_DELTA_COLOR
        ax_delta.add_patch(Rectangle((bi, 0), 1, 1, facecolor=color, alpha=0.85, edgecolor="white"))
        ax_delta.text(bi + 0.5, 0.5, f"{int(row['delta'])}", ha="center", va="center",
                      fontsize=7, color="white", fontweight="bold")
    ax_delta.set_xlim(0, n_bars)
    ax_delta.set_ylim(0, 1)
    ax_delta.set_yticks([0.5])
    ax_delta.set_yticklabels(["Delta"], fontsize=7)
    ax_delta.set_xticks([])

    for bi, row in summary_df.iterrows():
        intensity = 0.15 + 0.75 * (row["volume"] / max_bar_vol)
        ax_vol.add_patch(Rectangle((bi, 0), 1, 1, facecolor=mcolors.to_rgba(VOLUME_HEAT, alpha=intensity),
                                    edgecolor="white"))
        ax_vol.text(bi + 0.5, 0.5, f"{int(row['volume'])}", ha="center", va="center", fontsize=7)
    ax_vol.set_xlim(0, n_bars)
    ax_vol.set_ylim(0, 1)
    ax_vol.set_yticks([0.5])
    ax_vol.set_yticklabels(["Volume"], fontsize=7)
    ax_vol.set_xticks([i + 0.5 for i in range(n_bars)])
    ax_vol.set_xticklabels([b.strftime("%H:%M") for b in bars], rotation=45, fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Print a bid x ask footprint chart with delta coloring.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1min", help="Bar interval, e.g. 1min, 5min, 15min")
    parser.add_argument("--tick-size", type=float, default=1.0, help="Price bucket size per footprint row")
    parser.add_argument("--limit", type=int, default=1000, help="Recent trades to fetch (max 1000)")
    parser.add_argument("--max-bars", type=int, default=30, help="Most recent bars to draw")
    parser.add_argument("--out", default="footprint.png")
    parser.add_argument("--demo", action="store_true", help="Use synthetic offline data instead of Bybit")
    args = parser.parse_args()

    if args.demo:
        trades = synthetic_trades(tick_size=args.tick_size)
    else:
        trades = fetch_recent_trades(args.symbol, args.limit)

    footprint, summary_df = build_footprint(trades, bar_interval=args.interval, tick_size=args.tick_size)
    footprint, summary_df = limit_bars(footprint, summary_df, args.max_bars)
    out = plot_footprint(footprint, summary_df, title=f"{args.symbol} Footprint ({args.interval})", out_path=args.out)
    print(f"Saved footprint chart to {out}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
