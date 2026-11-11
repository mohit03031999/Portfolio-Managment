"""How long a memory should the skeleton have? Room forecasts vs history length.

The "each piece at its own speed" idea takes the top directions (market arrow,
sector room) from a long history and the sizes and volatilities from recent
days. This study asks the one question that decides the memory length: does a
longer history predict the *next* period's top room better, or does drift make
old data hurt?

For each decision date, the top-k room of the correlation matrix is estimated
from the last L days (L = 250 … 2000) and compared with the room of the next
250 days (the target, itself noisy). Room overlap = share of the target room
captured, 1 = same room, k/N = random. The same is done on shuffled real days,
where nothing drifts, so longer is always better: that curve is the noise-only
ceiling, and the gap between it and the real curve is the cost of drift.

A second check looks at the sizes: the variance of the recent 1000 days along
directions taken from the recent window itself (in-sample) and along directions
taken from a 2000-day history, each compared with the variance along the same
directions in the next 250 days. Directions borrowed from older data are not
fitted to the recent days, so their sizes should carry over more honestly. The
same is recorded direction by direction (by rank).

A third picture is the weather: the market's share of co-movement and the
sector room's share in each half-year block of the real market, to show how
fast the sizes move while the directions stay.

Outputs in output/sp500_memory_length/: overlap.csv, sizes.csv,
sizes_by_rank.csv, weather.csv, summary.txt, memory_length.png,
sizes_by_rank.png, weather.png.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import sp500_shuffle_test as shuffle  # noqa: E402
import sp500_simulated_backtest as simulated  # noqa: E402

OUT = ROOT / "output" / "sp500_memory_length"
WINDOW, HOLDING, FUTURE, BLOCK = 1000, 21, 250, 126
LENGTHS = (250, 500, 1000, 1500, 2000)
ROOMS = (1, 3, 10, 20)
GROUPS = (("market (1)", 0, 1), ("sector room (2–20)", 1, 20), ("quietest 50", -50, None))
SOURCES = {"recent 1000 (in-sample)": (WINDOW, 0), "history 2000 (includes the window)": (2000, 0), "1000 days before the window": (2000, WINDOW)}
REAL, SHUFFLED, HISTORY, MUTED, INK = "#2a6fdb", "#e07b00", "#1a9e5c", "#777777", "#222222"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    returns = simulated.load_returns(simulated.PRICES)
    overlap_rows, size_rows, rank_rows = [], [], []
    for panel in ("real", "shuffled"):
        data = (returns if panel == "real" else shuffle.shuffled(returns, args.seed)).to_numpy(dtype=float)
        overlap_rows += overlaps(data, panel)
        rows, by_rank = sizes(data, panel)
        size_rows += rows
        rank_rows += by_rank
    overlap, size, by_rank = pd.DataFrame(overlap_rows), pd.DataFrame(size_rows), pd.DataFrame(rank_rows)
    blocks = weather(returns)
    overlap.to_csv(OUT / "overlap.csv", index=False)
    size.to_csv(OUT / "sizes.csv", index=False)
    by_rank.to_csv(OUT / "sizes_by_rank.csv", index=False)
    blocks.to_csv(OUT / "weather.csv", index=False)
    report(overlap, size, by_rank, blocks)


def correlation(x: np.ndarray) -> np.ndarray:
    centred = x - x.mean(axis=0)
    vol = np.sqrt((centred**2).sum(axis=0) / (len(x) - 1))
    vol = np.where(vol > 0, vol, 1.0)
    z = centred / vol
    return z.T @ z / (len(x) - 1)


def top_vectors(c: np.ndarray) -> np.ndarray:
    _, vectors = np.linalg.eigh(c)
    return vectors[:, ::-1]


def overlaps(data: np.ndarray, panel: str) -> list[dict]:
    n_days, n = data.shape
    starts = range(max(LENGTHS), n_days - FUTURE + 1, HOLDING)
    rows = []
    for i, start in enumerate(starts):
        target = top_vectors(correlation(data[start : start + FUTURE]))
        for length in LENGTHS:
            vectors = top_vectors(correlation(data[start - length : start]))
            row = {"panel": panel, "date": i, "length": length}
            for k in ROOMS:
                row[f"room_{k}"] = float(np.sum((vectors[:, :k].T @ target[:, :k]) ** 2) / k)
            rows.append(row)
    return rows


def sizes(data: np.ndarray, panel: str) -> tuple[list[dict], list[dict]]:
    n_days, n = data.shape
    starts = list(range(max(LENGTHS), n_days - FUTURE + 1, HOLDING))
    rows = []
    totals = {source: np.zeros((2, n)) for source in SOURCES}
    for i, start in enumerate(starts):
        recent = correlation(data[start - WINDOW : start])
        future = correlation(data[start : start + FUTURE])
        for source, (back, until) in SOURCES.items():
            vectors = top_vectors(recent) if (back, until) == (WINDOW, 0) else top_vectors(correlation(data[start - back : start - until]))
            along_recent = np.einsum("ij,ik,kj->j", vectors, recent, vectors)
            along_future = np.einsum("ij,ik,kj->j", vectors, future, vectors)
            totals[source] += np.vstack([along_recent, along_future])
            for group, lo, hi in GROUPS:
                sl = slice(lo, hi)
                rows.append({"panel": panel, "date": i, "source": source, "group": group, "recent": float(along_recent[sl].sum()), "future": float(along_future[sl].sum())})
    by_rank = []
    for source, total in totals.items():
        mean = total / len(starts)
        by_rank += [{"panel": panel, "source": source, "rank": r + 1, "recent": float(mean[0, r]), "future": float(mean[1, r])} for r in range(n)]
    return rows, by_rank


def weather(returns: pd.DataFrame) -> pd.DataFrame:
    """Market and sector-room shares of co-movement in each half-year block of the real market."""
    data = returns.to_numpy(dtype=float)
    rows = []
    for end in range(len(data), BLOCK - 1, -BLOCK):
        block = data[end - BLOCK : end]
        values = np.sort(np.linalg.eigvalsh(correlation(block)))[::-1]
        rows.append(
            {
                "end": returns.index[end - 1],
                "market": float(values[0]),
                "second": float(values[1]),
                "sector_room": float(values[1:20].sum()),
                "market_vol": float(np.sqrt(252) * block.mean(axis=1).std()),
            }
        )
    return pd.DataFrame(rows).sort_values("end").reset_index(drop=True)


def report(overlap: pd.DataFrame, size: pd.DataFrame, by_rank: pd.DataFrame, blocks: pd.DataFrame) -> None:
    n_dates = overlap["date"].nunique()
    lines = [
        f"Room forecasts vs memory length. Target = top room of the next {FUTURE} days; {n_dates} decision dates (21 days apart), 456 stocks.",
        "Room overlap: 1 = same room, k/456 = random. 'shuffled' = same days in random order (no drift): the noise-only ceiling.",
        "",
    ]
    mean = overlap.groupby(["panel", "length"]).mean(numeric_only=True)
    for k in ROOMS:
        lines.append(f"top-{k} room (random level {k / 456:.3f}):")
        lines.append("  history days   " + "  ".join(f"{L:>6}" for L in LENGTHS))
        for panel in ("real", "shuffled"):
            lines.append(f"  {panel:<13}  " + "  ".join(f"{mean.loc[(panel, L), f'room_{k}']:6.3f}" for L in LENGTHS))
        lines.append("")
    lines.append("Sizes: variance (correlation units) along the top directions, summed per group, in the recent 1000 days and in the next 250 days.")
    lines.append("Directions from the recent window are fitted to those days (in-sample); directions from a 2000-day history are not.")
    agg = size.groupby(["panel", "source", "group"], sort=False)[["recent", "future"]].mean()
    for panel in ("real", "shuffled"):
        lines.append(f"  {panel}:")
        for source in SOURCES:
            parts = []
            for group, _, _ in GROUPS:
                r, f = agg.loc[(panel, source, group), "recent"], agg.loc[(panel, source, group), "future"]
                parts.append(f"{group}: recent {r:6.1f} → next {f:6.1f} ({f / r - 1:+.0%})")
            lines.append(f"    {source:<24} " + " | ".join(parts))
    lines.append("")
    lines.append("By rank (real market): variance along the direction of that rank, recent window → next 250 days (ratio next/recent).")
    lines.append("rank  | " + " | ".join(f"{source:^36}" for source in SOURCES))
    real = by_rank[by_rank["panel"] == "real"].set_index(["source", "rank"])
    for r in (1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400, 456):
        cells = []
        for source in SOURCES:
            row = real.loc[(source, r)]
            cells.append(f"{row['recent']:12.3f} → {row['future']:8.3f} ({row['future'] / row['recent']:5.2f})")
        lines.append(f"{r:5d} | " + " | ".join(f"{c:^36}" for c in cells))
    lines.append("")
    lines.append(f"Weather, {BLOCK}-day blocks of the real market (correlation units, average direction = 1):")
    lines.append(f"  market share: {blocks['market'].min():.0f}–{blocks['market'].max():.0f} (median {blocks['market'].median():.0f}); second eigenvalue {blocks['second'].min():.0f}–{blocks['second'].max():.0f}; sector room (2–20) {blocks['sector_room'].min():.0f}–{blocks['sector_room'].max():.0f}. (A {BLOCK}-day block cannot see the quiet end: its correlation matrix has rank {BLOCK - 1}.)")
    lines.append(f"  market share vs market volatility, correlation across blocks: {blocks['market'].corr(blocks['market_vol']):.2f}.")
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)
    plot_rooms(mean, OUT / "memory_length.png")
    plot_sizes(by_rank, OUT / "sizes_by_rank.png")
    plot_weather(blocks, OUT / "weather.png")


def plot_rooms(mean: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, len(ROOMS), figsize=(16, 4.4), sharex=True)
    for ax, k in zip(axes, ROOMS):
        for panel, color, label in (("real", REAL, "real market (drift + noise)"), ("shuffled", SHUFFLED, "shuffled days (noise only)")):
            ax.plot(LENGTHS, [mean.loc[(panel, L), f"room_{k}"] for L in LENGTHS], marker="o", color=color, lw=2, label=label)
        ax.axhline(k / 456, color=MUTED, lw=1, ls=":")
        ax.set_title(f"top-{k} room", loc="left", fontsize=11)
        ax.set_xlabel("days of history used")
        ax.set_xticks(LENGTHS)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(f"share of the next {FUTURE} days' room captured")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("How much of the next period's top room a longer history predicts", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_sizes(by_rank: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    real = by_rank[by_rank["panel"] == "real"]
    colors = dict(zip(SOURCES, (REAL, HISTORY, SHUFFLED)))
    ax = axes[0]
    for source, color in colors.items():
        part = real[real["source"] == source].sort_values("rank")
        ax.plot(part["rank"], part["recent"], color=color, lw=2, ls="--", label=f"{source}: inside the window")
        ax.plot(part["rank"], part["future"], color=color, lw=2, label=f"{source}: next 250 days")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("direction rank (1 = market, 456 = quietest)")
    ax.set_ylabel("variance along the direction (correlation units, log)")
    ax.set_title("1. Sizes along each direction: seen in the window vs realised next (real market)", loc="left", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    ax = axes[1]
    for panel, ls in (("real", "-"), ("shuffled", "--")):
        part = by_rank[by_rank["panel"] == panel]
        for source, color in colors.items():
            if panel == "shuffled" and source.startswith("history 2000"):
                continue
            p = part[part["source"] == source].sort_values("rank")
            ax.plot(p["rank"], p["future"] / p["recent"], color=color, lw=2, ls=ls, label=f"{source}, {panel}")
    ax.axhline(1, color=MUTED, lw=1)
    ax.text(1.1, 1.06, "above 1: realised more than seen (forecast too optimistic)", fontsize=8, color=MUTED)
    ax.text(1.1, 0.94, "below 1: realised less than seen", fontsize=8, color=MUTED, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("direction rank (1 = market, 456 = quietest)")
    ax.set_ylabel("realised ÷ seen (log)")
    ax.set_title("2. How honest the size along each direction is", loc="left", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_weather(blocks: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    x = pd.to_datetime(blocks["end"])
    ax = axes[0]
    ax.plot(x, blocks["market"], marker="o", color=REAL, lw=2, label="market (direction 1)")
    ax.plot(x, blocks["sector_room"], marker="s", color=HISTORY, lw=2, label="sector room (directions 2–20, summed)")
    ax.plot(x, blocks["second"], marker="^", color=SHUFFLED, lw=1.5, label="direction 2 alone")
    ax.set_ylabel("share of co-movement (average direction = 1)")
    ax.set_title(f"1. How much each room carries, per {BLOCK}-day block", loc="left", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False)
    ax = axes[1]
    ax.scatter(blocks["market_vol"] * 100, blocks["market"], color=REAL, s=36)
    for _, row in blocks.iterrows():
        ax.annotate(str(row["end"])[:7], (row["market_vol"] * 100, row["market"]), fontsize=7, color=MUTED, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("volatility of the equal-weight market in the block, % per year")
    ax.set_ylabel("market share of co-movement")
    ax.set_title("2. Stormy blocks are market-driven blocks", loc="left", fontsize=11)
    for ax in axes:
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
