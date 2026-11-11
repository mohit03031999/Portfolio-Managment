"""What carries over from one period to the next, piece by piece, and how much of the change is real?

The covariance of 456 S&P 500 stocks is split into pieces: each stock's
volatility, the pairwise correlations, the market eigenvector, the sector
eigenvectors 2–20 (one by one and as a subspace) and the middle modes
21–100 (as a subspace). History is cut into back-to-back windows that do
not overlap, and each piece's similarity between a window and a later one
is measured.

Two windows differ even when nothing in the market changed, because each
is a noisy sample. That noise-only difference is measured, not assumed, on
panels with no time order:

* ``shuffled``: the real days in random order (fat tails kept, no change);
* ``gaussian``: Gaussian days from the raw full-sample covariance (noise only).

real similarity < shuffled similarity  ⇒  the gap is genuine change in the market.

Similarity measures (1 = identical):

* volatilities: Spearman rank correlation of the stocks' standard deviations;
* correlations: Pearson correlation of the off-diagonal correlation entries;
* market: squared overlap of the top correlation eigenvectors;
* sector arrows: mean squared overlap of modes 2–20 matched by rank;
* sector room: ‖U₂₋₂₀ᵀ U′₂₋₂₀‖²_F / 19, the subspace overlap (random: 19/456 ≈ 0.04);
* middle room: the same for modes 21–100 (random: 80/456 ≈ 0.18).

Adjacent windows of 126, 250 and 500 days, and the fade with lag for
126-day windows (lags of 1 to 10 windows, i.e. up to five years).

    .venv/bin/python examples/sp500_persistence_study.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sp500_shuffle_test as shuffle  # noqa: E402
import sp500_simulated_backtest as simulated  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "sp500_persistence"
LENGTHS = (126, 250, 500)
LAG_LENGTH = 126
MAX_LAG = 10
PIECES = ("volatilities", "correlations", "market", "sector arrows", "sector room", "middle room")
WORLDS = ("real", "shuffled", "gaussian")
WORLD_LABELS = {"real": "real market", "shuffled": "real days, shuffled (no change)", "gaussian": "Gaussian days (noise only)"}
SEEDS = (1, 2, 3)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    returns = simulated.load_returns(simulated.PRICES)
    panels = {("real", 0): returns.to_numpy(dtype=float)}
    for seed in SEEDS:
        panels[("shuffled", seed)] = shuffle.shuffled(returns, seed).to_numpy(dtype=float)
        panels[("gaussian", seed)] = shuffle.gaussian_raw(returns, seed).to_numpy(dtype=float)

    rows = []
    for (world, seed), data in panels.items():
        for length in sorted(set(LENGTHS) | {LAG_LENGTH}):
            windows = [describe(data[i : i + length]) for i in range(0, len(data) - length + 1, length)]
            max_lag = MAX_LAG if length == LAG_LENGTH else 1
            for lag in range(1, max_lag + 1):
                for i in range(len(windows) - lag):
                    for piece, value in compare(windows[i], windows[i + lag]).items():
                        rows.append({"world": world, "seed": seed, "length": length, "lag": lag, "piece": piece, "value": value})
        print(f"{world} seed {seed} done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "similarities.csv", index=False)
    report(table)


def describe(x: np.ndarray) -> dict[str, np.ndarray]:
    x = x - x.mean(axis=0)
    vol = np.sqrt(np.sum(x**2, axis=0) / (len(x) - 1))
    z = x / vol
    corr = z.T @ z / (len(x) - 1)
    values, vectors = np.linalg.eigh(corr)
    vectors = vectors[:, ::-1]
    return {"vol": vol, "corr": corr[np.triu_indices_from(corr, k=1)], "vectors": vectors}


def _subspace(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum((a.T @ b) ** 2) / a.shape[1])


def compare(a: dict, b: dict) -> dict[str, float]:
    ua, ub = a["vectors"], b["vectors"]
    return {
        "volatilities": float(np.corrcoef(np.argsort(np.argsort(a["vol"])), np.argsort(np.argsort(b["vol"])))[0, 1]),
        "correlations": float(np.corrcoef(a["corr"], b["corr"])[0, 1]),
        "market": float((ua[:, 0] @ ub[:, 0]) ** 2),
        "sector arrows": float(np.mean(np.sum(ua[:, 1:20] * ub[:, 1:20], axis=0) ** 2)),
        "sector room": _subspace(ua[:, 1:20], ub[:, 1:20]),
        "middle room": _subspace(ua[:, 20:100], ub[:, 20:100]),
    }


def report(table: pd.DataFrame) -> None:
    adjacent = table[table["lag"] == 1]
    mean = adjacent.groupby(["length", "piece", "world"])["value"].mean().unstack("world")
    lines = [
        "How much of each piece carries over to the next window (adjacent windows that do not overlap).",
        "Similarity 1 = identical. 'change share' = (shuffled − real) / (1 − real): the part of the real drop that noise does not explain.",
        "Random level: sector room 0.04, middle room 0.18, market and arrows ~0.002.",
        "",
    ]
    for length in LENGTHS:
        lines.append(f"--- windows of {length} days ({len(adjacent[(adjacent.length == length) & (adjacent.world == 'real') & (adjacent.piece == 'market')])} pairs on the real market) ---")
        lines.append(f"{'piece':<16}{'real':>8}{'shuffled':>10}{'gaussian':>10}{'change share':>15}")
        for piece in PIECES:
            r, s, g = (mean.loc[(length, piece), w] for w in WORLDS)
            share = (s - r) / (1 - r) if r < 1 else float("nan")
            lines.append(f"{piece:<16}{r:8.3f}{s:10.3f}{g:10.3f}{share:15.0%}")
        lines.append("")
    lag = table[table["length"] == LAG_LENGTH].groupby(["piece", "world", "lag"])["value"].mean()
    lines.append(f"Fade with time, {LAG_LENGTH}-day windows (lag in half-years):")
    lines.append(f"{'piece':<16}{'world':<10}" + "".join(f"{k:>7}" for k in range(1, MAX_LAG + 1)))
    for piece in PIECES:
        for world in ("real", "shuffled"):
            lines.append(f"{piece:<16}{world:<10}" + "".join(f"{lag[(piece, world, k)]:7.3f}" for k in range(1, MAX_LAG + 1)))
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)
    plot(mean, lag, OUT / "persistence.png")


def plot(mean: pd.DataFrame, lag: pd.Series, path: Path) -> None:
    simulated._style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    colors = {"real": "#a61b1b", "shuffled": "#2a78d6", "gaussian": "#9a9a9a"}
    x = np.arange(len(PIECES))
    width = 0.27
    for k, world in enumerate(WORLDS):
        values = [mean.loc[(250, piece), world] for piece in PIECES]
        left.bar(x + (k - 1) * width, values, width=width, color=colors[world], label=WORLD_LABELS[world])
    left.set_xticks(x)
    left.set_xticklabels([p.replace(" ", "\n") for p in PIECES])
    left.set_ylim(0, 1.05)
    left.set_ylabel("similarity with the next window")
    left.set_title("How much carries over to the next year (250-day windows)")
    left.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    left.grid(axis="x", visible=False)
    piece_colors = dict(zip(PIECES, ("#eda100", "#6250d6", "#1f1f1f", "#1baf7a", "#2a78d6", "#e87ba4")))
    lags = np.arange(1, MAX_LAG + 1)
    for piece in PIECES:
        right.plot(lags, [lag[(piece, "real", k)] for k in lags], "-", color=piece_colors[piece], linewidth=2, marker="o", markersize=4, label=piece)
        right.plot(lags, [lag[(piece, "shuffled", k)] for k in lags], ":", color=piece_colors[piece], linewidth=1.5)
    right.set_xlabel("years apart")
    right.set_xticks(lags)
    right.set_xticklabels([f"{k / 2:g}" for k in lags])
    right.set_ylim(0, 1.05)
    right.set_ylabel("similarity")
    right.set_title("How fast each piece fades (126-day windows; dotted = no change)")
    right.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
