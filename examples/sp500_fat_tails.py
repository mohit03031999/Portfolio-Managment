"""Four pictures of the fat-tail problem, drawn from the S&P 500 panel.

1. The histogram of real daily returns against the bell curve with the same
   width, on a log scale so the tails are visible.
2. The daily returns of the equal-weight index through time with a rolling
   two-sigma band: calm years and stormy months, not one volatility.
3. The index histogram after dividing each day's return by the volatility of
   the previous 21 sessions. For the market as a whole the tail is almost
   entirely volatility clustering and disappears. For single stocks it does
   not: half of their extreme days are company news on calm market days.
4. How much of the total variance the most extreme days carry, real against
   Gaussian. A covariance is an average of squared returns, so a few wild
   days do most of the averaging.

    .venv/bin/python examples/sp500_fat_tails.py
"""

from __future__ import annotations

import os
from math import erf, sqrt
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "output" / "sp500_marchenko_pastur" / "prices.csv"
OUT = ROOT / "output" / "sp500_fat_tails"

REAL = "#2a78d6"
BELL = "#444444"
FIXED = "#1baf7a"
MUTED = "#52514e"
EPISODES = (
    ("Covid crash", "2020-02-19", "2020-03-23"),
    ("2022 bear market", "2022-01-03", "2022-10-12"),
)
ROLL = 21
FLOOR = 0.2  # rolling volatility is never allowed below this fraction of the long-run volatility


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICES, index_col=0, parse_dates=True).sort_index()
    returns = prices.pct_change(fill_method=None).iloc[1:].dropna(axis=1)
    returns = returns.loc[:, returns.std(ddof=1) > 0]
    n_days, n_stocks = returns.shape
    index = returns.mean(axis=1)

    # Every stock in its own long-run units, pooled: 1.35 million daily moves.
    z_stocks = (returns / returns.std(ddof=1)).to_numpy().ravel()
    z_stocks_roll = _standardised(returns).to_numpy().ravel()
    z_index = (index / index.std(ddof=1)).to_numpy()
    z_index_roll = _standardised(index).to_numpy()
    z_index_roll = z_index_roll[np.isfinite(z_index_roll)]
    z_index_roll = z_index_roll / z_index_roll.std()
    rng = np.random.default_rng(0)
    gauss = rng.standard_normal(z_stocks.size)

    stats = {
        "stocks": (_kurtosis(z_stocks), _beyond(z_stocks)),
        "stocks_roll": (_kurtosis(z_stocks_roll), _beyond(z_stocks_roll)),
        "index": (_kurtosis(z_index), _beyond(z_index)),
        "index_roll": (_kurtosis(z_index_roll), _beyond(z_index_roll)),
        "gauss": (_kurtosis(gauss), _beyond(gauss)),
    }
    expected5_stocks = z_stocks.size * 2 * (1 - _phi(5))
    expected5_index = z_index.size * 2 * (1 - _phi(5))
    share_real = _top_share(z_stocks, 0.05)
    share_gauss = _top_share(gauss, 0.05)
    worst = index.abs().idxmax()
    worst_share = float(index.loc[worst] ** 2 / (index**2).sum())
    worst_in_typical_days = float(index.loc[worst] ** 2 / np.median(index**2))
    # Split the extreme stock-days into those on market-wide panic days and those on ordinary days.
    big = (returns / returns.std(ddof=1)).abs() > 5
    panic_days = (index / index.std(ddof=1)).abs() > 3
    n_big = int(big.to_numpy().sum())
    n_big_panic = int(big.loc[panic_days].to_numpy().sum())
    share_company = 1 - n_big_panic / n_big

    summary = [
        f"Panel: {n_days} sessions × {n_stocks} tickers, {z_stocks.size:,} stock-days.",
        f"Single stocks pooled: excess kurtosis {stats['stocks'][0]:.1f}, {stats['stocks'][1]:,} stock-days beyond 5 sd (bell curve expects {expected5_stocks:.1f}).",
        f"  after dividing by the previous {ROLL} days' volatility: kurtosis {stats['stocks_roll'][0]:.1f}, {stats['stocks_roll'][1]:,} beyond 5 sd. The tail gets fatter, not thinner.",
        f"  of the {n_big:,} extreme stock-days, {n_big_panic:,} fall on market days beyond 3 sd and {n_big - n_big_panic:,} ({share_company:.0%}) on ordinary days: company news.",
        f"Equal-weight index: excess kurtosis {stats['index'][0]:.1f}, {stats['index'][1]} days beyond 5 sd (bell curve expects {expected5_index:.2f}).",
        f"  after dividing by the previous {ROLL} days' volatility: kurtosis {stats['index_roll'][0]:.1f}, {stats['index_roll'][1]} beyond 5 sd. The market-wide tail is volatility clustering.",
        f"Gaussian control: kurtosis {stats['gauss'][0]:.2f}, {stats['gauss'][1]} beyond 5 sd.",
        f"Share of total variance carried by the 5% most extreme days: real stocks {share_real:.0%}, Gaussian {share_gauss:.0%}.",
        f"Index worst day {worst.date()} ({index.loc[worst]:+.1%}) alone is {worst_share:.1%} of eleven years of variance, as much as {worst_in_typical_days:,.0f} typical days.",
    ]
    text = "\n".join(summary)
    print(text)
    (OUT / "summary.txt").write_text(text + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    _plot_histogram(axes[0, 0], z_stocks, stats["stocks"], expected5_stocks)
    _plot_clustering(axes[0, 1], index)
    _plot_fix(axes[1, 0], z_index, z_index_roll, stats, share_company)
    _plot_concentration(axes[1, 1], z_stocks, gauss, share_real, share_gauss, worst, index.loc[worst], worst_share, worst_in_typical_days)
    fig.suptitle(
        "Fat tails: real daily moves are not a bell curve, and why that hurts a covariance estimate",
        x=0.02, ha="left", fontsize=13, fontweight="medium",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fat_tails.png")
    plt.close(fig)
    print(f"Wrote {OUT}")


def _standardised(data):
    """Each day's return divided by the volatility of the previous ROLL sessions, floored."""
    rolling = data.rolling(ROLL).std(ddof=1).shift(1)
    if isinstance(data, pd.DataFrame):
        rolling = rolling.clip(lower=FLOOR * data.std(ddof=1), axis=1)
    else:
        rolling = rolling.clip(lower=FLOOR * data.std(ddof=1))
    return data / rolling


def _density(sample: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(-12, 12 + width, width)
    centers = (edges[:-1] + edges[1:]) / 2
    density, _ = np.histogram(sample[np.isfinite(sample)], bins=edges, density=True)
    return centers, density


def _bell(centers: np.ndarray) -> np.ndarray:
    return np.exp(-centers**2 / 2) / np.sqrt(2 * np.pi)


def _plot_histogram(ax, z, stat, expected5) -> None:
    centers, density = _density(z, 0.2)
    mask = density > 0
    ax.plot(centers[mask], density[mask], color=REAL, linewidth=2.0, marker="o", markersize=3.5, label="real daily moves, all 456 stocks pooled")
    ax.plot(centers, _bell(centers), color=BELL, linewidth=1.6, linestyle="--", label="bell curve with the same width")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1)
    ax.set_xlim(-12, 12)
    ax.set_xlabel("daily move, in standard deviations of that stock")
    ax.set_ylabel("how often, log scale")
    ax.set_title("1. The tails are fat: big days happen far more often than the bell curve says")
    ax.axvspan(5, 12, color="#000000", alpha=0.05, lw=0)
    ax.axvspan(-12, -5, color="#000000", alpha=0.05, lw=0)
    ax.text(5.3, 3e-2, f"beyond 5 standard deviations\nreal: {stat[1]:,} stock-days\nbell curve: {expected5:.1f} expected", fontsize=9.5, color=MUTED, va="top")
    ax.text(-11.5, 3e-2, f"excess kurtosis\nreal {stat[0]:.1f}\nbell curve 0", fontsize=9.5, color=MUTED, va="top")
    ax.legend(loc="lower center", fontsize=9.5)


def _plot_clustering(ax, index: pd.Series) -> None:
    sd = index.rolling(ROLL).std(ddof=1)
    ax.plot(index.index, index.to_numpy(), color=REAL, linewidth=0.7, alpha=0.9, label="equal-weight index, daily return")
    ax.fill_between(index.index, -2 * sd, 2 * sd, color=BELL, alpha=0.18, lw=0, label=f"± 2 × volatility of the previous {ROLL} days")
    for label, start, end in EPISODES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#000000", alpha=0.06, lw=0)
        ax.text(pd.Timestamp(start), 0.97, f" {label}", transform=ax.get_xaxis_transform(), fontsize=9, color="#555555", va="top")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("daily return")
    ax.set_title("2. Where the market's tail comes from: calm years and stormy months, not one volatility")
    ax.legend(loc="lower left", fontsize=9.5)


def _plot_fix(ax, z_index, z_index_roll, stats, share_company) -> None:
    centers, raw = _density(z_index, 0.5)
    _, fixed = _density(z_index_roll, 0.5)
    ax.plot(centers[raw > 0], raw[raw > 0], color=REAL, linestyle="", marker="o", markersize=5.5, label="index, long-run units")
    ax.plot(centers[fixed > 0], fixed[fixed > 0], color=FIXED, linestyle="", marker="D", markersize=4.5, label=f"index ÷ previous {ROLL}-day volatility")
    ax.plot(centers, _bell(centers), color=BELL, linewidth=1.6, linestyle="--", label="bell curve")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(-12, 12)
    ax.set_xlabel("daily move of the index, in standard deviations (each dot is one histogram bin)")
    ax.set_ylabel("how often, log scale")
    ax.set_title("3. For the market the fix works. For single stocks it does not: their extremes are company news")
    ax.text(
        -11.5, 2.5e-4,
        f"index, excess kurtosis\nlong-run units {stats['index'][0]:.1f}\nown-period units {stats['index_roll'][0]:.1f}\nbell curve 0",
        fontsize=9.5, color=MUTED, va="top",
    )
    ax.text(
        5.6, 2.5e-4,
        f"single stocks, same treatment\n{stats['stocks'][0]:.1f} → {stats['stocks_roll'][0]:.1f}: the tail gets fatter.\n"
        f"{share_company:.0%} of their 5-sd days fall on\nordinary market days.",
        fontsize=9.5, color=MUTED, va="top",
    )
    ax.legend(loc="upper left", fontsize=9)


def _plot_concentration(ax, z, gauss, share_real, share_gauss, worst, worst_return, worst_share, worst_in_typical_days) -> None:
    grid = np.linspace(0.0, 1.0, 401)
    for sample, color, style, label in ((z, REAL, "-", "real daily moves, all stocks"), (gauss, BELL, "--", "bell curve")):
        sq = np.sort(sample**2)[::-1]
        cum = np.cumsum(sq) / sq.sum()
        idx = np.clip((grid * sq.size).astype(int) - 1, 0, sq.size - 1)
        y = np.where(grid > 0, cum[idx], 0.0)
        ax.plot(grid * 100, y * 100, color=color, linewidth=2.0, linestyle=style, label=label)
    ax.axvline(5, color="#999999", linewidth=1.0, linestyle=":")
    ax.annotate(
        f"the 5% most extreme days carry\n{share_real:.0%} of the variance for real stocks,\n{share_gauss:.0%} for a bell curve",
        xy=(5, share_real * 100), xytext=(14, 36), fontsize=9.5, color=MUTED,
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
    )
    ax.text(
        0.98, 0.04,
        f"Equal-weight index: {worst.date()} ({worst_return:+.1%}) alone is {worst_share:.1%} of eleven years\nof variance, as much as {worst_in_typical_days:,.0f} typical days.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=MUTED,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("share of days, most extreme first")
    ax.set_ylabel("share of total variance they carry")
    ax.set_title("4. Why it hurts: a covariance is an average of squared moves, and a few days dominate it")
    ax.legend(loc="center right", fontsize=9.5)


def _kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    return float(np.mean(x**4) / np.mean(x**2) ** 2 - 3.0)


def _beyond(x: np.ndarray, k: float = 5.0) -> int:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return int(np.sum(np.abs(x / x.std()) > k))


def _phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _top_share(x: np.ndarray, frac: float) -> float:
    sq = np.sort(x**2)[::-1]
    k = max(1, int(frac * sq.size))
    return float(sq[:k].sum() / sq.sum())


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e8e8e8",
            "grid.linewidth": 0.8,
            "axes.titlelocation": "left",
            "axes.titleweight": "medium",
            "axes.titlesize": 10.5,
            "font.size": 10.5,
            "legend.frameon": False,
        }
    )


if __name__ == "__main__":
    main()
