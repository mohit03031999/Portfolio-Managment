"""Roll minimum-variance portfolios across the S&P 500, raw versus cleaned covariance.

Reuses the adjusted closes cached by ``examples/sp500_marchenko_pastur.py``
(run that script once first, or pass ``--prices``). Daily simple returns keep
their gaps, so a name joins a decision only once it has a complete
estimation window.

For each window length the script rebalances every 21 sessions into
unconstrained minimum-variance weights built from four covariance
estimators, plus an equal-weight control, and records predicted versus
realised volatility, turnover, and concentration. Tables and plots land in
``output/sp500_minvar_backtest/T<window>/`` with a cross-window comparison
one level up.

    python examples/sp500_minvar_backtest.py
    python examples/sp500_minvar_backtest.py --windows 500 1000 --holding 21
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter, PercentFormatter
import numpy as np
import pandas as pd

from market_analyzer.backtest import BacktestConfig, BacktestResult, run_backtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "sp500_minvar_backtest"
PRICES = ROOT / "output" / "sp500_marchenko_pastur" / "prices.csv"
WINDOWS = (500, 1000)
HOLDING = 21

LABELS = {
    "minvar_sample": "Sample covariance",
    "minvar_clipped": "MP clipped, fitted edge",
    "minvar_clipped_unit": "MP clipped, σ² = 1 edge",
    "minvar_clipped_market": "MP clipped, market edge",
    "minvar_diagonal": "Diagonal",
    "minvar_ledoit_wolf": "Ledoit-Wolf",
    "equal_weight": "Equal weight",
}
# Categorical slots in fixed order, one per strategy. Color follows the strategy everywhere.
COLORS = {
    "minvar_sample": "#2a78d6",
    "minvar_clipped": "#eb6834",
    "minvar_clipped_unit": "#1baf7a",
    "minvar_diagonal": "#eda100",
    "minvar_ledoit_wolf": "#e87ba4",
    "equal_weight": "#008300",
    "minvar_clipped_market": "#6250d6",
}
SHORT = {
    "minvar_sample": "Sample\ncovariance",
    "minvar_clipped": "MP clipped\nfitted edge",
    "minvar_clipped_unit": "MP clipped\nσ² = 1 edge",
    "minvar_clipped_market": "MP clipped\nmarket edge",
    "minvar_diagonal": "Diagonal",
    "minvar_ledoit_wolf": "Ledoit-\nWolf",
    "equal_weight": "Equal\nweight",
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
EPISODES = (
    ("Covid crash", "2020-02-19", "2020-03-23"),
    ("2022 bear market", "2022-01-03", "2022-10-12"),
)
COMPARISON = (
    ("ann_volatility", "Realised volatility, whole track", True),
    ("median_vol_ratio", "Median realised / predicted volatility", False),
    ("mean_turnover", "Turnover per rebalance", False),
    ("mean_gross_leverage", "Gross leverage", False),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", type=Path, default=PRICES, help="Adjusted-close CSV, dates down, tickers across.")
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS), help="Estimation windows in sessions.")
    parser.add_argument("--holding", type=int, default=HOLDING, help="Holding period in sessions. Default: 21.")
    parser.add_argument("--cost-bps", type=float, default=0.0, help="Cost per unit turnover in basis points. Default: 0.")
    parser.add_argument("--replot", action="store_true", help="Redraw the plots from saved results without rerunning.")
    args = parser.parse_args(argv)

    if not args.replot and not args.prices.is_file():
        print(
            f"No price file at {args.prices}. Run examples/sp500_marchenko_pastur.py once to cache "
            "the S&P 500 closes, or pass --prices.",
            file=sys.stderr,
        )
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    _style()

    returns = None
    if not args.replot:
        prices = pd.read_csv(args.prices, index_col=0, parse_dates=True).sort_index()
        returns = prices.pct_change(fill_method=None).iloc[1:]
        returns = returns.dropna(axis=1, how="all")
        print(
            f"Returns: {returns.shape[0]} sessions × {returns.shape[1]} names, "
            f"{returns.index[0].date()} → {returns.index[-1].date()}"
        )

    results: dict[int, BacktestResult] = {}
    for window in args.windows:
        folder = OUT / f"T{window}"
        if args.replot:
            result = BacktestResult.load(folder)
        else:
            config = BacktestConfig(window=window, holding=args.holding, cost_bps=args.cost_bps)
            result = run_backtest(returns, config)
            result.save(folder)
        print()
        print(result.report())
        _plot_growth(result, folder)
        _plot_rolling_volatility(result, folder)
        _plot_predicted_vs_realised(result, folder)
        _plot_vol_ratio(result, folder)
        _plot_turnover_and_leverage(result, folder)
        _plot_signal_modes(result, folder)
        results[window] = result

    summaries = {window: result.summary() for window, result in results.items()}
    pd.concat(summaries, names=["window", "strategy"]).to_csv(OUT / "summary_all_windows.csv")
    if len(results) > 1:
        # Longer windows start later, so compare every run over the dates they all cover.
        start = max(result.periods["hold_start"].min() for result in results.values())
        end = min(result.periods["hold_end"].max() for result in results.values())
        common = {window: result.restrict(start, end).summary() for window, result in results.items()}
        pd.concat(common, names=["window", "strategy"]).to_csv(OUT / "summary_common_period.csv")
        _plot_comparison(common, start, end)
        print()
        print(f"Common period {start.date()} to {end.date()}:")
        print(
            pd.concat(common, names=["window", "strategy"])
            .loc[:, ["n_periods", "ann_volatility", "median_vol_ratio", "mean_turnover", "mean_gross_leverage"]]
            .to_string(float_format=lambda value: f"{value:.4f}")
        )
    print()
    print(f"Wrote {OUT}")
    return 0


# ----------------------------------------------------------------------------- plots


def _plot_growth(result: BacktestResult, folder: Path) -> None:
    growth = (1.0 + result.daily_returns).cumprod()
    fig, axis = plt.subplots(figsize=(11, 5.2))
    for name in result.strategies:
        axis.plot(growth.index, growth[name], color=COLORS[name], linewidth=1.8, label=LABELS[name])
    axis.set_yscale("log")
    _shade_episodes(axis, growth.index)
    _label_ends(axis, growth, fmt=lambda value: f"{value:.2f}×")
    axis.set_ylabel("Growth of 1 (log scale)")
    axis.set_title(_title(result, "Growth of 1, weights rebalanced every hold"))
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(folder / "growth.png", dpi=160)
    plt.close(fig)


def _plot_rolling_volatility(result: BacktestResult, folder: Path) -> None:
    span = 63
    ppy = result.config.periods_per_year
    rolling = result.daily_returns.rolling(span).std(ddof=1) * np.sqrt(ppy)
    fig, axis = plt.subplots(figsize=(11, 5.2))
    for name in result.strategies:
        axis.plot(rolling.index, rolling[name], color=COLORS[name], linewidth=1.8, label=LABELS[name])
    _shade_episodes(axis, rolling.index)
    _label_ends(axis, rolling.dropna(how="all"), fmt=lambda value: f"{value:.0%}")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_ylabel(f"Realised volatility, trailing {span} sessions, annualised")
    axis.set_title(_title(result, "Realised volatility through time"))
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(folder / "rolling_volatility.png", dpi=160)
    plt.close(fig)


def _plot_predicted_vs_realised(result: BacktestResult, folder: Path) -> None:
    periods = result.periods
    names = result.strategies
    values = periods[["predicted_vol", "realised_vol"]].to_numpy(dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    low, high = float(positive.min()) / 1.4, float(positive.max()) * 1.4
    ticks = [tick for tick in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0) if low <= tick <= high]
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 4.1), sharex=True, sharey=True)
    for axis, name in zip(np.atleast_1d(axes), names):
        rows = periods.loc[periods["strategy"] == name]
        axis.plot([low, high], [low, high], color=MUTED, linewidth=1.0, linestyle="--")
        axis.scatter(
            rows["predicted_vol"],
            rows["realised_vol"],
            s=22,
            color=COLORS[name],
            edgecolors="white",
            linewidths=0.8,
            alpha=0.9,
        )
        median = float(rows["vol_ratio"].median())
        axis.set_title(f"{LABELS[name]}\nmedian ratio {median:.2f}", fontsize=9)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal")
        for scale in (axis.xaxis, axis.yaxis):
            scale.set_minor_formatter(NullFormatter())
        axis.set_xticks(ticks)
        axis.set_xticklabels([f"{tick:.0%}" for tick in ticks], fontsize=8)
        axis.set_yticks(ticks)
        axis.set_yticklabels([f"{tick:.0%}" for tick in ticks], fontsize=8)
        axis.set_xlabel("Predicted volatility")
    np.atleast_1d(axes)[0].set_ylabel("Realised volatility")
    fig.suptitle(
        _title(result, "Each point is one holding period, log scales. Honest models sit on the dashed line."),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(folder / "predicted_vs_realised.png", dpi=160)
    plt.close(fig)


def _plot_vol_ratio(result: BacktestResult, folder: Path) -> None:
    ratios = result.periods.pivot(index="hold_start", columns="strategy", values="vol_ratio")[result.strategies]
    fig, axis = plt.subplots(figsize=(11, 5.2))
    axis.axhline(1.0, color=MUTED, linewidth=1.0, linestyle="--")
    for name in result.strategies:
        axis.plot(ratios.index, ratios[name], color=COLORS[name], linewidth=1.8, label=LABELS[name])
    axis.set_yscale("log")
    _shade_episodes(axis, ratios.index)
    _label_ends(axis, ratios, fmt=lambda value: f"{value:.2f}")
    axis.set_ylabel("Realised / predicted volatility per hold (log scale)")
    axis.set_title(_title(result, "How honest was each risk model? Dashed line: prediction matched reality"))
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(folder / "vol_ratio.png", dpi=160)
    plt.close(fig)


def _plot_turnover_and_leverage(result: BacktestResult, folder: Path) -> None:
    periods = result.periods
    steady = periods.loc[~periods["first_period"]]
    turnover = steady.pivot(index="hold_start", columns="strategy", values="turnover")[result.strategies]
    leverage = periods.pivot(index="hold_start", columns="strategy", values="gross_leverage")[result.strategies]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True)
    for name in result.strategies:
        axes[0].plot(turnover.index, turnover[name], color=COLORS[name], linewidth=1.8, label=LABELS[name])
        axes[1].plot(leverage.index, leverage[name], color=COLORS[name], linewidth=1.8)
    axes[0].set_yscale("symlog", linthresh=0.01)
    axes[0].set_ylim(bottom=0)
    axes[1].set_yscale("log")
    axes[1].axhline(1.0, color=MUTED, linewidth=1.0, linestyle="--")
    for axis, frame in zip(axes, (turnover, leverage)):
        _shade_episodes(axis, frame.index)
        _label_ends(axis, frame, fmt=lambda value: f"{value:.2f}")
    axes[0].set_ylabel("Turnover per rebalance (log scale above 0.01)")
    axes[1].set_ylabel("Gross leverage, sum of |w| (log scale)")
    axes[0].set_title(_title(result, "Weight churn and how far the book is stretched"))
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(folder / "turnover_and_leverage.png", dpi=160)
    plt.close(fig)


def _plot_signal_modes(result: BacktestResult, folder: Path) -> None:
    if "n_signal" not in result.periods:
        return
    rows = result.periods.dropna(subset=["n_signal"])
    names = [name for name in result.strategies if name in set(rows["strategy"])]
    if not names:
        return
    counts = rows.pivot(index="hold_start", columns="strategy", values="n_signal")[names]
    fig, axis = plt.subplots(figsize=(11, 4.2))
    for name in names:
        axis.plot(counts.index, counts[name], color=COLORS[name], linewidth=1.8, label=LABELS[name])
    _shade_episodes(axis, counts.index)
    _label_ends(axis, counts, fmt=lambda value: f"{value:.0f}")
    axis.set_yscale("log")
    axis.set_ylabel("Eigenvalues above the noise edge (log scale)")
    q = float(rows["q"].median())
    n_assets = int(rows["n_assets"].median())
    axis.set_title(_title(result, f"Modes kept by each clipping rule, out of about {n_assets} names, median q = {q:.2f}"))
    if len(names) > 1:
        axis.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(folder / "signal_modes.png", dpi=160)
    plt.close(fig)


def _plot_comparison(summaries: dict[int, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> None:
    windows = sorted(summaries)
    names = list(next(iter(summaries.values())).index)
    hatches = ["", "//", "xx", ".."]
    width = 0.8 / len(windows)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    for axis, (column, title, percent) in zip(axes.ravel(), COMPARISON):
        for position, name in enumerate(names):
            for offset, window in enumerate(windows):
                value = float(summaries[window].loc[name, column])
                x = position + (offset - (len(windows) - 1) / 2) * width
                axis.bar(
                    x,
                    value,
                    width=width * 0.92,
                    color=COLORS[name],
                    hatch=hatches[offset % len(hatches)],
                    edgecolor="white",
                    linewidth=0.8,
                )
                text = f"{value:.1%}" if percent else f"{value:.2f}"
                axis.text(x, value, text, ha="center", va="bottom", fontsize=7, color=TEXT)
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels([SHORT.get(name, LABELS[name]) for name in names], fontsize=8)
        axis.set_title(title, fontsize=10)
        if percent:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        if column == "median_vol_ratio":
            axis.axhline(1.0, color=MUTED, linewidth=1.0, linestyle="--")
        top = max(float(summaries[window].loc[name, column]) for name in names for window in windows)
        axis.set_ylim(0, top * 1.22)
    handles = [
        Patch(facecolor="#d9d8d3", edgecolor="white", hatch=hatches[offset % len(hatches)], label=f"T = {window}")
        for offset, window in enumerate(windows)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(windows), frameon=False, fontsize=9)
    fig.suptitle(
        "Same strategies, different estimation windows, scored over the dates every run covers: "
        f"{start.date()} to {end.date()}. Hatching marks the window length.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "summary_comparison.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- helpers


def _title(result: BacktestResult, text: str) -> str:
    config = result.config
    n_periods = result.periods["hold_start"].nunique()
    first = result.periods["hold_start"].min().date()
    last = result.periods["hold_end"].max().date()
    return f"{text}\nT = {config.window}, hold {config.holding} sessions, {n_periods} periods, {first} to {last}"


def _shade_episodes(axis, index) -> None:
    start, end = pd.Timestamp(index.min()), pd.Timestamp(index.max())
    for label, begin, finish in EPISODES:
        lo, hi = pd.Timestamp(begin), pd.Timestamp(finish)
        if hi < start or lo > end:
            continue
        axis.axvspan(max(lo, start), min(hi, end), color="#efeeea", zorder=0)
        axis.text(
            max(lo, start),
            0.985,
            f" {label}",
            transform=axis.get_xaxis_transform(),
            fontsize=7,
            color=MUTED,
            va="top",
        )


def _label_ends(axis, frame: pd.DataFrame, fmt, min_gap_px: float = 11.0) -> None:
    """Direct labels at the last point of each line, pushed apart so they never overlap."""
    frame = frame.dropna(how="all")
    if frame.empty:
        return
    last_x = frame.index[-1]
    items = []
    for name in frame.columns:
        series = frame[name].dropna()
        if series.empty:
            continue
        items.append((name, series.index[-1], float(series.iloc[-1])))
    if not items:
        return
    axis.figure.canvas.draw()
    transform = axis.transData
    inverse = transform.inverted()
    x_num = matplotlib.dates.date2num(last_x) if isinstance(last_x, pd.Timestamp) else float(last_x)
    display = [(name, y, transform.transform((x_num, y))[1]) for name, _, y in items]
    display.sort(key=lambda item: item[2])
    pushed = []
    for name, y, pixel in display:
        if pushed and pixel - pushed[-1][2] < min_gap_px:
            pixel = pushed[-1][2] + min_gap_px
        pushed.append((name, y, pixel))
    for name, y, pixel in pushed:
        y_text = inverse.transform((transform.transform((x_num, y))[0], pixel))[1]
        axis.annotate(
            f"{LABELS[name]}  {fmt(y)}",
            xy=(last_x, y),
            xytext=(last_x, y_text),
            textcoords="data",
            xycoords="data",
            fontsize=7.5,
            color=MUTED,
            va="center",
            ha="left",
            annotation_clip=False,
        )
    # Leave room on the right for the labels. Computed from the data, so a
    # second call on a shared axis does not stretch it again.
    first = frame.index[0]
    left = matplotlib.dates.date2num(first) if isinstance(first, pd.Timestamp) else float(first)
    axis.set_xlim(left, x_num + (x_num - left) * 0.22)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "hatch.linewidth": 0.6,
        }
    )


if __name__ == "__main__":
    sys.exit(main())
