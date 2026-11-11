"""Run the backtest twice: on the S&P 500 panel and on Gaussian returns from one fixed covariance.

The rolling backtest reports, for every model, the ratio of realised to
predicted volatility. A ratio above one means the model promised less risk
than it delivered. That gap has three sources that the real data cannot
separate:

* model bias: the estimator would be wrong even with unlimited data in a
  market that never changes (a diagonal matrix in a correlated market);
* estimation noise: 500 sessions is a finite sample of 456 names, so the
  matrix, and the weights built from it, carry sampling error;
* market change: the covariance of the next 21 sessions is not the
  covariance of the previous 500, because the market itself moved.

Gaussian returns drawn from one fixed covariance matrix remove the third
source by construction. Running the same models on that panel measures bias
plus noise alone, and the ratio of the real result to the Gaussian result is
what the market's own movement adds. A seventh strategy uses the fixed
matrix itself at every decision date: on the Gaussian panel it is the
oracle that knows the truth and should score one, on the real panel it is
the simulation's truth applied to the market that produced it, with
look-ahead, and its ratio shows how far the real market strays from its own
eleven-year average covariance.

The simulated panel is the one from ``sp500_eigenvector_stability.py``: the
same tickers, the same dates, the same seed.

    .venv/bin/python examples/sp500_simulated_backtest.py
    .venv/bin/python examples/sp500_simulated_backtest.py --replot   # figures only, from the saved runs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# A 456 × 456 eigendecomposition takes 60 ms on one BLAS thread and several
# seconds on an oversubscribed pool, so pin the pool before numpy loads.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from market_analyzer.backtest import DEFAULT_STRATEGIES, BacktestConfig, BacktestResult, run_backtest  # noqa: E402
from market_analyzer.matrices import CorrelationModel  # noqa: E402
from market_analyzer.optimizer import CovarianceEstimate, Estimator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "output" / "sp500_marchenko_pastur" / "prices.csv"
OUT = ROOT / "output" / "sp500_simulated_backtest"

WINDOWS = (500, 1000)
HOLDING = 21
PANELS = ("real", "gaussian")
FIXED = "minvar_fixed"
STRATEGIES = DEFAULT_STRATEGIES + (FIXED,)
# Top-to-bottom order in the dot plots and left-to-right order in the small multiples.
ORDER = (
    "minvar_sample",
    "minvar_ledoit_wolf",
    "minvar_clipped",
    "minvar_clipped_unit",
    "minvar_diagonal",
    "equal_weight",
    FIXED,
)
LABELS = {
    "minvar_sample": "Sample covariance",
    "minvar_clipped": "MP clipped, fitted edge",
    "minvar_clipped_unit": "MP clipped, σ² = 1 edge",
    "minvar_diagonal": "Diagonal",
    "minvar_ledoit_wolf": "Ledoit-Wolf",
    "equal_weight": "Equal weight",
    FIXED: "Fixed full-sample covariance",
}
# Same categorical slots as sp500_minvar_backtest.py, so a strategy keeps its colour across figures.
COLORS = {
    "minvar_sample": "#2a78d6",
    "minvar_clipped": "#eb6834",
    "minvar_clipped_unit": "#1baf7a",
    "minvar_diagonal": "#eda100",
    "minvar_ledoit_wolf": "#e87ba4",
    "equal_weight": "#008300",
    FIXED: "#6250d6",
}
EPISODES = (
    ("Covid crash", "2020-02-19", "2020-03-23"),
    ("2022 bear market", "2022-01-03", "2022-10-12"),
)
EPISODE_LEGEND = "shaded: Covid crash (Feb–Mar 2020), 2022 bear market (Jan–Oct 2022)"
REAL_LABEL = "S&P 500 returns"
SIM_LABEL = "Gaussian returns from one fixed covariance"
MUTED = "#52514e"
LOG_TICKS = (0.3, 0.5, 1, 2, 5, 10, 20, 50, 100)
SYMLOG_TICKS = (0, 0.01, 0.1, 1, 10, 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", type=Path, default=PRICES)
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS))
    parser.add_argument("--holding", type=int, default=HOLDING)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replot", action="store_true", help="Redraw the figures from the saved runs instead of running the backtests again.")
    args = parser.parse_args()

    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        results = {
            (panel_name, window): BacktestResult.load(args.out / f"{panel_name}_T{window}")
            for window in args.windows
            for panel_name in PANELS
        }
        decomp = pd.read_csv(args.out / "decomposition.csv")
    else:
        results, decomp = run_all(args)

    primary = args.windows[0]
    plot_dumbbells(decomp, args.out / "honesty_decomposition.png")
    plot_series(
        results[("real", primary)],
        results[("gaussian", primary)],
        "vol_ratio",
        f"Realised / predicted volatility per 21-session hold, T = {primary}. Solid: {REAL_LABEL}. Dashed: {SIM_LABEL}.",
        "realised / predicted volatility",
        args.out / "vol_ratio_over_time.png",
        reference=1.0,
    )
    plot_series(
        results[("real", primary)],
        results[("gaussian", primary)],
        "turnover",
        f"Turnover per rebalance, T = {primary}. Solid: {REAL_LABEL}. Dashed: {SIM_LABEL}.",
        "sum of |Δw| at the rebalance",
        args.out / "turnover_over_time.png",
        steady_only=True,
        yscale="symlog",
    )
    print(f"Wrote {args.out}")


def run_all(args: argparse.Namespace) -> tuple[dict[tuple[str, int], BacktestResult], pd.DataFrame]:
    """Run every window on both panels, save each run, and return the runs plus the decomposition table."""
    returns = load_returns(args.prices)
    fake, truth = simulate(returns, args.seed)
    n_dates, n_assets = returns.shape
    lines = [
        f"Panel: {n_dates} sessions × {n_assets} tickers with a complete history, "
        f"{returns.index[0].date()} → {returns.index[-1].date()}",
        f"Simulation truth: cleaned covariance of the full real panel, edge rule {truth.edge_rule}, "
        f"{truth.n_signal} signal modes, q = {truth.q:.3f}. Gaussian returns drawn with seed {args.seed}.",
        f"Strategy {FIXED} uses that matrix at every decision date: the oracle on the Gaussian panel, "
        "a look-ahead reference on the real panel.",
        "",
    ]
    print("\n".join(lines))

    estimators: dict[str, Estimator] = {"fixed": fixed_estimator(truth.cleaned_covariance)}
    results: dict[tuple[str, int], BacktestResult] = {}
    rows = []
    for window in args.windows:
        config = BacktestConfig(window=window, holding=args.holding, strategies=STRATEGIES)
        for panel_name, panel in (("real", returns), ("gaussian", fake)):
            result = run_backtest(panel, config, estimators)
            result.save(args.out / f"{panel_name}_T{window}")
            results[(panel_name, window)] = result
            header = f"=== {REAL_LABEL if panel_name == 'real' else SIM_LABEL}, T = {window} ==="
            print(header)
            print(result.report())
            print()
            lines.extend([header, result.report(), ""])
        table = decomposition(results[("real", window)], results[("gaussian", window)], window, n_assets)
        rows.append(table)
        text = describe(table, window, n_assets)
        print(text)
        print()
        lines.extend([text, ""])

    decomp = pd.concat(rows, ignore_index=True)
    decomp.to_csv(args.out / "decomposition.csv", index=False)
    (args.out / "summary.txt").write_text("\n".join(lines))
    return results, decomp


def load_returns(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    returns = prices.pct_change(fill_method=None).iloc[1:]
    complete = returns.dropna(axis=1)
    complete = complete.loc[:, complete.std(ddof=1) > 0]
    return complete


def simulate(returns: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, CorrelationModel]:
    """Zero-mean Gaussian returns with the cleaned covariance of the full panel as the truth."""
    truth = CorrelationModel.from_returns(returns)
    chol = np.linalg.cholesky(truth.cleaned_covariance.to_numpy(dtype=float))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=returns.shape)
    fake = pd.DataFrame(noise @ chol.T, index=returns.index, columns=returns.columns)
    return fake, truth


def fixed_estimator(covariance: pd.DataFrame) -> Estimator:
    """An estimator that ignores the window and returns one fixed matrix for the window's tickers."""
    matrix = covariance.copy()
    matrix.index = [str(ticker) for ticker in matrix.index]
    matrix.columns = [str(ticker) for ticker in matrix.columns]

    def estimate(returns: pd.DataFrame) -> CovarianceEstimate:
        tickers = [str(ticker) for ticker in returns.columns]
        return CovarianceEstimate(matrix.loc[tickers, tickers])

    return estimate


def decomposition(real: BacktestResult, sim: BacktestResult, window: int, n_assets: int) -> pd.DataFrame:
    """One row per strategy: the real and Gaussian results side by side and their ratio."""
    a = real.summary()
    b = sim.summary()
    names = list(ORDER)
    table = pd.DataFrame(
        {
            "window": window,
            "strategy": names,
            "median_vol_ratio_real": a.loc[names, "median_vol_ratio"].to_numpy(),
            "median_vol_ratio_gaussian": b.loc[names, "median_vol_ratio"].to_numpy(),
            "ann_volatility_real": a.loc[names, "ann_volatility"].to_numpy(),
            "ann_volatility_gaussian": b.loc[names, "ann_volatility"].to_numpy(),
            "mean_turnover_real": a.loc[names, "mean_turnover"].to_numpy(),
            "mean_turnover_gaussian": b.loc[names, "mean_turnover"].to_numpy(),
            "mean_gross_leverage_real": a.loc[names, "mean_gross_leverage"].to_numpy(),
            "mean_gross_leverage_gaussian": b.loc[names, "mean_gross_leverage"].to_numpy(),
        }
    )
    table["market_change_factor"] = table["median_vol_ratio_real"] / table["median_vol_ratio_gaussian"]
    table["q"] = n_assets / window
    return table


def describe(table: pd.DataFrame, window: int, n_assets: int) -> str:
    q = n_assets / window
    show = table.set_index("strategy").loc[
        list(ORDER),
        [
            "median_vol_ratio_gaussian",
            "median_vol_ratio_real",
            "market_change_factor",
            "mean_turnover_gaussian",
            "mean_turnover_real",
        ],
    ]
    show.index = [LABELS[name] for name in show.index]
    show.columns = ["ratio Gaussian", "ratio real", "real / Gaussian", "turnover Gaussian", "turnover real"]
    lines = [
        f"--- Decomposition of the honesty ratio, T = {window}, q = N/T = {q:.3f} ---",
        "ratio Gaussian = model bias plus estimation noise (the market never changes in that panel).",
        "real / Gaussian = what the real market's own movement adds on top.",
        f"Random-matrix prediction for the sample covariance in a stationary Gaussian market: 1 / (1 - q) = {1.0 / (1.0 - q):.2f}.",
        "",
        show.to_string(float_format=lambda value: f"{value:.3f}"),
    ]
    return "\n".join(lines)


def _series(result: BacktestResult, name: str, column: str, steady_only: bool) -> pd.Series:
    rows = result.periods[result.periods["strategy"] == name]
    if steady_only:
        rows = rows[~rows["first_period"].astype(bool)]
    return pd.Series(rows[column].to_numpy(dtype=float), index=pd.DatetimeIndex(rows["hold_start"]))


def plot_dumbbells(decomp: pd.DataFrame, path: Path) -> None:
    """Hollow marker: Gaussian panel. Filled marker: real panel. The bar between them is the market's contribution."""
    metrics = (
        ("median_vol_ratio", "Median realised / predicted volatility", "log", 1.0),
        ("mean_turnover", "Turnover per rebalance", "symlog", None),
    )
    windows = sorted(decomp["window"].unique())
    fig, axes = plt.subplots(len(metrics), len(windows), figsize=(6.6 * len(windows), 3.6 * len(metrics)), squeeze=False)
    for row, (metric, title, scale, reference) in enumerate(metrics):
        for col, window in enumerate(windows):
            ax = axes[row, col]
            sub = decomp[decomp["window"] == window].set_index("strategy").reindex(list(ORDER))
            names = [name for name in ORDER if not (metric == "mean_turnover" and name in (FIXED, "equal_weight"))]
            values = []
            for i, name in enumerate(names):
                y = len(names) - 1 - i
                real = float(sub.loc[name, f"{metric}_real"])
                sim = float(sub.loc[name, f"{metric}_gaussian"])
                color = COLORS[name]
                ax.plot([sim, real], [y, y], color=color, linewidth=2.2, alpha=0.45, solid_capstyle="round")
                ax.scatter([sim], [y], s=64, facecolor="white", edgecolor=color, linewidth=2.0, zorder=3)
                ax.scatter([real], [y], s=64, facecolor=color, edgecolor="white", linewidth=1.2, zorder=4)
                low, high = sorted((sim, real))
                ax.annotate(_fmt(low), (low, y), xytext=(-10, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5, color=MUTED)
                ax.annotate(_fmt(high), (high, y), xytext=(10, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5, color=MUTED)
                values.extend([sim, real])
            if scale == "log":
                lo, hi = min(values) / 2.2, max(values) * 2.6
                ax.set_xscale("log")
                ax.set_xlim(lo, hi)
                _plain_ticks(ax.xaxis, [tick for tick in LOG_TICKS if lo < tick < hi])
            else:
                hi = max(values) * 3.0
                ax.set_xscale("symlog", linthresh=0.01, linscale=0.6)
                ax.set_xlim(-0.004, hi)
                _plain_ticks(ax.xaxis, [tick for tick in SYMLOG_TICKS if tick < hi])
            ax.minorticks_off()
            if reference is not None:
                ax.axvline(reference, color="#888888", linewidth=1.0, linestyle=":", zorder=1)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([LABELS[name] for name in reversed(names)], fontsize=9.5)
            ax.set_ylim(-0.7, len(names) - 0.3)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{title}, T = {window}")
            if metric == "mean_turnover" and col == 0:
                ax.text(
                    0.0, -0.16, "Equal weight and the fixed matrix never trade on this complete-history panel, so they are left out here.",
                    transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top",
                )
    handles = [
        Line2D([], [], marker="o", color="#333333", markerfacecolor="#333333", markeredgecolor="white", linestyle="", markersize=8, label=REAL_LABEL),
        Line2D([], [], marker="o", color="#333333", markerfacecolor="white", markeredgecolor="#333333", markeredgewidth=2, linestyle="", markersize=8, label=SIM_LABEL),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.915), fontsize=10)
    fig.suptitle(
        "Every model run on the real S&P 500 panel and on Gaussian returns from one fixed covariance.\n"
        "The bar between the two markers is what the market's own movement adds to each model's honesty gap.",
        x=0.02, ha="left", fontsize=12, fontweight="medium", y=0.995, linespacing=1.4,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    fig.savefig(path)
    plt.close(fig)


def plot_series(
    real: BacktestResult,
    sim: BacktestResult,
    column: str,
    title: str,
    ylabel: str,
    path: Path,
    reference: float | None = None,
    steady_only: bool = False,
    yscale: str = "log",
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.4), sharex=True, sharey=True)
    flat = axes.ravel()
    top = 0.0
    for i, (ax, name) in enumerate(zip(flat, ORDER)):
        r = _series(real, name, column, steady_only)
        s = _series(sim, name, column, steady_only)
        top = max(top, float(np.nanmax(r.to_numpy())), float(np.nanmax(s.to_numpy())))
        color = COLORS[name]
        ax.plot(s.index, s.to_numpy(), color=color, linewidth=1.3, linestyle="--", alpha=0.85)
        ax.plot(r.index, r.to_numpy(), color=color, linewidth=1.8)
        if reference is not None:
            ax.axhline(reference, color="#888888", linewidth=1.0, linestyle=":", zorder=1)
        ax.set_title(f"{LABELS[name]}\nmedian {_fmt(r.median())} real, {_fmt(s.median())} Gaussian", fontsize=10)
        _shade_episodes(ax)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if i % 4 == 0:
            ax.set_ylabel(ylabel)
    first = axes[0, 0]
    if yscale == "log":
        first.set_yscale("log")
        lo, hi = first.get_ylim()
        ticks = [tick for tick in LOG_TICKS if lo <= tick <= hi]
    else:
        first.set_yscale("symlog", linthresh=0.01, linscale=0.6)
        first.set_ylim(-0.003, top * 2.5)
        ticks = [tick for tick in SYMLOG_TICKS if tick < top * 2.5]
    for ax in flat[: len(ORDER)]:
        _plain_ticks(ax.yaxis, ticks)
        ax.minorticks_off()
    spare = flat[-1]
    spare.axis("off")
    handles = [
        Line2D([], [], color="#333333", linewidth=1.8, label=REAL_LABEL),
        Line2D([], [], color="#333333", linewidth=1.3, linestyle="--", label=SIM_LABEL),
    ]
    if reference is not None:
        handles.append(Line2D([], [], color="#888888", linewidth=1.0, linestyle=":", label="honest model: ratio 1"))
    handles.append(Patch(facecolor="#ebebeb", edgecolor="#d0d0d0", label=EPISODE_LEGEND))
    spare.legend(handles=handles, loc="center left", fontsize=9.5, frameon=False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, fontweight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def _plain_ticks(axis, ticks: list[float]) -> None:
    axis.set_ticks(ticks)
    axis.set_ticklabels([f"{tick:g}" for tick in ticks])


def _shade_episodes(ax) -> None:
    for _, start, end in EPISODES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#000000", alpha=0.06, lw=0)


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if abs(value) < 0.005:
        return "0"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}"


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
            "axes.titlesize": 11.5,
            "font.size": 10.5,
            "legend.frameon": False,
        }
    )


if __name__ == "__main__":
    main()
