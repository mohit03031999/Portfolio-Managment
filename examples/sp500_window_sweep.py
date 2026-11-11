"""Does a longer window keep helping on the real market the way it does in the Gaussian world?

``sp500_honesty_vs_window.py`` showed that in a market that never changes,
every extra day of history sharpens the covariance estimate and the
minimum-variance models become more honest and less risky without limit.
The real market is not like that: a day from six years ago may describe a
market that no longer exists. This script runs the rolling backtest at
several window lengths on the real S&P 500 panel and on the Gaussian panel
from one fixed covariance, then compares them over the same holding
periods, so the dates do not confound the window length.

Saved runs from ``sp500_simulated_backtest.py`` are reused when present.

    .venv/bin/python examples/sp500_window_sweep.py --panel real       # one panel, e.g. in parallel
    .venv/bin/python examples/sp500_window_sweep.py --panel gaussian
    .venv/bin/python examples/sp500_window_sweep.py --replot           # tables and figures from the saved runs
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
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from market_analyzer.backtest import DEFAULT_STRATEGIES, BacktestConfig, BacktestResult, run_backtest  # noqa: E402
from market_analyzer.matrices import CorrelationModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "output" / "sp500_marchenko_pastur" / "prices.csv"
SAVED = ROOT / "output" / "sp500_simulated_backtest"
OUT = ROOT / "output" / "sp500_window_sweep"

WINDOWS = (250, 500, 750, 1000, 1500, 2000)
HOLDING = 21
PANELS = ("real", "gaussian")
STRATEGIES = tuple(DEFAULT_STRATEGIES)
LABELS = {
    "minvar_sample": "Sample covariance",
    "minvar_clipped": "MP clipped, fitted edge",
    "minvar_clipped_unit": "MP clipped, σ² = 1 edge",
    "minvar_diagonal": "Diagonal",
    "minvar_ledoit_wolf": "Ledoit-Wolf",
    "equal_weight": "Equal weight",
}
# Same categorical slots as the other S&P figures, so a strategy keeps its colour.
COLORS = {
    "minvar_sample": "#2a78d6",
    "minvar_clipped": "#eb6834",
    "minvar_clipped_unit": "#1baf7a",
    "minvar_diagonal": "#eda100",
    "minvar_ledoit_wolf": "#e87ba4",
    "equal_weight": "#008300",
}
ORDER = ("minvar_sample", "minvar_ledoit_wolf", "minvar_clipped", "minvar_clipped_unit", "minvar_diagonal", "equal_weight")
REAL_LABEL = "S&P 500 returns"
SIM_LABEL = "Gaussian returns from one fixed covariance"
MUTED = "#52514e"
METRICS = (
    ("median_vol_ratio", "Honesty: median realised / predicted volatility", "ratio, log scale", "log", 1.0),
    ("ann_volatility", "Risk: realised volatility of the strategy, annualised", "annualised volatility", "linear", None),
    ("mean_turnover", "Trading: turnover per rebalance", "sum of |Δw|, log scale", "log", None),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", type=Path, default=PRICES)
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS))
    parser.add_argument("--holding", type=int, default=HOLDING)
    parser.add_argument("--panel", choices=("real", "gaussian", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replot", action="store_true", help="Only aggregate and draw from saved runs; run nothing.")
    args = parser.parse_args()

    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    panels = PANELS if args.panel == "both" else (args.panel,)
    if not args.replot:
        run_missing(args, panels)

    results = {}
    for panel in PANELS:
        for window in args.windows:
            folder = find_saved(panel, window, args.out)
            if folder is None:
                print(f"Missing {panel} T = {window}; run the other panel first, then --replot.")
                return
            results[(panel, window)] = BacktestResult.load(folder)

    common_start = max(result.periods["hold_start"].min() for result in results.values())
    agg = aggregate(results, common_start)
    agg.to_csv(args.out / "sweep_summary.csv", index=False)
    text = describe(agg, common_start, results)
    print(text)
    (args.out / "summary.txt").write_text(text + "\n")
    for metric, title, ylabel, yscale, reference in METRICS:
        plot_metric(agg, metric, title, ylabel, yscale, reference, common_start, args.out / f"{metric}_vs_window.png")
    print(f"Wrote {args.out}")


def run_missing(args: argparse.Namespace, panels: tuple[str, ...]) -> None:
    returns = load_returns(args.prices)
    data = {"real": returns}
    if "gaussian" in panels:
        data["gaussian"], _ = simulate(returns, args.seed)
    for panel in panels:
        for window in args.windows:
            if find_saved(panel, window, args.out) is not None:
                print(f"{panel} T = {window}: reusing saved run")
                continue
            config = BacktestConfig(window=window, holding=args.holding, strategies=STRATEGIES)
            result = run_backtest(data[panel], config)
            result.save(args.out / f"{panel}_T{window}")
            medians = result.summary()["median_vol_ratio"]
            print(f"{panel} T = {window}: {len(result.periods['hold_start'].unique())} periods, " + ", ".join(f"{LABELS[s].split(',')[0]} {medians[s]:.2f}" for s in ORDER))


def find_saved(panel: str, window: int, out: Path) -> Path | None:
    for base in (out, SAVED):
        folder = base / f"{panel}_T{window}"
        if (folder / "periods.csv").is_file():
            return folder
    return None


def load_returns(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    returns = prices.pct_change(fill_method=None).iloc[1:]
    complete = returns.dropna(axis=1)
    return complete.loc[:, complete.std(ddof=1) > 0]


def simulate(returns: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, CorrelationModel]:
    """Zero-mean Gaussian returns with the cleaned covariance of the full panel as the truth (same seed as the other studies)."""
    truth = CorrelationModel.from_returns(returns)
    chol = np.linalg.cholesky(truth.cleaned_covariance.to_numpy(dtype=float))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=returns.shape)
    return pd.DataFrame(noise @ chol.T, index=returns.index, columns=returns.columns), truth


def aggregate(results: dict[tuple[str, int], BacktestResult], common_start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for (panel, window), result in results.items():
        for scope, part in (("all", result), ("common", result.restrict(start=common_start))):
            summary = part.summary()
            for strategy in ORDER:
                if strategy not in summary.index:
                    continue
                row = summary.loc[strategy]
                rows.append(
                    {
                        "panel": panel,
                        "window": window,
                        "strategy": strategy,
                        "scope": scope,
                        "n_periods": int(row["n_periods"]),
                        "first_hold": part.periods["hold_start"].min(),
                        "median_vol_ratio": float(row["median_vol_ratio"]),
                        "ann_volatility": float(row["ann_volatility"]),
                        "mean_turnover": float(row["mean_turnover"]),
                        "mean_gross_leverage": float(row["mean_gross_leverage"]),
                    }
                )
    return pd.DataFrame(rows)


def _pivot(agg: pd.DataFrame, panel: str, scope: str, metric: str) -> pd.DataFrame:
    sub = agg[(agg["panel"] == panel) & (agg["scope"] == scope)]
    wide = sub.pivot(index="window", columns="strategy", values=metric).reindex(columns=list(ORDER))
    wide.columns = [LABELS[s] for s in wide.columns]
    wide.index.name = "T"
    return wide


def describe(agg: pd.DataFrame, common_start: pd.Timestamp, results: dict[tuple[str, int], BacktestResult]) -> str:
    common = agg[agg["scope"] == "common"]
    n_common = int(common["n_periods"].max())
    last = max(result.periods["hold_end"].max() for result in results.values())
    fmt3 = lambda v: f"{v:.3f}"  # noqa: E731
    fmt_pct = lambda v: f"{v:.2%}"  # noqa: E731
    lines = [
        f"Common evaluation period: {n_common} holding periods from {common_start.date()} to {last.date()}, "
        f"the span every window length covers. Each window is judged only on these dates.",
        "",
    ]
    for panel, label in (("real", REAL_LABEL), ("gaussian", SIM_LABEL)):
        lines += [f"=== {label}, common period ===", ""]
        lines += ["Median realised / predicted volatility (1 is honest):", _pivot(agg, panel, "common", "median_vol_ratio").to_string(float_format=fmt3), ""]
        lines += ["Realised volatility, annualised (lower is better):", _pivot(agg, panel, "common", "ann_volatility").to_string(float_format=fmt_pct), ""]
        lines += ["Turnover per rebalance:", _pivot(agg, panel, "common", "mean_turnover").to_string(float_format=fmt3), ""]
    factor = _pivot(agg, "real", "common", "median_vol_ratio") / _pivot(agg, "gaussian", "common", "median_vol_ratio")
    lines += ["=== Real / Gaussian honesty ratio, common period: what the real market adds at each window ===", factor.to_string(float_format=fmt3), ""]
    vol_gap = _pivot(agg, "real", "common", "ann_volatility") / _pivot(agg, "gaussian", "common", "ann_volatility")
    lines += ["=== Real / Gaussian realised volatility, common period ===", vol_gap.to_string(float_format=fmt3), ""]
    lines += [f"=== {REAL_LABEL}, every window on its own full span (periods differ, so compare with care) ===", ""]
    spans = agg[(agg["panel"] == "real") & (agg["scope"] == "all") & (agg["strategy"] == ORDER[0])].set_index("window")
    lines += ["First holding period and count per window: " + ", ".join(f"T={w}: {spans.loc[w, 'first_hold'].date()} ({spans.loc[w, 'n_periods']})" for w in spans.index), ""]
    lines += ["Median realised / predicted volatility:", _pivot(agg, "real", "all", "median_vol_ratio").to_string(float_format=fmt3), ""]
    lines += ["Realised volatility, annualised:", _pivot(agg, "real", "all", "ann_volatility").to_string(float_format=fmt_pct), ""]
    real_vol = _pivot(agg, "real", "common", "ann_volatility")
    best = real_vol.idxmin()
    lines += ["Window with the lowest realised volatility on the real market, common period: " + ", ".join(f"{name} T={int(w)}" for name, w in best.items())]
    return "\n".join(lines)


def plot_metric(agg: pd.DataFrame, metric: str, title: str, ylabel: str, yscale: str, reference: float | None, common_start: pd.Timestamp, path: Path) -> None:
    common = agg[agg["scope"] == "common"]
    windows = sorted(common["window"].unique())
    n_common = int(common["n_periods"].max())
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharex=True, sharey=True)
    for ax, strategy in zip(axes.ravel(), ORDER):
        color = COLORS[strategy]
        for panel, style in (("gaussian", "dashed"), ("real", "solid")):
            sub = common[(common["panel"] == panel) & (common["strategy"] == strategy)].set_index("window").reindex(windows)
            y = sub[metric].to_numpy(dtype=float)
            if panel == "real":
                ax.plot(windows, y, color=color, linewidth=2.0, marker="o", markersize=6, markeredgecolor="white", markeredgewidth=0.8)
                for x, value in zip(windows, y):
                    if np.isfinite(value):
                        ax.annotate(_fmt(metric, value), (x, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=MUTED)
            else:
                ax.plot(windows, y, color=color, linewidth=1.4, linestyle="--", marker="o", markersize=6, markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.4)
        if reference is not None:
            ax.axhline(reference, color="#888888", linewidth=1.0, linestyle=":", zorder=1)
        both = common[common["strategy"] == strategy][metric].to_numpy(dtype=float)
        if yscale == "log" and np.all(np.abs(both) < 1e-9):
            ax.text(0.5, 0.5, "never trades: the universe and the weights\nare fixed on this complete-history panel", transform=ax.transAxes, ha="center", va="center", fontsize=9.5, color=MUTED)
        ax.axvline(456, color="#cccccc", linewidth=1.0, linestyle="--", zorder=0)
        ax.set_title(LABELS[strategy], fontsize=10.5)
        ax.set_xscale("log")
        ax.set_xticks(windows)
        ax.set_xticklabels([str(w) for w in windows])
        ax.minorticks_off()
    first = axes[0, 0]
    if yscale == "log":
        first.set_yscale("log")
        lo, hi = first.get_ylim()
        ticks = [t for t in (0.3, 0.5, 1, 1.5, 2, 3, 5, 10, 20, 50) if lo <= t <= hi]
        for ax in axes.ravel():
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:g}" for t in ticks])
            ax.minorticks_off()
    else:
        for ax in axes.ravel():
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for ax in axes[1]:
        ax.set_xlabel("window length T in sessions")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    axes[0, 0].text(456, 0.03, " T = N = 456", transform=axes[0, 0].get_xaxis_transform(), fontsize=8.5, color="#999999", va="bottom")
    handles = [
        Line2D([], [], color="#333333", linewidth=2.0, marker="o", markersize=6, markeredgecolor="white", label=REAL_LABEL),
        Line2D([], [], color="#333333", linewidth=1.4, linestyle="--", marker="o", markersize=6, markerfacecolor="white", markeredgecolor="#333333", label=SIM_LABEL),
    ]
    if reference is not None:
        handles.append(Line2D([], [], color="#888888", linewidth=1.0, linestyle=":", label="honest model: ratio 1"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        f"{title}. Every window judged on the same {n_common} holding periods, {common_start.date()} onward.",
        x=0.02, ha="left", fontsize=12, fontweight="medium",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def _fmt(metric: str, value: float) -> str:
    if metric == "ann_volatility":
        return f"{value:.1%}"
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
