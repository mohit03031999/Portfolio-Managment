"""Does dividing each day by its own volatility help? Volatility standardisation through the three-world test.

The covariance is size × shape × size, ``Σ = D C D``. Every fix so far
worked on the shape ``C``. The size ``D`` has been one number per stock,
the standard deviation over the whole window, and it did two jobs with
that stale average: it set how loudly each day enters the correlation
sums, and it scaled the cleaned correlation back into a forecast.
``standardized_covariance`` replaces it with a running estimate: each
return is divided by the stock's exponentially weighted volatility from
earlier days (half-life 20 or 60 sessions) before the correlation is
built, and the forecast uses the latest volatility.

A third synthetic panel isolates what volatility clustering alone does.
"clocked" takes the Gaussian returns from the fixed covariance and
multiplies every day by the real market's volatility factor for that day
(the cross-sectional RMS of the real returns, each stock in units of its
own full-sample volatility, normalised to mean square one). It has the
Gaussian panel's correlation structure, no fat tails of its own, and the
market's exact storm calendar.

    gaussian  →  clocked  →  real
    noise + bias   + volatility clustering   + everything else (fat tails, structure change)

Window T = 1000, 21-session holds, 93 holding periods from Dec 2018. The
saved flat runs of the other strategies are reused on the real and
Gaussian panels; on the clocked panel every strategy is run.

    .venv/bin/python examples/sp500_standardisation_study.py --panel real      # the three panels can run in parallel
    .venv/bin/python examples/sp500_standardisation_study.py --panel gaussian
    .venv/bin/python examples/sp500_standardisation_study.py --panel clocked
    .venv/bin/python examples/sp500_standardisation_study.py --replot
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
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from market_analyzer.backtest import DEFAULT_STRATEGIES, BacktestConfig, BacktestResult, run_backtest  # noqa: E402
from market_analyzer.matrices import CorrelationModel  # noqa: E402
from market_analyzer.optimizer import Estimator, ewma_volatility, minimum_variance_weights, rie_covariance, standardized_covariance, standardized_estimator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "output" / "sp500_marchenko_pastur" / "prices.csv"
OUT = ROOT / "output" / "sp500_standardisation"
SAVED_BASE = ROOT / "output" / "sp500_simulated_backtest"
SAVED_RIE = ROOT / "output" / "sp500_rie"

WINDOW = 1000
HOLDING = 21
PANELS = ("gaussian", "clocked", "real")
NEW = ("minvar_std20_none", "minvar_std20_clipped", "minvar_std20_rie", "minvar_std60_rie")
# Second set: the market-wide factor, and the two jobs of the volatility taken one at a time.
PARTS = ("minvar_mkt20_rie", "minvar_mkt20_clipped", "minvar_std20w_rie", "minvar_lat20_rie")
BASE = tuple(DEFAULT_STRATEGIES) + ("minvar_rie",)
ORDER = (
    "minvar_sample",
    "minvar_std20_none",
    "minvar_clipped_unit",
    "minvar_std20_clipped",
    "minvar_rie",
    "minvar_std20_rie",
    "minvar_std60_rie",
    "minvar_std20w_rie",
    "minvar_lat20_rie",
    "minvar_mkt20_rie",
    "minvar_mkt20_clipped",
    "minvar_ledoit_wolf",
    "equal_weight",
)
LABELS = {
    "minvar_sample": "Sample covariance",
    "minvar_std20_none": "Sample, standardised (h = 20)",
    "minvar_clipped": "MP clipped, fitted edge",
    "minvar_clipped_unit": "MP clipped, σ² = 1 edge",
    "minvar_std20_clipped": "MP clipped σ² = 1, standardised (h = 20)",
    "minvar_rie": "RIE",
    "minvar_std20_rie": "RIE, standardised (h = 20)",
    "minvar_std60_rie": "RIE, standardised (h = 60)",
    "minvar_std20w_rie": "RIE, standardised corr, window scale (h = 20)",
    "minvar_lat20_rie": "RIE, raw corr, latest per-stock scale (h = 20)",
    "minvar_mkt20_rie": "RIE, market factor (h = 20)",
    "minvar_mkt20_clipped": "MP clipped σ² = 1, market factor (h = 20)",
    "minvar_ledoit_wolf": "Ledoit-Wolf",
    "minvar_diagonal": "Diagonal",
    "equal_weight": "Equal weight",
}
# Base strategies keep their slots from the other figures; standardised variants are purple shades.
COLORS = {
    "minvar_sample": "#2a78d6",
    "minvar_std20_none": "#7f77dd",
    "minvar_clipped": "#eb6834",
    "minvar_clipped_unit": "#1baf7a",
    "minvar_std20_clipped": "#7f77dd",
    "minvar_rie": "#a61b1b",
    "minvar_std20_rie": "#534ab7",
    "minvar_std60_rie": "#afa9ec",
    "minvar_std20w_rie": "#7f77dd",
    "minvar_lat20_rie": "#7f77dd",
    "minvar_mkt20_rie": "#0f6e56",
    "minvar_mkt20_clipped": "#5dcaa5",
    "minvar_ledoit_wolf": "#e87ba4",
    "minvar_diagonal": "#eda100",
    "equal_weight": "#008300",
}
WORLD_LABELS = {"gaussian": "Gaussian, constant volatility", "clocked": "Gaussian with the market's volatility clock", "real": "S&P 500 returns"}
EPISODES = (("Covid crash", "2020-02-19", "2020-03-23"), ("2022 bear market", "2022-01-03", "2022-10-12"))
EPISODE_LEGEND = "shaded: Covid crash (Feb–Mar 2020), 2022 bear market (Jan–Oct 2022)"
MUTED = "#52514e"
LOG_TICKS = (0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", type=Path, default=PRICES)
    parser.add_argument("--panel", choices=PANELS + ("all",), default="all")
    parser.add_argument("--set", choices=("new", "parts", "both"), default="both", help="Which strategy set to run: the standardised estimators, the decomposition variants, or both.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replot", action="store_true", help="Run nothing; build the tables and figures from what is saved.")
    args = parser.parse_args()

    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.replot:
        panels = PANELS if args.panel == "all" else (args.panel,)
        data = build_panels(load_returns(args.prices), args.seed)
        for panel in panels:
            if args.set in ("new", "both"):
                run_panel(args, panel, data[panel])
            if args.set in ("parts", "both"):
                run_panel(args, panel, data[panel], PARTS, suffix="_parts")
        if args.panel != "all":
            print(f"Panel {args.panel} done. Run with --replot once every panel is saved.")
            return
    report(args)


# ----------------------------------------------------------------------------- data


def load_returns(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    returns = prices.pct_change(fill_method=None).iloc[1:]
    complete = returns.dropna(axis=1)
    return complete.loc[:, complete.std(ddof=1) > 0]


def volatility_clock(returns: pd.DataFrame) -> pd.Series:
    """The market's daily volatility factor: cross-sectional RMS of returns in units of each stock's full-sample volatility, mean square one."""
    scaled = returns / returns.std(ddof=1)
    factor = np.sqrt((scaled**2).mean(axis=1))
    return factor / np.sqrt((factor**2).mean())


def build_panels(returns: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """real, gaussian (fixed covariance, same seed as the other studies) and clocked (gaussian × the market's volatility factor)."""
    truth = CorrelationModel.from_returns(returns)
    chol = np.linalg.cholesky(truth.cleaned_covariance.to_numpy(dtype=float))
    rng = np.random.default_rng(seed)
    gaussian = pd.DataFrame(rng.standard_normal(size=returns.shape) @ chol.T, index=returns.index, columns=returns.columns)
    clock = volatility_clock(returns)
    clocked = gaussian.mul(clock, axis=0)
    return {"real": returns, "gaussian": gaussian, "clocked": clocked}


def registry() -> dict[str, Estimator]:
    return {
        "std20_none": standardized_estimator(20.0, "none"),
        "std20_clipped": standardized_estimator(20.0, "clipped"),
        "std20_rie": standardized_estimator(20.0, "rie"),
        "std60_rie": standardized_estimator(60.0, "rie"),
        "mkt20_rie": standardized_estimator(20.0, "rie", mode="market"),
        "mkt20_clipped": standardized_estimator(20.0, "clipped", mode="market"),
        "std20w_rie": standardized_estimator(20.0, "rie", scale="window"),
        "lat20_rie": standardized_estimator(20.0, "rie", standardize=False),
    }


# ----------------------------------------------------------------------------- rolling runs


def run_panel(args: argparse.Namespace, panel: str, data: pd.DataFrame, strategies: tuple[str, ...] | None = None, suffix: str = "") -> None:
    folder = args.out / f"{panel}_T{WINDOW}{suffix}"
    if (folder / "periods.csv").is_file():
        print(f"{panel}{suffix}: reusing saved run")
        return
    if strategies is None:
        # Real and Gaussian panels already have the base strategies saved; the clocked panel needs everything.
        strategies = NEW if panel != "clocked" else BASE + NEW
    config = BacktestConfig(window=WINDOW, holding=HOLDING, strategies=strategies)
    result = run_backtest(data, config, registry())
    result.save(folder)
    summary = result.summary()
    print(f"{panel} T = {WINDOW}: {int(summary['n_periods'].max())} periods")
    for name in strategies:
        print(f"   {LABELS[name]:42s} ratio {summary.loc[name, 'median_vol_ratio']:.3f}  vol {summary.loc[name, 'ann_volatility']:.2%}  turnover {summary.loc[name, 'mean_turnover']:.3f}")


def load_runs(out: Path) -> dict[str, list[BacktestResult]] | None:
    """Per panel, the results holding every strategy of ORDER (new run plus the saved flat runs where needed)."""
    runs: dict[str, list[BacktestResult]] = {}
    for panel in PANELS:
        folder = out / f"{panel}_T{WINDOW}"
        if not (folder / "periods.csv").is_file():
            return None
        pieces = [BacktestResult.load(folder)]
        parts = out / f"{panel}_T{WINDOW}_parts"
        if (parts / "periods.csv").is_file():
            pieces.append(BacktestResult.load(parts))
        if panel != "clocked":
            for base in (SAVED_BASE, SAVED_RIE):
                saved = base / f"{panel}_T{WINDOW}"
                if not (saved / "periods.csv").is_file():
                    return None
                pieces.append(BacktestResult.load(saved))
        runs[panel] = pieces
    return runs


def summary_of(pieces: list[BacktestResult]) -> pd.DataFrame:
    table = pd.concat([piece.summary() for piece in pieces])
    table = table[~table.index.duplicated(keep="first")]
    return table.reindex([s for s in ORDER if s in table.index])


def series_of(pieces: list[BacktestResult], name: str, column: str, steady_only: bool) -> pd.Series:
    for piece in pieces:
        if name in piece.strategies:
            rows = piece.periods[piece.periods["strategy"] == name]
            if steady_only:
                rows = rows[~rows["first_period"].astype(bool)]
            return pd.Series(rows[column].to_numpy(dtype=float), index=pd.DatetimeIndex(rows["hold_start"]))
    raise KeyError(name)


# ----------------------------------------------------------------------------- report


def report(args: argparse.Namespace) -> None:
    lines: list[str] = []
    returns = load_returns(args.prices)
    lines += volume_section(returns, args.out / "daily_volume.png")
    lines += blame_section(returns, args.seed)
    runs = load_runs(args.out)
    if runs is None:
        lines.append("Not every panel is saved yet (--panel real|gaussian|clocked), or the flat runs of the earlier studies are missing.")
    else:
        table = three_worlds(runs)
        table.to_csv(args.out / "three_worlds.csv")
        lines += describe(table, runs)
        plot_three_worlds(table, args.out / "three_worlds.png")
        plot_series(runs, args.out / "vol_ratio_over_time.png")
    text = "\n".join(lines)
    print(text)
    (args.out / "summary.txt").write_text(text + "\n")
    print(f"Wrote {args.out}")


def volume_section(returns: pd.DataFrame, path: Path) -> list[str]:
    """How loud each day is, before and after standardisation, across the whole panel."""
    raw_volume = np.sqrt(((returns / returns.std(ddof=1)) ** 2).mean(axis=1))
    sigma = ewma_volatility(returns, 20)
    standardized = (returns / sigma).dropna(how="any")
    std_volume = np.sqrt((standardized**2).mean(axis=1))
    kurt_raw = float(((returns - returns.mean()) / returns.std()).stack().pipe(lambda z: (z**4).mean() / (z**2).mean() ** 2 - 3))
    kurt_std = float(((standardized - standardized.mean()) / standardized.std()).stack().pipe(lambda z: (z**4).mean() / (z**2).mean() ** 2 - 3))
    index_kurt_raw = float(returns.mean(axis=1).kurt())
    index_kurt_std = float(standardized.mean(axis=1).kurt())
    median_kurt_raw = float(returns.kurt().median())
    median_kurt_std = float(standardized.kurt().median())
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.8), sharex=True)
    axes[0].plot(raw_volume.index, raw_volume.to_numpy(), color="#2a78d6", linewidth=0.9)
    axes[0].axhline(1.0, color="#888888", linewidth=1.0, linestyle=":")
    axes[0].set_ylabel("RMS return across stocks\n(each in units of its full-sample σ)")
    axes[0].set_title("Raw returns: the volume of the market changes by a factor of ten")
    axes[1].plot(std_volume.index, std_volume.to_numpy(), color="#534ab7", linewidth=0.9)
    axes[1].axhline(1.0, color="#888888", linewidth=1.0, linestyle=":")
    axes[1].set_ylabel("RMS of standardised returns\n(each ÷ yesterday's EWMA σ, h = 20)")
    axes[1].set_title("Standardised returns: what feeds the correlation matrix after the fix")
    for ax in axes:
        for _, start, end in EPISODES:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#000000", alpha=0.05, zorder=0)
        ax.set_yscale("log")
        ax.set_yticks([0.3, 0.5, 1, 2, 3, 5])
        ax.set_yticklabels(["0.3", "0.5", "1", "2", "3", "5"])
        ax.minorticks_off()
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(
        f"How loud is each day? {returns.shape[1]} stocks, {returns.index[0].year}–{returns.index[-1].year}. "
        f"Excess kurtosis of the equal-weight index: raw {index_kurt_raw:.1f}, standardised {index_kurt_std:.1f}. "
        f"Median single stock: {median_kurt_raw:.1f} → {median_kurt_std:.1f}.",
        x=0.02, ha="left", fontsize=12, fontweight="medium",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)
    return [
        f"--- Daily volume of the market ({returns.shape[1]} stocks) ---",
        f"Raw: RMS return across stocks ranges {raw_volume.min():.2f}–{raw_volume.max():.2f} (in units of each stock's full-sample σ); days beyond twice the typical level: {(raw_volume > 2).mean():.1%}.",
        f"Standardised by yesterday's EWMA σ (half-life 20): RMS ranges {std_volume.min():.2f}–{std_volume.max():.2f}; days beyond 2: {(std_volume > 2).mean():.1%}.",
        f"Excess kurtosis (Gaussian 0): equal-weight index raw {index_kurt_raw:.1f} → standardised {index_kurt_std:.1f}; median single stock {median_kurt_raw:.1f} → {median_kurt_std:.1f}; "
        f"pooled single-stock {kurt_raw:.1f} → {kurt_std:.1f}, because the largest standardised moves are company events on calm days.",
        "",
    ]


def blame_section(returns: pd.DataFrame, seed: int) -> list[str]:
    """In the Gaussian world the truth is known, so the dishonesty of each estimator can be split between its volatilities and its correlation."""
    data = build_panels(returns, seed)
    truth = CorrelationModel.from_returns(returns).cleaned_covariance.to_numpy(dtype=float)
    d_true = np.sqrt(np.diag(truth))
    c_true = truth / np.outer(d_true, d_true)
    variants = (
        ("RIE, plain", lambda w: rie_covariance(w)),
        ("RIE, standardised (h = 60)", lambda w: standardized_covariance(w, 60)),
        ("RIE, standardised (h = 20)", lambda w: standardized_covariance(w, 20)),
        ("RIE, standardised corr, window scale (h = 20)", lambda w: standardized_covariance(w, 20, scale="window")),
        ("RIE, raw corr, latest per-stock scale (h = 20)", lambda w: standardized_covariance(w, 20, standardize=False)),
        ("RIE, market factor (h = 20)", lambda w: standardized_covariance(w, 20, mode="market")),
    )
    rows = []
    for panel_name in ("gaussian",):
        panel = data[panel_name]
        for start in (0, 650, 1300, 1950):
            window = panel.iloc[start : start + WINDOW]
            for label, make in variants:
                estimate = make(window)
                matrix = estimate.covariance.to_numpy()
                weights = minimum_variance_weights(estimate.covariance).to_numpy(dtype=float)
                d_hat = np.sqrt(np.diag(matrix))
                c_hat = matrix / np.outer(d_hat, d_hat)
                predicted = np.sqrt(weights @ matrix @ weights)
                with_true_vol = np.sqrt(weights @ (d_true[:, None] * c_hat * d_true[None, :]) @ weights)
                with_true_corr = np.sqrt(weights @ (d_hat[:, None] * c_true * d_hat[None, :]) @ weights)
                true = np.sqrt(weights @ truth @ weights)
                rows.append({"panel": panel_name, "estimator": label, "true / predicted": true / predicted, "with a perfect volatility forecast": true / with_true_vol, "with a perfect correlation": true / with_true_corr, "median |vol error|": float(np.median(np.abs(d_hat / d_true - 1.0)))})
    table = pd.DataFrame(rows)
    gaussian = table[table["panel"] == "gaussian"].groupby("estimator", sort=False).mean(numeric_only=True)
    return [
        f"--- Whose fault is the dishonesty? Gaussian world (truth known), mean over four windows of T = {WINDOW} ---",
        "true / predicted is the honesty gap of the estimator's own min-var portfolio; the next two columns replace the estimator's volatilities, then its correlation, by the truth.",
        "On the constant-volatility panel the volatility 'error' of the standardised estimators is pure estimation noise.",
        gaussian.to_string(float_format=lambda v: f"{v:.3f}"),
        "",
    ]


def three_worlds(runs: dict[str, list[BacktestResult]]) -> pd.DataFrame:
    frames = {}
    for panel, pieces in runs.items():
        summary = summary_of(pieces)
        for metric in ("median_vol_ratio", "ann_volatility", "mean_turnover", "mean_gross_leverage"):
            frames[(metric, panel)] = summary[metric]
    table = pd.DataFrame(frames)
    table.columns = pd.MultiIndex.from_tuples(table.columns, names=["metric", "panel"])
    return table.reindex([s for s in ORDER if s in table.index])


def describe(table: pd.DataFrame, runs: dict[str, list[BacktestResult]]) -> list[str]:
    first = runs["real"][0]
    n = int(first.periods.groupby("strategy").size().max())
    lines = [
        f"--- Three worlds, window T = {WINDOW}, 21-session holds, {n} holding periods {first.periods['hold_start'].min().date()} → {first.periods['hold_end'].max().date()} ---",
        "gaussian: fixed covariance, constant volatility (noise + bias). clocked: the same draws × the market's daily volatility factor (adds volatility clustering). real: the S&P 500.",
        "",
    ]
    fmt = {"median_vol_ratio": "{:.3f}", "ann_volatility": "{:.2%}", "mean_turnover": "{:.3f}"}
    titles = {"median_vol_ratio": "Median realised / predicted volatility (1 is honest; 0.98 over 21-day holds)", "ann_volatility": "Realised volatility, annualised (lower is better)", "mean_turnover": "Turnover per rebalance"}
    for metric in ("median_vol_ratio", "ann_volatility", "mean_turnover"):
        wide = table[metric].reindex(columns=list(PANELS)).copy()
        wide.index = [LABELS[s] for s in wide.index]
        wide.columns = [WORLD_LABELS[p] for p in wide.columns]
        lines += [titles[metric] + ":", wide.to_string(float_format=lambda v: fmt[metric].format(v)), ""]
    ratio = table["median_vol_ratio"]
    decomposition = pd.DataFrame(
        {
            "clustering: clocked / gaussian": ratio["clocked"] / ratio["gaussian"],
            "the rest: real / clocked": ratio["real"] / ratio["clocked"],
            "total: real / gaussian": ratio["real"] / ratio["gaussian"],
        }
    )
    decomposition.index = [LABELS[s] for s in decomposition.index]
    lines += ["What each world adds to the honesty ratio:", decomposition.to_string(float_format=lambda v: f"{v:.3f}"), ""]
    vol = table["ann_volatility"]
    vol_decomposition = pd.DataFrame({"clustering: clocked / gaussian": vol["clocked"] / vol["gaussian"], "the rest: real / clocked": vol["real"] / vol["clocked"]})
    vol_decomposition.index = [LABELS[s] for s in vol_decomposition.index]
    lines += ["What each world adds to realised volatility:", vol_decomposition.to_string(float_format=lambda v: f"{v:.3f}"), ""]
    return lines


def plot_three_worlds(table: pd.DataFrame, path: Path) -> None:
    metrics = (("median_vol_ratio", "Median realised / predicted volatility", "log", 1.0), ("ann_volatility", "Realised volatility, annualised", "linear", None))
    markers = {"gaussian": ("o", "white"), "clocked": ("D", "white"), "real": ("o", None)}
    names = [s for s in ORDER if s in table.index]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.4 * len(metrics), 0.56 * len(names) + 2.2))
    for ax, (metric, title, scale, reference) in zip(axes, metrics):
        values = []
        for i, name in enumerate(names):
            y = len(names) - 1 - i
            color = COLORS[name]
            xs = [float(table[(metric, panel)].loc[name]) for panel in PANELS]
            ax.plot([min(xs), max(xs)], [y, y], color=color, linewidth=2.2, alpha=0.4, solid_capstyle="round")
            for panel, x in zip(PANELS, xs):
                marker, face = markers[panel]
                ax.scatter([x], [y], s=62 if marker == "o" else 48, marker=marker, facecolor=face or color, edgecolor=color if face else "white", linewidth=1.8 if face else 1.0, zorder=4 if face is None else 3)
            fmt = (lambda v: f"{v:.1%}") if metric == "ann_volatility" else (lambda v: f"{v:.2f}" if v < 10 else f"{v:.0f}")
            ax.annotate(", ".join(fmt(x) for x in xs), (max(xs), y), xytext=(10, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5, color=MUTED)
            values.extend(xs)
        if scale == "log":
            lo, hi = min(values) / 1.4, max(values) * 3.2
            ax.set_xscale("log")
            ax.set_xlim(lo, hi)
            ticks = [t for t in LOG_TICKS if lo < t < hi]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
        else:
            ax.set_xlim(min(values) * 0.85, max(values) * 1.45)
            ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.minorticks_off()
        if reference is not None:
            ax.axvline(reference, color="#888888", linewidth=1.0, linestyle=":", zorder=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([LABELS[n] for n in reversed(names)], fontsize=9.5)
        ax.set_ylim(-0.7, len(names) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor="white", markeredgecolor="#333333", markeredgewidth=1.8, label=WORLD_LABELS["gaussian"]),
        Line2D([], [], marker="D", linestyle="none", markersize=7, markerfacecolor="white", markeredgecolor="#333333", markeredgewidth=1.8, label=WORLD_LABELS["clocked"]),
        Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor="#333333", markeredgecolor="white", label=WORLD_LABELS["real"]),
        Line2D([], [], color="#888888", linewidth=1.0, linestyle=":", label="honest model: ratio 1"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        f"Three worlds, window T = {WINDOW}. Hollow circle: constant volatility. Hollow diamond: plus the market's volatility clock. Filled: the real market.\nLabels list the three values in that order.",
        x=0.02, ha="left", fontsize=12, fontweight="medium",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    fig.savefig(path)
    plt.close(fig)


def plot_series(runs: dict[str, list[BacktestResult]], path: Path) -> None:
    picks = tuple(name for name in ("minvar_rie", "minvar_std60_rie", "minvar_mkt20_rie") if all(any(name in piece.strategies for piece in pieces) for pieces in runs.values()))
    rows = (("vol_ratio", "realised / predicted volatility", 1.0, False), ("turnover", "turnover per rebalance", None, True))
    fig, axes = plt.subplots(len(rows), len(picks), figsize=(4.6 * len(picks), 3.4 * len(rows)), sharex=True, sharey="row")
    for r, (column, ylabel, reference, steady) in enumerate(rows):
        for c, name in enumerate(picks):
            ax = axes[r, c]
            color = COLORS[name]
            clocked = series_of(runs["clocked"], name, column, steady)
            real = series_of(runs["real"], name, column, steady)
            ax.plot(clocked.index, clocked.to_numpy(), color=color, linewidth=1.1, linestyle="--", alpha=0.75)
            ax.plot(real.index, real.to_numpy(), color=color, linewidth=1.7)
            if reference is not None:
                ax.axhline(reference, color="#888888", linewidth=1.0, linestyle=":", zorder=1)
            for _, start, end in EPISODES:
                ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#000000", alpha=0.05, zorder=0)
            if r == 0:
                ax.set_title(LABELS[name], fontsize=10.5)
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        first = axes[r, 0]
        first.set_yscale("log")
        lo, hi = first.get_ylim()
        ticks = [t for t in (0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10) if lo <= t <= hi]
        first.set_yticks(ticks)
        first.set_yticklabels([f"{t:g}" for t in ticks])
        for ax in axes[r]:
            ax.minorticks_off()
    handles = [
        Line2D([], [], color="#333333", linewidth=1.7, label=WORLD_LABELS["real"]),
        Line2D([], [], color="#333333", linewidth=1.1, linestyle="--", label=WORLD_LABELS["clocked"]),
        Line2D([], [], color="#888888", linewidth=1.0, linestyle=":", label="honest model: ratio 1"),
        Patch(facecolor="#000000", alpha=0.08, label=EPISODE_LEGEND),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Per 21-session hold, window T = {WINDOW}. Solid: the real market. Dashed: Gaussian returns on the market's volatility clock.", x=0.02, ha="left", fontsize=12, fontweight="medium")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


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
