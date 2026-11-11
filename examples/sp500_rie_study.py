"""Put the optimal RIE (nonlinear shrinkage) through the three tests the earlier studies built.

``minvar_rie`` is the Ledoit-Péché / Bun-Bouchaud-Potters estimator added to
:mod:`market_analyzer.optimizer`: every correlation eigenvalue gets its own
corrected value, the smallest ones are lifted with the inverse-Wishart fix,
and the result is sorted. ``minvar_rie_raw`` is the bare formula, kept so
the fix can be measured. This script runs both through:

1. the rolling backtest at T = 500 and 1000 on the real S&P 500 panel and
   on Gaussian returns from one fixed covariance (``sp500_simulated_backtest.py``),
   which splits each model's honesty gap into noise + bias and market change;
2. the Monte Carlo window sweep T = 250 … 20000 plus the T = ∞ limit
   (``sp500_honesty_vs_window.py``), which splits noise from bias;
3. the rolling window sweep T = 250 … 2000 on both panels over a common
   period (``sp500_window_sweep.py``), which shows where stale data starts
   to hurt on the real market.

Only the two new strategies are run; the other strategies' saved runs are
reused, so the comparison is on identical dates and identical samples. A
fourth figure shows the cleaning functions themselves on one window: what
each method does to each eigenvalue.

    .venv/bin/python examples/sp500_rie_study.py --part rolling      # the three parts can run in parallel
    .venv/bin/python examples/sp500_rie_study.py --part montecarlo
    .venv/bin/python examples/sp500_rie_study.py --part sweep
    .venv/bin/python examples/sp500_rie_study.py --replot            # tables and figures from everything saved
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

from market_analyzer.backtest import BacktestConfig, BacktestResult, run_backtest  # noqa: E402
from market_analyzer.matrices import CorrelationModel, clip_eigenvalues  # noqa: E402
from market_analyzer.optimizer import ESTIMATORS, minimum_variance_weights, portfolio_variance  # noqa: E402
from market_analyzer.rie import rie_eigenvalues  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "output" / "sp500_marchenko_pastur" / "prices.csv"
OUT = ROOT / "output" / "sp500_rie"
SAVED_ROLLING = ROOT / "output" / "sp500_simulated_backtest"
SAVED_SWEEP = ROOT / "output" / "sp500_window_sweep"
SAVED_MC = ROOT / "output" / "sp500_honesty_vs_window"

NEW = ("minvar_rie_raw", "minvar_rie")
ROLLING_WINDOWS = (500, 1000)
SWEEP_WINDOWS = (250, 500, 750, 1000, 1500, 2000)
MC_WINDOWS = (250, 500, 1000, 2000, 5000, 20000)
HOLDING = 21
PANELS = ("real", "gaussian")
INFINITY = "inf"
FIXED = "minvar_fixed"
ORDER = (
    "minvar_sample",
    "minvar_ledoit_wolf",
    "minvar_clipped",
    "minvar_clipped_unit",
    "minvar_rie_raw",
    "minvar_rie",
    "minvar_diagonal",
    "equal_weight",
    FIXED,
)
MC_MODELS = ("minvar_sample", "minvar_ledoit_wolf", "minvar_clipped", "minvar_clipped_unit", "minvar_rie_raw", "minvar_rie")
FOCUS = ("minvar_ledoit_wolf", "minvar_clipped", "minvar_rie_raw", "minvar_rie")
LABELS = {
    "minvar_sample": "Sample covariance",
    "minvar_ledoit_wolf": "Ledoit-Wolf",
    "minvar_clipped": "MP clipped, fitted edge",
    "minvar_clipped_unit": "MP clipped, σ² = 1 edge",
    "minvar_rie_raw": "RIE, bare formula",
    "minvar_rie": "RIE, inverse-Wishart lift + sort",
    "minvar_diagonal": "Diagonal",
    "equal_weight": "Equal weight",
    FIXED: "Fixed full-sample covariance",
}
# Same categorical slots as the other S&P figures; the two RIE variants get the two new slots.
COLORS = {
    "minvar_sample": "#2a78d6",
    "minvar_ledoit_wolf": "#e87ba4",
    "minvar_clipped": "#eb6834",
    "minvar_clipped_unit": "#1baf7a",
    "minvar_rie_raw": "#b5651d",
    "minvar_rie": "#a61b1b",
    "minvar_diagonal": "#eda100",
    "equal_weight": "#008300",
    FIXED: "#6250d6",
}
MARKERS = {
    "minvar_sample": "o",
    "minvar_ledoit_wolf": "s",
    "minvar_clipped": "D",
    "minvar_clipped_unit": "^",
    "minvar_rie_raw": "x",
    "minvar_rie": "P",
    "minvar_diagonal": "v",
    "equal_weight": "*",
}
EPISODES = (
    ("Covid crash", "2020-02-19", "2020-03-23"),
    ("2022 bear market", "2022-01-03", "2022-10-12"),
)
EPISODE_LEGEND = "shaded: Covid crash (Feb–Mar 2020), 2022 bear market (Jan–Oct 2022)"
REAL_LABEL = "S&P 500 returns"
SIM_LABEL = "Gaussian returns from one fixed covariance"
MUTED = "#52514e"
LOG_TICKS = (0.3, 0.5, 1, 1.5, 2, 3, 5, 10, 20, 50, 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", type=Path, default=PRICES)
    parser.add_argument("--part", choices=("rolling", "montecarlo", "sweep", "all"), default="all")
    parser.add_argument("--holding", type=int, default=HOLDING)
    parser.add_argument("--reps", type=int, default=20, help="Monte Carlo draws per window up to T = 2000, as in sp500_honesty_vs_window.py.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replot", action="store_true", help="Run nothing; build the tables and figures from what is saved.")
    args = parser.parse_args()

    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.replot:
        parts = ("rolling", "montecarlo", "sweep") if args.part == "all" else (args.part,)
        returns = load_returns(args.prices)
        fake, truth = simulate(returns, args.seed)
        if "rolling" in parts:
            run_rolling(args, returns, fake, ROLLING_WINDOWS)
        if "sweep" in parts:
            run_rolling(args, returns, fake, tuple(w for w in SWEEP_WINDOWS if w not in ROLLING_WINDOWS))
        if "montecarlo" in parts:
            run_montecarlo(args, returns, truth)
        if args.part != "all":
            print(f"Part {args.part} done. Run with --replot once every part is saved.")
            return
    report(args)


# ----------------------------------------------------------------------------- data


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


def saved_run(panel: str, window: int, out: Path) -> Path | None:
    """Folder holding the earlier strategies' run for this panel and window, if any."""
    for base in (SAVED_ROLLING, SAVED_SWEEP, out):
        folder = base / f"{panel}_T{window}"
        if (folder / "periods.csv").is_file():
            return folder
    return None


# ----------------------------------------------------------------------------- part 1 and 3: rolling runs


def run_rolling(args: argparse.Namespace, returns: pd.DataFrame, fake: pd.DataFrame, windows: tuple[int, ...]) -> None:
    for window in windows:
        config = BacktestConfig(window=window, holding=args.holding, strategies=NEW)
        for panel_name, panel in (("real", returns), ("gaussian", fake)):
            folder = args.out / f"{panel_name}_T{window}"
            if (folder / "periods.csv").is_file():
                print(f"{panel_name} T = {window}: reusing saved RIE run")
                continue
            result = run_backtest(panel, config)
            result.save(folder)
            summary = result.summary()
            print(
                f"{panel_name} T = {window}: {int(summary['n_periods'].max())} periods, "
                + ", ".join(f"{LABELS[s]} ratio {summary.loc[s, 'median_vol_ratio']:.3f} vol {summary.loc[s, 'ann_volatility']:.2%}" for s in NEW)
            )


# ----------------------------------------------------------------------------- part 2: Monte Carlo sweep


def run_montecarlo(args: argparse.Namespace, returns: pd.DataFrame, truth: CorrelationModel) -> None:
    """Redraw the samples of sp500_honesty_vs_window.py (same seed, same order) and fit the RIE variants on them."""
    sigma = truth.cleaned_covariance.to_numpy(dtype=float)
    tickers = [str(column) for column in returns.columns]
    n_assets = len(tickers)
    ones = np.ones(n_assets)
    optimal_weights = np.linalg.solve(sigma, ones)
    optimal_weights /= optimal_weights.sum()
    optimal = float(np.sqrt(optimal_weights @ sigma @ optimal_weights))
    chol = np.linalg.cholesky(sigma)
    rng = np.random.default_rng(args.seed)
    saved = pd.read_csv(SAVED_MC / "draws.csv") if (SAVED_MC / "draws.csv").is_file() else None
    rows = []
    for window in MC_WINDOWS:
        reps = max(4, int(round(args.reps * min(1.0, 2000.0 / window))))
        for rep in range(reps):
            sample = pd.DataFrame(rng.standard_normal((window, n_assets)) @ chol.T, columns=tickers)
            if rep == 0 and saved is not None:
                # Same seed and draw order as the earlier sweep, so the sample estimator must reproduce its row exactly.
                check = _fit("sample", sample, sigma, optimal)["ratio"]
                stored = saved[(saved["window"] == window) & (saved["rep"] == 0) & (saved["model"] == "minvar_sample")]["ratio"]
                same = len(stored) == 1 and abs(float(stored.iloc[0]) - check) < 1e-9
                print(f"T = {window}: first draw {'matches' if same else 'DOES NOT match'} the saved sweep")
            for name in NEW:
                rows.append({"window": window, "rep": rep, "model": name, **_fit(name[len("minvar_") :], sample, sigma, optimal)})
        done = pd.DataFrame([row for row in rows if row["window"] == window])
        medians = done.groupby("model")["ratio"].median()
        print(f"T = {window:>6}, q = {n_assets / window:.3f}, {reps} draws: " + ", ".join(f"{LABELS[m]} {medians[m]:.3f}" for m in NEW))
    pd.DataFrame(rows).to_csv(args.out / "draws_rie.csv", index=False)
    # T = ∞: q → 0 makes 1 − q + q z g → 1, so ξ(λ) = λ and the estimator returns the truth.
    limits = []
    for name in NEW:
        limits.append(
            {
                "model": name,
                "ratio": 1.0,
                "excess": 1.0,
                "leverage": float(np.abs(optimal_weights).sum()),
                "predicted": optimal,
                "true": optimal,
                "n_signal": np.nan,
                "shrinkage": np.nan,
                "true_modes": float(truth.n_signal),
                "n_regularized": 0.0,
                "kappa": np.nan,
            }
        )
    pd.DataFrame(limits).to_csv(args.out / "limit_rie.csv", index=False)


def _fit(key: str, sample: pd.DataFrame, sigma: np.ndarray, optimal: float) -> dict[str, float]:
    estimate = ESTIMATORS[key](sample)
    weights = minimum_variance_weights(estimate.covariance)
    predicted = float(np.sqrt(max(portfolio_variance(weights, estimate.covariance), 0.0)))
    vector = weights.to_numpy(dtype=float)
    true = float(np.sqrt(vector @ sigma @ vector))
    return {
        "ratio": true / predicted if predicted > 0 else float("nan"),
        "excess": true / optimal,
        "leverage": float(np.abs(vector).sum()),
        "predicted": predicted,
        "true": true,
        "n_signal": float(estimate.info.get("n_signal", np.nan)),
        "shrinkage": float(estimate.info.get("shrinkage", np.nan)),
        "n_regularized": float(estimate.info.get("n_regularized", np.nan)),
        "kappa": float(estimate.info.get("kappa", np.nan)),
    }


# ----------------------------------------------------------------------------- report


def report(args: argparse.Namespace) -> None:
    lines: list[str] = []
    returns = load_returns(args.prices)
    fake, truth = simulate(returns, args.seed)
    n_assets = returns.shape[1]
    lines.append(
        f"Panel: {returns.shape[0]} sessions × {n_assets} tickers, {returns.index[0].date()} → {returns.index[-1].date()}. "
        f"Gaussian panel from the cleaned full-sample covariance ({truth.n_signal} signal modes), seed {args.seed}."
    )
    lines.append("")

    lines += shrinkage_section(returns, fake, truth, args.out / "shrinkage_function.png")

    rolling = {}
    for window in ROLLING_WINDOWS:
        for panel in PANELS:
            base, new = saved_run(panel, window, args.out), args.out / f"{panel}_T{window}"
            if base is not None and (new / "periods.csv").is_file():
                rolling[(panel, window)] = (BacktestResult.load(base), BacktestResult.load(new))
    if len(rolling) == 2 * len(ROLLING_WINDOWS):
        decomp = pd.concat([decomposition(*rolling[("real", w)], *rolling[("gaussian", w)], w, n_assets) for w in ROLLING_WINDOWS], ignore_index=True)
        decomp.to_csv(args.out / "decomposition.csv", index=False)
        for window in ROLLING_WINDOWS:
            lines += describe_rolling(decomp[decomp["window"] == window], window, n_assets)
            lines.append("")
        plot_dumbbells(decomp, args.out / "honesty_decomposition.png")
        plot_series(rolling[("real", ROLLING_WINDOWS[0])], rolling[("gaussian", ROLLING_WINDOWS[0])], ROLLING_WINDOWS[0], args.out / "vol_ratio_over_time.png")
    else:
        lines.append("Rolling part not run yet (--part rolling).")
        lines.append("")

    if (args.out / "draws_rie.csv").is_file() and (SAVED_MC / "draws.csv").is_file():
        draws = pd.concat([pd.read_csv(SAVED_MC / "draws.csv"), pd.read_csv(args.out / "draws_rie.csv")], ignore_index=True)
        limits = pd.concat([pd.read_csv(SAVED_MC / "limit.csv"), pd.read_csv(args.out / "limit_rie.csv")], ignore_index=True)
        table = summarize_montecarlo(draws, limits)
        table.to_csv(args.out / "summary_by_window.csv", index=False)
        lines += describe_montecarlo(table, draws, n_assets)
        lines.append("")
        plot_montecarlo(table, args.out / "honesty_vs_window.png")
    else:
        lines.append("Monte Carlo part not run yet (--part montecarlo).")
        lines.append("")

    sweep = {}
    for window in SWEEP_WINDOWS:
        for panel in PANELS:
            base, new = saved_run(panel, window, args.out), args.out / f"{panel}_T{window}"
            if base is not None and (new / "periods.csv").is_file():
                sweep[(panel, window)] = (BacktestResult.load(base), BacktestResult.load(new))
    if len(sweep) == 2 * len(SWEEP_WINDOWS):
        common_start = max(result.periods["hold_start"].min() for pair in sweep.values() for result in pair)
        agg = aggregate_sweep(sweep, common_start)
        agg.to_csv(args.out / "sweep_summary.csv", index=False)
        lines += describe_sweep(agg, common_start, sweep)
        lines.append("")
        plot_sweep(agg, common_start, args.out / "window_sweep.png")
    else:
        missing = [f"{p} T={w}" for w in SWEEP_WINDOWS for p in PANELS if (p, w) not in sweep]
        lines.append("Window sweep incomplete, missing: " + ", ".join(missing) + " (--part sweep, and --part rolling for T = 500 and 1000).")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    (args.out / "summary.txt").write_text(text + "\n")
    print(f"Wrote {args.out}")


# ----------------------------------------------------------------------------- the cleaning functions on one window


def shrinkage_section(returns: pd.DataFrame, fake: pd.DataFrame, truth: CorrelationModel, path: Path) -> list[str]:
    """Sample eigenvalue against cleaned eigenvalue for each method, on the last T = 500 window of both panels."""
    window = ROLLING_WINDOWS[0]
    frames = {}
    for panel_name, panel in (("real", returns), ("gaussian", fake)):
        sample = panel.iloc[-window:]
        model = CorrelationModel.from_returns(sample)
        eigenvalues = model.eigenvalues["eigenvalue"].to_numpy(dtype=float)
        matrix = model.correlation.to_numpy(dtype=float)
        _, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
        eigenvectors = eigenvectors[:, ::-1]
        q = model.q
        unit = clip_eigenvalues(eigenvalues, q, 1.0)[0]
        raw = rie_eigenvalues(eigenvalues, q, regularize=False, sort=False)
        rie = rie_eigenvalues(eigenvalues, q)
        frame = pd.DataFrame(
            {
                "sample": eigenvalues,
                "clipped_fitted": model.eigenvalues["clipped_eigenvalue"].to_numpy(dtype=float),
                "clipped_unit": unit,
                "rie_raw": raw.cleaned,
                "rie": rie.cleaned,
            },
            index=pd.Index(range(1, len(eigenvalues) + 1), name="mode"),
        )
        if panel_name == "gaussian":
            vols = np.sqrt(np.diag(truth.cleaned_covariance.to_numpy(dtype=float)))
            true_corr = truth.cleaned_covariance.to_numpy(dtype=float) / np.outer(vols, vols)
            frame["oracle"] = np.einsum("ij,jk,ki->i", eigenvectors.T, true_corr, eigenvectors)
        frame.attrs = {"q": q, "n_signal": model.n_signal, "lambda_plus": model.lambda_plus, "kappa": rie.kappa, "alpha": rie.alpha, "n_regularized": rie.n_regularized, "trace_scale": rie.trace_scale, "unit_signal": int(np.sum(eigenvalues > clip_eigenvalues(eigenvalues, q, 1.0)[3]))}
        frames[panel_name] = frame
        frame.to_csv(path.parent / f"shrinkage_{panel_name}_T{window}.csv")
    plot_shrinkage(frames, window, path)

    lines = [f"--- The cleaning functions on the last T = {window} window ({returns.index[-window].date()} → {returns.index[-1].date()}) ---"]
    for panel_name, frame in frames.items():
        a = frame.attrs
        lines.append(
            f"{REAL_LABEL if panel_name == 'real' else SIM_LABEL}: q = {a['q']:.3f}; fitted edge keeps {a['n_signal']} modes, unit edge {a['unit_signal']}; "
            f"RIE lifted {a['n_regularized']} eigenvalues (κ = {a['kappa']:.2f}, α = {a['alpha']:.3f}), trace rescale {a['trace_scale']:.4f}."
        )
        bottom = frame.iloc[-20:]
        lines.append(
            f"  smallest 20 eigenvalues, mean: sample {bottom['sample'].mean():.3f}, clipped fitted {bottom['clipped_fitted'].mean():.3f}, "
            f"clipped unit {bottom['clipped_unit'].mean():.3f}, RIE bare {bottom['rie_raw'].mean():.3f}, RIE {bottom['rie'].mean():.3f}"
            + (f", oracle {bottom['oracle'].mean():.3f}" if "oracle" in frame else "")
        )
        top = frame.iloc[:3]
        lines.append(
            "  largest 3: sample " + ", ".join(f"{v:.1f}" for v in top["sample"]) + "; RIE " + ", ".join(f"{v:.1f}" for v in top["rie"])
            + ("; oracle " + ", ".join(f"{v:.1f}" for v in top["oracle"]) if "oracle" in frame else "")
        )
    lines.append("")
    return lines


def plot_shrinkage(frames: dict[str, pd.DataFrame], window: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), sharey=True)
    for ax, (panel_name, frame) in zip(axes, frames.items()):
        x = frame["sample"].to_numpy()
        lo = max(min(x.min(), frame["rie_raw"].min()) * 0.7, 1e-3)
        hi = x.max() * 1.6
        ax.plot([lo, hi], [lo, hi], color="#999999", linewidth=1.0, linestyle=":", zorder=1, label="no cleaning: ξ = λ")
        ax.plot(x, frame["clipped_fitted"], color=COLORS["minvar_clipped"], linewidth=2.0, zorder=2, label=LABELS["minvar_clipped"])
        ax.plot(x, frame["clipped_unit"], color=COLORS["minvar_clipped_unit"], linewidth=2.0, zorder=2, label=LABELS["minvar_clipped_unit"])
        ax.scatter(x, frame["rie_raw"], s=14, marker="x", color=COLORS["minvar_rie_raw"], linewidth=1.0, zorder=3, label=LABELS["minvar_rie_raw"])
        ax.scatter(x, frame["rie"], s=16, color=COLORS["minvar_rie"], edgecolor="white", linewidth=0.4, zorder=4, label=LABELS["minvar_rie"])
        if "oracle" in frame:
            ax.scatter(x, frame["oracle"], s=30, facecolor="none", edgecolor=COLORS[FIXED], linewidth=1.0, zorder=5, label="oracle: truth seen through the sample eigenvectors")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ticks = [t for t in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300) if lo <= t <= hi]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        a = frame.attrs
        ax.set_title(f"{REAL_LABEL if panel_name == 'real' else SIM_LABEL}. q = {a['q']:.2f}, RIE lifted {a['n_regularized']} eigenvalues")
        ax.set_xlabel("sample correlation eigenvalue λ (log)")
        ax.axvline(a["lambda_plus"], color=COLORS["minvar_clipped"], linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(a["lambda_plus"], lo * 1.15, " fitted edge", fontsize=8, color=COLORS["minvar_clipped"], va="bottom")
        ax.legend(loc="upper left", fontsize=8.5)
    axes[0].set_ylabel("cleaned eigenvalue ξ (log)")
    fig.suptitle(
        f"What each method does to each eigenvalue: last T = {window} window, {len(frames['real'])} tickers.\n"
        "Clipping flattens the bulk to one value; the RIE gives every eigenvalue its own, and the inverse-Wishart lift stops the smallest from diving.",
        x=0.02, ha="left", fontsize=12, fontweight="medium",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- rolling tables and figures


def merged_summary(base: BacktestResult, new: BacktestResult) -> pd.DataFrame:
    return pd.concat([base.summary(), new.summary()]).reindex([s for s in ORDER if s in set(base.strategies) | set(new.strategies)])


def decomposition(real_base: BacktestResult, real_new: BacktestResult, sim_base: BacktestResult, sim_new: BacktestResult, window: int, n_assets: int) -> pd.DataFrame:
    a = merged_summary(real_base, real_new)
    b = merged_summary(sim_base, sim_new)
    names = list(a.index)
    table = pd.DataFrame(
        {
            "window": window,
            "strategy": names,
            "median_vol_ratio_real": a["median_vol_ratio"].to_numpy(),
            "median_vol_ratio_gaussian": b.loc[names, "median_vol_ratio"].to_numpy(),
            "ann_volatility_real": a["ann_volatility"].to_numpy(),
            "ann_volatility_gaussian": b.loc[names, "ann_volatility"].to_numpy(),
            "mean_turnover_real": a["mean_turnover"].to_numpy(),
            "mean_turnover_gaussian": b.loc[names, "mean_turnover"].to_numpy(),
            "mean_gross_leverage_real": a["mean_gross_leverage"].to_numpy(),
            "mean_gross_leverage_gaussian": b.loc[names, "mean_gross_leverage"].to_numpy(),
        }
    )
    table["market_change_factor"] = table["median_vol_ratio_real"] / table["median_vol_ratio_gaussian"]
    table["q"] = n_assets / window
    return table


def describe_rolling(table: pd.DataFrame, window: int, n_assets: int) -> list[str]:
    show = table.set_index("strategy").loc[
        :, ["median_vol_ratio_gaussian", "median_vol_ratio_real", "market_change_factor", "ann_volatility_gaussian", "ann_volatility_real", "mean_turnover_real"]
    ]
    show.index = [LABELS[name] for name in show.index]
    show.columns = ["ratio Gaussian", "ratio real", "real / Gaussian", "vol Gaussian", "vol real", "turnover real"]
    return [
        f"--- Rolling backtest, T = {window}, q = N/T = {n_assets / window:.3f}, 21-session holds ---",
        "ratio = median realised / predicted volatility (1 is honest; 0.98 is what an honest model shows over 21-day holds).",
        "ratio Gaussian = bias + estimation noise. real / Gaussian = what the market's own movement adds. vol = annualised realised volatility.",
        "",
        show.to_string(float_format=lambda value: f"{value:.3f}"),
    ]


def _series(result: BacktestResult, name: str, column: str, steady_only: bool) -> pd.Series:
    rows = result.periods[result.periods["strategy"] == name]
    if steady_only:
        rows = rows[~rows["first_period"].astype(bool)]
    return pd.Series(rows[column].to_numpy(dtype=float), index=pd.DatetimeIndex(rows["hold_start"]))


def _pick(pair: tuple[BacktestResult, BacktestResult], name: str) -> BacktestResult:
    base, new = pair
    return new if name in new.strategies else base


def plot_series(real: tuple[BacktestResult, BacktestResult], sim: tuple[BacktestResult, BacktestResult], window: int, path: Path) -> None:
    rows = (
        ("vol_ratio", "realised / predicted volatility", "log", 1.0, False),
        ("turnover", "turnover per rebalance", "symlog", None, True),
    )
    fig, axes = plt.subplots(len(rows), len(FOCUS), figsize=(4.1 * len(FOCUS), 3.4 * len(rows)), sharex=True, sharey="row")
    for r, (column, ylabel, yscale, reference, steady) in enumerate(rows):
        for c, name in enumerate(FOCUS):
            ax = axes[r, c]
            color = COLORS[name]
            a = _series(_pick(real, name), name, column, steady)
            b = _series(_pick(sim, name), name, column, steady)
            ax.plot(b.index, b.to_numpy(), color=color, linewidth=1.1, linestyle="--", alpha=0.75)
            ax.plot(a.index, a.to_numpy(), color=color, linewidth=1.7)
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
        if yscale == "log":
            first.set_yscale("log")
            lo, hi = first.get_ylim()
            ticks = [t for t in LOG_TICKS if lo <= t <= hi]
            first.set_yticks(ticks)
            first.set_yticklabels([f"{t:g}" for t in ticks])
        else:
            first.set_yscale("symlog", linthresh=0.1, linscale=0.8)
            first.set_yticks([0, 0.1, 0.3, 1, 3, 10])
            first.set_yticklabels(["0", "0.1", "0.3", "1", "3", "10"])
        for ax in axes[r]:
            ax.minorticks_off()
    handles = [
        Line2D([], [], color="#333333", linewidth=1.7, label=REAL_LABEL),
        Line2D([], [], color="#333333", linewidth=1.1, linestyle="--", label=SIM_LABEL),
        Line2D([], [], color="#888888", linewidth=1.0, linestyle=":", label="honest model: ratio 1"),
        Patch(facecolor="#000000", alpha=0.08, label=EPISODE_LEGEND),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        f"Per 21-session hold, T = {window}. Solid: {REAL_LABEL}. Dashed: {SIM_LABEL}.",
        x=0.02, ha="left", fontsize=12, fontweight="medium",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_dumbbells(decomp: pd.DataFrame, path: Path) -> None:
    """Hollow marker: Gaussian panel. Filled marker: real panel. The bar between them is the market's contribution."""
    metrics = (
        ("median_vol_ratio", "Median realised / predicted volatility", "log", 1.0),
        ("ann_volatility", "Realised volatility, annualised", "linear", None),
    )
    windows = sorted(decomp["window"].unique())
    fig, axes = plt.subplots(len(metrics), len(windows), figsize=(6.8 * len(windows), 4.2 * len(metrics)), squeeze=False)
    for row, (metric, title, scale, reference) in enumerate(metrics):
        for col, window in enumerate(windows):
            ax = axes[row, col]
            sub = decomp[decomp["window"] == window].set_index("strategy")
            names = [name for name in ORDER if name in sub.index]
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
                fmt = (lambda v: f"{v:.1%}") if metric == "ann_volatility" else _fmt
                close = (high / low < 1.12) if scale == "log" else (high - low < 0.03)
                if close:
                    # Two markers almost on top of each other: one label to the right, low first.
                    ax.annotate(f"{fmt(low)}, {fmt(high)}", (high, y), xytext=(10, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5, color=MUTED)
                else:
                    ax.annotate(fmt(low), (low, y), xytext=(-10, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5, color=MUTED)
                    ax.annotate(fmt(high), (high, y), xytext=(10, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5, color=MUTED)
                values.extend([sim, real])
            if scale == "log":
                lo, hi = min(values) / 2.2, max(values) * 2.6
                ax.set_xscale("log")
                ax.set_xlim(lo, hi)
                ticks = [tick for tick in LOG_TICKS if lo < tick < hi]
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{t:g}" for t in ticks])
            else:
                ax.set_xlim(min(values) * 0.68, max(values) * 1.3)
                ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.minorticks_off()
            if reference is not None:
                ax.axvline(reference, color="#888888", linewidth=1.0, linestyle=":", zorder=1)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([LABELS[name] for name in reversed(names)], fontsize=9.5)
            ax.set_ylim(-0.7, len(names) - 0.3)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{title}, T = {window}")
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor="white", markeredgecolor="#333333", markeredgewidth=2.0, label=SIM_LABEL),
        Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor="#333333", markeredgecolor="white", label=REAL_LABEL),
        Line2D([], [], color="#888888", linewidth=1.0, linestyle=":", label="honest model: ratio 1"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        "Honesty and risk of every strategy, with the two RIE variants added. Hollow: Gaussian world (bias + noise). Filled: real market.",
        x=0.02, ha="left", fontsize=12, fontweight="medium",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- Monte Carlo tables and figure


def summarize_montecarlo(draws: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
    grouped = draws.groupby(["window", "model"])
    table = pd.DataFrame(
        {
            "ratio_median": grouped["ratio"].median(),
            "ratio_q10": grouped["ratio"].quantile(0.10),
            "ratio_q90": grouped["ratio"].quantile(0.90),
            "excess_median": grouped["excess"].median(),
            "excess_q10": grouped["excess"].quantile(0.10),
            "excess_q90": grouped["excess"].quantile(0.90),
            "leverage_median": grouped["leverage"].median(),
            "n_regularized_mean": grouped["n_regularized"].mean() if "n_regularized" in draws else np.nan,
            "draws": grouped.size(),
        }
    ).reset_index()
    limit_rows = limits.rename(columns={"ratio": "ratio_median", "excess": "excess_median", "leverage": "leverage_median"})
    limit_rows["window"] = INFINITY
    limit_rows["draws"] = 0
    limit_rows["n_regularized_mean"] = np.nan
    for column in ("ratio_q10", "ratio_q90"):
        limit_rows[column] = limit_rows["ratio_median"]
    for column in ("excess_q10", "excess_q90"):
        limit_rows[column] = limit_rows["excess_median"]
    table["window"] = table["window"].astype(str)
    table = pd.concat([table, limit_rows[table.columns]], ignore_index=True)
    bias = table[table["window"] == INFINITY].set_index("model")["ratio_median"]
    table["bias"] = table["model"].map(bias)
    table["noise_factor"] = table["ratio_median"] / table["bias"]
    return table


def describe_montecarlo(table: pd.DataFrame, draws: pd.DataFrame, n_assets: int) -> list[str]:
    windows = [str(w) for w in sorted(draws["window"].unique())] + [INFINITY]

    def pivot(column: str, fmt: str, models: tuple[str, ...] = MC_MODELS) -> str:
        wide = table.pivot(index="window", columns="model", values=column).reindex(windows)[list(models)]
        wide.columns = [LABELS[m] for m in wide.columns]
        wide.index.name = "T"
        return wide.to_string(float_format=lambda v: fmt.format(v), na_rep="-")

    lines = [
        "--- Monte Carlo window sweep: independent Gaussian samples of T days from the fixed truth, same draws for every model ---",
        "Honesty ratio = true volatility of the fitted minimum-variance portfolio / the volatility the model predicted (median across draws). T = inf hands the estimator the truth.",
        pivot("ratio_median", "{:.3f}"),
        "",
        "Noise factor = ratio(T) / ratio(inf); the T = inf row above is the bias.",
        pivot("noise_factor", "{:.3f}"),
        "",
        "True volatility / best possible volatility (median across draws): how far from the optimum the portfolio sits.",
        pivot("excess_median", "{:.3f}"),
        "",
        "Eigenvalues the inverse-Wishart step lifted (mean across draws):",
        pivot("n_regularized_mean", "{:.1f}", ("minvar_rie",)),
    ]
    return lines


def _positions(windows: list[str]) -> dict[str, float]:
    finite = [int(w) for w in windows if w != INFINITY]
    positions = {str(w): float(w) for w in finite}
    positions[INFINITY] = max(finite) * 5.0
    return positions


def plot_montecarlo(table: pd.DataFrame, path: Path) -> None:
    windows = sorted([w for w in table["window"].unique() if w != INFINITY], key=int) + [INFINITY]
    pos = _positions(windows)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    panels = (
        ("ratio", "Honesty: true volatility / predicted volatility", 1.0),
        ("excess", "Cost: true volatility / best possible volatility", 1.0),
    )
    for ax, (column, title, reference) in zip(axes, panels):
        for model in MC_MODELS:
            sub = table[table["model"] == model].set_index("window").reindex(windows)
            y = sub[f"{column}_median"].to_numpy(dtype=float)
            x = np.array([pos[w] for w in windows])
            finite = slice(0, len(windows) - 1)
            color = COLORS[model]
            ax.plot(x[finite], y[finite], color=color, linewidth=2.0, marker=MARKERS[model], markersize=6.5, markeredgecolor="white", markeredgewidth=0.6, label=LABELS[model])
            ax.plot(x[-2:], y[-2:], color=color, linewidth=1.2, linestyle="--")
            ax.scatter([x[-1]], [y[-1]], color=color, s=40, marker=MARKERS[model], zorder=4)
            if model in NEW:
                ax.fill_between(x[finite], sub[f"{column}_q10"].to_numpy(dtype=float)[finite], sub[f"{column}_q90"].to_numpy(dtype=float)[finite], color=color, alpha=0.12, linewidth=0)
        ax.axhline(reference, color="#888888", linewidth=1.0, linestyle=":", zorder=1)
        ax.axvline(456, color="#cccccc", linewidth=1.0, linestyle="--", zorder=0)
        ax.text(456, 0.02, " T = N = 456", transform=ax.get_xaxis_transform(), fontsize=8.5, color="#999999", va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([pos[w] for w in windows])
        ax.set_xticklabels([w if w != INFINITY else "∞" for w in windows])
        lo, hi = ax.get_ylim()
        hi = min(hi, 6.0)
        ax.set_ylim(max(lo, 0.9), hi)
        ticks = [t for t in LOG_TICKS if max(lo, 0.9) <= t <= hi]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel("window length T in sessions (∞: the estimator is handed the truth)")
        ax.legend(fontsize=9, loc="upper right")
    axes[0].set_ylabel("ratio, log scale")
    fig.suptitle(
        "Noise vs bias per estimator, Gaussian world. Bands: 10th–90th percentile of the RIE draws. The sample covariance at T = 500 (11.1) is above the frame.",
        x=0.02, ha="left", fontsize=12, fontweight="medium",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- window sweep tables and figure


def aggregate_sweep(sweep: dict[tuple[str, int], tuple[BacktestResult, BacktestResult]], common_start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for (panel, window), pair in sweep.items():
        for scope in ("all", "common"):
            parts = pair if scope == "all" else tuple(result.restrict(start=common_start) for result in pair)
            summary = merged_summary(*parts)
            for strategy, row in summary.iterrows():
                rows.append(
                    {
                        "panel": panel,
                        "window": window,
                        "strategy": strategy,
                        "scope": scope,
                        "n_periods": int(row["n_periods"]),
                        "median_vol_ratio": float(row["median_vol_ratio"]),
                        "ann_volatility": float(row["ann_volatility"]),
                        "mean_turnover": float(row["mean_turnover"]),
                        "mean_gross_leverage": float(row["mean_gross_leverage"]),
                    }
                )
    return pd.DataFrame(rows)


def _pivot(agg: pd.DataFrame, panel: str, scope: str, metric: str, models: tuple[str, ...]) -> pd.DataFrame:
    sub = agg[(agg["panel"] == panel) & (agg["scope"] == scope)]
    wide = sub.pivot(index="window", columns="strategy", values=metric).reindex(columns=list(models))
    wide.columns = [LABELS[s] for s in wide.columns]
    wide.index.name = "T"
    return wide


def describe_sweep(agg: pd.DataFrame, common_start: pd.Timestamp, sweep: dict) -> list[str]:
    common = agg[agg["scope"] == "common"]
    n_common = int(common["n_periods"].max())
    last = max(result.periods["hold_end"].max() for pair in sweep.values() for result in pair)
    models = ("minvar_sample", "minvar_ledoit_wolf", "minvar_clipped", "minvar_clipped_unit", "minvar_rie_raw", "minvar_rie")
    fmt3 = lambda v: f"{v:.3f}"  # noqa: E731
    fmt_pct = lambda v: f"{v:.2%}"  # noqa: E731
    lines = [f"--- Rolling window sweep, common period: {n_common} holding periods from {common_start.date()} to {last.date()} ---", ""]
    for panel, label in (("real", REAL_LABEL), ("gaussian", SIM_LABEL)):
        lines += [f"=== {label} ===", ""]
        lines += ["Median realised / predicted volatility (1 is honest):", _pivot(agg, panel, "common", "median_vol_ratio", models).to_string(float_format=fmt3), ""]
        lines += ["Realised volatility, annualised (lower is better):", _pivot(agg, panel, "common", "ann_volatility", models).to_string(float_format=fmt_pct), ""]
        lines += ["Turnover per rebalance:", _pivot(agg, panel, "common", "mean_turnover", models).to_string(float_format=fmt3), ""]
    real_vol = _pivot(agg, "real", "common", "ann_volatility", models)
    best = real_vol.idxmin()
    lines += ["Window with the lowest realised volatility on the real market: " + ", ".join(f"{name} T={int(w)}" for name, w in best.items())]
    return lines


def plot_sweep(agg: pd.DataFrame, common_start: pd.Timestamp, path: Path) -> None:
    common = agg[agg["scope"] == "common"]
    windows = sorted(common["window"].unique())
    n_common = int(common["n_periods"].max())
    metrics = (
        ("median_vol_ratio", "honesty: median realised / predicted", "log", 1.0),
        ("ann_volatility", "risk: realised volatility", "linear", None),
        ("mean_turnover", "trading: turnover per rebalance", "log", None),
    )
    fig, axes = plt.subplots(len(metrics), len(FOCUS), figsize=(4.0 * len(FOCUS), 3.3 * len(metrics)), sharex=True, sharey="row")
    for r, (metric, ylabel, yscale, reference) in enumerate(metrics):
        for c, strategy in enumerate(FOCUS):
            ax = axes[r, c]
            color = COLORS[strategy]
            for panel, style in (("gaussian", "dashed"), ("real", "solid")):
                sub = common[(common["panel"] == panel) & (common["strategy"] == strategy)].set_index("window").reindex(windows)
                y = sub[metric].to_numpy(dtype=float)
                if panel == "real":
                    ax.plot(windows, y, color=color, linewidth=2.0, marker="o", markersize=6, markeredgecolor="white", markeredgewidth=0.8)
                    for x, value in zip(windows, y):
                        if np.isfinite(value):
                            ax.annotate(_fmt_metric(metric, value), (x, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5, color=MUTED)
                else:
                    ax.plot(windows, y, color=color, linewidth=1.3, linestyle="--", marker="o", markersize=6, markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.3)
            if reference is not None:
                ax.axhline(reference, color="#888888", linewidth=1.0, linestyle=":", zorder=1)
            ax.axvline(456, color="#cccccc", linewidth=1.0, linestyle="--", zorder=0)
            if r == 0:
                ax.set_title(LABELS[strategy], fontsize=10.5)
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.set_xticks(windows)
            ax.set_xticklabels([str(w) for w in windows], fontsize=8.5)
            ax.minorticks_off()
        first = axes[r, 0]
        if yscale == "log":
            first.set_yscale("log")
            lo, hi = first.get_ylim()
            ticks = [t for t in LOG_TICKS if lo <= t <= hi]
            first.set_yticks(ticks)
            first.set_yticklabels([f"{t:g}" for t in ticks])
        else:
            first.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        for ax in axes[r]:
            ax.minorticks_off()
    for ax in axes[-1]:
        ax.set_xlabel("window length T in sessions")
    axes[0, 0].text(456, 0.03, " T = N", transform=axes[0, 0].get_xaxis_transform(), fontsize=8, color="#999999", va="bottom")
    handles = [
        Line2D([], [], color="#333333", linewidth=2.0, marker="o", markersize=6, markeredgecolor="white", label=REAL_LABEL),
        Line2D([], [], color="#333333", linewidth=1.3, linestyle="--", marker="o", markersize=6, markerfacecolor="white", markeredgecolor="#333333", label=SIM_LABEL),
        Line2D([], [], color="#888888", linewidth=1.0, linestyle=":", label="honest model: ratio 1"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        f"Does a longer window help? Every window judged on the same {n_common} holding periods, {common_start.date()} onward.",
        x=0.02, ha="left", fontsize=12, fontweight="medium",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- helpers


def _fmt(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _fmt_metric(metric: str, value: float) -> str:
    if metric == "ann_volatility":
        return f"{value:.1%}"
    return _fmt(value)


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
