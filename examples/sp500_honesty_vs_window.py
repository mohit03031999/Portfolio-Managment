"""Sweep the window length in the Gaussian world: how much of each model's gap is noise, how much is bias?

``sp500_simulated_backtest.py`` showed that the models under-state their
risk almost as much on Gaussian returns from one fixed covariance as on the
real market. What is left in that world has two sources:

* estimation noise, which shrinks as the window ``T`` grows;
* model bias, which an unlimited sample would not remove (a diagonal matrix
  in a correlated market, or an edge rule that files true modes under noise).

This script separates them. For each window length it draws independent
Gaussian samples of ``T`` days from the truth Σ, fits every estimator, and
compares the minimum-variance portfolio's true volatility ``sqrt(wᵀΣw)``
with the volatility the model predicted, ``sqrt(wᵀΣ̂w)``. Σ is known, so
nothing about a 21-day hold enters. A ``T = ∞`` column hands each estimator
the truth itself (sample eigenvalues equal the true ones, ``q = 0``): what
remains there is bias, and the excess at each finite ``T`` is noise.

A second measure, true volatility over the best possible volatility
``1 / sqrt(1ᵀΣ⁻¹1)``, says how far each model's portfolio is from the
optimum the truth allows, which honesty alone does not tell.

    .venv/bin/python examples/sp500_honesty_vs_window.py
    .venv/bin/python examples/sp500_honesty_vs_window.py --replot
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from market_analyzer.matrices import CorrelationModel  # noqa: E402
from market_analyzer.optimizer import ESTIMATORS, minimum_variance_weights, portfolio_variance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "output" / "sp500_marchenko_pastur" / "prices.csv"
OUT = ROOT / "output" / "sp500_honesty_vs_window"
ROLLING = ROOT / "output" / "sp500_simulated_backtest" / "decomposition.csv"

WINDOWS = (250, 500, 1000, 2000, 5000, 20000)
INFINITY = "inf"
MODELS = (
    "minvar_sample",
    "minvar_ledoit_wolf",
    "minvar_clipped",
    "minvar_clipped_unit",
    "minvar_diagonal",
    "equal_weight",
)
KEYS = {name: name[len("minvar_"):] for name in MODELS if name != "equal_weight"}
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
MARKERS = {
    "minvar_sample": "o",
    "minvar_ledoit_wolf": "s",
    "minvar_clipped": "D",
    "minvar_clipped_unit": "^",
    "minvar_diagonal": "v",
    "equal_weight": "P",
}
MUTED = "#52514e"
LOG_TICKS = (0.5, 1, 1.5, 2, 3, 5, 10, 20, 50)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", type=Path, default=PRICES)
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS))
    parser.add_argument("--reps", type=int, default=20, help="Draws per window up to T = 2000; longer windows use fewer, never below 4.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replot", action="store_true", help="Redraw the figures from the saved sweep instead of drawing new samples.")
    args = parser.parse_args()

    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        draws = pd.read_csv(args.out / "draws.csv")
        limits = pd.read_csv(args.out / "limit.csv")
    else:
        draws, limits = run_sweep(args)
    table = summarize(draws, limits)
    table.to_csv(args.out / "summary_by_window.csv", index=False)
    rolling = load_rolling()
    text = describe(table, rolling, draws, limits)
    print(text)
    (args.out / "summary.txt").write_text(text + "\n")
    plot_sweep(table, rolling, args.out / "honesty_vs_window.png")
    plot_modes(table, limits, args.out / "signal_modes_vs_window.png")
    print(f"Wrote {args.out}")


def run_sweep(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = load_returns(args.prices)
    truth = CorrelationModel.from_returns(returns)
    sigma = truth.cleaned_covariance.to_numpy(dtype=float)
    tickers = [str(column) for column in returns.columns]
    n_assets = len(tickers)
    optimal = optimal_volatility(sigma)
    print(
        f"Truth: cleaned covariance of {returns.shape[0]} sessions × {n_assets} tickers, edge rule {truth.edge_rule}, "
        f"{truth.n_signal} signal modes. Best possible daily volatility {optimal:.5f} ({optimal * np.sqrt(252):.3%} annualised)."
    )
    chol = np.linalg.cholesky(sigma)
    rng = np.random.default_rng(args.seed)
    rows = []
    for window in args.windows:
        reps = max(4, int(round(args.reps * min(1.0, 2000.0 / window))))
        for rep in range(reps):
            sample = pd.DataFrame(rng.standard_normal((window, n_assets)) @ chol.T, columns=tickers)
            for name, record in fit_all(sample, sigma, optimal).items():
                rows.append({"window": window, "rep": rep, "model": name, **record})
        done = pd.DataFrame([row for row in rows if row["window"] == window])
        medians = done.groupby("model")["ratio"].median()
        print(f"T = {window:>6}, q = {n_assets / window:.3f}, {reps} draws: " + ", ".join(f"{LABELS[m].split(',')[0]} {medians[m]:.2f}" for m in MODELS))
    draws = pd.DataFrame(rows)
    draws.to_csv(args.out / "draws.csv", index=False)
    limits = pd.DataFrame([{"model": name, **record} for name, record in limit_all(sigma, optimal, truth.n_signal).items()])
    limits.to_csv(args.out / "limit.csv", index=False)
    return draws, limits


def load_returns(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    returns = prices.pct_change(fill_method=None).iloc[1:]
    complete = returns.dropna(axis=1)
    return complete.loc[:, complete.std(ddof=1) > 0]


def optimal_volatility(sigma: np.ndarray) -> float:
    ones = np.ones(len(sigma))
    return float(np.sqrt(1.0 / (ones @ np.linalg.solve(sigma, ones))))


def evaluate(weights: np.ndarray, predicted: float, sigma: np.ndarray, optimal: float) -> dict[str, float]:
    true = float(np.sqrt(weights @ sigma @ weights))
    return {
        "ratio": true / predicted if predicted > 0 else float("nan"),
        "excess": true / optimal,
        "leverage": float(np.abs(weights).sum()),
        "predicted": predicted,
        "true": true,
    }


def fit_all(sample: pd.DataFrame, sigma: np.ndarray, optimal: float) -> dict[str, dict[str, float]]:
    """Fit every estimator on one sample; return the honesty ratio and excess risk per model."""
    out = {}
    n_assets = sample.shape[1]
    for name, key in KEYS.items():
        estimate = ESTIMATORS[key](sample)
        weights = minimum_variance_weights(estimate.covariance)
        predicted = float(np.sqrt(max(portfolio_variance(weights, estimate.covariance), 0.0)))
        record = evaluate(weights.to_numpy(dtype=float), predicted, sigma, optimal)
        record["n_signal"] = float(estimate.info.get("n_signal", np.nan))
        record["shrinkage"] = float(estimate.info.get("shrinkage", np.nan))
        out[name] = record
    equal = np.full(n_assets, 1.0 / n_assets)
    sample_cov = sample.cov().to_numpy(dtype=float)
    record = evaluate(equal, float(np.sqrt(equal @ sample_cov @ equal)), sigma, optimal)
    record["n_signal"] = np.nan
    record["shrinkage"] = np.nan
    out["equal_weight"] = record
    return out


def limit_all(sigma: np.ndarray, optimal: float, true_modes: int) -> dict[str, dict[str, float]]:
    """The T = ∞ limit: each estimator applied to the truth itself (q = 0)."""
    n_assets = len(sigma)
    ones = np.ones(n_assets)
    volatilities = np.sqrt(np.diag(sigma))
    correlation = sigma / np.outer(volatilities, volatilities)
    eigenvalues, eigenvectors = np.linalg.eigh((correlation + correlation.T) / 2.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    def minvar(matrix: np.ndarray) -> tuple[np.ndarray, float]:
        weights = np.linalg.solve(matrix, ones)
        weights = weights / weights.sum()
        return weights, float(np.sqrt(weights @ matrix @ weights))

    out = {}
    # With unlimited data the sample covariance is the truth, Ledoit-Wolf's
    # intensity goes to zero, and the fitted edge keeps every true mode
    # (q → 0 shrinks the noise band to nothing), so all three become the truth.
    weights, predicted = minvar(sigma)
    for name in ("minvar_sample", "minvar_ledoit_wolf", "minvar_clipped"):
        record = evaluate(weights, predicted, sigma, optimal)
        record["n_signal"] = float(true_modes) if name == "minvar_clipped" else np.nan
        record["shrinkage"] = 0.0 if name == "minvar_ledoit_wolf" else np.nan
        out[name] = record
    # The unit edge tends to λ₊ = 1 as q → 0: it keeps the true eigenvalues above
    # one and averages the rest, exactly as clip_eigenvalues would with q = 0.
    n_signal = int(np.sum(eigenvalues > 1.0))
    clipped = eigenvalues.copy()
    clipped[n_signal:] = eigenvalues[n_signal:].mean()
    cleaned = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    scale = np.sqrt(np.diag(cleaned))
    cleaned = cleaned / np.outer(scale, scale)
    np.fill_diagonal(cleaned, 1.0)
    matrix = volatilities[:, None] * cleaned * volatilities[None, :]
    weights, predicted = minvar(matrix)
    out["minvar_clipped_unit"] = {**evaluate(weights, predicted, sigma, optimal), "n_signal": float(n_signal), "shrinkage": np.nan}
    weights, predicted = minvar(np.diag(np.diag(sigma)))
    out["minvar_diagonal"] = {**evaluate(weights, predicted, sigma, optimal), "n_signal": np.nan, "shrinkage": np.nan}
    equal = ones / n_assets
    out["equal_weight"] = {**evaluate(equal, float(np.sqrt(equal @ sigma @ equal)), sigma, optimal), "n_signal": np.nan, "shrinkage": np.nan}
    for record in out.values():
        record["true_modes"] = float(true_modes)
    return out


def summarize(draws: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
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
            "n_signal_mean": grouped["n_signal"].mean(),
            "shrinkage_mean": grouped["shrinkage"].mean(),
            "draws": grouped.size(),
        }
    ).reset_index()
    limit_rows = limits.rename(columns={"ratio": "ratio_median", "excess": "excess_median", "leverage": "leverage_median", "n_signal": "n_signal_mean", "shrinkage": "shrinkage_mean"})
    limit_rows["window"] = INFINITY
    limit_rows["draws"] = 0
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


def load_rolling() -> pd.DataFrame | None:
    """Medians of the rolling Gaussian backtest at T = 500 and 1000, if that study has been run."""
    if not ROLLING.is_file():
        return None
    frame = pd.read_csv(ROLLING)
    frame = frame[frame["strategy"].isin(MODELS)]
    return frame[["window", "strategy", "median_vol_ratio_gaussian"]].rename(columns={"strategy": "model", "median_vol_ratio_gaussian": "ratio"})


def describe(table: pd.DataFrame, rolling: pd.DataFrame | None, draws: pd.DataFrame, limits: pd.DataFrame) -> str:
    windows = [str(w) for w in sorted(draws["window"].unique())] + [INFINITY]
    n_assets = None
    lines = ["Honesty ratio = true volatility of the fitted minimum-variance portfolio over the volatility the model predicted."]
    lines.append("Each cell is the median over independent Gaussian samples of T days from the fixed truth; T = inf feeds the estimator the truth itself.")
    lines.append("")

    def pivot(column: str, fmt: str) -> str:
        wide = table.pivot(index="window", columns="model", values=column).reindex(windows)[list(MODELS)]
        wide.columns = [LABELS[m] for m in wide.columns]
        wide.index.name = "T"
        return wide.to_string(float_format=lambda v: fmt.format(v), na_rep="-")

    lines += ["--- Honesty ratio (median across draws) ---", pivot("ratio_median", "{:.3f}"), ""]
    lines += ["--- Noise factor = ratio(T) / ratio(inf). Bias is the T = inf row above. ---", pivot("noise_factor", "{:.3f}"), ""]
    lines += ["--- True volatility / best possible volatility (median across draws) ---", pivot("excess_median", "{:.3f}"), ""]
    lines += ["--- Modes kept above the edge (mean across draws); the truth has {:d} ---".format(int(limits["true_modes"].iloc[0])), pivot("n_signal_mean", "{:.1f}"), ""]
    lines += ["--- Ledoit-Wolf shrinkage intensity (mean across draws) ---", pivot("shrinkage_mean", "{:.4f}"), ""]
    lines += ["--- Gross leverage (median across draws) ---", pivot("leverage_median", "{:.2f}"), ""]
    sample = table[table["model"] == "minvar_sample"].set_index("window")
    lines.append("--- Sample covariance against the random-matrix law 1 / (1 - q), q = N / T ---")
    n_assets = 456
    for window in windows[:-1]:
        q = n_assets / int(window)
        law = f"{1.0 / (1.0 - q):.3f}" if q < 1 else "singular (q > 1, pseudo-inverse)"
        lines.append(f"T = {window:>6}: simulated {sample.loc[window, 'ratio_median']:.3f}, law {law}")
    lines.append("")
    if rolling is not None:
        lines.append("--- Cross-check: medians of the rolling Gaussian backtest (21-day holds, one path) ---")
        for window, group in rolling.groupby("window"):
            lines.append(f"T = {int(window):>6}: " + ", ".join(f"{LABELS[m]} {group.set_index('model').loc[m, 'ratio']:.3f}" for m in MODELS if m in set(group['model'])))
        lines.append("An honest model shows about 0.98 there, not 1.00: the median of a 21-day sample standard deviation sits below its true value.")
    return "\n".join(lines)


def _positions(windows: list[str]) -> dict[str, float]:
    finite = [int(w) for w in windows if w != INFINITY]
    positions = {str(w): float(w) for w in finite}
    positions[INFINITY] = max(finite) * 5.0
    return positions


def plot_sweep(table: pd.DataFrame, rolling: pd.DataFrame | None, path: Path) -> None:
    windows = [w for w in table["window"].unique() if w != INFINITY]
    windows = sorted(windows, key=int) + [INFINITY]
    pos = _positions(windows)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    panels = (
        ("ratio", "Honesty: true volatility / predicted volatility", 1.0),
        ("excess", "Cost: true volatility / best possible volatility", 1.0),
    )
    for ax, (column, title, reference) in zip(axes, panels):
        for name in MODELS:
            sub = table[table["model"] == name].set_index("window").reindex(windows)
            x = [pos[w] for w in windows]
            color = COLORS[name]
            med = sub[f"{column}_median"].to_numpy(dtype=float)
            ax.fill_between(x, sub[f"{column}_q10"].to_numpy(dtype=float), sub[f"{column}_q90"].to_numpy(dtype=float), color=color, alpha=0.12, linewidth=0)
            ax.plot(x[:-1], med[:-1], color=color, linewidth=1.9, marker=MARKERS[name], markersize=6, markeredgecolor="white", markeredgewidth=0.8)
            ax.plot(x[-2:], med[-2:], color=color, linewidth=1.4, linestyle=(0, (2, 2)))
            ax.plot(x[-1:], med[-1:], color=color, linestyle="", marker=MARKERS[name], markersize=7, markeredgecolor="white", markeredgewidth=0.8)
        if column == "ratio" and rolling is not None:
            for _, row in rolling.iterrows():
                if str(int(row["window"])) not in pos:
                    continue
                ax.plot(pos[str(int(row["window"]))], row["ratio"], marker="s", markersize=9, markerfacecolor="none", markeredgecolor=COLORS[row["model"]], markeredgewidth=1.6, linestyle="")
        ax.axhline(reference, color="#888888", linewidth=1.0, linestyle=":", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([pos[w] for w in windows])
        ax.set_xticklabels([w if w != INFINITY else "∞" for w in windows])
        ax.axvline(pos[INFINITY] / 2.2, color="#cccccc", linewidth=1.0, linestyle="--", zorder=0)
        lo, hi = ax.get_ylim()
        ticks = [t for t in LOG_TICKS if lo <= t <= hi]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("window length T in sessions (N = 456 tickers)")
        ax.set_title(title)
    axes[0].set_ylabel("ratio, log scale")
    handles = [
        Line2D([], [], color=COLORS[name], linewidth=1.9, marker=MARKERS[name], markersize=6, markeredgecolor="white", label=LABELS[name])
        for name in MODELS
    ]
    extras = [Line2D([], [], color="#888888", linewidth=1.0, linestyle=":", label="honest model / optimal portfolio")]
    if rolling is not None:
        extras.append(Line2D([], [], marker="s", markersize=9, markerfacecolor="none", markeredgecolor="#333333", markeredgewidth=1.6, linestyle="", label="rolling Gaussian backtest, 21-day holds"))
    fig.legend(handles=handles + extras, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Gaussian returns from one fixed covariance, independent samples per window length. Band: 10th to 90th percentile across draws.\n"
        "The ∞ column applies each estimator to the truth itself, so whatever is left there is model bias, and the rest at each T is estimation noise.",
        x=0.02, ha="left", fontsize=11.5, fontweight="medium", linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)


def plot_modes(table: pd.DataFrame, limits: pd.DataFrame, path: Path) -> None:
    windows = [w for w in table["window"].unique() if w != INFINITY]
    windows = sorted(windows, key=int) + [INFINITY]
    pos = _positions(windows)
    true_modes = int(limits["true_modes"].iloc[0])
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ("minvar_clipped", "minvar_clipped_unit"):
        sub = table[table["model"] == name].set_index("window").reindex(windows)
        x = [pos[w] for w in windows]
        y = sub["n_signal_mean"].to_numpy(dtype=float)
        color = COLORS[name]
        ax.plot(x[:-1], y[:-1], color=color, linewidth=1.9, marker=MARKERS[name], markersize=6, markeredgecolor="white", label=LABELS[name])
        ax.plot(x[-2:], y[-2:], color=color, linewidth=1.4, linestyle=(0, (2, 2)))
        ax.plot(x[-1:], y[-1:], color=color, linestyle="", marker=MARKERS[name], markersize=7, markeredgecolor="white")
        ax.annotate(f"{y[-1]:.0f}", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9.5, color=color)
        if y[0] > true_modes:
            ax.annotate(
                f"T < N: every non-zero eigenvalue\ncounts as signal ({y[0]:.0f} of {windows[0]} possible)",
                (x[0], y[0]), xytext=(14, -4), textcoords="offset points", va="top", fontsize=9, color=color,
            )
    ax.axhline(true_modes, color="#555555", linewidth=1.0, linestyle=":")
    ax.text(pos[windows[0]], true_modes - 3, f"the truth has {true_modes} modes above its own noise average", fontsize=9.5, color="#555555", va="top")
    ax.axvline(pos[INFINITY] / 2.2, color="#cccccc", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([pos[w] for w in windows])
    ax.set_xticklabels([w if w != INFINITY else "∞" for w in windows])
    ax.minorticks_off()
    top = max(float(true_modes), float(table.loc[table["model"].isin(("minvar_clipped", "minvar_clipped_unit")), "n_signal_mean"].max()))
    ax.set_ylim(0, top * 1.12)
    ax.set_xlabel("window length T in sessions (N = 456 tickers)")
    ax.set_ylabel("eigenvalues kept above the edge")
    ax.set_title("How many of the truth's modes each edge rule resolves, Gaussian samples from the fixed truth")
    ax.legend(loc="center right", fontsize=9.5)
    fig.tight_layout()
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
