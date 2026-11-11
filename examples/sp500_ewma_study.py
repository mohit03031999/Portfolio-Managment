"""Does forgetting old days help? Exponentially weighted covariance with the RIE as the eigenvalue map.

The window sweeps showed that on the real market data older than about two
years costs the clipped models more than it saves, while in the Gaussian
world every extra day helps. Exponential weighting is the cheap answer:
keep a long window, but let a day's weight halve every ``halflife``
sessions. The price is noise, because fewer days effectively count, and
Pafka, Potters and Kondor (2004) showed the noise is not Marchenko-Pastur
any more: the spectrum of a weighted pure-noise matrix has a softer lower
edge and a longer upper tail. :mod:`market_analyzer.rie` carries that
noise model, so the RIE knows exactly how much noise each half-life buys.

Three parts, each saved so the comparison can be rebuilt with ``--replot``:

1. Monte Carlo on the fixed Gaussian truth, window T = 1000: the price of
   forgetting when nothing changes, per half-life, for the weighted sample
   covariance, clipping at the effective q, the RIE with the exact weighted
   noise, the RIE with the flat formula at the effective q, and the bare RIE.
2. The rolling backtest at window T = 1000 on the real panel and on the
   Gaussian panel, half-lives 63, 126, 252 and 504 sessions, against the
   saved flat runs at T = 1000 and T = 500 on the same holding periods.
3. Figures of the noise model itself: the weighted pure-noise spectrum
   against its theory and against Marchenko-Pastur, and the cleaning
   functions on one real window.

    .venv/bin/python examples/sp500_ewma_study.py --part montecarlo
    .venv/bin/python examples/sp500_ewma_study.py --part rolling
    .venv/bin/python examples/sp500_ewma_study.py --replot
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
from market_analyzer.matrices import CorrelationModel, marchenko_pastur_density  # noqa: E402
from market_analyzer.optimizer import Estimator, ewma_covariance, ewma_estimator, minimum_variance_weights, portfolio_variance  # noqa: E402
from market_analyzer.rie import effective_observations, exponential_weights, rie_eigenvalues, weighted_r_transform  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "output" / "sp500_marchenko_pastur" / "prices.csv"
OUT = ROOT / "output" / "sp500_ewma"
SAVED_FLAT = {"base": ROOT / "output" / "sp500_simulated_backtest", "rie": ROOT / "output" / "sp500_rie"}

WINDOW = 1000
HOLDING = 21
HALFLIVES = (63, 126, 252, 504)
FLAT = "flat"
PANELS = ("real", "gaussian")
CLEANINGS = ("none", "clipped", "rie")
# Monte Carlo variants: (label key, cleaning, options)
VARIANTS = {
    "sample": ("none", {}),
    "clipped": ("clipped", {}),
    "rie": ("rie", {}),
    "rie_effective": ("rie", {"noise": "effective"}),
    "rie_bare": ("rie", {"regularize": False, "sort": False}),
}
VARIANT_LABELS = {
    "sample": "Weighted sample covariance",
    "clipped": "MP clipped at the effective q",
    "rie": "RIE, weighted-noise model + lift",
    "rie_effective": "RIE, flat formula at the effective q + lift",
    "rie_bare": "RIE, weighted-noise model, bare",
}
CLEANING_LABELS = {"none": "Weighted sample covariance", "clipped": "MP clipped at the effective q", "rie": "RIE, weighted noise + lift"}
# Same slots as the other S&P figures: sample blue, clipped green (unit edge), RIE dark red, bare RIE copper.
COLORS = {"sample": "#2a78d6", "clipped": "#1baf7a", "rie": "#a61b1b", "rie_effective": "#a61b1b", "rie_bare": "#b5651d", "none": "#2a78d6"}
MARKERS = {"sample": "o", "clipped": "^", "rie": "P", "rie_effective": "s", "rie_bare": "x", "none": "o"}
FLAT_REFERENCE = {"minvar_rie": "RIE, flat T = 1000", "minvar_clipped": "MP clipped fitted, flat T = 1000", "minvar_sample": "Sample, flat T = 1000"}
EPISODES = (("Covid crash", "2020-02-19", "2020-03-23"), ("2022 bear market", "2022-01-03", "2022-10-12"))
EPISODE_LEGEND = "shaded: Covid crash (Feb–Mar 2020), 2022 bear market (Jan–Oct 2022)"
REAL_LABEL = "S&P 500 returns"
SIM_LABEL = "Gaussian returns from one fixed covariance"
MUTED = "#52514e"
LOG_TICKS = (0.3, 0.5, 1, 1.5, 2, 3, 5, 10, 20, 50)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", type=Path, default=PRICES)
    parser.add_argument("--part", choices=("montecarlo", "rolling", "all"), default="all")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replot", action="store_true", help="Run nothing; build the tables and figures from what is saved.")
    args = parser.parse_args()

    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.replot:
        parts = ("montecarlo", "rolling") if args.part == "all" else (args.part,)
        returns = load_returns(args.prices)
        fake, truth = simulate(returns, args.seed)
        if "montecarlo" in parts:
            run_montecarlo(args, returns, truth)
        if "rolling" in parts:
            run_rolling(args, returns, fake)
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


def strategy_name(halflife: int | str, cleaning: str) -> str:
    return f"minvar_ewma{halflife}_{cleaning}"


def registry() -> dict[str, Estimator]:
    return {f"ewma{h}_{c}": ewma_estimator(h, c) for h in HALFLIVES for c in CLEANINGS}


# ----------------------------------------------------------------------------- part 1: Monte Carlo


def run_montecarlo(args: argparse.Namespace, returns: pd.DataFrame, truth: CorrelationModel) -> None:
    sigma = truth.cleaned_covariance.to_numpy(dtype=float)
    tickers = [str(column) for column in returns.columns]
    n_assets = len(tickers)
    ones = np.ones(n_assets)
    optimal_weights = np.linalg.solve(sigma, ones)
    optimal_weights /= optimal_weights.sum()
    optimal = float(np.sqrt(optimal_weights @ sigma @ optimal_weights))
    chol = np.linalg.cholesky(sigma)
    rng = np.random.default_rng(args.seed + 1)
    rows = []
    for rep in range(args.reps):
        sample = pd.DataFrame(rng.standard_normal((WINDOW, n_assets)) @ chol.T, columns=tickers)
        for halflife in HALFLIVES + (None,):
            for key, (cleaning, options) in VARIANTS.items():
                estimate = ewma_covariance(sample, halflife, cleaning=cleaning, **options)
                weights = minimum_variance_weights(estimate.covariance)
                predicted = float(np.sqrt(max(portfolio_variance(weights, estimate.covariance), 0.0)))
                vector = weights.to_numpy(dtype=float)
                true = float(np.sqrt(vector @ sigma @ vector))
                rows.append(
                    {
                        "rep": rep,
                        "halflife": FLAT if halflife is None else halflife,
                        "variant": key,
                        "ratio": true / predicted,
                        "excess": true / optimal,
                        "leverage": float(np.abs(vector).sum()),
                        "q": estimate.info["q"],
                        "effective_observations": estimate.info["effective_observations"],
                        "n_regularized": estimate.info.get("n_regularized", np.nan),
                        "n_signal": estimate.info.get("n_signal", np.nan),
                        "min_cleaned": estimate.info.get("min_cleaned", np.nan),
                    }
                )
        done = pd.DataFrame([r for r in rows if r["rep"] == rep])
        print(f"draw {rep + 1}/{args.reps}: " + ", ".join(f"h={h}: RIE {done[(done['halflife'] == h) & (done['variant'] == 'rie')]['ratio'].iloc[0]:.2f}" for h in [*HALFLIVES, FLAT]))
    pd.DataFrame(rows).to_csv(args.out / "draws.csv", index=False)


# ----------------------------------------------------------------------------- part 2: rolling backtests


def run_rolling(args: argparse.Namespace, returns: pd.DataFrame, fake: pd.DataFrame) -> None:
    strategies = tuple(strategy_name(h, c) for h in HALFLIVES for c in CLEANINGS)
    config = BacktestConfig(window=WINDOW, holding=HOLDING, strategies=strategies)
    estimators = registry()
    for panel_name, panel in (("real", returns), ("gaussian", fake)):
        folder = args.out / f"{panel_name}_T{WINDOW}"
        if (folder / "periods.csv").is_file():
            print(f"{panel_name}: reusing saved run")
            continue
        result = run_backtest(panel, config, estimators)
        result.save(folder)
        summary = result.summary()
        print(f"{panel_name} T = {WINDOW}: {int(summary['n_periods'].max())} periods")
        for h in HALFLIVES:
            print("   h = %4d: " % h + ", ".join(f"{c} ratio {summary.loc[strategy_name(h, c), 'median_vol_ratio']:.3f} vol {summary.loc[strategy_name(h, c), 'ann_volatility']:.2%}" for c in CLEANINGS))


# ----------------------------------------------------------------------------- report


def report(args: argparse.Namespace) -> None:
    lines: list[str] = []
    returns = load_returns(args.prices)
    fake, truth = simulate(returns, args.seed)
    n_assets = returns.shape[1]
    lines.append(
        f"Panel: {returns.shape[0]} sessions × {n_assets} tickers, {returns.index[0].date()} → {returns.index[-1].date()}. "
        f"Window T = {WINDOW}; half-lives {', '.join(map(str, HALFLIVES))} sessions and flat."
    )
    weights = {h: exponential_weights(WINDOW, h) for h in HALFLIVES}
    lines.append("Effective observations (Kish) and effective q = N Σ w² at T = 1000: " + ", ".join(f"h={h}: T_eff {effective_observations(w):.0f}, q {n_assets * np.sum(w**2):.2f}" for h, w in weights.items()) + f"; flat: T_eff 1000, q {n_assets / WINDOW:.2f}.")
    lines.append("")

    lines += noise_section(n_assets, args.out / "noise_spectrum.png")
    lines += cleaning_section(returns, fake, truth, args.out / "cleaning_function.png")

    if (args.out / "draws.csv").is_file():
        draws = pd.read_csv(args.out / "draws.csv")
        table = summarize_montecarlo(draws)
        table.to_csv(args.out / "summary_by_halflife.csv", index=False)
        lines += describe_montecarlo(table)
        plot_montecarlo(table, args.out / "halflife_sweep_gaussian.png")
    else:
        lines += ["Monte Carlo part not run yet (--part montecarlo).", ""]

    rolling = load_rolling(args.out)
    if rolling is not None:
        agg = aggregate_rolling(rolling)
        agg.to_csv(args.out / "rolling_summary.csv", index=False)
        lines += describe_rolling(agg, rolling)
        plot_rolling(agg, args.out / "rolling_halflife.png")
        plot_series(rolling, args.out / "vol_ratio_over_time.png")
    else:
        lines += ["Rolling part not run yet (--part rolling), or the flat runs of sp500_rie_study.py / sp500_simulated_backtest.py are missing.", ""]

    text = "\n".join(lines)
    print(text)
    (args.out / "summary.txt").write_text(text + "\n")
    print(f"Wrote {args.out}")


# ----------------------------------------------------------------------------- the noise model


def theory_density(grid: np.ndarray, weights: np.ndarray, n_assets: int, eta: float = 2e-3, iterations: int = 4000) -> np.ndarray:
    """Density of the weighted pure-noise spectrum from the fixed point ``g = 1 / (z − R(g))``."""
    z = grid - 1j * eta
    g = 1.0 / (z - 1.0)
    for _ in range(iterations):
        g = 0.5 * g + 0.5 / (z - weighted_r_transform(g, weights, n_assets))
    return np.maximum(np.imag(g) / np.pi, 0.0)


def noise_section(n_assets: int, path: Path) -> list[str]:
    rng = np.random.default_rng(123)
    cases = (126, 252)
    fig, axes = plt.subplots(1, len(cases), figsize=(7.2 * len(cases), 5.2), sharey=True)
    lines = ["--- The weighted pure-noise spectrum (N = %d, T = %d) ---" % (n_assets, WINDOW)]
    for ax, halflife in zip(axes, cases):
        w = exponential_weights(WINDOW, halflife)
        q_eff = float(n_assets * np.sum(w**2))
        values = []
        for _ in range(4):
            y = rng.standard_normal((WINDOW, n_assets))
            values.append(np.linalg.eigvalsh((y * w[:, None]).T @ y))
        eigenvalues = np.concatenate(values)
        top = max(eigenvalues.max() * 1.05, (1 + np.sqrt(q_eff)) ** 2 * 1.1)
        grid = np.linspace(1e-3, top, 400)
        theory = theory_density(grid, w, n_assets)
        mp = marchenko_pastur_density(grid, q_eff)
        ax.hist(eigenvalues, bins=60, range=(0, top), density=True, color="#cfd5dc", edgecolor="white", linewidth=0.4, label="simulated weighted noise, 4 draws")
        ax.plot(grid, theory, color=COLORS["rie"], linewidth=2.2, label="weighted-noise theory (R-transform)")
        ax.plot(grid, mp, color=COLORS["clipped"], linewidth=1.8, linestyle="--", label=f"Marchenko-Pastur at the same q = {q_eff:.2f}")
        ax.set_title(f"half-life {halflife}: T_eff = {effective_observations(w):.0f}, q = {q_eff:.2f}")
        ax.set_xlabel("eigenvalue of the weighted correlation matrix")
        ax.legend(fontsize=9)
        lower, upper = (1 - np.sqrt(q_eff)) ** 2, (1 + np.sqrt(q_eff)) ** 2
        lines.append(
            f"half-life {halflife}: q = {q_eff:.3f}; simulated eigenvalues span {eigenvalues.min():.3f}–{eigenvalues.max():.3f}, "
            f"Marchenko-Pastur edges at the same q {lower:.3f}–{upper:.3f}; variance {eigenvalues.var():.3f} vs q {q_eff:.3f}."
        )
    axes[0].set_ylabel("density")
    fig.suptitle(
        "Weighting changes the shape of the noise, not just its amount: a softer lower edge and a longer upper tail than Marchenko-Pastur at the same variance.",
        x=0.02, ha="left", fontsize=12, fontweight="medium",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)
    lines.append("")
    return lines


def cleaning_section(returns: pd.DataFrame, fake: pd.DataFrame, truth: CorrelationModel, path: Path) -> list[str]:
    """Flat RIE against weighted RIE (exact and effective-q) on the last T = 1000 window of both panels."""
    halflife = 126
    vols = np.sqrt(np.diag(truth.cleaned_covariance.to_numpy(dtype=float)))
    true_corr = truth.cleaned_covariance.to_numpy(dtype=float) / np.outer(vols, vols)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), sharey=True)
    lines = [f"--- Cleaning functions on the last T = {WINDOW} window, half-life {halflife} ---"]
    for ax, (panel_name, panel) in zip(axes, (("real", returns), ("gaussian", fake))):
        sample = panel.iloc[-WINDOW:]
        x = sample.to_numpy(dtype=float)
        n_assets = x.shape[1]
        series = {}
        for label, w in (("flat", np.full(WINDOW, 1.0 / WINDOW)), ("ewma", exponential_weights(WINDOW, halflife))):
            xc = x - w @ x
            cov = (xc * w[:, None]).T @ xc / (1 - np.sum(w**2))
            d = np.sqrt(np.diag(cov))
            corr = cov / np.outer(d, d)
            eigenvalues, eigenvectors = np.linalg.eigh((corr + corr.T) / 2)
            eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
            q = float(n_assets * np.sum(w**2))
            exact = rie_eigenvalues(eigenvalues, q, weights=None if label == "flat" else w)
            series[label] = {
                "lam": eigenvalues,
                "exact": exact.cleaned,
                "effective": rie_eigenvalues(eigenvalues, q).cleaned if label == "ewma" else None,
                "oracle": np.einsum("ij,jk,ki->i", eigenvectors.T, true_corr, eigenvectors) if panel_name == "gaussian" else None,
                "q": q,
                "lifted": exact.n_regularized,
            }
        flat, ewma = series["flat"], series["ewma"]
        lo = max(min(flat["lam"].min(), ewma["lam"].min()) * 0.7, 1e-3)
        hi = max(flat["lam"].max(), ewma["lam"].max()) * 1.6
        ax.plot([lo, hi], [lo, hi], color="#999999", linewidth=1.0, linestyle=":", label="no cleaning: ξ = λ")
        ax.scatter(flat["lam"], flat["exact"], s=14, color="#888888", edgecolor="white", linewidth=0.3, zorder=3, label=f"flat window, RIE (q = {flat['q']:.2f})")
        ax.scatter(ewma["lam"], ewma["effective"], s=26, facecolor="none", edgecolor=COLORS["rie_bare"], linewidth=0.9, zorder=4, label=f"half-life {halflife}, flat formula at q = {ewma['q']:.2f} + lift")
        ax.scatter(ewma["lam"], ewma["exact"], s=14, color=COLORS["rie"], edgecolor="white", linewidth=0.3, zorder=5, label=f"half-life {halflife}, weighted-noise RIE + lift")
        if ewma["oracle"] is not None:
            ax.scatter(ewma["lam"], ewma["oracle"], s=34, facecolor="none", edgecolor="#6250d6", linewidth=0.9, zorder=6, label="oracle for the weighted matrix's eigenvectors")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ticks = [t for t in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300) if lo <= t <= hi]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_title(REAL_LABEL if panel_name == "real" else SIM_LABEL)
        ax.set_xlabel("sample correlation eigenvalue λ (log)")
        ax.legend(loc="upper left", fontsize=8.5)
        bottom = slice(-20, None)
        lines.append(
            f"{panel_name}: flat q {flat['q']:.2f} lifted {flat['lifted']}; weighted q {ewma['q']:.2f} lifted {ewma['lifted']}. "
            f"Smallest 20 eigenvalues: weighted sample {ewma['lam'][bottom].mean():.3f} → exact RIE {ewma['exact'][bottom].mean():.3f}, "
            f"effective-q RIE {ewma['effective'][bottom].mean():.3f}" + (f", oracle {ewma['oracle'][bottom].mean():.3f}" if ewma["oracle"] is not None else "")
            + f". Largest: weighted sample {ewma['lam'][0]:.1f} → exact {ewma['exact'][0]:.1f}, effective-q {ewma['effective'][0]:.1f}" + (f", oracle {ewma['oracle'][0]:.1f}" if ewma["oracle"] is not None else "") + "."
        )
    axes[0].set_ylabel("cleaned eigenvalue ξ (log)")
    fig.suptitle(
        f"Weighting shrinks the window to about {effective_observations(exponential_weights(WINDOW, halflife)):.0f} effective days, so the RIE has to clean harder.\n"
        "Filled dark red: the weighted-noise model. Hollow copper: the flat formula at the same effective q. Grey: the flat window.",
        x=0.02, ha="left", fontsize=12, fontweight="medium",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)
    lines.append("")
    return lines


# ----------------------------------------------------------------------------- Monte Carlo tables and figure


def summarize_montecarlo(draws: pd.DataFrame) -> pd.DataFrame:
    grouped = draws.groupby(["halflife", "variant"])
    table = pd.DataFrame(
        {
            "ratio_median": grouped["ratio"].median(),
            "ratio_q10": grouped["ratio"].quantile(0.10),
            "ratio_q90": grouped["ratio"].quantile(0.90),
            "excess_median": grouped["excess"].median(),
            "excess_q10": grouped["excess"].quantile(0.10),
            "excess_q90": grouped["excess"].quantile(0.90),
            "leverage_median": grouped["leverage"].median(),
            "q": grouped["q"].first(),
            "effective_observations": grouped["effective_observations"].first(),
            "n_regularized_mean": grouped["n_regularized"].mean(),
            "n_signal_mean": grouped["n_signal"].mean(),
            "draws": grouped.size(),
        }
    ).reset_index()
    table["halflife"] = table["halflife"].astype(str)
    return table


def _halflife_order(values) -> list[str]:
    finite = sorted({int(v) for v in values if str(v) != FLAT})
    return [str(v) for v in finite] + [FLAT]


def describe_montecarlo(table: pd.DataFrame) -> list[str]:
    order = _halflife_order(table["halflife"])

    def pivot(column: str, fmt: str) -> str:
        wide = table.pivot(index="halflife", columns="variant", values=column).reindex(order)[list(VARIANTS)]
        wide.columns = [VARIANT_LABELS[v] for v in wide.columns]
        wide.index.name = "half-life"
        return wide.to_string(float_format=lambda v: fmt.format(v), na_rep="-")

    return [
        f"--- Monte Carlo on the fixed Gaussian truth, window T = {WINDOW}: the price of forgetting when the market never changes ---",
        "Honesty ratio = true volatility of the fitted minimum-variance portfolio / predicted (median across draws; 1 is honest).",
        pivot("ratio_median", "{:.3f}"),
        "",
        "True volatility / best possible volatility (median across draws).",
        pivot("excess_median", "{:.3f}"),
        "",
        "Eigenvalues lifted by the inverse-Wishart step (mean across draws), and modes kept by clipping:",
        pivot("n_regularized_mean", "{:.0f}"),
        pivot("n_signal_mean", "{:.1f}"),
        "",
    ]


def plot_montecarlo(table: pd.DataFrame, path: Path) -> None:
    order = _halflife_order(table["halflife"])
    pos = {h: (float(h) if h != FLAT else 504 * 2.5) for h in order}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))
    panels = (("ratio", "Honesty: true volatility / predicted volatility"), ("excess", "Cost: true volatility / best possible volatility"))
    for ax, (column, title) in zip(axes, panels):
        for variant in VARIANTS:
            sub = table[table["variant"] == variant].set_index("halflife").reindex(order)
            x = np.array([pos[h] for h in order])
            y = sub[f"{column}_median"].to_numpy(dtype=float)
            style = dict(color=COLORS[variant], marker=MARKERS[variant], markersize=7, linewidth=2.0, label=VARIANT_LABELS[variant])
            if variant == "rie_effective":
                style.update(linestyle="--", linewidth=1.4, markerfacecolor="white", markeredgewidth=1.4)
            if variant == "rie_bare":
                style.update(linestyle=":", linewidth=1.6)
            ax.plot(x[:-1], y[:-1], **style)
            ax.plot(x[-2:], y[-2:], color=COLORS[variant], linewidth=1.0, linestyle="--", alpha=0.6)
            ax.scatter([x[-1]], [y[-1]], color=COLORS[variant], marker=MARKERS[variant], s=48, zorder=4)
            if variant == "rie":
                ax.fill_between(x[:-1], sub[f"{column}_q10"].to_numpy(dtype=float)[:-1], sub[f"{column}_q90"].to_numpy(dtype=float)[:-1], color=COLORS[variant], alpha=0.12, linewidth=0)
        ax.axhline(1.0, color="#888888", linewidth=1.0, linestyle=":", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([pos[h] for h in order])
        ax.set_xticklabels([h if h != FLAT else "flat" for h in order])
        lo, hi = ax.get_ylim()
        ticks = [t for t in LOG_TICKS if lo <= t <= hi]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel(f"half-life in sessions, window T = {WINDOW} (flat: every day counts the same)")
        ax.legend(fontsize=9, loc="upper right")
    axes[0].set_ylabel("ratio, log scale")
    fig.suptitle(
        "Gaussian world, nothing changes: forgetting can only cost. Band: 10th–90th percentile of the weighted-noise RIE across draws.",
        x=0.02, ha="left", fontsize=12, fontweight="medium",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- rolling tables and figures


def load_rolling(out: Path) -> dict[str, dict[str, BacktestResult]] | None:
    """Per panel: the EWMA run, the flat T = 1000 runs (base strategies and RIE) and the flat T = 500 runs, all restricted to the EWMA run's holds."""
    results: dict[str, dict[str, BacktestResult]] = {}
    for panel in PANELS:
        folder = out / f"{panel}_T{WINDOW}"
        if not (folder / "periods.csv").is_file():
            return None
        ewma = BacktestResult.load(folder)
        start = ewma.periods["hold_start"].min()
        entry = {"ewma": ewma}
        for kind, base in SAVED_FLAT.items():
            for window in (WINDOW, 500):
                flat_folder = base / f"{panel}_T{window}"
                if not (flat_folder / "periods.csv").is_file():
                    return None
                entry[f"{kind}_T{window}"] = BacktestResult.load(flat_folder).restrict(start=start)
        results[panel] = entry
    return results


def aggregate_rolling(rolling: dict[str, dict[str, BacktestResult]]) -> pd.DataFrame:
    rows = []
    for panel, entry in rolling.items():
        for source, result in entry.items():
            summary = result.summary()
            for strategy, row in summary.iterrows():
                if source == "ewma":
                    _, rest = strategy.split("minvar_ewma", 1)
                    halflife, cleaning = rest.split("_", 1)
                    window = WINDOW
                else:
                    halflife, cleaning = FLAT, strategy
                    window = int(source.split("_T")[1])
                rows.append(
                    {
                        "panel": panel,
                        "source": source,
                        "window": window,
                        "halflife": halflife,
                        "cleaning": cleaning,
                        "strategy": strategy,
                        "n_periods": int(row["n_periods"]),
                        "median_vol_ratio": float(row["median_vol_ratio"]),
                        "ann_volatility": float(row["ann_volatility"]),
                        "mean_turnover": float(row["mean_turnover"]),
                        "mean_gross_leverage": float(row["mean_gross_leverage"]),
                    }
                )
    return pd.DataFrame(rows)


FLAT_ROWS = (
    ("minvar_sample", WINDOW, "Sample covariance, flat T = 1000"),
    ("minvar_clipped", WINDOW, "MP clipped fitted, flat T = 1000"),
    ("minvar_clipped_unit", WINDOW, "MP clipped σ² = 1, flat T = 1000"),
    ("minvar_ledoit_wolf", WINDOW, "Ledoit-Wolf, flat T = 1000"),
    ("minvar_rie", WINDOW, "RIE, flat T = 1000"),
    ("minvar_rie", 500, "RIE, flat T = 500"),
    ("minvar_clipped", 500, "MP clipped fitted, flat T = 500"),
)


def _rolling_table(agg: pd.DataFrame, panel: str, metric: str) -> pd.DataFrame:
    sub = agg[agg["panel"] == panel]
    ewma = sub[sub["halflife"] != FLAT].copy()
    ewma["halflife"] = ewma["halflife"].astype(int)
    wide = ewma.pivot(index="halflife", columns="cleaning", values=metric).reindex(columns=list(CLEANINGS))
    wide.columns = [CLEANING_LABELS[c] for c in wide.columns]
    wide.index = [f"half-life {h}" for h in wide.index]
    flat_rows = {}
    for strategy, window, label in FLAT_ROWS:
        match = sub[(sub["halflife"] == FLAT) & (sub["strategy"] == strategy) & (sub["window"] == window)]
        if len(match):
            flat_rows[label] = float(match[metric].iloc[0])
    flat = pd.DataFrame({"": flat_rows})
    return wide, flat


def describe_rolling(agg: pd.DataFrame, rolling: dict) -> list[str]:
    real = rolling["real"]["ewma"]
    n = int(agg[agg["panel"] == "real"]["n_periods"].max())
    lines = [f"--- Rolling backtest, window T = {WINDOW}, 21-session holds: {n} holding periods, {real.periods['hold_start'].min().date()} → {real.periods['hold_end'].max().date()}; flat runs restricted to the same holds ---", ""]
    fmt = {"median_vol_ratio": "{:.3f}", "ann_volatility": "{:.2%}", "mean_turnover": "{:.3f}"}
    titles = {"median_vol_ratio": "Median realised / predicted volatility (1 is honest)", "ann_volatility": "Realised volatility, annualised (lower is better)", "mean_turnover": "Turnover per rebalance"}
    for panel, label in (("real", REAL_LABEL), ("gaussian", SIM_LABEL)):
        lines += [f"=== {label} ===", ""]
        for metric in ("median_vol_ratio", "ann_volatility", "mean_turnover"):
            wide, flat = _rolling_table(agg, panel, metric)
            lines += [titles[metric] + ":", wide.to_string(float_format=lambda v: fmt[metric].format(v)), "flat references: " + ", ".join(f"{k} {fmt[metric].format(v)}" for k, v in flat[""].items()), ""]
    real_vol, _ = _rolling_table(agg, "real", "ann_volatility")
    best = real_vol.idxmin()
    lines.append("Lowest realised volatility on the real market per cleaning: " + ", ".join(f"{c}: {h}" for c, h in best.items()))
    lines.append("")
    return lines


def plot_rolling(agg: pd.DataFrame, path: Path) -> None:
    metrics = (
        ("median_vol_ratio", "honesty: median realised / predicted", "log", 1.0),
        ("ann_volatility", "risk: realised volatility", "linear", None),
        ("mean_turnover", "trading: turnover per rebalance", "log", None),
    )
    halflives = list(HALFLIVES)
    pos = {**{h: float(h) for h in halflives}, FLAT: 504 * 2.5}
    xs = [pos[h] for h in halflives] + [pos[FLAT]]
    fig, axes = plt.subplots(len(metrics), len(CLEANINGS), figsize=(4.4 * len(CLEANINGS), 3.3 * len(metrics)), sharex=True, sharey="row")
    flat_key = {"none": "minvar_sample", "clipped": "minvar_clipped", "rie": "minvar_rie"}
    for r, (metric, ylabel, yscale, reference) in enumerate(metrics):
        for c, cleaning in enumerate(CLEANINGS):
            ax = axes[r, c]
            color = COLORS[cleaning]
            for panel, style in (("gaussian", "dashed"), ("real", "solid")):
                sub = agg[(agg["panel"] == panel)]
                ewma = sub[(sub["halflife"] != FLAT) & (sub["cleaning"] == cleaning)].copy()
                ewma["halflife"] = ewma["halflife"].astype(int)
                ewma = ewma.set_index("halflife").reindex(halflives)
                flat = sub[(sub["halflife"] == FLAT) & (sub["strategy"] == flat_key[cleaning]) & (sub["window"] == WINDOW)]
                y = list(ewma[metric].to_numpy(dtype=float)) + [float(flat[metric].iloc[0]) if len(flat) else np.nan]
                if panel == "real":
                    ax.plot(xs[:-1], y[:-1], color=color, linewidth=2.0, marker="o", markersize=6, markeredgecolor="white", markeredgewidth=0.8)
                    ax.plot(xs[-2:], y[-2:], color=color, linewidth=1.0, linestyle="--", alpha=0.6)
                    ax.scatter([xs[-1]], [y[-1]], color=color, s=40, zorder=4)
                    for x, value in zip(xs, y):
                        if np.isfinite(value):
                            ax.annotate(_fmt_metric(metric, value), (x, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5, color=MUTED)
                else:
                    ax.plot(xs[:-1], y[:-1], color=color, linewidth=1.3, linestyle="--", marker="o", markersize=6, markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.3)
                    ax.scatter([xs[-1]], [y[-1]], facecolor="white", edgecolor=color, s=40, linewidth=1.3, zorder=4)
                # Flat T = 500 of the same cleaning on the real market, the earlier sweet spot, as a reference line.
                if panel == "real" and cleaning in ("clipped", "rie"):
                    ref = sub[(sub["halflife"] == FLAT) & (sub["strategy"] == flat_key[cleaning]) & (sub["window"] == 500)]
                    if len(ref):
                        ax.axhline(float(ref[metric].iloc[0]), color=color, linewidth=1.0, linestyle=(0, (1, 2)), alpha=0.9)
            if reference is not None:
                ax.axhline(reference, color="#888888", linewidth=1.0, linestyle=":", zorder=1)
            if r == 0:
                ax.set_title(CLEANING_LABELS[cleaning], fontsize=10.5)
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels([str(h) for h in halflives] + ["flat"], fontsize=8.5)
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
        ax.set_xlabel(f"half-life in sessions, window T = {WINDOW}")
    handles = [
        Line2D([], [], color="#333333", linewidth=2.0, marker="o", markersize=6, markeredgecolor="white", label=REAL_LABEL),
        Line2D([], [], color="#333333", linewidth=1.3, linestyle="--", marker="o", markersize=6, markerfacecolor="white", markeredgecolor="#333333", label=SIM_LABEL),
        Line2D([], [], color="#333333", linewidth=1.0, linestyle=(0, (1, 2)), label="same cleaning, flat T = 500 on the real market"),
        Line2D([], [], color="#888888", linewidth=1.0, linestyle=":", label="honest model: ratio 1"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
    n = int(agg["n_periods"].max())
    fig.suptitle(f"Does forgetting help on the real market? Window T = {WINDOW}, every half-life judged on the same {n} holding periods.", x=0.02, ha="left", fontsize=12, fontweight="medium")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def _series(result: BacktestResult, name: str, column: str, steady_only: bool) -> pd.Series:
    rows = result.periods[result.periods["strategy"] == name]
    if steady_only:
        rows = rows[~rows["first_period"].astype(bool)]
    return pd.Series(rows[column].to_numpy(dtype=float), index=pd.DatetimeIndex(rows["hold_start"]))


def plot_series(rolling: dict[str, dict[str, BacktestResult]], path: Path) -> None:
    picks = (
        ("rie_T1000", "minvar_rie", "RIE, flat T = 1000", COLORS["rie"]),
        ("ewma", strategy_name(252, "rie"), "RIE, half-life 252", "#d95f02"),
        ("ewma", strategy_name(126, "rie"), "RIE, half-life 126", "#1f77b4"),
    )
    rows = (("vol_ratio", "realised / predicted volatility", "log", 1.0, False), ("turnover", "turnover per rebalance", "log", None, True))
    fig, axes = plt.subplots(len(rows), len(picks), figsize=(4.6 * len(picks), 3.4 * len(rows)), sharex=True, sharey="row")
    for r, (column, ylabel, yscale, reference, steady) in enumerate(rows):
        for c, (source, name, label, color) in enumerate(picks):
            ax = axes[r, c]
            a = _series(rolling["real"][source], name, column, steady)
            b = _series(rolling["gaussian"][source], name, column, steady)
            ax.plot(b.index, b.to_numpy(), color=color, linewidth=1.1, linestyle="--", alpha=0.75)
            ax.plot(a.index, a.to_numpy(), color=color, linewidth=1.7)
            if reference is not None:
                ax.axhline(reference, color="#888888", linewidth=1.0, linestyle=":", zorder=1)
            for _, start, end in EPISODES:
                ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#000000", alpha=0.05, zorder=0)
            if r == 0:
                ax.set_title(label, fontsize=10.5)
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        first = axes[r, 0]
        first.set_yscale("log")
        lo, hi = first.get_ylim()
        ticks = [t for t in (0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 3, 5, 10) if lo <= t <= hi]
        first.set_yticks(ticks)
        first.set_yticklabels([f"{t:g}" for t in ticks])
        for ax in axes[r]:
            ax.minorticks_off()
    handles = [
        Line2D([], [], color="#333333", linewidth=1.7, label=REAL_LABEL),
        Line2D([], [], color="#333333", linewidth=1.1, linestyle="--", label=SIM_LABEL),
        Line2D([], [], color="#888888", linewidth=1.0, linestyle=":", label="honest model: ratio 1"),
        Patch(facecolor="#000000", alpha=0.08, label=EPISODE_LEGEND),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Per 21-session hold, window T = {WINDOW}. Solid: {REAL_LABEL}. Dashed: {SIM_LABEL}.", x=0.02, ha="left", fontsize=12, fontweight="medium")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- helpers


def _fmt_metric(metric: str, value: float) -> str:
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
