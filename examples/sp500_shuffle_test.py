"""Is the real market's extra gap caused by time order or by the shape of the returns? Shuffle the days and see.

Every estimator we built is less honest and riskier on the S&P 500 than on
Gaussian returns from a fixed covariance. Two things differ between those
panels. The *distribution* of a day: fat tails, company jumps, the real
cross-section. And the *order* of the days: volatility clustering and a
correlation structure that drifts. Pafka and Kondor (2003) split the two
by assigning alternate days to estimation and test, and Bongiorno and
Challet (2023) by shuffling days; both found the real-market excess
disappears, so it comes from time order.

This script repeats the test on our panel. It permutes the 2,900 days of
the real returns at random. Each day keeps its 456 real returns (fat tails,
jumps and the real cross-section survive), but any window and the hold
after it are now draws from one fixed distribution: no clustering, no
drift. The exact truth of a shuffled panel is the full-sample covariance
of the real returns, so a second oracle uses that matrix.

    gaussian   →  gaussian, raw truth  →  shuffled real  →  real
    noise + bias   + a harder truth          + fat tails      + the order of the days

The saved Gaussian panel was drawn from the *cleaned* full-sample
covariance. Its spectrum is smoother than the raw one, so a second
Gaussian panel drawn from the raw full-sample covariance (``--panel
gaussian_raw``) separates "the truth is harder to estimate" from "real
days have fat tails".

Three shuffles, window T = 1000, 21-session holds, the same 93 holding
periods as the saved runs.

    .venv/bin/python examples/sp500_shuffle_test.py --seed 1     # the runs can go in parallel
    .venv/bin/python examples/sp500_shuffle_test.py --seed 2
    .venv/bin/python examples/sp500_shuffle_test.py --seed 3
    .venv/bin/python examples/sp500_shuffle_test.py --panel gaussian_raw --seed 1
    .venv/bin/python examples/sp500_shuffle_test.py --panel gaussian_raw --seed 2
    .venv/bin/python examples/sp500_shuffle_test.py --replot
"""

from __future__ import annotations

import argparse
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
import sp500_simulated_backtest as simulated  # noqa: E402

from market_analyzer.backtest import BacktestConfig, BacktestResult, run_backtest  # noqa: E402
from market_analyzer.matrices import CorrelationModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "output" / "sp500_marchenko_pastur" / "prices.csv"
OUT = ROOT / "output" / "sp500_shuffle"
SAVED = {
    "gaussian": [ROOT / "output" / "sp500_simulated_backtest" / "gaussian_T1000", ROOT / "output" / "sp500_rie" / "gaussian_T1000"],
    "clocked": [ROOT / "output" / "sp500_standardisation" / "clocked_T1000"],
    "real": [ROOT / "output" / "sp500_simulated_backtest" / "real_T1000", ROOT / "output" / "sp500_rie" / "real_T1000"],
}
WINDOW, HOLDING = 1000, 21
STRATEGIES = ("minvar_sample", "minvar_ledoit_wolf", "minvar_clipped", "minvar_clipped_unit", "minvar_rie", "minvar_fixed", "minvar_fixed_raw", "equal_weight")
LABELS = {
    "minvar_sample": "Sample covariance",
    "minvar_ledoit_wolf": "Ledoit-Wolf",
    "minvar_clipped": "MP clipped, fitted edge",
    "minvar_clipped_unit": "MP clipped, σ² = 1 edge",
    "minvar_rie": "RIE",
    "minvar_fixed": "Oracle: cleaned full-sample covariance",
    "minvar_fixed_raw": "Oracle: raw full-sample covariance",
    "equal_weight": "Equal weight",
}
WORLDS = ("gaussian", "clocked", "gaussian_raw", "shuffled", "real")
WORLD_LABELS = {"gaussian": "Gaussian", "clocked": "Gaussian × clock", "gaussian_raw": "Gaussian, raw truth", "shuffled": "real days, shuffled", "real": "real market"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", type=Path, default=PRICES)
    parser.add_argument("--panel", choices=("shuffled", "gaussian_raw"), default="shuffled")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    returns = simulated.load_returns(args.prices)
    if not args.replot:
        run(args, returns)
    report(args, returns)


def shuffled(returns: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The same days in a random order, re-stamped with the original dates so the holding calendar is unchanged."""
    order = np.random.default_rng(seed).permutation(len(returns))
    return pd.DataFrame(returns.to_numpy()[order], index=returns.index, columns=returns.columns)


def gaussian_raw(returns: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Gaussian returns whose truth is the raw full-sample covariance, the exact truth of a shuffled panel."""
    chol = np.linalg.cholesky(returns.cov(ddof=1).to_numpy(dtype=float))
    noise = np.random.default_rng(1000 + seed).standard_normal(size=returns.shape)
    return pd.DataFrame(noise @ chol.T, index=returns.index, columns=returns.columns)


def estimators(returns: pd.DataFrame) -> dict:
    truth = CorrelationModel.from_returns(returns).cleaned_covariance
    raw = returns.cov(ddof=1)
    return {"fixed": simulated.fixed_estimator(truth), "fixed_raw": simulated.fixed_estimator(raw)}


def run(args: argparse.Namespace, returns: pd.DataFrame) -> None:
    folder = args.out / f"{args.panel}_s{args.seed}_T{WINDOW}"
    if (folder / "periods.csv").is_file():
        print(f"{args.panel} seed {args.seed}: reusing saved run")
        return
    config = BacktestConfig(window=WINDOW, holding=HOLDING, strategies=STRATEGIES)
    panel = shuffled(returns, args.seed) if args.panel == "shuffled" else gaussian_raw(returns, args.seed)
    result = run_backtest(panel, config, estimators(returns))
    result.save(folder)
    summary = result.summary()
    for name in STRATEGIES:
        print(f"seed {args.seed}  {LABELS[name]:40s} ratio {summary.loc[name, 'median_vol_ratio']:.3f}  vol {summary.loc[name, 'ann_volatility']:.2%}")


def saved_summary(world: str) -> pd.DataFrame | None:
    frames = [BacktestResult.load(f).summary() for f in SAVED[world] if (f / "periods.csv").is_file()]
    if not frames:
        return None
    table = pd.concat(frames)
    return table[~table.index.duplicated(keep="first")]


def report(args: argparse.Namespace, returns: pd.DataFrame) -> None:
    metrics = ("median_vol_ratio", "ann_volatility", "mean_gross_leverage")
    tables, runs, spread = {}, [], []
    for world in ("shuffled", "gaussian_raw"):
        found = [r for r in sorted(args.out.glob(f"{world}_s*_T{WINDOW}")) if (r / "periods.csv").is_file()]
        if not found:
            continue
        per_seed = [BacktestResult.load(r).summary() for r in found]
        tables[world] = sum(s[list(metrics)] for s in per_seed) / len(per_seed)
        runs += found
        spread += [s["median_vol_ratio"] for s in per_seed]
    if "shuffled" not in tables:
        print("No shuffled run saved yet.")
        return
    spread = pd.concat(spread, axis=1)
    for world in ("gaussian", "clocked", "real"):
        summary = saved_summary(world)
        if summary is not None:
            tables[world] = summary
    lines = [
        f"--- Shuffled days: random orders of the {len(returns)} real sessions × {returns.shape[1]} tickers, window T = {WINDOW}, 21-session holds; runs: {', '.join(r.name for r in runs)} ---",
        "Each shuffled day keeps its real returns (fat tails, jumps, the real cross-section) but the order is random: no volatility clustering, no drift.",
        "The raw full-sample covariance is the exact truth of a shuffled panel; the cleaned one is the truth of the Gaussian and clocked panels.",
        "",
    ]
    titles = {"median_vol_ratio": "Median realised / predicted volatility (1 is honest; 0.98 over 21-day holds)", "ann_volatility": "Realised volatility, annualised", "mean_gross_leverage": "Gross leverage"}
    fmt = {"median_vol_ratio": "{:.3f}", "ann_volatility": "{:.2%}", "mean_gross_leverage": "{:.2f}"}
    combined = {}
    for metric in metrics:
        wide = pd.DataFrame({WORLD_LABELS[w]: tables[w][metric] for w in WORLDS if w in tables}).reindex(list(STRATEGIES))
        combined[metric] = wide
        shown = wide.copy()
        shown.index = [LABELS[s] for s in shown.index]
        lines += [titles[metric] + ":", shown.to_string(float_format=lambda v: fmt[metric].format(v), na_rep="—"), ""]
    seeds = spread.copy()
    seeds.columns = [r.name for r in runs]
    seeds.index = [LABELS[s] for s in seeds.index]
    lines += ["Honesty ratio per run (spread from one draw or shuffle to the next):", seeds.to_string(float_format=lambda v: f"{v:.3f}"), ""]
    h, v = combined["median_vol_ratio"], combined["ann_volatility"]
    G, GR, S, R = (WORLD_LABELS[w] for w in ("gaussian", "gaussian_raw", "shuffled", "real"))
    steps = {}
    for metric, frame in (("honesty", h), ("risk", v)):
        if GR in frame:
            steps[f"{metric}: harder truth (Gaussian raw / Gaussian)"] = frame[GR] / frame[G]
            steps[f"{metric}: fat tails (shuffled / Gaussian raw)"] = frame[S] / frame[GR]
        else:
            steps[f"{metric}: shape of a day (shuffled / Gaussian)"] = frame[S] / frame[G]
        steps[f"{metric}: order of the days (real / shuffled)"] = frame[R] / frame[S]
    split = pd.DataFrame(steps).reindex([s for s in STRATEGIES if s not in ("minvar_fixed", "minvar_fixed_raw")])
    split.index = [LABELS[s] for s in split.index]
    lines += ["What each step adds (ratios; 1 means no change):", split.T.to_string(float_format=lambda x: f"{x:.3f}", na_rep="—"), ""]
    if GR in v and WORLD_LABELS["clocked"] in v:
        steps_table = staircase(v, "minvar_rie")
        lines += [
            "Where the RIE's realised risk comes from (the clustering step uses the clocked panel's ratio, so the split of the last step is an estimate):",
            steps_table.to_string(float_format=lambda x: f"{x:.4f}"),
            "",
        ]
        plot_staircase(steps_table, args.out / "risk_staircase.png")
    text = "\n".join(lines)
    print(text)
    (args.out / "summary.txt").write_text(text + "\n")
    pd.concat(combined, axis=1).to_csv(args.out / "worlds.csv")


def staircase(vol: pd.DataFrame, name: str) -> pd.DataFrame:
    """From the best possible risk to the real market, one cause at a time."""
    G, C, GR, S, R = (WORLD_LABELS[w] for w in ("gaussian", "clocked", "gaussian_raw", "shuffled", "real"))
    best = float(vol.loc["minvar_fixed_raw", GR])
    noise = float(vol.loc[name, GR])
    tails = float(vol.loc[name, S])
    real = float(vol.loc[name, R])
    clustering = tails * float(vol.loc[name, C] / vol.loc[name, G])
    clustering = min(clustering, real)
    rows = [
        ("best possible (knows the true covariance)", best),
        ("+ estimation noise, Gaussian days", noise),
        ("+ fat-tailed real days (order shuffled)", tails),
        ("+ volatility clustering (estimated)", clustering),
        ("+ drift in the structure (the rest)", real),
    ]
    table = pd.DataFrame(rows, columns=["step", "realised volatility"]).set_index("step")
    table["added"] = table["realised volatility"] / table["realised volatility"].shift(1) - 1.0
    return table


def plot_staircase(table: pd.DataFrame, path: Path) -> None:
    values = table["realised volatility"].to_numpy()
    labels = ["Best possible\n(knows the truth)", "Estimation noise\n(Gaussian days)", "Fat tails\n(real days, shuffled)", "Volatility clustering\n(estimated)", "Structure drift\n(the rest)"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    previous = 0.0
    for i, value in enumerate(values):
        bottom = 0.0 if i == 0 else previous
        height = value - bottom
        color = "#6b6b6b" if i == 0 else "#a61b1b"
        ax.bar(i, height, bottom=bottom, width=0.62, color=color, edgecolor="white", linewidth=2)
        text = f"{value:.1%}" if i == 0 else f"+{height / previous:.0%}\n→ {value:.1%}"
        ax.text(i, value + 0.0015, text, ha="center", va="bottom", fontsize=10, color="#1f1f1f")
        if i:
            ax.plot([i - 1 + 0.31, i - 0.31], [previous, previous], color="#9a9a9a", linewidth=1, linestyle=":")
        previous = value
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, values.max() * 1.18)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("realised volatility, annualised")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6")
    ax.set_axisbelow(True)
    ax.set_title("Why the RIE's portfolio is riskier than the best possible: one cause at a time (S&P 500, T = 1000)", fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
