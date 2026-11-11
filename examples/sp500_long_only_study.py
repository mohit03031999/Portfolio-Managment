"""Long-only minimum variance: the same covariance estimates with shorting forbidden.

The unconstrained optimiser hunts for the quietest combination of 456 stocks
and hedges with shorts. Every error in the covariance becomes a position, and
the book runs at 5 to 8 times gross leverage. Long-only keeps the objective
and forbids negative weights, so leverage is exactly 1 and the optimiser loses
the tool it uses to exploit noise. Jagannathan and Ma (2003) showed the
constraint acts as a data-driven shrinkage: it lowers the covariances of the
stocks the optimiser wanted to short until it no longer wants to.

Per window length (1000 and 250 sessions) the script runs the library backtest
on the same decision dates with five strategies: unconstrained and long-only
minimum variance on the sample covariance and on the RIE, plus equal weight.
Worlds: ``real`` (the S&P 500 panel) and ``gaussian_raw`` (Gaussian days drawn
from the full-sample covariance, where the true risk of every portfolio is
known; an oracle on the true matrix is added, unconstrained and long-only).

Outputs in output/sp500_long_only/: {panel}[_s{seed}]/T{window}/ holds the
backtest files (periods.csv, daily_returns.csv, weights_*.csv, config.json,
plus true_risk.csv in the lab); --report writes summary.txt and long_only.png.

    python examples/sp500_long_only_study.py
    python examples/sp500_long_only_study.py --panel gaussian_raw --seed 1
    python examples/sp500_long_only_study.py --report
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import sp500_shuffle_test as shuffle  # noqa: E402
import sp500_simulated_backtest as simulated  # noqa: E402
from market_analyzer.backtest import BacktestConfig, BacktestResult, run_backtest  # noqa: E402

OUT = ROOT / "output" / "sp500_long_only"
WINDOWS = (1000, 250)
HOLDING, ANNUAL = 21, 252
STRATEGIES = ("minvar_sample", "longonly_sample", "minvar_rie", "longonly_rie", "equal_weight")
LAB_EXTRA = ("minvar_oracle", "longonly_oracle")
PANEL_LABELS = {"real": "real market", "gaussian_raw": "Gaussian days"}
LABELS = {
    "minvar_sample": "sample covariance, unconstrained",
    "longonly_sample": "sample covariance, long-only",
    "minvar_rie": "RIE, unconstrained",
    "longonly_rie": "RIE, long-only",
    "equal_weight": "equal weight",
    "minvar_oracle": "true covariance, unconstrained",
    "longonly_oracle": "true covariance, long-only",
}
SHORT = {
    "minvar_sample": "sample\nunconstr.",
    "longonly_sample": "sample\nlong-only",
    "minvar_rie": "RIE\nunconstr.",
    "longonly_rie": "RIE\nlong-only",
    "equal_weight": "equal\nweight",
    "minvar_oracle": "truth\nunconstr.",
    "longonly_oracle": "truth\nlong-only",
}
# one fixed colour per strategy; the long-only version of an estimator is the pale shade of its colour
COLORS = {
    "minvar_sample": "#2a6fdb",
    "longonly_sample": "#9dc0f0",
    "minvar_rie": "#e07b00",
    "longonly_rie": "#f4c48a",
    "equal_weight": "#1a9e5c",
    "minvar_oracle": "#555555",
    "longonly_oracle": "#b8b8b8",
}
MUTED, INK = "#777777", "#222222"
COLUMNS = ["ann_volatility", "median_vol_ratio", "mean_turnover", "mean_gross_leverage", "mean_n_positions", "mean_max_weight", "mean_effective_n"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", choices=tuple(PANEL_LABELS), default="real")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS))
    parser.add_argument("--report", action="store_true", help="aggregate saved runs, write summary.txt and the figure")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report:
        report()
        return
    returns = simulated.load_returns(simulated.PRICES)
    run(returns, args.panel, args.seed, tuple(args.windows))


# ----------------------------------------------------------------------------- the run


def run(returns: pd.DataFrame, panel: str, seed: int, windows: tuple[int, ...]) -> None:
    if panel == "gaussian_raw":
        data = shuffle.gaussian_raw(returns, seed)
        truth = returns.cov(ddof=1)
        estimators = {"oracle": simulated.fixed_estimator(truth)}
        strategies = STRATEGIES + LAB_EXTRA
    else:
        data, truth, estimators, strategies = returns, None, None, STRATEGIES
    folder = OUT / ("real" if panel == "real" else f"{panel}_s{seed}")
    longest = max(windows)
    for window in windows:
        clock = time.perf_counter()
        sub = data.iloc[longest - window :]  # every window decides on the same dates
        result = run_backtest(sub, BacktestConfig(window=window, holding=HOLDING, strategies=strategies), estimators)
        target = folder / f"T{window}"
        result.save(target)
        if truth is not None:
            sigma = truth.to_numpy(dtype=float)
            tickers = list(truth.index)
            rows = []
            for name, frame in result.weights.items():
                w = frame.reindex(columns=tickers).fillna(0.0).to_numpy(dtype=float)
                variance = np.einsum("ij,jk,ik->i", w, sigma, w)
                rows += [{"strategy": name, "hold_start": d, "true_variance": float(v)} for d, v in zip(frame.index, variance)]
            pd.DataFrame(rows).to_csv(target / "true_risk.csv", index=False)
        print(f"{panel} T={window}: {time.perf_counter() - clock:.0f}s")
        print(result.report())
        print(flush=True)
    print(f"saved {folder}")


# ----------------------------------------------------------------------------- the report


def load_runs() -> dict[tuple[str, str, int], BacktestResult]:
    runs = {}
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir():
            continue
        panel = "real" if folder.name == "real" else folder.name.rsplit("_s", 1)[0]
        if panel not in PANEL_LABELS:
            continue
        for sub in sorted(folder.glob("T*")):
            if (sub / "periods.csv").exists():
                runs[(panel, folder.name, int(sub.name[1:]))] = BacktestResult.load(sub)
    return runs


def table(result: BacktestResult, folder: Path | None = None) -> pd.DataFrame:
    summary = result.summary().loc[:, COLUMNS].copy()
    if folder is not None and (folder / "true_risk.csv").exists():
        true = pd.read_csv(folder / "true_risk.csv").groupby("strategy")["true_variance"].mean()
        summary["true_volatility"] = np.sqrt(ANNUAL * true.reindex(summary.index))
    return summary


def _fmt_row(name: str, row: pd.Series, base: pd.Series | None) -> str:
    text = f"  {LABELS[name]:<36} {row['ann_volatility']:7.2%}"
    text += f"  {(row['ann_volatility'] / base['ann_volatility'] - 1):+7.1%}" if base is not None else "         "
    if "true_volatility" in row.index and np.isfinite(row["true_volatility"]):
        text += f"  {row['true_volatility']:7.2%}"
    text += f"  {row['median_vol_ratio']:7.2f}  {row['mean_turnover']:8.2f}  {row['mean_gross_leverage']:8.2f}  {row['mean_n_positions']:9.0f}  {row['mean_max_weight']:9.1%}  {row['mean_effective_n']:11.0f}"
    return text


def report() -> None:
    runs = load_runs()
    if not runs:
        print("nothing saved yet")
        return
    lines = ["Long-only minimum variance against the unconstrained optimiser. Holds of 21 days, same decision dates for every window.", ""]
    tables: dict[tuple[str, int], pd.DataFrame] = {}
    for (panel, run_name, window), result in runs.items():
        folder = OUT / run_name / f"T{window}"
        summary = table(result, folder)
        tables[(panel, window)] = summary
        first = result.periods["hold_start"].min()
        last = result.periods["hold_end"].max()
        n_holds = result.periods["hold_start"].nunique()
        lines.append(f"=== {PANEL_LABELS[panel]} ({run_name}), T = {window}: {n_holds} holds, {pd.Timestamp(first).date()} to {pd.Timestamp(last).date()} ===")
        truth_col = "  true vol" if "true_volatility" in summary else ""
        lines.append(f"  {'strategy':<36} {'realised':>7}  {'vs unc.':>7}{truth_col}  {'honesty':>7}  {'turnover':>8}  {'leverage':>8}  {'positions':>9}  {'max weight':>9}  {'effective N':>11}")
        for name in [n for n in LABELS if n in summary.index]:
            base = summary.loc[name.replace("longonly_", "minvar_")] if name.startswith("longonly_") else None
            lines.append(_fmt_row(name, summary.loc[name], base))
        lines.append("")
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)
    plot(runs, tables, OUT / "long_only.png")


def plot(runs: dict[tuple[str, str, int], BacktestResult], tables: dict[tuple[str, int], pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    _bars(axes[0, 0], tables, "real", "ann_volatility", "1. Real market: realised volatility, % per year", percent=True)
    _bars(axes[0, 1], tables, "real", "median_vol_ratio", "2. Real market: honesty, realised / predicted (1 = honest)", percent=False, line=1.0)
    _bars(axes[0, 2], tables, "gaussian_raw", "ann_volatility", "3. Gaussian lab: realised volatility (dot = true risk)", percent=True, truth=True)
    real_1000 = next((r for (p, _, w), r in runs.items() if p == "real" and w == 1000), None)
    if real_1000 is not None:
        _rolling(axes[1, 0], real_1000, ("minvar_rie", "longonly_rie", "longonly_sample", "equal_weight"), "4. Real market, T = 1000: trailing 63-day realised volatility")
        _growth(axes[1, 1], real_1000, ("minvar_rie", "longonly_rie", "longonly_sample", "equal_weight"), "5. Real market, T = 1000: growth of 1")
    _positions(axes[1, 2], runs, "6. Real market: how many stocks the long-only portfolios hold")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def _bars(ax, tables, panel, column, title, percent, line=None, truth=False) -> None:
    windows = sorted({w for (p, w) in tables if p == panel}, reverse=True)
    if not windows:
        ax.set_visible(False)
        return
    names = [n for n in LABELS if n in tables[(panel, windows[0])].index]
    width = 0.8 / len(windows)
    hatches = ["", "//"]
    for offset, window in enumerate(windows):
        summary = tables[(panel, window)]
        for position, name in enumerate(names):
            value = float(summary.loc[name, column])
            x = position + (offset - (len(windows) - 1) / 2) * width
            ax.bar(x, value, width=width * 0.92, color=COLORS[name], hatch=hatches[offset % 2], edgecolor="white", linewidth=0.8)
            ax.text(x, value, f"{value:.1%}" if percent else f"{value:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
            if truth and "true_volatility" in summary and np.isfinite(summary.loc[name, "true_volatility"]):
                ax.plot([x], [summary.loc[name, "true_volatility"]], marker="o", color=INK, markersize=4, zorder=5)
    if line is not None:
        ax.axhline(line, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([SHORT[n] for n in names], fontsize=8)
    if percent:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    top = max(float(tables[(panel, w)].loc[n, column]) for w in windows for n in names)
    ax.set_ylim(0, top * 1.18)
    ax.set_title(title, loc="left", fontsize=11)
    handles = [Patch(facecolor="#d9d8d3", edgecolor="white", hatch=hatches[i % 2], label=f"T = {w}") for i, w in enumerate(windows)]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")


def _rolling(ax, result: BacktestResult, names, title) -> None:
    rolling = result.daily_returns.rolling(63).std(ddof=1) * np.sqrt(ANNUAL)
    for name in names:
        if name in rolling:
            ax.plot(rolling.index, rolling[name], color=COLORS[name], lw=1.6, label=LABELS[name])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def _growth(ax, result: BacktestResult, names, title) -> None:
    growth = (1.0 + result.daily_returns).cumprod()
    for name in names:
        if name in growth:
            ax.plot(growth.index, growth[name], color=COLORS[name], lw=1.6, label=f"{LABELS[name]}  {growth[name].iloc[-1]:.2f}×")
    ax.set_yscale("log")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def _positions(ax, runs, title) -> None:
    styles = {1000: "-", 250: ":"}
    drawn = False
    for (panel, _, window), result in runs.items():
        if panel != "real":
            continue
        counts = result.periods.pivot(index="hold_start", columns="strategy", values="n_positions")
        for name in ("longonly_sample", "longonly_rie"):
            if name in counts:
                ax.plot(counts.index, counts[name], color=COLORS[name.replace("longonly_", "minvar_")], lw=1.6, ls=styles.get(window, "-"), label=f"{LABELS[name]}, T = {window}")
                drawn = True
    if not drawn:
        ax.set_visible(False)
        return
    ax.set_ylim(bottom=0)
    ax.set_ylabel("stocks with a positive weight")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


if __name__ == "__main__":
    main()
