"""The gross-exposure dial: from long-only (cap 1) to the unconstrained optimiser (no cap).

The long-only test showed the two ends of the dial. Shorting carried real
hedging information (the unconstrained RIE realised less risk) and also the
noise the optimiser exploits (its forecasts were 1.3 to 2.6 times too
optimistic). A cap on total long plus short exposure, ``Σ|w| ≤ L`` with the
weights still summing to one, gives the optimiser a budget for shorting and
lets the size of the payoff decide which hedges survive (Fan, Zhang and Yu
2012). The gap between predicted and realised risk is bounded by the worst
single-entry error of the covariance times L², so the damage from noise grows
with the square of the stretch while the benefit of hedging flattens out.

For each window length (1000 and 250 sessions, same decision dates) the script
runs the library backtest with caps 1, 1.25, 1.5, 2, 3, 5 and none, on the
sample covariance and on the RIE, plus equal weight. Worlds: ``real`` and
``gaussian_raw`` (Gaussian days from the full-sample covariance, with an
oracle on the true matrix at every cap, so the pure price of the constraint
is visible). The report also splits the real track into first and second
halves, so a cap chosen on one half can be judged on the other.

Outputs in output/sp500_gross_cap/: {panel}[_s{seed}]/T{window}/ holds the
backtest files (plus true_risk.csv in the lab); --report writes summary.txt
and gross_cap.png.

    python examples/sp500_gross_cap_study.py
    python examples/sp500_gross_cap_study.py --panel gaussian_raw --seed 1
    python examples/sp500_gross_cap_study.py --report
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import sp500_shuffle_test as shuffle  # noqa: E402
import sp500_simulated_backtest as simulated  # noqa: E402
from market_analyzer.backtest import BacktestConfig, BacktestResult, run_backtest  # noqa: E402

OUT = ROOT / "output" / "sp500_gross_cap"
WINDOWS = (1000, 250)
HOLDING, ANNUAL = 21, 252
CAPS = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, float("inf"))
ESTIMATORS = ("sample", "rie")
LAB_ESTIMATORS = ("sample", "rie", "oracle")
PANEL_LABELS = {"real": "real market", "gaussian_raw": "Gaussian days"}
ESTIMATOR_LABELS = {"sample": "sample covariance", "rie": "RIE", "oracle": "true covariance"}
COLORS = {"sample": "#2a6fdb", "rie": "#e07b00", "oracle": "#555555"}
STYLES = {1000: "-", 250: ":"}
MUTED, INK = "#777777", "#222222"


def strategy_name(cap: float, estimator: str) -> str:
    if cap == 1.0:
        return f"longonly_{estimator}"
    if not np.isfinite(cap):
        return f"minvar_{estimator}"
    return f"gross{cap:g}_{estimator}"


def parse_name(name: str) -> tuple[float, str] | None:
    if name.startswith("longonly_"):
        return 1.0, name[len("longonly_") :]
    if name.startswith("minvar_"):
        return float("inf"), name[len("minvar_") :]
    match = re.match(r"^gross(\d+(?:\.\d+)?)_(.+)$", name)
    if match:
        return float(match.group(1)), match.group(2)
    return None


def cap_label(cap: float) -> str:
    return "none" if not np.isfinite(cap) else f"{cap:g}"


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
        names = [strategy_name(cap, est) for est in LAB_ESTIMATORS for cap in CAPS]
    else:
        data, truth, estimators = returns, None, None
        names = [strategy_name(cap, est) for est in ESTIMATORS for cap in CAPS]
    names.append("equal_weight")
    folder = OUT / ("real" if panel == "real" else f"{panel}_s{seed}")
    longest = max(windows)
    for window in windows:
        clock = time.perf_counter()
        sub = data.iloc[longest - window :]  # every window decides on the same dates
        result = run_backtest(sub, BacktestConfig(window=window, holding=HOLDING, strategies=tuple(names)), estimators)
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


def table(result: BacktestResult, folder: Path) -> pd.DataFrame:
    """One row per (estimator, cap): realised risk, honesty, turnover, positions, and halves of the track."""
    summary = result.summary()
    daily = result.daily_returns
    middle = daily.index[len(daily) // 2]
    first = daily.loc[daily.index < middle].std(ddof=1) * np.sqrt(ANNUAL)
    second = daily.loc[daily.index >= middle].std(ddof=1) * np.sqrt(ANNUAL)
    true = None
    if (folder / "true_risk.csv").exists():
        true = pd.read_csv(folder / "true_risk.csv").groupby("strategy")["true_variance"].mean()
    periods = result.periods
    rows = []
    for name in summary.index:
        parsed = parse_name(name)
        if parsed is None:
            continue
        cap, estimator = parsed
        mine = periods.loc[periods["strategy"] == name]
        row = {
            "estimator": estimator,
            "cap": cap,
            "strategy": name,
            "volatility": float(summary.loc[name, "ann_volatility"]),
            "honesty": float(summary.loc[name, "median_vol_ratio"]),
            "turnover": float(summary.loc[name, "mean_turnover"]),
            "leverage": float(summary.loc[name, "mean_gross_leverage"]),
            "n_long": float((mine["n_positions"] - mine["n_short"]).mean()),
            "n_short": float(mine["n_short"].mean()),
            "max_weight": float(summary.loc[name, "mean_max_weight"]),
            "first_half": float(first[name]),
            "second_half": float(second[name]),
        }
        if true is not None:
            row["true_volatility"] = float(np.sqrt(ANNUAL * true[name]))
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["estimator", "cap"]).reset_index(drop=True)
    frame.attrs["split_date"] = pd.Timestamp(middle).date()
    frame.attrs["equal_weight"] = float(summary.loc["equal_weight", "ann_volatility"]) if "equal_weight" in summary.index else float("nan")
    return frame


def report() -> None:
    runs = load_runs()
    if not runs:
        print("nothing saved yet")
        return
    lines = ["The gross-exposure dial, cap 1 (long-only) to none (unconstrained). Holds of 21 days, same decision dates for every window.", ""]
    tables: dict[tuple[str, int], pd.DataFrame] = {}
    for (panel, run_name, window), result in runs.items():
        frame = table(result, OUT / run_name / f"T{window}")
        tables[(panel, window)] = frame
        first = pd.Timestamp(result.periods["hold_start"].min()).date()
        last = pd.Timestamp(result.periods["hold_end"].max()).date()
        lines.append(f"=== {PANEL_LABELS[panel]} ({run_name}), T = {window}: {result.periods['hold_start'].nunique()} holds, {first} to {last}; halves split at {frame.attrs['split_date']} ===")
        has_truth = "true_volatility" in frame
        header = f"  {'estimator':<16} {'cap':>5}  {'realised':>8}" + (f"  {'true':>7}" if has_truth else "") + f"  {'honesty':>7}  {'turnover':>8}  {'leverage':>8}  {'longs':>5}  {'shorts':>6}  {'max w':>6}  {'1st half':>8}  {'2nd half':>8}"
        lines.append(header)
        for _, row in frame.iterrows():
            text = f"  {ESTIMATOR_LABELS[row['estimator']]:<16} {cap_label(row['cap']):>5}  {row['volatility']:8.2%}"
            if has_truth:
                text += f"  {row['true_volatility']:7.2%}"
            text += f"  {row['honesty']:7.2f}  {row['turnover']:8.2f}  {row['leverage']:8.2f}  {row['n_long']:5.0f}  {row['n_short']:6.0f}  {row['max_weight']:6.1%}  {row['first_half']:8.2%}  {row['second_half']:8.2%}"
            lines.append(text)
        lines.append(f"  equal weight: {frame.attrs['equal_weight']:.2%}")
        lines.append("")
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)
    plot(tables, OUT / "gross_cap.png")


def _positions(caps: list[float]) -> list[float]:
    return [float(i) for i in range(len(caps))]


def _curve(ax, frame: pd.DataFrame, estimator: str, column: str, window: int, label: str, caps: list[float]) -> None:
    rows = frame.loc[frame["estimator"] == estimator].set_index("cap").reindex(caps)
    ax.plot(_positions(caps), rows[column].to_numpy(), color=COLORS[estimator], ls=STYLES.get(window, "-"), lw=1.8, marker="o", ms=4, label=label)


def plot(tables: dict[tuple[str, int], pd.DataFrame], path: Path) -> None:
    caps = list(CAPS)
    ticks = [cap_label(c) for c in caps]
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    real = {w: f for (p, w), f in tables.items() if p == "real"}
    lab = {w: f for (p, w), f in tables.items() if p == "gaussian_raw"}
    # 1. real: realised volatility vs cap
    ax = axes[0, 0]
    for window, frame in sorted(real.items(), reverse=True):
        for est in ESTIMATORS:
            _curve(ax, frame, est, "volatility", window, f"{ESTIMATOR_LABELS[est]}, T = {window}", caps)
    ax.set_title("1. Real market: realised volatility against the cap", loc="left", fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    # 2. real: honesty
    ax = axes[0, 1]
    for window, frame in sorted(real.items(), reverse=True):
        for est in ESTIMATORS:
            _curve(ax, frame, est, "honesty", window, f"{ESTIMATOR_LABELS[est]}, T = {window}", caps)
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_title("2. Real market: honesty, realised / predicted (1 = honest)", loc="left", fontsize=11)
    # 3. lab: true volatility
    ax = axes[0, 2]
    for window, frame in sorted(lab.items(), reverse=True):
        for est in LAB_ESTIMATORS:
            if "true_volatility" in frame:
                _curve(ax, frame, est, "true_volatility", window, f"{ESTIMATOR_LABELS[est]}, T = {window}", caps)
    ax.set_title("3. Gaussian lab: true risk of the chosen portfolio", loc="left", fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    # 4. real: turnover
    ax = axes[1, 0]
    for window, frame in sorted(real.items(), reverse=True):
        for est in ESTIMATORS:
            _curve(ax, frame, est, "turnover", window, f"{ESTIMATOR_LABELS[est]}, T = {window}", caps)
    ax.set_title("4. Real market: turnover per rebalance", loc="left", fontsize=11)
    # 5. real: longs and shorts, RIE
    ax = axes[1, 1]
    for window, frame in sorted(real.items(), reverse=True):
        rows = frame.loc[frame["estimator"] == "rie"].set_index("cap").reindex(caps)
        ax.plot(_positions(caps), rows["n_long"], color=COLORS["rie"], ls=STYLES.get(window, "-"), lw=1.8, marker="o", ms=4, label=f"longs, T = {window}")
        ax.plot(_positions(caps), rows["n_short"], color=COLORS["sample"], ls=STYLES.get(window, "-"), lw=1.8, marker="s", ms=4, label=f"shorts, T = {window}")
    ax.set_title("5. Real market, RIE: stocks held long and short", loc="left", fontsize=11)
    # 6. real: halves, RIE
    ax = axes[1, 2]
    for window, frame in sorted(real.items(), reverse=True):
        rows = frame.loc[frame["estimator"] == "rie"].set_index("cap").reindex(caps)
        ax.plot(_positions(caps), rows["first_half"], color="#1a9e5c", ls=STYLES.get(window, "-"), lw=1.8, marker="o", ms=4, label=f"first half, T = {window}")
        ax.plot(_positions(caps), rows["second_half"], color="#8e44ad", ls=STYLES.get(window, "-"), lw=1.8, marker="o", ms=4, label=f"second half, T = {window}")
    ax.set_title("6. Real market, RIE: realised volatility in each half of the track", loc="left", fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for ax in axes.ravel():
        ax.set_xticks(_positions(caps))
        ax.set_xticklabels(ticks)
        ax.set_xlabel("gross-exposure cap (1 = long-only)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
