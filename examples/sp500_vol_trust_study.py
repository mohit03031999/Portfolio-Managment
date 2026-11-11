"""Can a stock's own recent volatility be trusted a little? The middle path between the market factor and per-stock forecasts.

The standardisation study ended with two findings about the size ``D`` of
``Σ = D C D``. Each stock's own exponentially weighted volatility
(half-life 20) is a fine forecast for that stock and a terrible input for
the optimiser, which finds the stocks whose volatility reads low by luck
and bets on them. One market-wide factor per day, pooled over 456 stocks
and almost noise-free, keeps honesty and risk and calms the portfolio.

The middle path keeps the market factor and believes only a fraction of
each stock's own deviation from it. With ``σ̄_i`` the window standard
deviation, ``m`` the market factor and ``σ_i`` the stock's own fast
volatility, the ratio ``ρ_i = σ_i / (σ̄_i m)`` says how much calmer or
stormier the stock has been than the market mood explains, and the
forecast is ``σ̄_i · m · (1 + α (ρ_i − 1))``. ``α = 0`` is the market
factor alone, ``α = 1`` is the per-stock forecast.

``α`` is not a matter of taste. It is measured from history: at every
rebalance date, line up each stock's ``ρ`` against the realised ratio
``R`` over the next 21 sessions (its realised volatility divided by its
window σ̄ and the realised market factor) and fit ``R − 1 = α (ρ − 1)``
through the origin, pooled over stocks and dates. The Gaussian panel
(constant volatility) and the clocked panel (market-wide clustering only)
are the controls: neither has stock-specific volatility spells, so there
the measured α is only the small weight that 58 recent days deserve
against a 1000-day σ̄ that is itself estimated with noise.

The rolling runs then test the dial at α = 0.25, 0.5, 1 and at the
measured value, window T = 1000, 21-session holds, on all three panels.
The saved runs of the market factor alone (α = 0), the per-stock forecast
with the raw correlation, and the plain RIE are reused.

    .venv/bin/python examples/sp500_vol_trust_study.py --part measure
    .venv/bin/python examples/sp500_vol_trust_study.py --part rolling --panel real      # the three panels can run in parallel
    .venv/bin/python examples/sp500_vol_trust_study.py --part rolling --panel gaussian
    .venv/bin/python examples/sp500_vol_trust_study.py --part rolling --panel clocked
    .venv/bin/python examples/sp500_vol_trust_study.py --replot
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp500_standardisation_study as study  # noqa: E402

from market_analyzer.backtest import BacktestConfig, BacktestResult, run_backtest  # noqa: E402
from market_analyzer.optimizer import standardized_estimator  # noqa: E402

OUT = study.ROOT / "output" / "sp500_vol_trust"
WINDOW, HOLDING, PANELS = study.WINDOW, study.HOLDING, study.PANELS
VOL_HALFLIFE = 20.0
FIXED_TRUSTS = (0.25, 0.5, 1.0)
ACTIVE_SHARE = 0.95  # a (date, stock) pair counts only if the stock traded on at least this share of the window's days
WORLD_COLORS = {"gaussian": "#2a78d6", "clocked": "#eda100", "real": "#a61b1b"}
WORLD_STYLE = {"gaussian": dict(linestyle="--", marker="o", markerfacecolor="white"), "clocked": dict(linestyle="-.", marker="D"), "real": dict(linestyle="-", marker="o")}
MUTED = study.MUTED


def strategy_name(trust: float) -> str:
    return f"minvar_mkt20_t{round(100 * trust):03d}_rie"


def trust_of(name: str) -> float | None:
    match = re.fullmatch(r"minvar_mkt20_t(\d{3})_rie", name)
    return int(match.group(1)) / 100 if match else None


def label_of(name: str) -> str:
    trust = trust_of(name)
    if trust is None:
        return study.LABELS[name]
    return f"RIE, market factor + stock dial α = {trust:.2f} (h = 20)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", type=Path, default=study.PRICES)
    parser.add_argument("--part", choices=("measure", "rolling", "all"), default="all")
    parser.add_argument("--panel", choices=PANELS + ("all",), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replot", action="store_true", help="Run nothing; build the tables and figures from what is saved.")
    args = parser.parse_args()
    study._style()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.replot:
        returns = study.load_returns(args.prices)
        data = study.build_panels(returns, args.seed)
        if args.part in ("measure", "all"):
            measurement(args, data)
        if args.part in ("rolling", "all"):
            trusts = trust_set(args.out)
            for panel in PANELS if args.panel == "all" else (args.panel,):
                run_panel(args, panel, data[panel], trusts)
    report(args)


# ----------------------------------------------------------------------------- measuring the trust


def pairs(panel: pd.DataFrame, halflife: float = VOL_HALFLIFE, window: int = WINDOW, holding: int = HOLDING, step: int = HOLDING) -> pd.DataFrame:
    """One row per (rebalance date, stock): the ratio ρ read from the window, and the realised ratio R over the next hold."""
    r = panel.to_numpy(dtype=float)
    n_days = r.shape[0]
    weights = 0.5 ** (np.arange(window)[::-1] / halflife)
    weights /= weights.sum()
    rows = []
    for t in range(window, n_days - holding + 1, step):
        past, future = r[t - window : t], r[t : t + holding]
        sbar = past.std(axis=0, ddof=1)
        active = (past != 0).mean(axis=0) >= ACTIVE_SHARE
        m2 = weights @ np.mean((past / sbar) ** 2, axis=1)
        own = np.maximum(np.sqrt(weights @ past**2), 0.1 * sbar)  # the estimator's floor
        rho = own / (sbar * np.sqrt(m2))
        fut = (future / sbar) ** 2
        realised = np.sqrt(fut.mean(axis=0) / fut.mean())
        rows.append(pd.DataFrame({"date": panel.index[t], "ticker": panel.columns, "rho": rho, "realised": realised, "active": active}))
    return pd.concat(rows, ignore_index=True)


def fit(table: pd.DataFrame) -> dict[str, float]:
    """R − 1 = α (ρ − 1) through the origin, and the same in logs."""
    x = table["rho"].to_numpy(dtype=float) - 1.0
    y = table["realised"].to_numpy(dtype=float) - 1.0
    lx, ly = np.log(table["rho"].to_numpy(dtype=float)), np.log(np.maximum(table["realised"].to_numpy(dtype=float), 1e-3))
    return {
        "pairs": float(len(table)),
        "trust_linear": float(x @ y / (x @ x)),
        "trust_log": float(lx @ ly / (lx @ lx)),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "rho_spread": float(np.std(x)),
        "realised_spread": float(np.std(y)),
    }


def binned(table: pd.DataFrame, bins: int) -> pd.DataFrame:
    groups = table.groupby(pd.qcut(table["rho"], bins, labels=False, duplicates="drop"))
    out = pd.DataFrame({"mean rho": groups["rho"].mean(), "mean realised": groups["realised"].mean(), "pairs": groups.size()})
    out.index = [f"{i + 1}" for i in range(len(out))]
    return out


def measurement(args: argparse.Namespace, data: dict[str, pd.DataFrame]) -> None:
    fits, curves = {}, {}
    for panel in PANELS:
        table = pairs(data[panel])
        dropped = int((~table["active"]).sum())
        table = table[table["active"]]
        fits[panel] = {**fit(table), "dropped_inactive": float(dropped)}
        curves[panel] = binned(table, 20)
        if panel == "real":
            binned(table, 10).to_csv(args.out / "trust_deciles_real.csv")
            forecast_error(table).to_csv(args.out / "trust_forecast_error_real.csv")
            by_year = pd.DataFrame({year: fit(rows) for year, rows in table.groupby(table["date"].dt.year)}).T
            by_year.to_csv(args.out / "trust_by_year_real.csv")
    summary = pd.DataFrame(fits).T
    summary.to_csv(args.out / "trust_measurement.csv")
    plot_measurement(summary, curves, args.out / "trust_measurement.png")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))


def forecast_error(table: pd.DataFrame, trusts: tuple[float, ...] = (0.0, 0.25, 0.41, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """How wrong each stock's volatility forecast is, pair by pair, at each setting of the dial: forecast = 1 + α (ρ − 1) against the realised R."""
    rho = table["rho"].to_numpy(dtype=float)
    realised = np.maximum(table["realised"].to_numpy(dtype=float), 1e-3)
    rows = {}
    for trust in trusts:
        forecast = 1.0 + trust * (rho - 1.0)
        error = np.log(forecast / realised)
        rows[f"{trust:.2f}"] = {"rms log error": float(np.sqrt(np.mean(error**2))), "median |error|": float(np.median(np.abs(np.expm1(error)))), "mean forecast / realised": float(np.mean(forecast / realised))}
    return pd.DataFrame(rows).T.rename_axis("trust")


def trust_set(out: Path) -> tuple[float, ...]:
    path = out / "trust_measurement.csv"
    if not path.is_file():
        raise SystemExit("Measure the trust first (--part measure); the rolling runs include the measured value.")
    measured = round(float(pd.read_csv(path, index_col=0).loc["real", "trust_linear"]), 2)
    return tuple(sorted(set(FIXED_TRUSTS) | {measured}))


def plot_measurement(summary: pd.DataFrame, curves: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)
    grid = np.linspace(0.4, 2.2, 50)
    for ax, panel in zip(axes, PANELS):
        curve = curves[panel]
        alpha = float(summary.loc[panel, "trust_linear"])
        ax.axhline(1.0, color="#888888", linewidth=1.0, linestyle=":")
        ax.plot(grid, grid, color="#888888", linewidth=1.0, linestyle="--")
        ax.plot(grid, 1.0 + alpha * (grid - 1.0), color=WORLD_COLORS[panel], linewidth=1.6)
        ax.plot(curve["mean rho"], curve["mean realised"], linestyle="none", marker=WORLD_STYLE[panel]["marker"], markerfacecolor=WORLD_STYLE[panel].get("markerfacecolor", WORLD_COLORS[panel]), markeredgecolor=WORLD_COLORS[panel], markersize=6)
        ax.text(0.03, 0.95, f"α = {alpha:.2f}", transform=ax.transAxes, va="top", color=WORLD_COLORS[panel], fontsize=11)
        ax.set_title(study.WORLD_LABELS[panel], fontsize=11)
        ax.set_xlabel("ρ: recent volatility ÷ market-factor value")
    axes[0].set_ylabel("R: realised next 21 days ÷ the market")
    axes[0].set_xlim(0.4, 2.2)
    axes[0].set_ylim(0.6, 1.8)
    handles = [
        Line2D([], [], color="#888888", linestyle=":", label="trust 0: the market factor alone"),
        Line2D([], [], color="#888888", linestyle="--", label="trust 1: the stock's own volatility"),
        Line2D([], [], color=MUTED, linestyle="-", label="fitted: 1 + α (ρ − 1)"),
        Line2D([], [], color=MUTED, linestyle="none", marker="o", label="mean of a 5% bin of ρ (20 bins)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("How much of a stock's own calm or storm is still there next month?", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- rolling runs


def registry(trusts: tuple[float, ...]) -> dict:
    return {strategy_name(t)[len("minvar_") :]: standardized_estimator(VOL_HALFLIFE, "rie", mode="market", trust=t) for t in trusts}


def run_panel(args: argparse.Namespace, panel: str, data: pd.DataFrame, trusts: tuple[float, ...]) -> None:
    folder = args.out / f"{panel}_T{WINDOW}"
    if (folder / "periods.csv").is_file():
        print(f"{panel}: reusing saved run")
        return
    names = tuple(strategy_name(t) for t in trusts)
    config = BacktestConfig(window=WINDOW, holding=HOLDING, strategies=names)
    result = run_backtest(data, config, registry(trusts))
    result.save(folder)
    summary = result.summary()
    print(f"{panel} T = {WINDOW}: {int(summary['n_periods'].max())} periods")
    for name in names:
        print(f"   {label_of(name):52s} ratio {summary.loc[name, 'median_vol_ratio']:.3f}  vol {summary.loc[name, 'ann_volatility']:.2%}  turnover {summary.loc[name, 'mean_turnover']:.3f}  leverage {summary.loc[name, 'mean_gross_leverage']:.2f}")


def load_runs(out: Path) -> dict[str, list[BacktestResult]] | None:
    """Per panel: the trust runs, the saved standardisation parts (trust 0 and the per-stock forecast) and the saved plain RIE."""
    runs: dict[str, list[BacktestResult]] = {}
    for panel in PANELS:
        folder = out / f"{panel}_T{WINDOW}"
        if not (folder / "periods.csv").is_file():
            return None
        pieces = [BacktestResult.load(folder)]
        parts = study.OUT / f"{panel}_T{WINDOW}_parts"
        base = study.OUT / f"{panel}_T{WINDOW}" if panel == "clocked" else study.SAVED_RIE / f"{panel}_T{WINDOW}"
        for saved in (parts, base):
            if (saved / "periods.csv").is_file():
                pieces.append(BacktestResult.load(saved))
        runs[panel] = pieces
    return runs


def order_of(runs: dict[str, list[BacktestResult]]) -> list[str]:
    trusts = sorted(t for t in (trust_of(n) for n in runs["real"][0].strategies) if t is not None)
    return ["minvar_rie", "minvar_mkt20_rie"] + [strategy_name(t) for t in trusts] + ["minvar_lat20_rie"]


def summary_of(pieces: list[BacktestResult], order: list[str]) -> pd.DataFrame:
    table = pd.concat([piece.summary() for piece in pieces])
    table = table[~table.index.duplicated(keep="first")]
    return table.reindex([s for s in order if s in table.index])


def three_worlds(runs: dict[str, list[BacktestResult]], order: list[str]) -> pd.DataFrame:
    frames = {}
    for panel, pieces in runs.items():
        summary = summary_of(pieces, order)
        for metric in ("median_vol_ratio", "ann_volatility", "mean_turnover", "mean_gross_leverage"):
            frames[(metric, panel)] = summary[metric]
    table = pd.DataFrame(frames)
    table.columns = pd.MultiIndex.from_tuples(table.columns, names=["metric", "panel"])
    return table.reindex([s for s in order if s in table.index])


# ----------------------------------------------------------------------------- report


def report(args: argparse.Namespace) -> None:
    lines: list[str] = []
    lines += measurement_section(args.out)
    runs = load_runs(args.out)
    if runs is None:
        lines.append("Not every panel is saved yet (--part rolling --panel real|gaussian|clocked).")
    else:
        order = order_of(runs)
        table = three_worlds(runs, order)
        table.to_csv(args.out / "three_worlds.csv")
        lines += describe(table, runs)
        plot_three_worlds(table, args.out / "three_worlds.png")
    text = "\n".join(lines)
    print(text)
    (args.out / "summary.txt").write_text(text + "\n")
    print(f"Wrote {args.out}")


def measurement_section(out: Path) -> list[str]:
    path = out / "trust_measurement.csv"
    if not path.is_file():
        return ["The trust has not been measured yet (--part measure).", ""]
    summary = pd.read_csv(path, index_col=0)
    summary.index = [study.WORLD_LABELS[p] for p in summary.index]
    lines = [
        f"--- How much of a stock's own volatility deviation survives into the next hold? (window {WINDOW}, half-life {VOL_HALFLIFE:g}, {HOLDING}-session holds) ---",
        "ρ = stock's EWMA volatility ÷ (window σ̄ × market factor), read at each rebalance; R = the same ratio realised over the next hold.",
        "trust_linear is α in R − 1 = α (ρ − 1), fitted through the origin over all (date, stock) pairs; trust_log the same in logs.",
        f"Pairs where the stock did not trade on {1 - ACTIVE_SHARE:.0%} or more of the window's days are dropped (price series filled forward before listing).",
        summary.to_string(float_format=lambda v: f"{v:.3f}"),
        "",
    ]
    deciles = out / "trust_deciles_real.csv"
    if deciles.is_file():
        table = pd.read_csv(deciles, index_col=0)
        lines += ["S&P 500, by decile of ρ (does a stock that reads calm stay calm?):", table.to_string(float_format=lambda v: f"{v:.3f}"), ""]
    errors = out / "trust_forecast_error_real.csv"
    if errors.is_file():
        table = pd.read_csv(errors, index_col=0)
        lines += ["S&P 500, how wrong each stock's volatility forecast is at each setting of the dial (over the same pairs):", table.to_string(float_format=lambda v: f"{v:.3f}"), ""]
    by_year = out / "trust_by_year_real.csv"
    if by_year.is_file():
        table = pd.read_csv(by_year, index_col=0)[["pairs", "trust_linear", "trust_log", "correlation"]]
        lines += ["S&P 500, by year of the rebalance date:", table.to_string(float_format=lambda v: f"{v:.3f}"), ""]
    return lines


def describe(table: pd.DataFrame, runs: dict[str, list[BacktestResult]]) -> list[str]:
    first = runs["real"][0]
    n = int(first.periods.groupby("strategy").size().max())
    lines = [
        f"--- Three worlds, window T = {WINDOW}, 21-session holds, {n} holding periods {first.periods['hold_start'].min().date()} → {first.periods['hold_end'].max().date()} ---",
        "trust 0 is the market factor alone; trust 1 is each stock's own EWMA volatility; the last row is the per-stock forecast with the raw correlation, from the standardisation study.",
        "",
    ]
    fmt = {"median_vol_ratio": "{:.3f}", "ann_volatility": "{:.2%}", "mean_turnover": "{:.3f}", "mean_gross_leverage": "{:.2f}"}
    titles = {
        "median_vol_ratio": "Median realised / predicted volatility (1 is honest; 0.98 over 21-day holds)",
        "ann_volatility": "Realised volatility, annualised (lower is better)",
        "mean_turnover": "Turnover per rebalance",
        "mean_gross_leverage": "Gross leverage (sum of |weights|)",
    }
    for metric in fmt:
        wide = table[metric].reindex(columns=list(PANELS)).copy()
        wide.index = [label_of(s) for s in wide.index]
        wide.columns = [study.WORLD_LABELS[p] for p in wide.columns]
        lines += [titles[metric] + ":", wide.to_string(float_format=lambda v: fmt[metric].format(v)), ""]
    return lines


def plot_three_worlds(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    dial = [s for s in table.index if s == "minvar_mkt20_rie" or trust_of(s) is not None]
    xs = [0.0 if s == "minvar_mkt20_rie" else trust_of(s) for s in dial]
    for ax, metric, title in zip(axes, ("median_vol_ratio", "ann_volatility"), ("Honesty: realised ÷ predicted volatility", "Realised volatility, annualised")):
        for panel in PANELS:
            ys = table[(metric, panel)].reindex(dial).to_numpy(dtype=float)
            style = dict(WORLD_STYLE[panel])
            face = style.pop("markerfacecolor", WORLD_COLORS[panel])
            ax.plot(xs, ys, color=WORLD_COLORS[panel], markerfacecolor=face, markeredgecolor=WORLD_COLORS[panel], markersize=7, linewidth=1.6, **style, label=study.WORLD_LABELS[panel])
            if ("minvar_rie" in table.index):
                plain = float(table[(metric, panel)].loc["minvar_rie"])
                ax.axhline(plain, color=WORLD_COLORS[panel], linewidth=0.9, linestyle=":", alpha=0.8)
        ax.set_xlabel("trust α in each stock's own volatility (0 = market factor alone)")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(sorted(set(xs)))
        ax.set_xticklabels([f"{x:.2f}" for x in sorted(set(xs))])
        ax.grid(axis="y", color="#e6e6e6")
    axes[0].axhline(0.98, color="#888888", linewidth=1.0, linestyle="--")
    axes[0].text(1.0, 0.98, " honest", va="center", ha="left", color="#888888", fontsize=9, clip_on=False)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, linestyle=":", label="dotted: plain RIE (flat window σ) in that world"))
    labels.append("dotted: plain RIE (flat window σ) in that world")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Turning the stock dial: what the optimiser does with each stock's own volatility", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
