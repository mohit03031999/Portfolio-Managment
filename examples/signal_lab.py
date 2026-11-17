"""The signal lab: which simple return signals predict next month, across the whole market pack?

For each signal in ``market_analyzer.signals`` the script ranks every stock in
the pack at each month-end and scores the ranking against the stocks' returns
over the following 21 sessions: information coefficient (rank correlation),
decile portfolios and their top-minus-bottom spread, the calibration slope in
return per standard deviation of signal, the decay of the coefficient over the
months ahead, top-decile turnover and coverage. Two controls: the historical
mean (the textbook forecast, expected to carry nothing) and momentum scored
against shuffled outcomes (must be zero). Every signal is also scored inside
the ``--large`` most liquid stocks as of each date (trailing six-month median
dollar volume from the raw cache, a point-in-time split; today's market cap
would pick the period's winners).

Caveats printed with the results: the pack holds stocks listed today, so
companies that were delisted since 2018 are absent (survivorship, visible in
the pack's own equal-weight return by year, which the script prints); there
are no trading costs and no price floor.

Outputs in output/signal_lab/: summary.csv / summary.txt (one row per signal
and universe), ic_monthly.csv, spread_monthly.csv, decay.csv, ic_by_year.csv,
deciles/<key>.csv, and the figures spread_growth.png, ic_by_year.png,
deciles.png, decay.png.

    python examples/signal_lab.py
    python examples/signal_lab.py --replot
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_analyzer.market_pack import load_pack  # noqa: E402
from market_analyzer.signal_lab import SignalScore, decision_dates, score_signal, summary_table  # noqa: E402
from market_analyzer.signals import SIGNALS, compute_signals  # noqa: E402

OUT = ROOT / "output" / "signal_lab"
MIN_COVERAGE = 500
COLOURS = {
    "momentum": "#0072B2",
    "reversal": "#D55E00",
    "low_vol": "#009E73",
    "low_beta": "#CC79A7",
    "mean_1y": "#E69F00",
    "mean_3y": "#56B4E9",
}
CAVEATS = (
    "Caveats: the pack holds stocks listed today, so companies delisted since 2018 are absent",
    "(survivorship flatters every signal, and high-volatility and loser-buying ones most);",
    "no trading costs, no price floor; months are 21-session holds from month-ends;",
    "t-statistics treat months as independent; decile returns are equal-weight means, which",
    "a few extreme winners can dominate (the IC is rank-based and does not have this problem).",
)


def run(pack_dir: Path, raw_dir: Path, out: Path, horizon: int, large: int) -> None:
    started = time.time()
    pack = load_pack(pack_dir)
    returns = pack.returns.astype(float)
    broken = int((returns <= -0.99).sum().sum())
    returns = returns.where(returns > -0.99)
    print(f"pack: {returns.shape[1]} stocks, {returns.index[0].date()} to {returns.index[-1].date()}; "
          f"{broken} returns at or below -99% treated as missing")

    signals = compute_signals(returns)
    print(f"signals computed in {time.time() - started:.0f} s")

    all_dates = decision_dates(returns.index)
    momentum_cover = signals["momentum"].reindex(all_dates).notna().sum(axis=1)
    last_scoreable = returns.index[-horizon - 1]
    dates = all_dates[(momentum_cover >= MIN_COVERAGE) & (all_dates <= last_scoreable)]
    print(f"decision dates: {len(dates)} month-ends, {dates[0].date()} to {dates[-1].date()}")

    liquid = liquidity_mask(raw_dir, returns, dates, large)
    if liquid is None:
        print(f"no raw cache at {raw_dir}: the liquid-universe rows are skipped")

    scores: list[SignalScore] = []
    for key, spec in SIGNALS.items():
        in_units = key.startswith("mean_")
        scores.append(
            score_signal(signals[key], returns, dates, key=key, label=spec.label, horizon=horizon, in_return_units=in_units)
        )
        line = f"  {spec.label:28s} IC {scores[-1].ic.mean():+.3f} (t {scores[-1].summary()['ic_t']:+.1f})"
        if liquid is not None:
            scores.append(
                score_signal(
                    signals[key], returns, dates, key=f"{key}_liquid", label=f"{spec.label} · liquid {large}",
                    horizon=horizon, mask=liquid, in_return_units=in_units,
                )
            )
            line += f" liquid {scores[-1].ic.mean():+.3f}"
        print(f"{line}   {time.time() - started:.0f} s")
    control = score_signal(
        signals["momentum"], returns, dates, key="control", label="Momentum vs shuffled outcomes",
        horizon=horizon, shuffle_seed=0,
    )
    scores.append(control)
    print(f"  {'control (shuffled)':28s} IC {control.ic.mean():+.3f}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "deciles").mkdir(exist_ok=True)
    table = summary_table(scores)
    table.to_csv(out / "summary.csv")
    pd.DataFrame({s.key: s.ic for s in scores}).to_csv(out / "ic_monthly.csv")
    pd.DataFrame({s.key: s.spread for s in scores}).to_csv(out / "spread_monthly.csv")
    pd.DataFrame({s.key: s.decay for s in scores}).to_csv(out / "decay.csv")
    pd.DataFrame({s.key: s.ic_by_year() for s in scores}).to_csv(out / "ic_by_year.csv")
    for s in scores:
        s.deciles.to_csv(out / "deciles" / f"{s.key}.csv")
    yearly = yearly_equal_weight_return(returns)
    yearly.to_csv(out / "pack_equal_weight_by_year.csv")
    (out / "summary.txt").write_text(format_summary(table, dates, returns.shape[1], large if liquid is not None else 0, yearly))
    print((out / "summary.txt").read_text())
    plot(out)
    print(f"done in {time.time() - started:.0f} s; outputs in {out}")


def liquidity_mask(raw_dir: Path, returns: pd.DataFrame, dates: pd.DatetimeIndex, top: int) -> pd.DataFrame | None:
    """True for the ``top`` stocks by trailing six-month median dollar volume as of each date."""
    close_file, volume_file = raw_dir / "adj_close.parquet", raw_dir / "volume.parquet"
    if not (close_file.exists() and volume_file.exists()):
        return None
    close = pd.read_parquet(close_file).reindex(columns=returns.columns)
    volume = pd.read_parquet(volume_file).reindex(columns=returns.columns)
    dollars = (close * volume).reindex(returns.index)
    trailing = dollars.rolling(126, min_periods=60).median().reindex(dates)
    ranks = trailing.rank(axis=1, ascending=False)
    return ranks <= top


def yearly_equal_weight_return(returns: pd.DataFrame) -> pd.Series:
    market = returns.mean(axis=1, skipna=True)
    return ((1.0 + market).groupby(market.index.year).prod() - 1.0).rename("equal_weight_return")


def format_summary(table: pd.DataFrame, dates: pd.DatetimeIndex, n_stocks: int, n_liquid: int, yearly: pd.Series) -> str:
    universe = (f"liquid {n_liquid} = the {n_liquid} stocks with the highest trailing six-month dollar volume as of each date"
                if n_liquid else "no liquid-universe rows (raw cache absent)")
    lines = [
        f"Signal lab: {n_stocks} stocks in the pack, {len(dates)} month-ends {dates[0].date()} to {dates[-1].date()},",
        f"21-session forward returns; {universe}.",
        "",
        "Equal-weight return of every stock in the pack, by year (a survivorship tell when far above real baskets):",
        "  " + "  ".join(f"{year} {value*100:+.0f}%" for year, value in yearly.items()),
        "",
        f"{'signal':34s} {'IC':>7s} {'t':>6s} {'%>0':>5s} {'1st':>7s} {'2nd':>7s} {'spread/yr':>10s} {'Sharpe':>14s} "
        f"{'bp/sd':>7s} {'trust':>7s} {'turn':>6s} {'n':>6s} {'months':>6s}",
    ]
    for label, row in table.iterrows():
        trust = f"{row['trust_fraction']:+.2f}" if "trust_fraction" in row and pd.notna(row.get("trust_fraction")) else "     ·"
        lines.append(
            f"{label:34s} {row['ic_mean']:+.3f} {row['ic_t']:+6.1f} {row['ic_positive_share']*100:5.0f} "
            f"{row['ic_first_half']:+.3f} {row['ic_second_half']:+.3f} {row['spread_annual_return']*100:+9.1f}% "
            f"{row['spread_sharpe']:+5.2f} ± {row['spread_sharpe_se']:4.2f} {row['slope_per_sd']*1e4:+7.0f} {trust:>7s} "
            f"{row['top_decile_turnover']:6.2f} {row['coverage']:6.0f} {row['months']:6.0f}"
        )
    decay_cols = [c for c in table.columns if c.startswith("ic_month_")]
    lines += ["", "Decay: IC against the k-th month ahead", f"{'signal':34s} " + " ".join(f"{c[9:]:>7s}" for c in decay_cols)]
    for label, row in table.iterrows():
        lines.append(f"{label:34s} " + " ".join(f"{row[c]:+7.3f}" for c in decay_cols))
    lines += [
        "",
        "IC = rank correlation between the signal and the next month's return across stocks, averaged over months;",
        "t = mean / standard error over months; %>0 = share of months with a positive IC; 1st/2nd = IC in each half;",
        "spread/yr = annualised return of top decile minus bottom decile, equal weight; Sharpe with its error bar;",
        "bp/sd = next-month relative return per one standard deviation of the signal, basis points;",
        "trust = slope of realised on forecast for signals already in return units (1 = trust fully, 0 = nothing);",
        "turn = share of the top decile replaced each month; n = stocks scored per month.",
        "",
    ] + list(CAVEATS)
    return "\n".join(lines) + "\n"


def plot(out: Path) -> None:
    table = pd.read_csv(out / "summary.csv", index_col=0)
    spreads = pd.read_csv(out / "spread_monthly.csv", index_col=0, parse_dates=True)
    monthly = pd.read_csv(out / "ic_monthly.csv", index_col=0, parse_dates=True)
    months_in_year = monthly.groupby(monthly.index.year).size()
    by_year = monthly.groupby(monthly.index.year).mean().loc[months_in_year[months_in_year >= 6].index]
    decay = pd.read_csv(out / "decay.csv", index_col=0)
    keys = list(COLOURS)
    labels = {key: SIGNALS[key].label for key in keys}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key in keys:
        growth = (1.0 + spreads[key].fillna(0.0)).cumprod()
        ax.plot(growth.index, growth.to_numpy(), color=COLOURS[key], lw=2, label=labels[key])
        ax.annotate(f"{labels[key]} {growth.iloc[-1]:.2f}", (growth.index[-1], growth.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", fontsize=8, va="center", color=COLOURS[key])
    control = (1.0 + spreads["control"].fillna(0.0)).cumprod()
    ax.plot(control.index, control.to_numpy(), color="#9CA3AF", lw=1.5, ls="--", label="Momentum vs shuffled outcomes")
    ax.axhline(1.0, color="#9CA3AF", lw=0.8)
    ax.set_title("Growth of 1 in each signal's top-decile minus bottom-decile spread (equal weight, monthly, no costs)")
    ax.set_ylabel("growth of 1")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(out / "spread_growth.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    years = by_year.index.to_numpy()
    width = 0.8 / len(keys)
    for i, key in enumerate(keys):
        ax.bar(np.arange(len(years)) + (i - len(keys) / 2 + 0.5) * width, by_year[key].to_numpy(), width,
               color=COLOURS[key], label=labels[key])
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set_xticks(np.arange(len(years)), [str(y) for y in years])
    ax.set_ylabel("mean IC in the year")
    ax.set_title("Information coefficient by calendar year (years with at least six months)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "ic_by_year.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    for ax, key in zip(axes.ravel(), keys):
        deciles = pd.read_csv(out / "deciles" / f"{key}.csv", index_col=0, parse_dates=True)
        annual = deciles.mean() * 12 * 100
        ax.bar(np.arange(1, 11), annual.to_numpy(), color=COLOURS[key], width=0.7)
        ax.set_title(labels[key], fontsize=10)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("decile (10 = signal says best)")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("annualised mean return, %")
    axes[1, 0].set_ylabel("annualised mean return, %")
    fig.suptitle("Average annual return of each decile portfolio (equal weight, no costs)")
    fig.tight_layout()
    fig.savefig(out / "deciles.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key in keys:
        ax.plot(decay.index.to_numpy(), decay[key].to_numpy(), marker="o", ms=5, lw=2, color=COLOURS[key], label=labels[key])
    ax.plot(decay.index.to_numpy(), decay["control"].to_numpy(), marker="o", ms=4, lw=1.5, ls="--", color="#9CA3AF",
            label="Momentum vs shuffled outcomes")
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set_xlabel("month ahead")
    ax.set_ylabel("mean IC")
    ax.set_title("How fast each signal's information fades")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "decay.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", default=str(ROOT / "data" / "pack"))
    parser.add_argument("--raw", default=str(ROOT / "data" / "raw"),
                        help="raw cache with adj_close/volume parquet files, for the liquid universe")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--large", type=int, default=1000, help="size of the liquid universe")
    parser.add_argument("--replot", action="store_true", help="redraw the figures from the saved tables")
    args = parser.parse_args()
    if args.replot:
        plot(Path(args.out))
        return
    run(Path(args.pack), Path(args.raw), Path(args.out), args.horizon, args.large)


if __name__ == "__main__":
    main()
