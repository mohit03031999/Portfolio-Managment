"""The customer's risk dial: asked for X% yearly swings, got how much?

The dial splits the money between a stock basket and cash so the mix swings
by a chosen amount a year (see ``market_analyzer.risk_dial``). The split is
``target / forecast``, so the dial is exactly as good as its forecast of the
basket's volatility. This script replays the dial over the baskets saved by
``sp500_gross_cap_study.py`` (window 1000 sessions, monthly rebalance, 456
S&P 500 stocks, 2018-12 to 2026-10) and asks three questions.

1. Which forecast keeps the promise? The risk model's own number, the
   basket's behaviour over the last quarter, or a recent-weighted average,
   and does it help to reset the dial weekly or daily instead of monthly?
   Scored without a cap on the basket share, so the forecast is judged alone.
2. Is the miss about the dial or about the market? The same dials run on
   Gaussian days drawn from one fixed covariance, a world where risk never
   changes.
3. What would a customer have lived through? Targets from 4% to 12% a year
   with no borrowing, under the slow dial and the recommended one.

Outputs in output/sp500_risk_dial/: forecast_grid.csv, by_year.csv,
customer_table.csv, summary.txt, promise.png, through_time.png. If the saved
baskets are missing the script runs the backtest it needs first.

    python examples/sp500_risk_dial_study.py
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import sp500_shuffle_test as shuffle  # noqa: E402
import sp500_simulated_backtest as simulated  # noqa: E402
from market_analyzer.backtest import BacktestConfig, BacktestResult, run_backtest  # noqa: E402
from market_analyzer.risk_dial import DialConfig, DialResult, dial_backtest  # noqa: E402

OUT = ROOT / "output" / "sp500_risk_dial"
SAVED = ROOT / "output" / "sp500_gross_cap"
WINDOW, HOLDING, ANNUAL, LAB_SEED = 1000, 21, 252, 1
BASKETS = {
    "longonly_rie": "Long-only, cleaned (RIE)",
    "gross2_rie": "150/50, cleaned (RIE)",
    "minvar_rie": "Unconstrained, cleaned (RIE)",
    "equal_weight": "Equal weight, no optimiser",
}
LAB_ONLY = {"longonly_oracle": "Long-only, true covariance"}
RECENT = dict(lookback=252, halflife=20)
DIALS: dict[str, tuple[str, dict]] = {
    "model_monthly": ("Risk model, reset monthly", dict(forecast="model", dial_every=None)),
    "quarter_monthly": ("Last quarter, reset monthly", dict(forecast="recent", lookback=63, halflife=None, dial_every=None)),
    "quarter_weekly": ("Last quarter, reset weekly", dict(forecast="recent", lookback=63, halflife=None, dial_every=5)),
    "recent_monthly": ("Recent-weighted, reset monthly", dict(forecast="recent", dial_every=None, **RECENT)),
    "recent_weekly": ("Recent-weighted, reset weekly", dict(forecast="recent", dial_every=5, **RECENT)),
    "recent_daily": ("Recent-weighted, reset daily", dict(forecast="recent", dial_every=1, **RECENT)),
    "higher_weekly": ("Higher of model and recent-weighted, weekly", dict(forecast="higher", dial_every=5, **RECENT)),
    "higher_daily": ("Higher of model and recent-weighted, daily", dict(forecast="higher", dial_every=1, **RECENT)),
}
SLOW, CHOSEN = "model_monthly", "higher_weekly"
HEADLINE = "longonly_rie"
PROBE_TARGET = 0.05  # with no cap the score does not depend on the target
TARGETS = (0.04, 0.06, 0.08, 0.10, 0.12)
SHOWN_TARGET = 0.08
BLUE, RED, GREY, INK = "#0072B2", "#D55E00", "#9CA3AF", "#374151"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    returns = simulated.load_returns(simulated.PRICES)
    worlds = {"real": (returns, load_or_run(returns))}
    lab_folder = SAVED / f"gaussian_raw_s{LAB_SEED}" / f"T{WINDOW}"
    if (lab_folder / "periods.csv").is_file():
        worlds["lab"] = (shuffle.gaussian_raw(returns, LAB_SEED), BacktestResult.load(lab_folder))

    grid, yearly = [], []
    for world, (panel, backtest) in worlds.items():
        names = list(BASKETS) + (list(LAB_ONLY) if world == "lab" else [])
        for basket in names:
            if basket not in backtest.weights:
                continue
            for dial, (_, options) in DIALS.items():
                config = DialConfig(PROBE_TARGET, max_exposure=1e6, **options)
                result = dial_backtest(backtest, basket, panel, config)
                check = (result.daily["basket_return"] - backtest.daily_returns[basket]).abs().max()
                assert check < 1e-9, f"{world} {basket}: basket returns differ from the saved backtest by {check}"
                grid.append({"world": world, "basket": basket, "dial": dial, **score(result)})
                table = result.by_year().reset_index()
                table.insert(0, "dial", dial)
                table.insert(0, "basket", basket)
                table.insert(0, "world", world)
                yearly.append(table)
    grid = pd.DataFrame(grid)
    yearly = pd.concat(yearly, ignore_index=True)
    grid.to_csv(OUT / "forecast_grid.csv", index=False)
    yearly.to_csv(OUT / "by_year.csv", index=False)

    panel, backtest = worlds["real"]
    customers, tracks = [], {}
    for dial in (SLOW, CHOSEN):
        for target in TARGETS:
            result = dial_backtest(backtest, HEADLINE, panel, DialConfig(target, max_exposure=1.0, **DIALS[dial][1]))
            customers.append({"dial": dial, **result.summary()})
            if target == SHOWN_TARGET:
                tracks[dial] = result
    customers = pd.DataFrame(customers)
    customers.to_csv(OUT / "customer_table.csv", index=False)

    basket_turnover = backtest.summary().loc[HEADLINE, "mean_turnover"] * ANNUAL / HOLDING
    text = report(grid, customers, backtest, basket_turnover, "lab" in worlds)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)
    plot_promise(grid, OUT / "promise.png")
    plot_through_time(tracks, OUT / "through_time.png")
    print(f"\nwrote {OUT}")


def load_or_run(returns: pd.DataFrame) -> BacktestResult:
    """The saved baskets from the gross-cap study, or a fresh backtest of the four baskets."""
    folder = SAVED / "real" / f"T{WINDOW}"
    if (folder / "periods.csv").is_file():
        saved = BacktestResult.load(folder)
        if all(name in saved.weights for name in BASKETS):
            return saved
    own = OUT / "backtest"
    if (own / "periods.csv").is_file():
        return BacktestResult.load(own)
    print("No saved baskets found, running the backtest (a few minutes)...")
    result = run_backtest(returns, BacktestConfig(window=WINDOW, holding=HOLDING, strategies=tuple(BASKETS)))
    result.save(own)
    return result


def score(result: DialResult) -> dict[str, float]:
    """Delivered over asked: whole track, each half, calmest and wildest calendar year, and how hard the dial trades."""
    target = result.config.target_vol
    track = result.daily["return"]
    middle = len(track) // 2
    yearly = result.by_year()["delivered_over_target"]
    exposure = result.daily["exposure"]
    years = len(track) / ANNUAL
    ratio = lambda series: float(series.std(ddof=1)) * math.sqrt(ANNUAL) / target  # noqa: E731
    return {
        "delivered_over_target": ratio(track),
        "first_half": ratio(track.iloc[:middle]),
        "second_half": ratio(track.iloc[middle:]),
        "without_2020": ratio(track[track.index.year != 2020]),
        "calmest_year": float(yearly.min()),
        "wildest_year": float(yearly.max()),
        "wildest_year_is": int(yearly.idxmax()),
        "trades_per_year": float(exposure.diff().abs().sum() / exposure.mean() / years),
    }


# ----------------------------------------------------------------------------- the report


def report(grid: pd.DataFrame, customers: pd.DataFrame, backtest: BacktestResult, basket_turnover: float, has_lab: bool) -> str:
    first, last = backtest.periods["hold_start"].min(), backtest.periods["hold_end"].max()
    lines = [
        "The customer's risk dial: asked for X% yearly swings, got how much?",
        f"Baskets: window {WINDOW} sessions, rebalanced every {HOLDING}, {first.date()} to {last.date()}.",
        "Every number in parts 1 and 2 is delivered volatility divided by the target. 1.00 is a promise kept,",
        "1.50 means the customer got one and a half times the swings they asked for.",
        "",
        "1. Which forecast keeps the promise? Real market, no cap on the basket share.",
        "   trades/yr = how many times the dial alone turns the stock holding over in a year.",
    ]
    for basket, label in BASKETS.items():
        lines += ["", f"   {label}"] + grid_block(grid, "real", basket)
    lines += [
        "",
        f"   For scale: reshuffling the stocks inside the long-only basket trades {basket_turnover:.1f} times the holding a year.",
    ]
    if has_lab:
        lines += [
            "",
            "2. Same dials in a lab where risk never changes (Gaussian days from one fixed covariance).",
        ]
        for basket, label in {**BASKETS, **LAB_ONLY}.items():
            lines += ["", f"   {label}"] + grid_block(grid, "lab", basket)
    lines += [
        "",
        f"3. What a customer would have lived through. Basket: {BASKETS[HEADLINE]}. No borrowing: at most 100% in the basket.",
        "   The worst day and the largest fall are flattered: this stock list only holds companies that survived.",
    ]
    for dial in (SLOW, CHOSEN):
        lines += ["", f"   Dial: {DIALS[dial][0]}", customer_block(customers.loc[customers["dial"] == dial])]
    return "\n".join(lines)


def grid_block(grid: pd.DataFrame, world: str, basket: str) -> list[str]:
    rows = grid.loc[(grid["world"] == world) & (grid["basket"] == basket)].set_index("dial")
    if rows.empty:
        return ["   (not in the saved run)"]
    out = [f"   {'dial':<46}{'whole':>7}{'1st half':>10}{'2nd half':>10}{'calmest yr':>12}{'wildest yr':>12}{'trades/yr':>11}"]
    for dial, (label, _) in DIALS.items():
        row = rows.loc[dial]
        wild = f"{row['wildest_year']:.2f} ({int(row['wildest_year_is'])})"
        out.append(
            f"   {label:<46}{row['delivered_over_target']:>7.2f}{row['first_half']:>10.2f}{row['second_half']:>10.2f}"
            f"{row['calmest_year']:>12.2f}{wild:>12}{row['trades_per_year']:>11.1f}"
        )
    return out


def customer_block(rows: pd.DataFrame) -> str:
    header = (
        f"   {'asked':>6}{'got':>8}{'got/asked':>11}{'calmest yr':>12}{'wildest yr':>12}"
        f"{'avg in basket':>15}{'least':>8}{'days at 100%':>14}{'worst day':>11}{'largest fall':>14}"
    )
    out = [header]
    for _, row in rows.iterrows():
        out.append(
            f"   {row['target_vol']:>6.0%}{row['delivered_vol']:>8.1%}{row['delivered_over_target']:>11.2f}"
            f"{row['calmest_year_vol']:>12.1%}{row['wildest_year_vol']:>12.1%}{row['mean_exposure']:>15.0%}"
            f"{row['min_exposure']:>8.0%}{row['share_at_cap']:>14.0%}{row['worst_day']:>11.1%}{row['max_drawdown']:>14.1%}"
        )
    return "\n".join(out)


# ----------------------------------------------------------------------------- the figures


def plot_promise(grid: pd.DataFrame, path: Path) -> None:
    """One row per dial: dot = whole history, bar = calmest to wildest calendar year."""
    worlds = [world for world in ("real", "lab") if (grid["world"] == world).any()]
    titles = {"real": "Real market", "lab": "Lab where risk never changes"}
    order = list(DIALS)[::-1]
    fig, axes = plt.subplots(1, len(worlds), figsize=(7.0 * len(worlds), 4.6), sharex=True, sharey=True, squeeze=False)
    for ax, world in zip(axes[0], worlds):
        rows = grid.loc[(grid["world"] == world) & (grid["basket"] == HEADLINE)].set_index("dial")
        for y, dial in enumerate(order):
            row = rows.loc[dial]
            color = BLUE if dial == CHOSEN else (RED if dial == SLOW else INK)
            ax.plot([row["calmest_year"], row["wildest_year"]], [y, y], color=color, lw=2, solid_capstyle="round", alpha=0.45)
            ax.plot(row["delivered_over_target"], y, "o", ms=8, color=color, mec="white", mew=1.5, zorder=3)
            ax.annotate(
                f"{row['delivered_over_target']:.2f} overall, {row['wildest_year']:.2f} in the wildest year",
                (row["wildest_year"], y),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=INK,
            )
        ax.axvline(1.0, color=GREY, lw=1, ls="--", zorder=0)
        ax.set_title(titles[world], fontsize=11, loc="left", color=INK)
        ax.set_xlabel("Swings delivered ÷ swings asked for (1 = promise kept)", fontsize=9, color=INK)
        ax.set_yticks(range(len(order)), [DIALS[dial][0] for dial in order], fontsize=9)
        ax.tick_params(axis="x", labelsize=9, colors=INK)
        ax.tick_params(axis="y", length=0, colors=INK)
        ax.grid(axis="x", color="#E5E7EB", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GREY)
    axes[0][0].set_xlim(0.5, float(grid.loc[grid["basket"] == HEADLINE, "wildest_year"].max()) + 2.6)
    fig.suptitle(
        f"Does the dial keep its promise? {BASKETS[HEADLINE]} basket. Dot: whole history. Bar: calmest to wildest year.",
        fontsize=11,
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_through_time(tracks: dict[str, DialResult], path: Path) -> None:
    """Rolling three-month volatility the customer felt, and the share of money in the basket."""
    styles = {SLOW: (RED, "Slow dial: risk model, reset monthly"), CHOSEN: (BLUE, "Fast dial: higher of model and recent, reset weekly")}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True, gridspec_kw={"height_ratios": (3, 2)})
    for dial, (color, label) in styles.items():
        daily = tracks[dial].daily
        rolling = daily["return"].rolling(63).std(ddof=1) * math.sqrt(ANNUAL)
        top.plot(rolling.index, rolling.to_numpy(), color=color, lw=2, label=label)
        bottom.plot(daily.index, daily["exposure"].to_numpy(), color=color, lw=1.6, label=label)
    top.axhline(SHOWN_TARGET, color=INK, lw=1, ls="--", label=f"What the customer asked for: {SHOWN_TARGET:.0%}")
    top.set_title(
        f"A customer who asked for {SHOWN_TARGET:.0%} yearly swings: what they felt over each three months",
        fontsize=11, loc="left", color=INK,
    )
    top.set_ylabel("Volatility, yearly", fontsize=9, color=INK)
    top.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    top.legend(frameon=False, fontsize=9, loc="upper right")
    bottom.set_title("Share of the money in the stock basket (the rest is cash)", fontsize=11, loc="left", color=INK)
    bottom.set_ylim(0, 1.05)
    bottom.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for ax in (top, bottom):
        ax.grid(axis="y", color="#E5E7EB", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9, colors=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GREY)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
