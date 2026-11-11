"""Fit the Marchenko-Pastur law on the S&P 500 at three sample lengths.

Uses one fixed set of current S&P 500 names: every name that has a complete
adjusted-close history over the most recent 2,001 sessions. The three windows
are the last T = 500, 1,000, and 2,000 daily returns of that same panel, so
N stays fixed and only q = N / (T - 1) changes.

Writes prices, returns, spectra, and plots under output/sp500_marchenko_pastur/.
Reuses prices.csv in that folder unless --refresh is passed.

    python examples/sp500_marchenko_pastur.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import io
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from market_analyzer.matrices import (
    CorrelationModel,
    marchenko_pastur_density,
    marchenko_pastur_edges,
)
from market_analyzer.prices import download_prices
from market_analyzer.universe import tickers_from_html_table, wikipedia_yahoo_symbol

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "sp500_marchenko_pastur"
WIKIPEDIA = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WINDOWS = (500, 1000, 2000)
LONGEST = max(WINDOWS)
PRICE_START = "2015-01-01"
CHUNK = 80

SIGNAL_COLOR = "#b23a48"
BULK_COLOR = "#3d6f8e"
GUIDE_COLOR = "#6e6e6e"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Download prices again.")
    parser.add_argument("--start", default=PRICE_START, help="Price history start. Default: 2015-01-01.")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    prices, requested = _load_prices(args.start, refresh=args.refresh)
    panel = _dense_return_panel(prices, LONGEST)
    panel.to_csv(OUT / "returns_T2000.csv")
    pd.Series(panel.columns, name="ticker").to_csv(OUT / "tickers.csv", index=False)

    print(
        f"Wikipedia names: {requested}   "
        f"with a complete last {LONGEST} returns: {panel.shape[1]}"
    )
    print(f"Sample: {panel.index[0].date()} → {panel.index[-1].date()}")

    models: dict[int, CorrelationModel] = {}
    windows: dict[int, pd.DataFrame] = {}
    for length in WINDOWS:
        window = panel.iloc[-length:]
        windows[length] = window
        window.to_csv(OUT / f"returns_T{length}.csv")
        model = CorrelationModel.from_returns(window)
        models[length] = model
        model.eigenvalues.to_csv(OUT / f"eigenvalues_T{length}.csv")
        model.signal_loadings.to_csv(OUT / f"signal_loadings_T{length}.csv")
        _plot_window(model, window)
        print(model.note())

    summary = _summary_table(models, windows, requested)
    summary.to_csv(OUT / "summary.csv", index=False)
    _plot_comparison(models)
    _plot_variance_share(models)
    _plot_market_loadings(models)
    _plot_summary_bars(summary)
    _plot_correlation_matrices(windows, _sector_by_ticker())
    print(f"Wrote {OUT}")
    return 0


def _load_prices(start: str, refresh: bool) -> tuple[pd.DataFrame, int]:
    cache = OUT / "prices.csv"
    tickers = tickers_from_html_table(
        WIKIPEDIA,
        column="Symbol",
        wikipedia_class_shares=True,
    )
    if cache.is_file() and not refresh:
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"Loaded cached prices {prices.shape} from {cache}")
        return prices, len(tickers)

    print(f"Downloading {len(tickers)} S&P 500 names from {start}")
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for offset in range(0, len(tickers), CHUNK):
        batch = tickers[offset : offset + CHUNK]
        try:
            prices, missed = download_prices(batch, start=start)
        except RuntimeError as exc:
            print(f"  chunk {offset // CHUNK + 1} failed: {exc}")
            missing.extend(batch)
            continue
        frames.append(prices)
        missing.extend(missed)
        print(f"  {min(offset + CHUNK, len(tickers))}/{len(tickers)}  missing {len(missing)}")
    if not frames:
        raise RuntimeError("Yahoo Finance returned no S&P 500 prices.")
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index()
    prices.index.name = "Date"
    prices.to_csv(cache)
    pd.Series(missing, name="ticker").to_csv(OUT / "missing_tickers.csv", index=False)
    print(f"Saved {prices.shape[1]} price columns, {len(missing)} names missing")
    return prices, len(tickers)


def _dense_return_panel(prices: pd.DataFrame, n_returns: int) -> pd.DataFrame:
    n_prices = n_returns + 1
    if len(prices) < n_prices:
        raise RuntimeError(
            f"Need {n_prices} price sessions for {n_returns} returns, found {len(prices)}. "
            "Pass an earlier --start."
        )
    tail = prices.iloc[-n_prices:]
    complete = [column for column in tail.columns if tail[column].notna().all()]
    if len(complete) < 2:
        raise RuntimeError("Fewer than two names have a complete history over the longest window.")
    returns = tail.loc[:, complete].pct_change().iloc[1:]
    usable = returns.columns[returns.std(ddof=1) > 0]
    returns = returns.loc[:, usable]
    if len(returns) != n_returns:
        raise RuntimeError(f"Expected {n_returns} return rows, built {len(returns)}.")
    return returns


def _plot_window(model: CorrelationModel, returns: pd.DataFrame) -> None:
    eigenvalues = model.eigenvalues["eigenvalue"].to_numpy()
    bulk = eigenvalues[~model.eigenvalues["above_noise_edge"].to_numpy()]
    length = model.observations
    _, sigma_edge = marchenko_pastur_edges(model.q, 1.0)
    x_max = max(sigma_edge, model.lambda_plus) * 1.12

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    ranks = np.arange(1, len(eigenvalues) + 1)
    above_fitted = model.eigenvalues["above_noise_edge"].to_numpy()
    above_unit = eigenvalues > sigma_edge
    between = above_fitted & ~above_unit
    inside = ~above_fitted
    axes[0].scatter(ranks[inside], eigenvalues[inside], s=12, color=BULK_COLOR, label="Inside the fitted band", zorder=2)
    axes[0].scatter(
        ranks[between],
        eigenvalues[between],
        s=14,
        color="#d68910",
        label="Above the fitted edge only",
        zorder=3,
    )
    axes[0].scatter(
        ranks[above_unit],
        eigenvalues[above_unit],
        s=22,
        color=SIGNAL_COLOR,
        label="Above the σ² = 1 edge",
        zorder=4,
    )
    axes[0].axhline(model.lambda_plus, color=SIGNAL_COLOR, linestyle="--", linewidth=1.2, label=f"Fitted edge {_fmt(model.lambda_plus)}")
    axes[0].axhline(sigma_edge, color=GUIDE_COLOR, linestyle=":", linewidth=1.2, label=f"σ² = 1 edge {_fmt(sigma_edge)}")
    axes[0].set_yscale("log")
    positive = eigenvalues[eigenvalues > 1e-4]
    if len(positive):
        axes[0].set_ylim(positive.min() * 0.7, eigenvalues.max() * 1.4)
    axes[0].set_xlabel("Eigenvalue rank (largest first)")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Spectrum")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    _draw_bulk(axes[1], bulk, model, x_max, legend=False)
    axes[1].set_title("Noise band")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.72, 0.01), frameon=False, fontsize=8, ncol=1)

    fig.suptitle(_window_title(model, returns), fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / f"window_T{length}.png", dpi=160)
    plt.close(fig)


def _plot_comparison(models: dict[int, CorrelationModel]) -> None:
    x_max = max(marchenko_pastur_edges(model.q, 1.0)[1] for model in models.values()) * 1.05
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for axis, length in zip(axes, WINDOWS):
        model = models[length]
        bulk = model.eigenvalues.loc[~model.eigenvalues["above_noise_edge"], "eigenvalue"].to_numpy()
        _draw_bulk(axis, bulk, model, x_max, legend=False)
        unit_edge = marchenko_pastur_edges(model.q, 1.0)[1]
        axis.set_title(f"T = {length}    q = {model.q:.2f}\nσ² = 1 edge {unit_edge:.2f}")
    axes[0].set_ylabel("Density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Dashed curve: noise band if σ² = 1. It widens when T is shorter.", fontsize=13)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(OUT / "bulk_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _draw_bulk(axis, bulk: np.ndarray, model: CorrelationModel, x_max: float, legend: bool = True) -> None:
    bins = np.linspace(0, x_max, 36)
    if len(bulk):
        counts, _ = np.histogram(bulk, bins=bins, density=True)
        peak = float(counts.max()) if counts.size else 1.0
        axis.hist(bulk, bins=bins, density=True, color=BULK_COLOR, alpha=0.55, label="Eigenvalues inside the band")
    else:
        peak = 1.0
    grid = np.linspace(max(x_max / 800, 1e-4), x_max, 600)
    fitted = marchenko_pastur_density(grid, model.q, model.sigma2)
    naive = marchenko_pastur_density(grid, model.q, 1.0)
    axis.plot(grid, fitted, color=SIGNAL_COLOR, linewidth=1.8, label="Marchenko-Pastur, fitted to the bulk")
    axis.plot(grid, naive, color=GUIDE_COLOR, linestyle="--", linewidth=1.3, label="Marchenko-Pastur, σ² = 1")
    axis.axvline(model.lambda_plus, color=SIGNAL_COLOR, linestyle=":", linewidth=1.0)
    axis.set_xlim(0, x_max)
    axis.set_ylim(0, max(peak, np.nanpercentile(fitted, 98)) * 1.35)
    axis.set_xlabel("Eigenvalue")
    if legend:
        axis.legend(frameon=False, fontsize=8, loc="upper right")


def _plot_variance_share(models: dict[int, CorrelationModel]) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for length, color in zip(WINDOWS, ("#1b4f72", "#148f77", "#b9770e")):
        share = models[length].eigenvalues["cumulative_share"].to_numpy()
        axis.plot(np.arange(1, len(share) + 1), share, color=color, linewidth=1.8, label=f"T = {length}")
    axis.set_xlim(1, 40)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Number of leading eigenvalues kept")
    axis.set_ylabel("Share of correlation-matrix trace")
    axis.set_title("How fast the leading eigenvalues explain the correlation matrix")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "variance_share.png", dpi=160)
    plt.close(fig)


def _plot_market_loadings(models: dict[int, CorrelationModel]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for axis, length in zip(axes, WINDOWS):
        weights = models[length].signal_loadings.iloc[:, 0].to_numpy()
        if np.nanmean(weights) < 0:
            weights = -weights
        axis.hist(weights, bins=30, color=BULK_COLOR, alpha=0.85)
        axis.axvline(np.mean(weights), color=SIGNAL_COLOR, linestyle="--", linewidth=1.2, label="Average weight")
        axis.axvline(1 / np.sqrt(len(weights)), color=GUIDE_COLOR, linestyle=":", linewidth=1.3, label="Equal weight, 1/√N")
        axis.set_title(f"T = {length}")
        axis.set_xlabel("Market-mode weight")
    axes[0].set_ylabel("Number of stocks")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Leading eigenvector. Most names load together, close to equal weight.", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "market_mode_loadings.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_bars(summary: pd.DataFrame) -> None:
    labels = [f"T = {int(length)}" for length in summary["T"]]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    specs = (
        ("q", "q = N / (T − 1)", BULK_COLOR),
        ("lambda_plus_if_sigma2_is_1", "Noise edge if σ² = 1", SIGNAL_COLOR),
        ("n_above_sigma1", "Eigenvalues above the σ² = 1 edge", "#1b4f72"),
    )
    for axis, (column, title, color) in zip(axes, specs):
        axis.bar(labels, summary[column], color=color, width=0.65)
        axis.set_title(title)
        integer = column.startswith("n_")
        for label, value in zip(labels, summary[column]):
            text = f"{int(value)}" if integer else f"{value:.2f}"
            axis.text(label, value, text, ha="center", va="bottom", fontsize=9)
        axis.set_ylim(0, max(summary[column]) * 1.28)
    fig.suptitle(f"S&P 500 names kept for every window: N = {int(summary['N'].iloc[0])}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "summary_metrics.png", dpi=160)
    plt.close(fig)


def _plot_correlation_matrices(windows: dict[int, pd.DataFrame], sectors: dict[str, str]) -> None:
    """Heatmaps of the three correlation matrices, stocks kept in one sector order."""
    order = _sector_order(windows[LONGEST].columns, sectors)
    matrices = {length: windows[length].corr().loc[order, order] for length in WINDOWS}
    values = np.concatenate([matrix.to_numpy()[~np.eye(len(order), dtype=bool)] for matrix in matrices.values()])
    norm = TwoSlopeNorm(vmin=float(values.min()), vcenter=0.0, vmax=1.0)
    groups = _sector_groups(order, sectors)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.8))
    image = None
    for index, (axis, length) in enumerate(zip(axes, WINDOWS)):
        image = _draw_correlation(axis, matrices[length], groups, norm, label_sectors=index == 0)
        start = str(windows[length].index[0].date())
        end = str(windows[length].index[-1].date())
        axis.set_title(f"T = {length}\n{start} to {end}")
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Correlation")
    fig.suptitle("Correlation matrices, same stocks, grouped by GICS sector", fontsize=13)
    fig.savefig(OUT / "correlation_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    for length in WINDOWS:
        fig, axis = plt.subplots(figsize=(8.4, 7.6))
        image = _draw_correlation(axis, matrices[length], groups, norm, label_sectors=True)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02, label="Correlation")
        start = str(windows[length].index[0].date())
        end = str(windows[length].index[-1].date())
        axis.set_title(f"Correlation matrix, T = {length} ({start} to {end})")
        fig.tight_layout()
        fig.savefig(OUT / f"correlation_T{length}.png", dpi=160)
        plt.close(fig)


def _draw_correlation(axis, matrix: pd.DataFrame, groups: list[tuple[str, int, int]], norm, label_sectors: bool):
    image = axis.imshow(
        matrix.to_numpy(),
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    for _, start, _ in groups[1:]:
        axis.axhline(start - 0.5, color="white", linewidth=0.5)
        axis.axvline(start - 0.5, color="white", linewidth=0.5)
    names = [_short_sector(name) for name, _, _ in groups]
    centers = [(start + stop - 1) / 2 for _, start, stop in groups]
    if label_sectors:
        axis.set_yticks(centers)
        axis.set_yticklabels(names, fontsize=7)
        axis.set_xticks([])
    else:
        axis.set_xticks([])
        axis.set_yticks([])
    axis.tick_params(length=0)
    return image


def _sector_by_ticker() -> dict[str, str]:
    request = urllib.request.Request(WIKIPEDIA, headers={"User-Agent": "market-analyzer/0.1 (research)"})
    with urllib.request.urlopen(request, timeout=30) as response:
        table = pd.read_html(io.BytesIO(response.read()))[0]
    columns = {str(name).strip().lower(): name for name in table.columns}
    symbol_column = columns["symbol"]
    sector_column = next(columns[name] for name in columns if "sector" in name)
    mapping = {}
    for symbol, sector in zip(table[symbol_column], table[sector_column], strict=True):
        mapping[wikipedia_yahoo_symbol(str(symbol))] = str(sector)
    return mapping


def _sector_order(tickers, sectors: dict[str, str]) -> list[str]:
    frame = pd.DataFrame({"ticker": [str(ticker) for ticker in tickers]})
    frame["sector"] = frame["ticker"].map(lambda ticker: sectors.get(ticker, "Unknown"))
    counts = frame["sector"].value_counts()
    frame["sector_size"] = frame["sector"].map(counts)
    frame = frame.sort_values(["sector_size", "sector", "ticker"], ascending=[False, True, True])
    return frame["ticker"].tolist()


def _sector_groups(order: list[str], sectors: dict[str, str]) -> list[tuple[str, int, int]]:
    groups: list[tuple[str, int, int]] = []
    start = 0
    current = sectors.get(order[0], "Unknown")
    for index, ticker in enumerate(order[1:], start=1):
        sector = sectors.get(ticker, "Unknown")
        if sector != current:
            groups.append((current, start, index))
            current = sector
            start = index
    groups.append((current, start, len(order)))
    return groups


def _short_sector(name: str) -> str:
    return {
        "Information Technology": "Technology",
        "Consumer Discretionary": "Discretionary",
        "Consumer Staples": "Staples",
        "Communication Services": "Communication",
        "Health Care": "Health care",
        "Real Estate": "Real estate",
    }.get(name, name)


def _summary_table(
    models: dict[int, CorrelationModel],
    windows: dict[int, pd.DataFrame],
    requested: int,
) -> pd.DataFrame:
    rows = []
    for length in WINDOWS:
        model = models[length]
        window = windows[length]
        naive_lower, naive_upper = marchenko_pastur_edges(model.q, 1.0)
        market = model.signal_loadings.iloc[:, 0].to_numpy()
        market = market / np.linalg.norm(market)
        rows.append(
            {
                "T": length,
                "N": model.assets,
                "wikipedia_names": requested,
                "q": model.q,
                "sigma2": model.sigma2,
                "lambda_minus": model.lambda_minus,
                "lambda_plus": model.lambda_plus,
                "lambda_minus_if_sigma2_is_1": naive_lower,
                "lambda_plus_if_sigma2_is_1": naive_upper,
                "lambda_max": float(model.eigenvalues["eigenvalue"].iloc[0]),
                "market_variance_share": float(model.eigenvalues["variance_share"].iloc[0]),
                "n_signal": model.n_signal,
                "n_above_sigma1": int(np.sum(model.eigenvalues["eigenvalue"].to_numpy() > naive_upper)),
                "market_participation": float(1.0 / np.sum(market**4)),
                "start": str(window.index[0].date()),
                "end": str(window.index[-1].date()),
            }
        )
    return pd.DataFrame(rows)


def _window_title(model: CorrelationModel, returns: pd.DataFrame) -> str:
    return (
        f"S&P 500 correlation spectrum, T = {model.observations} sessions "
        f"({returns.index[0].date()} to {returns.index[-1].date()})\n"
        f"N = {model.assets}, q = {model.q:.3f}, fitted σ² = {model.sigma2:.3f}, "
        f"{model.n_signal} eigenvalues above the edge"
    )


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "0.85",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 12,
        }
    )


if __name__ == "__main__":
    sys.exit(main())
