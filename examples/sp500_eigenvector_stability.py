"""Do the eigenvectors of the S&P 500 correlation matrix stay put as the window rolls?

The backtest re-estimates the covariance matrix every 21 sessions from the
previous 500. Every estimator except ``diagonal`` builds its matrix from the
eigenvectors of that window, so the question "are the eigenvectors fixed or
moving?" decides how much each model's weights can move.

For every backtest decision date this script eigendecomposes the correlation
matrix of the tickers with a complete history and measures three things:

* the overlap ``|v_k(t) · v_k(t')|`` of mode ``k`` with the same mode at a
  later decision date, for consecutive dates and as a function of the lag;
* the share of the signal subspace that survives from one decision date to
  the next, for the market mode alone, for the modes the unit edge keeps,
  and for the modes the fitted edge keeps;
* the loadings of the first two eigenvectors on a few familiar stocks, with
  signs chained so the vector can be followed through time.

The same measurements are repeated on fake Gaussian returns drawn from one
fixed covariance matrix, the cleaned covariance of the full panel. In that
world nothing about the market changes, so any eigenvector movement is
estimation noise. The gap between the real and the fake curves is the part
of the movement that is the market itself changing.

    .venv/bin/python examples/sp500_eigenvector_stability.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from market_analyzer.matrices import CorrelationModel, clip_eigenvalues  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "output" / "sp500_marchenko_pastur" / "prices.csv"
OUT = ROOT / "output" / "sp500_eigenvector_stability"

KEEP = 150
HEAT_MODES = 150
LAG_MODES = (1, 2, 3, 5, 10, 20, 50)
MAX_LAG = 40
TABLE_MODES = (1, 2, 3, 5, 10, 20, 50, 100)
LOADING_TICKERS = ("NVDA", "AAPL", "JPM", "XOM", "PG", "DUK")
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")
RULE_COLORS = {"market": "#444444", "unit": "#1baf7a", "fitted": "#eb6834"}
EPISODES = (
    ("Covid crash", "2020-02-19", "2020-03-23"),
    ("2022 bear market", "2022-01-03", "2022-10-12"),
)
REAL_LABEL = "S&P 500 returns"
SIM_LABEL = "Gaussian returns from one fixed covariance"


@dataclass
class RollingSpectra:
    """Eigen-decompositions of every rolling window, largest eigenvalue first."""

    dates: pd.DatetimeIndex
    tickers: list[str]
    eigenvalues: np.ndarray
    vectors: np.ndarray
    cov_vectors: np.ndarray
    n_signal: dict[str, np.ndarray]

    @property
    def n_assets(self) -> int:
        return len(self.tickers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", type=Path, default=PRICES)
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--step", type=int, default=21)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    _style()
    args.out.mkdir(parents=True, exist_ok=True)

    returns = load_returns(args.prices)
    fake, truth = simulate(returns, args.seed)
    print(
        f"Panel: {returns.shape[0]} sessions × {returns.shape[1]} tickers with a complete history, "
        f"{returns.index[0].date()} → {returns.index[-1].date()}"
    )
    print(
        f"Simulation truth: cleaned covariance of the full panel, edge rule {truth.edge_rule}, "
        f"{truth.n_signal} signal modes, q = {truth.q:.3f}"
    )

    real = roll(returns, args.window, args.step)
    sim = roll(fake, args.window, args.step)
    print(f"Decision dates: {len(real.dates)} from {real.dates[0].date()} to {real.dates[-1].date()}, window {args.window}, step {args.step}")
    print()

    summary = describe(real, sim, args.window, args.step)
    print(summary)
    (args.out / "summary.txt").write_text(summary + "\n")

    save_tables(real, sim, args.out, args.step)
    plot_heatmap(real, sim, args.out / "eigenvector_overlap_heatmap.png")
    plot_lag(real, sim, args.window, args.step, args.out / "eigenvector_overlap_by_lag.png")
    plot_subspace(real, sim, args.out / "signal_subspace_stability.png")
    plot_loadings(real, args.out / "eigenvector_loadings.png")
    print()
    print(f"Wrote {args.out}")


def load_returns(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    returns = prices.pct_change(fill_method=None).iloc[1:]
    complete = returns.dropna(axis=1)
    complete = complete.loc[:, complete.std(ddof=1) > 0]
    return complete


def simulate(returns: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, CorrelationModel]:
    """Zero-mean Gaussian returns with the cleaned covariance of the full panel as the truth."""
    truth = CorrelationModel.from_returns(returns)
    chol = np.linalg.cholesky(truth.cleaned_covariance.to_numpy(dtype=float))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=returns.shape)
    fake = pd.DataFrame(noise @ chol.T, index=returns.index, columns=returns.columns)
    return fake, truth


def roll(returns: pd.DataFrame, window: int, step: int) -> RollingSpectra:
    values = returns.to_numpy(dtype=float)
    n_dates, n_assets = values.shape
    q = n_assets / (window - 1)
    starts = list(range(window, n_dates - step + 1, step))
    keep = min(KEEP, n_assets)
    eigenvalues = np.empty((len(starts), n_assets))
    vectors = np.empty((len(starts), n_assets, keep))
    cov_vectors = np.empty((len(starts), n_assets, keep))
    n_signal = {"market": np.ones(len(starts), dtype=int), "unit": np.empty(len(starts), dtype=int), "fitted": np.empty(len(starts), dtype=int)}
    dates = []
    for i, start in enumerate(starts):
        x = values[start - window : start]
        x = x - x.mean(axis=0)
        cov = x.T @ x / (window - 1)
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        w, v = np.linalg.eigh(corr)
        order = np.argsort(w)[::-1]
        w, v = w[order], v[:, order]
        wc, vc = np.linalg.eigh(cov)
        vc = vc[:, np.argsort(wc)[::-1]]
        eigenvalues[i] = w
        vectors[i] = v[:, :keep]
        cov_vectors[i] = vc[:, :keep]
        n_signal["unit"][i] = clip_eigenvalues(w, q, 1.0)[4]
        n_signal["fitted"][i] = clip_eigenvalues(w, q, "fitted")[4]
        dates.append(returns.index[start - 1])
    return RollingSpectra(pd.DatetimeIndex(dates), [str(c) for c in returns.columns], eigenvalues, vectors, cov_vectors, n_signal)


def consecutive_overlap(vectors: np.ndarray, modes: int) -> np.ndarray:
    """``|v_k(t) · v_k(t+1)|`` for the first ``modes`` modes; shape (windows - 1) × modes."""
    a = vectors[:-1, :, :modes]
    b = vectors[1:, :, :modes]
    return np.abs(np.einsum("tik,tik->tk", a, b))


def lag_overlap(vectors: np.ndarray, modes: tuple[int, ...], max_lag: int) -> pd.DataFrame:
    """Mean ``|v_k(t) · v_k(t+lag)|`` over t; index is the lag in decision dates."""
    table = {}
    for k in modes:
        column = vectors[:, :, k - 1]
        table[k] = [float(np.abs(np.sum(column[:-lag] * column[lag:], axis=1)).mean()) for lag in range(1, max_lag + 1)]
    frame = pd.DataFrame(table, index=pd.Index(range(1, max_lag + 1), name="lag"))
    frame.columns.name = "mode"
    return frame


def subspace_preserved(vectors: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Share of the top-k subspace at t that lies inside the top-k subspace at t+1, k from ``counts``."""
    out = np.full(len(counts) - 1, np.nan)
    for t in range(len(counts) - 1):
        k = int(min(counts[t], vectors.shape[2]))
        if k <= 0:
            continue
        cross = vectors[t, :, :k].T @ vectors[t + 1, :, :k]
        out[t] = float((cross**2).sum() / k)
    return out


def aligned_loadings(vectors: np.ndarray, mode: int) -> np.ndarray:
    """Mode ``mode`` through time with signs chained so consecutive vectors agree."""
    v = vectors[:, :, mode - 1].copy()
    if v[0].sum() < 0:
        v[0] = -v[0]
    for t in range(1, len(v)):
        if v[t] @ v[t - 1] < 0:
            v[t] = -v[t]
    return v


def describe(real: RollingSpectra, sim: RollingSpectra, window: int, step: int) -> str:
    n = real.n_assets
    random_vector = np.sqrt(2.0 / (np.pi * n))
    disjoint = int(np.ceil(window / step))
    lines = []

    def block(title: str, lag: int) -> None:
        rows = []
        for k in TABLE_MODES:
            if k > real.vectors.shape[2]:
                continue
            rows.append(
                {
                    "mode": k,
                    "S&P corr": _median_lag(real.vectors, k, lag),
                    "S&P cov": _median_lag(real.cov_vectors, k, lag),
                    "Gaussian corr": _median_lag(sim.vectors, k, lag),
                    "Gaussian cov": _median_lag(sim.cov_vectors, k, lag),
                }
            )
        table = pd.DataFrame(rows).set_index("mode")
        lines.append(title)
        lines.append(table.to_string(float_format=lambda value: f"{value:.3f}"))
        lines.append("")

    block(f"Median |overlap| with the same mode one decision date later ({step} of {window} sessions replaced):", 1)
    block(f"Median |overlap| {disjoint} decision dates later (the two windows share no sessions):", disjoint)
    lines.append(f"Two random unit vectors in {n} dimensions overlap by about {random_vector:.3f}.")
    lines.append("")

    rows = []
    for rule, label in (("market", "market mode only"), ("unit", "σ² = 1 edge"), ("fitted", "fitted edge")):
        pres_real = subspace_preserved(real.vectors, real.n_signal[rule])
        pres_sim = subspace_preserved(sim.vectors, sim.n_signal[rule])
        k_real = int(np.median(real.n_signal[rule]))
        k_sim = int(np.median(sim.n_signal[rule]))
        rows.append(
            {
                "rule": label,
                "modes kept, S&P (median)": k_real,
                "modes kept, Gaussian (median)": k_sim,
                "subspace preserved, S&P": float(np.nanmedian(pres_real)),
                "subspace preserved, Gaussian": float(np.nanmedian(pres_sim)),
                "random subspace": k_real / n,
            }
        )
    lines.append("Share of the signal subspace preserved from one decision date to the next (median):")
    lines.append(pd.DataFrame(rows).set_index("rule").to_string(float_format=lambda value: f"{value:.3f}"))
    return "\n".join(lines)


def _median_lag(vectors: np.ndarray, mode: int, lag: int) -> float:
    column = vectors[:, :, mode - 1]
    return float(np.median(np.abs(np.sum(column[:-lag] * column[lag:], axis=1))))


def save_tables(real: RollingSpectra, sim: RollingSpectra, out: Path, step: int) -> None:
    for name, spec in (("real", real), ("gaussian", sim)):
        overlap = pd.DataFrame(
            consecutive_overlap(spec.vectors, min(HEAT_MODES, spec.vectors.shape[2])),
            index=spec.dates[1:],
            columns=pd.Index(range(1, min(HEAT_MODES, spec.vectors.shape[2]) + 1), name="mode"),
        )
        overlap.index.name = "decision_date"
        overlap.to_csv(out / f"consecutive_overlap_{name}.csv")
        lag = lag_overlap(spec.vectors, LAG_MODES, MAX_LAG)
        lag.index = pd.Index(lag.index * step, name="sessions_apart")
        lag.to_csv(out / f"lag_overlap_{name}.csv")
        counts = pd.DataFrame(spec.n_signal, index=spec.dates)
        counts.index.name = "decision_date"
        counts.to_csv(out / f"signal_modes_{name}.csv")
        preserved = pd.DataFrame(
            {rule: subspace_preserved(spec.vectors, spec.n_signal[rule]) for rule in ("market", "unit", "fitted")},
            index=spec.dates[1:],
        )
        preserved.index.name = "decision_date"
        preserved.to_csv(out / f"subspace_preserved_{name}.csv")
    for mode in (1, 2):
        frame = pd.DataFrame(aligned_loadings(real.vectors, mode), index=real.dates, columns=real.tickers)
        frame.index.name = "decision_date"
        frame.to_csv(out / f"loadings_mode{mode}_real.csv")


def plot_heatmap(real: RollingSpectra, sim: RollingSpectra, path: Path) -> None:
    modes = min(HEAT_MODES, real.vectors.shape[2])
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.5), sharex=True)
    mesh = None
    for ax, spec, title in zip(axes, (real, sim), (REAL_LABEL, SIM_LABEL)):
        overlap = consecutive_overlap(spec.vectors, modes)
        x = mdates.date2num(spec.dates[1:])
        edges = np.concatenate([[x[0] - (x[1] - x[0]) / 2], (x[:-1] + x[1:]) / 2, [x[-1] + (x[-1] - x[-2]) / 2]])
        y_edges = np.arange(0.5, modes + 1.5)
        mesh = ax.pcolormesh(edges, y_edges, overlap.T, cmap="Blues", vmin=0, vmax=1, shading="flat")
        ax.step(spec.dates[1:], spec.n_signal["unit"][1:], where="mid", color=RULE_COLORS["unit"], lw=2)
        ax.step(spec.dates[1:], spec.n_signal["fitted"][1:], where="mid", color=RULE_COLORS["fitted"], lw=2)
        ax.set_ylim(modes + 0.5, 0.5)
        ax.set_ylabel("mode (1 = market mode)")
        ax.set_title(title)
        ax.grid(False)
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(
        handles=[
            Line2D([], [], color=RULE_COLORS["unit"], lw=2, label="modes kept by the σ² = 1 edge"),
            Line2D([], [], color=RULE_COLORS["fitted"], lw=2, label="modes kept by the fitted edge"),
        ],
        loc="lower left",
        frameon=True,
        framealpha=0.92,
    )
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("|overlap| with the same mode at the previous decision date")
    fig.suptitle("How much does each eigenvector move between consecutive decision dates?", x=0.01, ha="left", fontsize=13, fontweight="medium")
    fig.text(
        0.01,
        0.935,
        "Dark = the mode points the same way 21 sessions later. Lines show how many modes each clipping rule keeps.",
        fontsize=10,
        color="#555555",
    )
    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.07, right=0.90, hspace=0.18)
    fig.savefig(path)
    plt.close(fig)


def plot_lag(real: RollingSpectra, sim: RollingSpectra, window: int, step: int, path: Path) -> None:
    real_lag = lag_overlap(real.vectors, LAG_MODES, MAX_LAG)
    sim_lag = lag_overlap(sim.vectors, LAG_MODES, MAX_LAG)
    sessions = real_lag.index.to_numpy() * step
    shades = plt.get_cmap("Blues")(np.linspace(1.0, 0.45, len(LAG_MODES)))
    random_vector = np.sqrt(2.0 / (np.pi * real.n_assets))

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    items = []
    for k, color in zip(LAG_MODES, shades):
        ax.plot(sessions, real_lag[k], color=color, lw=2.2)
        ax.plot(sessions, sim_lag[k], color=color, lw=1.4, ls="--")
        items.append((sessions[-1], float(real_lag[k].iloc[-1]), f"mode {k}", color))
    ax.axhline(random_vector, color="#888888", ls=":", lw=1.3)
    ax.text(sessions[0], random_vector + 0.012, f"two random directions in {real.n_assets} dimensions", color="#666666", fontsize=9.5)
    ax.axvline(window, color="#888888", lw=1)
    ax.text(window - 10, 0.83, "windows no longer share any session", color="#666666", fontsize=9.5, ha="right", va="center")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, sessions[-1] * 1.12)
    ax.set_xlabel("sessions between the two decision dates")
    ax.set_ylabel("mean |overlap| of the same mode")
    ax.set_title("Eigenvector memory: how long does mode k keep pointing the same way?")
    _label_ends(ax, items)
    handles = [
        Line2D([], [], color="#333333", lw=2.2, label=REAL_LABEL),
        Line2D([], [], color="#333333", lw=1.4, ls="--", label=SIM_LABEL),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.93))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_subspace(real: RollingSpectra, sim: RollingSpectra, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    items = []
    for rule, text in (("market", "market mode only"), ("unit", "σ² = 1 edge"), ("fitted", "fitted edge")):
        color = RULE_COLORS[rule]
        y_real = subspace_preserved(real.vectors, real.n_signal[rule])
        y_sim = subspace_preserved(sim.vectors, sim.n_signal[rule])
        ax.plot(real.dates[1:], y_real, color=color, lw=2)
        ax.plot(sim.dates[1:], y_sim, color=color, lw=1.2, ls="--")
        k = int(np.median(real.n_signal[rule]))
        items.append((real.dates[-1], float(y_real[-1]), f"{text} (k ≈ {k})", color))
    ax.set_ylim(0.5, 1.01)
    _shade_episodes(ax)
    _extend_right(ax, 0.16)
    ax.set_ylabel("share of the signal subspace that survives\nto the next decision date")
    ax.set_title("Is the space the clipping rules keep stable, even when individual modes rotate inside it?")
    _label_ends(ax, items)
    handles = [
        Line2D([], [], color="#333333", lw=2, label=REAL_LABEL),
        Line2D([], [], color="#333333", lw=1.2, ls="--", label=SIM_LABEL),
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_loadings(real: RollingSpectra, path: Path) -> None:
    present = [ticker for ticker in LOADING_TICKERS if ticker in real.tickers]
    index = [real.tickers.index(ticker) for ticker in present]
    equal = 1.0 / np.sqrt(real.n_assets)
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.5), sharex=True)
    for ax, mode in zip(axes, (1, 2)):
        loadings = aligned_loadings(real.vectors, mode)[:, index]
        items = []
        for j, (ticker, color) in enumerate(zip(present, PALETTE)):
            ax.plot(real.dates, loadings[:, j], color=color, lw=1.8)
            items.append((real.dates[-1], float(loadings[-1, j]), ticker, color))
        ax.axhline(0, color="#888888", lw=0.8)
        if mode == 1:
            ax.axhline(equal, color="#888888", ls=":", lw=1.2)
            ax.text(real.dates[0], equal + 0.004, f"equal loading 1/√{real.n_assets}", color="#666666", fontsize=9.5)
            ax.set_title("Mode 1, the market mode: every loading positive, reshuffled when Covid enters the window")
        else:
            ax.set_title("Mode 2: a long-short axis across sectors, with sign chained between dates")
        ax.set_ylabel(f"loading of mode {mode}")
        _shade_episodes(ax)
        _extend_right(ax, 0.07)
        _label_ends(ax, items)
    fig.suptitle("Where the first two eigenvectors point, decision date by decision date", x=0.01, ha="left", fontsize=13, fontweight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def _shade_episodes(ax) -> None:
    for label, start, end in EPISODES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#000000", alpha=0.06, lw=0)
        lo, hi = ax.get_ylim()
        ax.text(pd.Timestamp(start), lo + (hi - lo) * 0.015, f" {label}", fontsize=9, color="#555555", va="bottom")


def _extend_right(ax, frac: float) -> None:
    """Make room for end labels without adding tick marks beyond the data."""
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + (hi - lo) * frac)
    ax.set_xticks([tick for tick in ax.get_xticks() if lo <= tick <= hi])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _label_ends(ax, items: list[tuple], min_gap_frac: float = 0.04) -> None:
    """Direct labels at the right end of each line, nudged apart so they do not collide."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    ordered = sorted(items, key=lambda item: item[1])
    ys = [item[1] for item in ordered]
    for j in range(1, len(ys)):
        if ys[j] - ys[j - 1] < gap:
            ys[j] = ys[j - 1] + gap
    overflow = ys[-1] - (hi - gap / 2)
    if overflow > 0:
        ys = [y - overflow for y in ys]
    for (x, _, text, color), y in zip(ordered, ys):
        ax.annotate(text, xy=(x, y), xytext=(6, 0), textcoords="offset points", va="center", ha="left", fontsize=9.5, color=color, annotation_clip=False)


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
