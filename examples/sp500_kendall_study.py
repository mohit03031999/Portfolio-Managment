"""Kendall's tau as the source of directions: does it aim better than Pearson, and does it cut risk?

Recipe from Espana, Le Coz and Smerlak (2024, arXiv 2410.17366): remove each
stock's mean, divide each day by its cross-sectional size sqrt(sum_j x_jt^2),
then take Kendall's tau between every pair of stocks (for every pair of days,
each stock votes "up or down", so a huge day counts like any other). No sine
transform. The paper's plain "Kendall" estimator uses that matrix as the
correlation; its best variant keeps Kendall's eigenvectors and sets the
eigenvalues by cross-validation.

What we test, per decision date (window T = 1000, holds of 21 days):

* ``rie``                the Pearson RIE we use everywhere (baseline);
* ``kendall``            the paper's plain Kendall matrix, rescaled by sample vols;
* ``kendall_raw_days``   the same without the per-day standardisation;
* ``kendall_rank_rie``   Kendall's eigenvectors with the RIE's eigenvalues by rank;
* ``kendall_top_k``      the RIE, with the top-k room re-positioned where Kendall
                         puts it (the RIE's own structure inside that room), for
                         k = 1, 3, 10, 20, 50, 100 — the Kendall twin of the
                         "true top subspace" hybrid in sp500_eigenvector_error.py.

Worlds: ``gaussian_raw`` (truth = raw full-sample covariance), ``shuffled``
(real fat-tailed days in random order; truth = covariance of the days outside
the window) and ``real`` (the market itself; realised risk from the next 21
days). In the two exact-truth worlds we also measure how well the Pearson and
Kendall directions line up with the truth, mode by mode and room by room.

Outputs in output/sp500_kendall/: {panel}_s{seed}/risk.csv, overlap.csv,
(real: returns.csv); summary.txt; kendall_study.png.
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
sys.path.insert(0, str(ROOT / "examples"))

import sp500_shuffle_test as shuffle  # noqa: E402
import sp500_simulated_backtest as simulated  # noqa: E402
from market_analyzer.optimizer import rie_covariance  # noqa: E402

OUT = ROOT / "output" / "sp500_kendall"
ERROR_STUDY = ROOT / "output" / "sp500_eigenvector_error"
WINDOW, HOLDING, ANNUAL = 1000, 21, 252
COUNTS = (1, 3, 10, 20, 50, 100)
ROOMS = (1, 3, 10, 20)
N_OVERLAP = 20
PANEL_LABELS = {"gaussian_raw": "Gaussian days", "shuffled": "real days, shuffled", "real": "real market"}
PEARSON, KENDALL, TRUTH, INK, MUTED = "#2a6fdb", "#e07b00", "#1a9e5c", "#222222", "#777777"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", choices=tuple(PANEL_LABELS), default="shuffled")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--every", type=int, default=1, help="use every k-th decision date (quick tests)")
    parser.add_argument("--report", action="store_true", help="aggregate saved runs and draw the figure")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report:
        report()
        return
    returns = simulated.load_returns(simulated.PRICES)
    run(returns, args.panel, args.seed, args.every)


# ----------------------------------------------------------------------------- estimators


def kendall_matrix(x: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Kendall's tau between every pair of columns of ``x`` (days × stocks).

    Paper eq. (2) with the sign kernel: sum over day pairs t < s of
    sign(x_t − x_s)·sign(y_t − y_s), divided by sqrt(sum sign² · sum sign²), so
    tied days (zero returns) drop out and the diagonal is exactly 1.
    """
    t, n = x.shape
    ii, jj = np.triu_indices(t, 1)
    acc = np.zeros((n, n), dtype=np.float64)
    for s in range(0, len(ii), chunk):
        d = np.sign(x[ii[s : s + chunk]] - x[jj[s : s + chunk]]).astype(np.float32)
        acc += d.T @ d
    norm = np.sqrt(np.diag(acc))
    norm = np.where(norm > 0, norm, 1.0)
    tau = acc / np.outer(norm, norm)
    np.fill_diagonal(tau, 1.0)
    return tau


def day_standardised(centred: np.ndarray) -> np.ndarray:
    """Divide each day by its cross-sectional size sqrt(sum_j x_jt^2) (the paper's step ii)."""
    scale = np.sqrt((centred**2).sum(axis=1, keepdims=True))
    return centred / np.where(scale > 0, scale, 1.0)


def _descending(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(matrix)
    return values[::-1], vectors[:, ::-1]


def _minvar(matrix: np.ndarray) -> np.ndarray:
    raw = np.linalg.solve(matrix, np.ones(matrix.shape[0]))
    return raw / raw.sum()


def _swap_block(estimate: np.ndarray, vectors: np.ndarray, block: np.ndarray) -> np.ndarray:
    """``vectors · block · vectorsᵀ + P · estimate · P`` with ``P = I − vectors vectorsᵀ``."""
    a = vectors.T @ estimate
    inner = a @ vectors
    rest = estimate - vectors @ a - a.T @ vectors.T + vectors @ inner @ vectors.T
    return vectors @ block @ vectors.T + rest


def estimates(window: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """Covariance estimates for one window, the direction sets used, and timings."""
    centred = window - window.mean(axis=0)
    vol = centred.std(axis=0, ddof=1)
    scale = np.outer(vol, vol)
    sample_corr = (centred / vol).T @ (centred / vol) / (len(window) - 1)
    _, pearson_vectors = _descending(sample_corr)

    tic = time.perf_counter()
    rie = rie_covariance(pd.DataFrame(window)).covariance.to_numpy(dtype=float)
    rie_corr = rie / scale
    rie_values, _ = _descending(rie_corr)
    t_rie = time.perf_counter() - tic

    tic = time.perf_counter()
    tau = kendall_matrix(day_standardised(centred))
    t_kendall = time.perf_counter() - tic
    tau_raw = kendall_matrix(centred)
    kendall_values, kendall_vectors = _descending(tau)

    out = {
        "rie": rie,
        "kendall": scale * tau,
        "kendall_raw_days": scale * tau_raw,
        "kendall_rank_rie": scale * ((kendall_vectors * rie_values) @ kendall_vectors.T),
    }
    for k in COUNTS:
        top = kendall_vectors[:, :k]
        out[f"kendall_top_{k}"] = scale * _swap_block(rie_corr, top, top.T @ rie_corr @ top)
    directions = {"pearson": pearson_vectors, "kendall": kendall_vectors, "kendall_values": kendall_values}
    return out, directions, {"t_rie": t_rie, "t_kendall": t_kendall}


# ----------------------------------------------------------------------------- the run


def run(returns: pd.DataFrame, panel: str, seed: int, every: int) -> None:
    if panel == "gaussian_raw":
        data = shuffle.gaussian_raw(returns, seed).to_numpy(dtype=float)
        fixed_truth = returns.cov(ddof=1).to_numpy(dtype=float)
    elif panel == "shuffled":
        data = shuffle.shuffled(returns, seed).to_numpy(dtype=float)
        fixed_truth = None
    else:
        data = returns.to_numpy(dtype=float)
        fixed_truth = None
    n_days, n = data.shape
    starts = list(range(WINDOW, n_days - HOLDING + 1, HOLDING))[::every]
    truth_vectors = _truth_directions(fixed_truth) if fixed_truth is not None else None

    risk_rows, overlap_rows, return_rows = [], [], []
    clock = time.perf_counter()
    for i, start in enumerate(starts):
        window = data[start - WINDOW : start]
        matrices, directions, timing = estimates(window)
        if panel == "real":
            forward = data[start : start + HOLDING]
            for name, matrix in matrices.items():
                w = _minvar(matrix)
                daily = forward @ w
                risk_rows.append({"estimator": name, "date": i, "realised": float(np.mean(daily**2)), "predicted": float(w @ matrix @ w)})
                return_rows += [{"estimator": name, "date": i, "day": d, "ret": float(r)} for d, r in enumerate(daily)]
            overlap_rows.append({"date": i, "kendall_top": float(directions["kendall_values"][0]), **timing})
        else:
            if fixed_truth is None:
                outside = np.concatenate([data[: start - WINDOW], data[start:]])
                sigma = np.cov(outside, rowvar=False)
                truth_vectors = _truth_directions(sigma)
            else:
                sigma = fixed_truth
            w_star = _minvar(sigma)
            risk_rows.append({"estimator": "oracle", "date": i, "realised": float(w_star @ sigma @ w_star), "predicted": float(w_star @ sigma @ w_star)})
            for name, matrix in matrices.items():
                w = _minvar(matrix)
                risk_rows.append({"estimator": name, "date": i, "realised": float(w @ sigma @ w), "predicted": float(w @ matrix @ w)})
            overlap_rows.append({"date": i, **_overlaps(directions, truth_vectors), **timing})
        if i % 10 == 0:
            print(f"{panel} seed {seed}: date {i + 1} of {len(starts)}, {time.perf_counter() - clock:.0f}s elapsed, kendall {timing['t_kendall']:.1f}s", flush=True)

    folder = OUT / (f"{panel}_s{seed}" if panel != "real" else "real")
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(risk_rows).to_csv(folder / "risk.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(folder / "overlap.csv", index=False)
    if return_rows:
        pd.DataFrame(return_rows).to_csv(folder / "returns.csv", index=False)
    print(f"saved {folder}")


def _truth_directions(sigma: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.diag(sigma))
    _, vectors = _descending(sigma / np.outer(vol, vol))
    return vectors


def _overlaps(directions: dict[str, np.ndarray], truth: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("pearson", "kendall"):
        vectors = directions[name]
        for k in range(N_OVERLAP):
            out[f"{name}_mode_{k + 1}"] = float((vectors[:, k] @ truth[:, k]) ** 2)
        for k in ROOMS:
            out[f"{name}_room_{k}"] = float(np.sum((vectors[:, :k].T @ truth[:, :k]) ** 2) / k)
    return out


# ----------------------------------------------------------------------------- the report


def _runs() -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for panel in PANEL_LABELS:
        pattern = f"{panel}_s*" if panel != "real" else "real"
        found[panel] = sorted(path for path in OUT.glob(pattern) if (path / "risk.csv").exists())
    return found


def _table(risk: pd.DataFrame) -> pd.DataFrame:
    risk = risk.copy()
    risk["ratio"] = np.sqrt(risk["realised"] / risk["predicted"])
    table = risk.groupby("estimator").agg(realised=("realised", "mean"), honesty=("ratio", "median"))
    table["volatility"] = np.sqrt(ANNUAL * table["realised"])
    return table


def report() -> None:
    runs = _runs()
    lines = ["Kendall's tau for the directions. Window T = 1000, holds of 21 days.", ""]
    labelled = {"rie": "RIE (Pearson)", "kendall": "Kendall, plain (paper)", "kendall_raw_days": "Kendall, no day standardisation", "kendall_rank_rie": "Kendall directions + RIE sizes by rank"}
    curves: dict[str, dict] = {}
    overlaps: dict[str, pd.Series] = {}
    real_table = None
    for panel, folders in runs.items():
        if not folders:
            continue
        risk = pd.concat([pd.read_csv(f / "risk.csv").assign(run=f.name) for f in folders])
        overlap = pd.concat([pd.read_csv(f / "overlap.csv").assign(run=f.name) for f in folders])
        table = _table(risk)
        lines.append(f"=== {PANEL_LABELS[panel]} ({', '.join(f.name for f in folders)}; {risk['date'].nunique()} dates) ===")
        base = float(table.loc["rie", "volatility"])
        if panel == "real":
            daily = pd.concat([pd.read_csv(f / "returns.csv") for f in folders])
            overall = daily.groupby("estimator")["ret"].std() * np.sqrt(ANNUAL)
            lines.append("Realised volatility over all holds (median honesty):")
            for name in list(labelled) + [f"kendall_top_{k}" for k in COUNTS]:
                lines.append(f"  {labelled.get(name, name):<40} {overall[name]:7.2%}  ({overall[name] / overall['rie'] - 1:+.1%} vs RIE)   honesty {table.loc[name, 'honesty']:.2f}")
            real_table = overall, table
            lines.append(f"Kendall top eigenvalue (correlation units): mean {overlap['kendall_top'].mean():.1f}; Kendall matrix took {overlap['t_kendall'].mean():.1f}s per window, RIE {overlap['t_rie'].mean():.1f}s.")
        else:
            oracle = float(table.loc["oracle", "volatility"])
            lines.append(f"Volatility: RIE {base:.2%}, truth {oracle:.2%}; the RIE is {base / oracle - 1:+.1%} above the truth.")
            lines.append("Estimator                                  volatility   gap closed   honesty")
            for name in labelled:
                vol = float(table.loc[name, "volatility"])
                lines.append(f"  {labelled[name]:<40} {vol:7.2%}   {(base - vol) / (base - oracle):+7.0%}     {table.loc[name, 'honesty']:.3f}")
            lines.append("The RIE with its top-k room moved to where Kendall puts it (gap closed; in brackets the perfect room from sp500_eigenvector_error):")
            reference = _reference_curve(panel)
            points = []
            for k in COUNTS:
                vol = float(table.loc[f"kendall_top_{k}", "volatility"])
                closed = (base - vol) / (base - oracle)
                ref = reference.get(k)
                extra = f" (perfect room {ref:+.0%})" if ref is not None else ""
                lines.append(f"  k = {k:<4} {vol:7.2%}   gap closed {closed:+6.0%}{extra}   honesty {table.loc[f'kendall_top_{k}', 'honesty']:.3f}")
                points.append((k, vol, closed))
            curves[panel] = {"points": points, "base": base, "oracle": oracle, "kendall": float(table.loc["kendall", "volatility"]), "reference": reference}
            mean = overlap.mean(numeric_only=True)
            overlaps[panel] = mean
            lines.append("Squared overlap with the true direction, mode by mode (1 = same, ~0.002 = random):")
            lines.append("  mode     " + " ".join(f"{k:>5}" for k in range(1, N_OVERLAP + 1)))
            for name in ("pearson", "kendall"):
                lines.append(f"  {name:<8} " + " ".join(f"{mean[f'{name}_mode_{k}']:5.2f}" for k in range(1, N_OVERLAP + 1)))
            lines.append("Room overlap (share of the true top-k room captured; 1 = same room, k/456 = random):")
            for name in ("pearson", "kendall"):
                lines.append(f"  {name:<8} " + "  ".join(f"top {k}: {mean[f'{name}_room_{k}']:.3f}" for k in ROOMS))
        lines.append("")
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)
    if curves or real_table is not None:
        plot(curves, overlaps, real_table, OUT / "kendall_study.png")


def _reference_curve(panel: str) -> dict[int, float]:
    """Gap closed by the perfect top-k room, from the earlier hand-over study (same panel, same dates)."""
    folders = sorted(path for path in ERROR_STUDY.glob(f"{panel}_s*") if (path / "risk.csv").exists())
    if not folders:
        return {}
    risk = pd.concat(pd.read_csv(f / "risk.csv") for f in folders)
    table = risk.groupby(["estimator", "version", "count"])["realised"].mean()
    oracle, base = table[("oracle", "truth", 456)], table[("rie", "as is", 0)]
    out = {}
    for k in COUNTS:
        key = ("rie", "true top subspace", k)
        if key in table.index:
            out[k] = float((np.sqrt(base) - np.sqrt(table[key])) / (np.sqrt(base) - np.sqrt(oracle)))
    return out


def plot(curves: dict, overlaps: dict, real_table, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    modes = np.arange(1, N_OVERLAP + 1)
    ax = axes[0]
    for panel, style in (("shuffled", "-"), ("gaussian_raw", "--")):
        if panel not in overlaps:
            continue
        mean = overlaps[panel]
        ax.plot(modes, [mean[f"pearson_mode_{k}"] for k in modes], style, color=PEARSON, lw=2, label=f"Pearson, {PANEL_LABELS[panel]}")
        ax.plot(modes, [mean[f"kendall_mode_{k}"] for k in modes], style, color=KENDALL, lw=2, label=f"Kendall, {PANEL_LABELS[panel]}")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("direction (1 = market)")
    ax.set_ylabel("squared overlap with the true direction")
    ax.set_title("1. How well each direction is aimed", loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    for panel, style in (("shuffled", "-"), ("gaussian_raw", "--")):
        if panel not in curves:
            continue
        c = curves[panel]
        ks = [k for k, _, _ in c["points"]]
        ax.plot(ks, [closed for _, _, closed in c["points"]], style, marker="o", color=KENDALL, lw=2, label=f"Kendall room, {PANEL_LABELS[panel]}")
        if c["reference"]:
            ax.plot(list(c["reference"]), list(c["reference"].values()), style, marker="s", color=TRUTH, lw=1.5, label=f"perfect room, {PANEL_LABELS[panel]}")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("size of the room moved (top k directions)")
    ax.set_ylabel("share of the RIE-to-truth gap closed")
    ax.set_title("2. Moving the RIE's top room to Kendall's", loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    if real_table is not None:
        overall, table = real_table
        names = ["rie", "kendall", "kendall_raw_days", "kendall_rank_rie", "kendall_top_20"]
        labels = ["RIE\n(Pearson)", "Kendall\nplain", "Kendall\nraw days", "Kendall dirs\n+ RIE sizes", "RIE, top-20\nroom → Kendall"]
        vols = [overall[n] * 100 for n in names]
        colors = [PEARSON] + [KENDALL] * 4
        bars = ax.bar(range(len(names)), vols, color=colors, width=0.62)
        for b, n in zip(bars, names):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, f"{overall[n]:.1%}\nh {table.loc[n, 'honesty']:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(min(vols) - 1.5, max(vols) + 1.2)
        ax.set_ylabel("realised volatility, % per year")
        ax.set_title("3. Real market, same holds (h = honesty)", loc="left", fontsize=11)
    for ax in axes:
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
