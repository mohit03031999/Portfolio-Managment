"""How much noise the RIE removes along each sample direction.

For one 1000-day window the sample correlation has eigenvalues λ_i along
directions u_i. The RIE keeps every u_i and replaces λ_i by its estimate ξ_i
of the true variance along u_i. "Noise removed" along direction i is λ_i − ξ_i
(positive: noise had inflated it; negative: noise had shrunk it).

Two windows are shown:

* real: the last 1000 days of the S&P 500 panel (truth unknown);
* lab: 1000 Gaussian days whose truth is the full-sample covariance, so the
  true variance along each sample direction, u_iᵀ C u_i (the "oracle"), is
  known and the RIE can be graded.

Outputs in output/sp500_rie_noise_by_direction/: noise_by_direction.csv,
summary.txt, noise_by_direction.png.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from market_analyzer.rie import rie_eigenvalues  # noqa: E402
from sp500_simulated_backtest import PRICES, load_returns  # noqa: E402

WINDOW = 1000
OUT = ROOT / "output" / "sp500_rie_noise_by_direction"
REAL, LAB, ORACLE, INK, MUTED = "#2a6fdb", "#e07b00", "#1a9e5c", "#222222", "#666666"


def correlation(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0)
    x = x / np.sqrt((x**2).sum(axis=0) / (len(x) - 1))
    c = x.T @ x / (len(x) - 1)
    return (c + c.T) / 2


def decompose(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(c)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def clean(x: np.ndarray) -> dict:
    c = correlation(x)
    values, vectors = decompose(c)
    spectrum = rie_eigenvalues(values, x.shape[1] / (len(x) - 1))
    return {"sample": values, "rie": spectrum.cleaned, "vectors": vectors}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    returns = load_returns(PRICES)
    n = returns.shape[1]

    real = clean(returns.iloc[-WINDOW:].to_numpy(dtype=float))

    cov = returns.cov(ddof=1).to_numpy(dtype=float)
    sd = np.sqrt(np.diag(cov))
    truth = cov / np.outer(sd, sd)
    true_values = decompose(truth)[0]
    rng = np.random.default_rng(7)
    lab_x = rng.standard_normal((WINDOW, n)) @ np.linalg.cholesky(cov).T
    lab = clean(lab_x)
    oracle = np.einsum("ij,ik,kj->j", lab["vectors"], truth, lab["vectors"])

    rank = np.arange(1, n + 1)
    table = pd.DataFrame(
        {
            "rank": rank,
            "real_sample": real["sample"],
            "real_rie": real["rie"],
            "real_noise_removed": real["sample"] - real["rie"],
            "lab_sample": lab["sample"],
            "lab_rie": lab["rie"],
            "lab_oracle": oracle,
            "lab_true_eigenvalue": true_values,
        }
    )
    table.to_csv(OUT / "noise_by_direction.csv", index=False)

    lines = [
        f"Window {WINDOW} days, {n} stocks, q = N/T = {n / (WINDOW - 1):.3f}. Correlation units: average direction = 1.",
        "noise removed = sample eigenvalue − RIE eigenvalue (positive: noise had inflated it).",
        "",
        "rank   | real: sample   RIE   removed  kept% | lab: sample   RIE   oracle  removed  RIE error",
    ]
    for r in (1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400, n):
        i = r - 1
        lines.append(
            f"{r:6d} | {real['sample'][i]:12.3f} {real['rie'][i]:6.3f} {real['sample'][i] - real['rie'][i]:8.3f}"
            f" {100 * real['rie'][i] / real['sample'][i]:5.0f}% |"
            f" {lab['sample'][i]:10.3f} {lab['rie'][i]:6.3f} {oracle[i]:7.3f} {lab['sample'][i] - lab['rie'][i]:8.3f}"
            f" {100 * (lab['rie'][i] / oracle[i] - 1):+8.0f}%"
        )
    lines.append("")
    for name, d in (("real", real), ("lab", lab)):
        s, x = d["sample"], d["rie"]
        cross = int(np.argmax(x > s)) + 1
        lines.append(
            f"{name}: RIE lowers ranks 1–{cross - 1} and raises ranks {cross}–{n}; "
            f"variance moved from top to bottom = {np.sum(np.clip(s - x, 0, None)):.1f} of total {s.sum():.0f}; "
            f"quietest 50 directions: sample {s[-50:].mean():.3f} → RIE {x[-50:].mean():.3f}"
        )
    lines.append(f"lab oracle, quietest 50 sample directions: {oracle[-50:].mean():.3f}; top direction oracle {oracle[0]:.2f}")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    a = ax[0]
    a.plot(rank, lab["sample"], color=LAB, lw=1.5, ls="--", label="sample eigenvalue (what the window saw)")
    a.plot(rank, lab["rie"], color=LAB, lw=2.2, label="RIE (its guess of the truth)")
    a.plot(rank, oracle, color=ORACLE, lw=1.5, ls=":", label="oracle: true variance along that direction")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("direction rank (1 = market, 456 = quietest)"); a.set_ylabel("variance along the direction (log)")
    a.set_title("1. Lab (truth known): sample vs RIE vs oracle", loc="left", fontsize=11)
    a.legend(fontsize=9, frameon=False)

    a = ax[1]
    a.plot(rank, real["rie"] / real["sample"], color=REAL, lw=2, label="real market, last 1000 days")
    a.plot(rank, lab["rie"] / lab["sample"], color=LAB, lw=2, label="lab: RIE")
    a.plot(rank, oracle / lab["sample"], color=ORACLE, lw=1.2, ls=":", label="lab: oracle (the right answer)")
    a.axhline(1, color=MUTED, lw=1)
    a.text(1.1, 1.05, "above 1: noise had shrunk it, RIE raises it", fontsize=8, color=MUTED)
    a.text(1.1, 0.9, "below 1: noise had inflated it, RIE lowers it", fontsize=8, color=MUTED, va="top")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("direction rank (1 = market, 456 = quietest)"); a.set_ylabel("RIE ÷ sample  (share kept, log)")
    a.set_title("2. What the RIE did to each direction", loc="left", fontsize=11)
    a.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.88))
    for a in ax:
        a.grid(alpha=0.25, which="both"); [a.spines[s].set_visible(False) for s in ("top", "right")]
    fig.tight_layout()
    fig.savefig(OUT / "noise_by_direction.png", dpi=130)


if __name__ == "__main__":
    main()
