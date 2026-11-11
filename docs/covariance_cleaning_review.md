# Cleaning a large covariance matrix: what exists, what we have, what to try next

*Survey written 2026-10-30. Nothing here has been implemented or run beyond what the repo already contains. Sources marked **[full text]** were read in full; the rest come from abstracts, published tables of contents and package documentation, and should be checked against the paper before anything is built on a detail.*

## 0. Where we stand

The repo estimates a covariance matrix from a rolling window of daily returns and feeds it to an unconstrained minimum-variance optimiser. It has five estimators: the plain sample covariance, Marchenko-Pastur eigenvalue clipping with three edge rules (fitted, σ² = 1, market), the diagonal matrix, and Ledoit-Wolf linear shrinkage toward a scaled identity, plus equal weight as the no-model control.

Four studies on the S&P 500 panel (456 complete-history tickers, 2015 to 2026) established what a new method has to fix:

1. **The honesty gap is estimation noise, not market change.** On Gaussian returns drawn from one fixed covariance the models under-promise risk almost exactly as much as on the real market (clipped fitted 1.71 vs 1.83 at T = 500). The optimiser bets on directions that merely looked quiet in the sample.
2. **Eigenvectors beyond roughly mode 20 are redrawn every month at T = 500**, even in a stationary world. Every rotationally invariant method keeps those eigenvectors, so eigenvalue cleaning alone cannot remove that part of the noise.
3. **The fitted edge has zero bias, the σ² = 1 edge has a permanent 13.5 % bias**, because it files 136 of the truth's 191 modes under noise. The sample covariance follows the 1/(1−q) law exactly.
4. **Stale data costs more than noise saves beyond about two years.** On the real market the clipped models' realised risk is lowest at T ≈ 500 and rises for longer windows, while in the Gaussian world it keeps falling.
5. **Fat tails matter to self-tuning methods.** Real excess kurtosis is about 11 against 0 for Gaussian data; Ledoit-Wolf shrinks twice as hard on real data for that reason.

So there are three distinct enemies: **noise** (eigenvalues and eigenvectors), **staleness** (non-stationarity), and **fat tails**. Each method below is judged by which of these it attacks.

## 1. The families

| Family | Representative methods | What it changes | Attacks | Keeps sample eigenvectors | In repo |
|---|---|---|---|---|---|
| Eigenvalue clipping | Laloux, Cizeau, Bouchaud, Potters 1999 | Bulk eigenvalues → one constant | noise (eigenvalues) | yes | yes |
| Linear shrinkage | Ledoit-Wolf 2003, 2004; OAS 2010 | Convex blend with a target | noise | yes (identity target) | identity target only |
| Nonlinear shrinkage / optimal RIE | Ledoit-Péché 2011; Bun-Bouchaud-Potters 2017; Ledoit-Wolf 2017, 2020, 2022 | Each eigenvalue individually, by a formula | noise (eigenvalues) | yes | **no** |
| Factor models | PCA factors; POET (Fan-Liao-Mincheva 2013); denoise + detone (López de Prado 2020) | K factors plus a sparse or diagonal residual | noise | leading ones only | no |
| Hierarchical clustering filters | Tumminello-Lillo-Mantegna 2007; BAHC (Bongiorno-Challet 2021), k-BAHC 2022 | Replaces correlations by cluster averages, bootstrapped | noise (eigenvectors too) | **no** | no |
| Sparse precision | Graphical lasso (Friedman-Hastie-Tibshirani 2008); glasso/tlasso for min-var (Torri et al. 2019) | Estimates Σ⁻¹ directly with an L1 penalty | noise | no | no |
| Time-varying | EWMA + RMT (Pafka-Potters-Kondor 2004); DCC-NL (Engle-Ledoit-Wolf 2019) | Weights recent days more; dynamic correlations | **staleness** | n/a | no |
| Robust / heavy tails | Student ensemble (Biroli-Bouchaud-Potters 2007); Tyler + shrinkage (Yang-Couillet-McKay 2015); Gerber statistic 2022; volatility standardisation | Changes what is averaged | **fat tails** | varies | no |
| Regularise the portfolio | Long-only (Jagannathan-Ma 2003); norm constraints (DeMiguel et al. 2009) | Constrains w instead of Σ | noise, via the optimiser | n/a | planned (phase 2) |
| Entrywise | Power mapping (Guhr-Kälber 2003) | Raises each correlation to a power ≈ 1.5 | noise | no | no |

### 1.1 Rotationally invariant estimators: clean the eigenvalues, keep the eigenvectors

**Clipping** is what we have. Bun, Bouchaud and Potters **[full text]** describe it as the first RMT estimator, note that the effective q that fits real spectra differs from N/T because of autocorrelation and an imperfect null, and list two fixes: a corrected edge from a fitted q (our "fitted" rule is in this spirit) or a tuned fraction α_c of eigenvalues to keep, which "is not based on any theoretical rule". López de Prado's "constant residual eigenvalue" method is the same clip with σ² fitted by matching a kernel density of the eigenvalues to the Marchenko-Pastur density; skfolio ships it as `DenoiseCovariance`.

**Linear shrinkage** blends the sample matrix with a target. We have the scaled-identity target of Ledoit-Wolf 2004. The 2003 paper uses a single-index (market) model as target and the 2004 *Journal of Portfolio Management* paper a constant-correlation target; both are one-line changes to the estimator we already have and both inject structure the identity does not. OAS (Chen, Wiesel, Eldar, Hero 2010) is a variant tuned for Gaussian data.

**Nonlinear shrinkage, the "Bouchaud method".** This is the method you remembered. Instead of one constant for the bulk and the raw value for everything else, every eigenvalue gets its own corrected value from a formula that needs nothing but the sample spectrum. Ledoit and Péché (2011) showed that the best matrix you can build on the sample eigenvectors, the "oracle", has the eigenvalues

    ξ(λ) = λ / | 1 − q + q · λ · g(λ − iη) |²

where g is the Stieltjes transform of the empirical spectrum, g(z) = (1/N) Σ_j 1/(z − λ_j), and η is a small regulariser. Bun, Bouchaud and Potters **[full text, eq. 6.5 and 6.26]** show the same formula holds for the outliers (the signal modes), recommend η = N^(−1/2), and prove the finite-N error is of order 1/√T. The one known defect is at the small end of the spectrum: the formula dives toward zero for the smallest eigenvalues, exactly where the minimum-variance optimiser looks. Their fix **[eq. 8.4]** rescales those eigenvalues by the factor that would be exact if the truth were an inverse-Wishart matrix, calibrated so the model's lower edge matches the observed smallest eigenvalue:

    ξ_reg = ξ · max(1, Γ_iw)

Ledoit and Wolf reached the same oracle by a different route: QuEST (2012, 2015), the *Review of Financial Studies* 2017 paper "Markowitz meets Goldilocks" (N free parameters, "just right"), the 2020 *Annals of Statistics* analytical version that estimates g with a kernel density and its Hilbert transform and runs about a thousand times faster than QuEST, and the 2022 *Bernoulli* quadratic-inverse shrinkage (QIS) tuned for inverse-type losses such as portfolio variance.

What it buys, from the review's own test **[Table 4, full text]**: on US stocks the minimum-variance portfolio realised 10.4 % volatility with the regularised RIE, 10.6 % with clipping, 10.5 % with Ledoit-Wolf linear shrinkage, 11.6 % with the raw sample matrix and 15.0 % with the identity. The RIE won in every test but one (Japan, minimum variance, where linear shrinkage won), and by more with return predictors than with minimum variance. The gain over clipping is real but modest for our use. It attacks eigenvalue noise only; the eigenvectors stay as noisy as we measured. Code: `nonlinshrink` on PyPI (analytical Ledoit-Wolf, numpy only), `pyRMT` (RIE), MATLAB `covShrinkage` (QIS). The Bun et al. formula is about forty lines of numpy and fits the estimator registry as it is.

### 1.2 Factor models

POET (Fan, Liao, Mincheva 2013, *JRSS-B*) keeps the K leading principal components as factors and thresholds the residual covariance to make it sparse; the sample matrix, a pure factor model and pure thresholding are special cases. R package `POET`. López de Prado's "detoning" removes the market eigenvector after denoising, which he warns makes the matrix singular and unfit for direct mean-variance use; it is a clustering tool more than a risk model.

### 1.3 Hierarchical clustering filters

Tumminello, Lillo and Mantegna (2007) compared spectral filters with hierarchical-clustering filters and found the spectral ones more informative but the clustering ones more stable under statistical uncertainty. Bongiorno and Challet built on that: **BAHC** (*PLOS ONE* 2021) averages hierarchical-clustering filters over many bootstrap copies of the data and reports lower realised minimum-variance risk than linear and nonlinear shrinkage when assets outnumber days; **k-BAHC** (*European Journal of Finance* 2022) extends it and beats nonlinear shrinkage and DCC-NL after costs in their tests, and does better in non-stationary universes. This is the one family that does not keep the sample eigenvectors, so it is the only one that speaks to finding 2. Hierarchical risk parity (López de Prado 2016) sidesteps inversion altogether and skfolio ships it, with nested-clusters optimisation.

### 1.4 Sparse precision

Minimum variance needs Σ⁻¹, not Σ. The graphical lasso (Friedman, Hastie, Tibshirani 2008) estimates a sparse Σ⁻¹ with an L1 penalty; Torri, Giacometti and Paterlini (2019) apply it, and a Student-t version (tlasso), to minimum-variance portfolios and report gains over RMT and Ledoit-Wolf. scikit-learn and skfolio ship `GraphicalLassoCV`. Runs slowly at N = 456 and the penalty must be cross-validated.

### 1.5 Time variation: the staleness problem

**Exponential weighting plus RMT.** Pafka, Potters and Kondor (2004) replaced the flat window by exponentially decaying weights, derived the Marchenko-Pastur-type spectrum for exponentially weighted matrices so the clip edge is still known, and found the combination beat both EWMA alone and flat-window RMT filtering. Potters, Bouchaud and Laloux (2005) report the same spectrum. This is the direct answer to finding 4: a long memory for the stable eigenvalues with more weight on the recent structure. skfolio has `EWCovariance` and a regime-adjusted version.

**DCC with nonlinear shrinkage.** Engle, Ledoit and Wolf (2019, *JBES*) combine GARCH-type dynamics with dynamic conditional correlations whose target matrix is nonlinearly shrunk; De Nard, Engle, Ledoit and Wolf (2022) extend it with intraday data. Heavy machinery, MATLAB code from the authors' page.

**Tests, not estimators.** Bouchaud, Mastromatteo, Potters and Tikhonov (2022, "Excess out-of-sample risk and fleeting modes") and Riabov, Tikhonov and Bouchaud (2025) give exact null distributions for eigenvector overlaps between two sample matrices, including overlapping windows. They do analytically what our Gaussian-panel simulation did numerically, and they report that in equities momentum is a source of excess out-of-sample risk.

### 1.6 Fat tails

Biroli, Bouchaud and Potters (2007) derive the eigenvalue density for Student-t returns, which replaces Marchenko-Pastur and moves the edge; part of why our fitted σ² differs from one. Yang, Couillet and McKay (2015) combine Tyler's robust M-estimator with shrinkage and tune the intensity to minimise portfolio risk under heavy tails. The Gerber statistic (Gerber, Markowitz, Ernst, Miao, Javid, Sargen 2022, *JPM*) counts co-movements beyond a threshold, so single extreme days cannot dominate; skfolio ships `GerberCovariance`. The cheapest robust step is the one Bun et al. use throughout: standardise each return by a rolling volatility before estimating the correlation. On our panel this removes the market-wide tail (equal-weight index excess kurtosis 18 → 3 with a 21-day volatility) but not the single-stock tail (pooled stocks 15 → 24, because about half of the stock-days beyond five standard deviations fall on ordinary market days and are company news, which scaling by a calm month makes larger, not smaller). So standardisation, Tyler's estimator and the Student-t edge all address the shared-scale (elliptical) tail: they rescale or re-weight the whole day. Idiosyncratic jumps need pairwise rank or threshold measures (Kendall's tau, the Gerber statistic) or per-stock winsorising, because in 456 dimensions one stock's jump barely changes the length of the day's return vector (a 10-sigma jump in a stock with R² = 0.5 raises the squared Mahalanobis length from about 456 to about 656, so a t or Tyler weight only falls to about 0.7, while a day when every stock moves 3 sigma falls to about 0.1). See `examples/sp500_fat_tails.py` and `output/sp500_fat_tails/`.

### 1.7 Regularise the portfolio, not the matrix

Jagannathan and Ma (2003) proved that a long-only constraint is equivalent to shrinking the large covariances; DeMiguel, Garlappi, Nogales and Uppal (2009) generalise to norm constraints that nest 1/N and long-only. Bongiorno and Challet also argue that nonlinear shrinkage, optimal for the matrix, is not optimal for the portfolio built from it. This is our planned phase 2.

### 1.8 Entrywise: power mapping

Guhr and Kälber (2003) raise the absolute value of every correlation to a power, about 1.5 in later work by Andersson, Öberg and Guhr (2005), which suppresses small spurious correlations and "prolongs" the series. Parameter-free and cheap; rarely benchmarked against the methods above.

## 2. Which enemy each method fights

| Finding | Methods that address it | Methods that do not |
|---|---|---|
| Eigenvalue noise exploited by the optimiser | RIE / nonlinear shrinkage, linear shrinkage with structured targets, POET, graphical lasso, long-only and norm constraints | diagonal (bias instead) |
| Eigenvector noise beyond mode 20 | BAHC and hierarchical filters, factor models (residual block), graphical lasso, portfolio constraints | every rotationally invariant method, including RIE |
| Stale data beyond two years | EWMA + RMT, DCC-NL, regime-adjusted EW | all flat-window estimators |
| Fat tails | volatility standardisation, Student ensemble edge, Tyler + shrinkage, Gerber | Gaussian-calibrated edges and intensities |

## 3. A suggested order, for when we do run things

1. **Optimal RIE with the inverse-Wishart fix, plus the Ledoit-Wolf analytical version as a cross-check.** *Done 2026-11-02, see 3.1; the Ledoit-Wolf analytical cross-check is still open.* It is the natural successor to the clipping we have, costs forty lines, and the review's table says it should shave a few tenths of a percent of realised risk. It is also the cleanest test of how much eigenvalue noise is left after clipping.
2. **Exponential weighting under the same estimators.** The only cheap attack on staleness, and the T-sweep says staleness is now the larger problem for the clipped models. *Done 2026-11-02, see 3.2.*
3. **Volatility standardisation before the correlation step, plus a robust treatment of single-stock jumps.** *Done 2026-11-02 for the standardisation part, see 3.3.* Standardisation is cheap and standard in the Bouchaud school and removes the market-wide clustering; the company-news jumps that remain need a pairwise measure such as Kendall's tau or the Gerber statistic, or per-stock winsorising; Tyler and the Student-t edge belong with standardisation, since they treat the whole day.
4. **BAHC.** The strongest recent challenger in minimum-variance tests and the only method here that changes eigenvectors.
5. **POET or a graphical lasso**, if 1 to 4 leave something on the table.
6. **Long-only and norm constraints**, already planned, which attack the same noise from the optimiser's side.

Each should go through the protocol we already built: real panel, Gaussian panel from the fixed truth, and the window sweep, so its gain splits into noise, bias and staleness the same way the current estimators did.

### 3.1 Status: the optimal RIE is built and tested (2026-11-02)

`market_analyzer.rie` implements the Ledoit-Péché formula with `η = N^(-1/2)` and the Bun-Bouchaud-Potters inverse-Wishart lift, sorting and trace preservation (their Algorithm 1; the κ-factor the printed algorithm drops in front of the square root is restored, and when the smallest eigenvalue sits above the Marchenko-Pastur edge the pure-noise prior κ = ∞ is used). For T ≤ N the zero eigenvalues take the Ledoit-Wolf null value. Registered as `minvar_rie`; the bare formula is `minvar_rie_raw`. `examples/sp500_rie_study.py` ran both through the three tests on the saved runs and samples of the earlier studies; outputs in `output/sp500_rie/`.

What came out, in plain words:

* **In the Gaussian world it tracks the oracle.** On one T = 500 window the lifted eigenvalues sit on top of `uᵢᵀ C uᵢ` across the whole spectrum (`shrinkage_function.png`); the 20 smallest average 0.298 against an oracle 0.303, where the bare formula gives 0.019 and clipping 0.22 (fitted) or 0.46 (unit).
* **Honesty improves a lot, with zero bias.** Monte Carlo medians of true/predicted volatility: 1.37 (RIE) vs 1.69 (clipped, fitted) and 1.58 (unit) at T = 500; 1.21 vs 1.42 at T = 1000; 1.11 vs 1.21 at T = 2000; all → 1.00 at T = ∞. Cost (true / best possible): 1.23 vs 1.28 at T = 500, 1.14 vs 1.19 at T = 1000.
* **The lift is essential near T ≈ N.** Without it the formula drives the smallest eigenvalues to ~0.01 and the optimiser piles into them: at T = 500 the bare formula scores a ratio of 3.4 and 20% realised volatility against 1.4 and 10% with the lift. By T ≥ 1500 the two variants coincide. For T < N the lift never triggers, so the two also coincide there.
* **On the real market it is the most honest and the least risky min-var model at every window from 500 on.** Rolling T = 500: ratio 1.44 vs 1.83 (clipped, fitted), 1.56 (unit), 2.25 (Ledoit-Wolf); realised 12.1% vs 12.2%, 13.3%, 13.0%; turnover 1.45 vs 1.72. Common-period sweep (46 holds from Dec 2022): realised volatility 9.7% (T = 500), 10.1%, 10.2%, 10.3%, 10.3% (T = 2000) against clipping's 10.3%, 10.5%, 10.9%, 11.3%, 11.1%. The risk gain is a few tenths of a percent, as in Bun et al.'s Table 4; the honesty gain is the large one.
* **Stale data hurts it less than it hurts clipping.** Clipping's realised risk rises by 0.8 points from T = 500 to 2000 on the real market while falling in the Gaussian world; the RIE's rises by 0.6 and is flat from T = 750. Real / Gaussian realised volatility at T = 2000: 1.11 for the RIE against 1.18 for clipping.
* **What it does not do.** It keeps the sample eigenvectors, so the month-to-month redraw of modes beyond 20 (the eigenvector study) is untouched, and the real-market ratio is still 1.3–1.5 at T = 500–1000: the remaining gap is market change plus the eigenvector noise no eigenvalue map can fix. The real-market smallest eigenvalues are lifted to about 0.17, while dual-class pairs (GOOG/GOOGL, FOX/FOXA, NWS/NWSA) have true eigenvalues near 0.01; Bun et al.'s footnote 28 flags exactly this limit of edge-calibrated lifts.

Next in the order: exponential weighting (item 2), now with the RIE as the eigenvalue map.

### 3.2 Status: exponential weighting with the RIE is built and tested (2026-11-02)

`rie.py` now carries the noise model of a weighted sample. Each weighted rank-one term `w_t y_t y_tᵀ` is free of the others, so the noise R-transform is `R(x) = Σ_t w_t / (1 − N w_t x)`, its S-transform solves `S · R(ωS) = 1`, and the optimal cleaning for any multiplicative noise is `ξ(λ) = Im[ω S(ω)] / Im g(z)` with `ω = z g(z) − 1`, which collapses to the Ledoit-Péché formula for flat weights. For exponential weights with an infinite past, `S(ω) = (1 − e^(−q_e ω)) / (q_e ω)` with `q_e = N (1 − ε)`, which is Pafka, Potters and Kondor's spectrum: variance `q_eff = N Σ w²` like Marchenko-Pastur, but a softer lower edge and a longer upper tail (`output/sp500_ewma/noise_spectrum.png`). The S-transform root is followed by continuation from `S(0) = 1`; a plain Newton start from `1 / (1 + q_eff ω)` diverges when `q_eff > 1`, which a half-life of 126 sessions already gives with 456 names. The inverse-Wishart lift is still the flat-noise one at `q_eff`, a regulariser only. `ewma_covariance(returns, halflife, cleaning)` in `optimizer.py` builds the weighted covariance (weighted means and variances too, divided by `1 − Σ w²`) and cleans it with nothing, clipping at `q_eff`, or the RIE; `minvar_rie_ewma` (half-life 252) is registered; `ewma_estimator(halflife, cleaning)` builds the rest. Study: `examples/sp500_ewma_study.py`, outputs in `output/sp500_ewma/`.

What came out, in plain words:

* **Forgetting costs noise, and the weighted-noise model is what keeps that cost in check.** Gaussian truth, window 1000, median honesty of the weighted RIE: 1.22 (flat), 1.23 (half-life 504), 1.27 (252), 1.47 (126), 2.07 (63). The flat formula at the effective q matches it down to half-life 252 and collapses at 63 (5.3), where `q_eff = 2.5`. Clipping at the effective q: 1.41 → 2.05.
* **On the real market a one-year half-life buys the short window's risk with the long window's honesty and turnover.** Window 1000, 93 holds from Dec 2018: realised volatility 13.0% at half-lives 126 and 252 against 13.3% flat at T = 1000 and 12.9% flat at T = 500; honesty 1.29 at half-life 252 against 1.31 (flat 1000) and 1.46 (flat 500); turnover 1.34 against 1.34 and 1.53. Half-life 126 has the same risk but worse honesty (1.45) and turnover (1.69); 63 is bad on every count (2.45, 15.0%, 4.2).
* **Forgetting removes the market-change part of the gap.** Real / Gaussian honesty for the weighted RIE: 1.07 flat, 1.05 at half-life 504, 1.00 at 252, 0.99 at 126. With a half-life of a year or less the model is exactly as honest on the real market as on Gaussian returns from a fixed truth; what remains is the estimation noise of a short effective sample plus the eigenvector churn. Real / Gaussian realised risk falls from 1.37 (flat) to 1.29 (252) and 1.23 (126).
* **The net gain is small because noise and staleness trade almost one for one.** The Gaussian world prices half-life 252 at +0.4 points of risk over flat; the real market pays back about 0.7, net −0.3. Pafka, Potters and Kondor's conclusion holds here too: the optimum is a long-ish memory, not the RiskMetrics 0.94 (half-life 11 days).
* **Known limitation.** The lift is the flat-noise one. In the Gaussian panel at half-life 126 the 20 smallest cleaned eigenvalues average 0.15 against an oracle 0.27 (`cleaning_function.png`), so the weighted RIE still under-lifts its bottom when `q_eff > 1`; a weighted-noise lift would need the inverse-Wishart prior pushed through the same subordination (or a simulated stand-in truth).

Next in the order: volatility standardisation before the correlation step (item 3), or the weighted-noise lift above if short half-lives are wanted.

### 3.3 Status: volatility standardisation is built and tested (2026-11-02)

The covariance is `Σ = D C D`, and every fix so far worked on `C`. `D` was one number per stock, the window standard deviation, doing two jobs with the same stale average: setting how loudly each day enters the correlation sums, and scaling the cleaned correlation back into a forecast. `ewma_volatility` (RiskMetrics recursion, past days only, burn-in dropped, floored at 10% of the window volatility for price series filled forward before a listing) and `standardized_covariance` in `optimizer.py` separate the two jobs (`standardize`, `scale`) and offer two volatility recipes (`mode="stock"`: each stock's own estimate; `mode="market"`: one cross-sectional factor per day, as Bun, Bouchaud and Potters do, with the forecast being the window volatility times the factor's exponentially weighted level). Registered: `minvar_rie_standardized` (stock mode, half-life 20). Study: `examples/sp500_standardisation_study.py`, outputs in `output/sp500_standardisation/`, with a third panel, "clocked": the Gaussian draws multiplied by the real market's daily volatility factor, so volatility clustering is isolated from fat tails and structure change.

What came out, in plain words (window 1000, 93 holds from Dec 2018, RIE cleaning unless said):

* **Volatility clustering alone costs the plain RIE 9% of honesty and 12% of realised risk** (Gaussian 1.22 and 9.7% → clocked 1.33 and 10.9%). Everything else the real market adds (fat tails, single-stock jumps, structure change) leaves the honesty ratio where it is (real 1.31) but raises the realised risk by another 22% (13.3%). The real market's extra cost shows up as more risk, not as more dishonesty.
* **Per-stock standardisation with a short half-life is harmful, everywhere.** Honesty 1.71 / 1.96 / 2.49 in the three worlds against 1.22 / 1.33 / 1.31 for the plain RIE, realised risk up 0.8–2.4 points, turnover tripled. Half-life 60 loses less (1.36 / 1.51 / 1.88). The culprit is the forecast, not the correlation: with the truth known, replacing the estimator's volatilities by the true ones brings the h = 20 version from 1.67 to 1.09, replacing its correlation only to 1.50. A 6% median error in each stock's volatility becomes a 1.5× error in predicted portfolio risk, because the optimiser overweights exactly the stocks whose volatility happens to read low. The same error-maximisation we saw in the correlation, now through the volatility channel.
* **Standardising the correlation's inputs is harmless, nothing more.** Standardised correlation with the window scale: 1.21 / 1.24 / 1.30 and 9.7% / 11.1% / 13.5%, against 1.22 / 1.33 / 1.31 and 9.7% / 10.9% / 13.3% plain. It removes most of the clustering effect on honesty in the clocked world and gains nothing on the real market.
* **The market-wide factor keeps honesty and risk and calms the portfolio.** 1.30 / 1.32 / 1.35 and 9.8% / 11.3% / 13.5%, with the clock adding only 1% (the factor is what the clock is), gross leverage 4.6 against 6.9 and turnover 0.81 against 1.34 on the real market. For clipping it also cuts realised risk (15.0% → 14.3%). It starts 7% less honest in the constant-volatility world because the factor forecast is noise there.
* **What it does not fix.** The equal-weight index's kurtosis falls from 18 to 4 and the market's daily volume flattens, but the pooled single-stock kurtosis stays at 15: the largest standardised moves are company events on calm days (a 64σ Danaher day in 2016, a 43σ Globe Life day in 2024), as the fat-tail study predicted. Standardised returns still have 15% more variance than the forecast implied (the estimate lags).
* **Data note found on the way.** Three tickers of the 456-name panel (SW 51%, FERG 37%, AMCR 34% zero-return days) were filled forward before their listing. The saved min-var runs put at most 4.5% of gross leverage on them, so earlier conclusions stand, but the universe filter should require actual trading, not just non-zero variance.

Where this leaves the order: a volatility forecast for the optimiser must be pooled or shrunk (the market factor, or a GARCH-style pull toward the long-run level); per-stock short-window volatilities should not feed a min-var optimiser directly. Next: BAHC (item 4), the only family that changes eigenvectors, or long-only (item 6).

### 3.4 Status: the stock dial on the market factor is built and tested (2026-11-03)

The middle path between the market factor (which refuses to tell stocks apart) and each stock's own fast volatility (which the optimiser exploits): keep the market factor `m` and believe a fraction `α` of each stock's own deviation from it. With `σ̄_i` the window volatility, `σ_i` the stock's exponentially weighted volatility (half-life 20) and `ρ_i = σ_i / (σ̄_i m)`, the forecast is `σ̄_i · m · (1 + α (ρ_i − 1))`; `α = 0` is the market factor, `α = 1` the per-stock forecast. Built as the `trust` option of `standardized_covariance(mode="market")` in `optimizer.py` (the correlation is still built from the market-standardised returns; only the forecast moves). Nothing new registered. Study: `examples/sp500_vol_trust_study.py` (`--part measure` measures `α` from history in ten seconds; `--part rolling --panel …` runs the three worlds, about four minutes each; `--replot`), outputs in `output/sp500_vol_trust/`.

`α` is measured, not chosen: at every rebalance date each stock's `ρ` is lined up against the realised ratio `R` over the next 21 sessions (its realised volatility over its window `σ̄` and the realised market factor), and `R − 1 = α (ρ − 1)` is fitted through the origin over all 42,000 (date, stock) pairs. The Gaussian and clocked panels are the controls: no stock has a volatility spell of its own there.

What came out, in plain words (window 1000, 93 holds from Dec 2018, RIE cleaning, market factor with half-life 20):

* **Stock-specific calm and storm are real, and about 40% of it survives into the next month.** Measured `α`: −0.01 on the Gaussian panel (pure noise, as it should be), 0.11 on the clocked panel (the heavy clock makes the window `σ̄` itself noisier, and the recent reading corrects it), 0.41 on the S&P 500 (0.54 in logs; correlation 0.27; 0.29–0.57 by year). By decile: stocks reading 33% calmer than the market factor explains were 22% calmer the next month; stocks reading 42% stormier were 10% stormier. Calm persists more than storm, because storms are mostly short company events.
* **Yet it barely improves the per-stock forecast.** Over the same pairs the RMS log error of each stock's forecast is 0.309 with the market factor alone, 0.290 at the best dial setting (0.5), 0.303 at `α = 1`. Most of a stock's deviation over 21 days is not predictable from its last few weeks.
* **And the optimiser gets worse at every setting, in every world.** Honesty on the real market 1.35 (`α = 0`) → 1.39 (0.25) → 1.50 (0.41, the measured value) → 1.57 (0.5) → 2.26 (1); Gaussian 1.30 → 1.33 → 1.37 → 1.41 → 1.78; clocked 1.32 → 1.36 → 1.39 → 1.43 → 1.87. Realised risk on the real market 13.46% → 13.31% → 13.39% → 13.46% → 14.04%; turnover 0.81 → 1.66 → 2.19 → 2.41 → 2.80; gross leverage stays near 4.6. The only thing `α = 0.25` buys is 0.15 points of realised risk, back to the plain RIE's 13.30%, for double the turnover and 3% of honesty.
* **The damage grows with the square of the noise let in.** On the Gaussian panel the dial injects pure noise of `α × 8.3%` into each stock's volatility: 2.1% costs 2.3% of honesty, 4.2% costs 8%, 8.3% costs 36%. The optimiser's sensitivity to noise in the volatilities is of the same kind as its sensitivity to noise in the correlation: it finds the entries that are accidentally low.
* **Lesson.** A forecast that is right on average for each stock is still a bad optimiser input if it carries noise. Forecast quality (RMS error) and portfolio quality (honesty) are different objectives, and here the second is far more sensitive to the noise than the first is to the signal. The size side of `Σ = D C D` ends at the market factor for now. A per-stock dial could only pay with a much less noisy per-stock estimate (a GARCH with long memory, or option-implied volatilities), not with a 20-day EWMA. The remaining choice on the size side is market factor (1.35, 13.5%, turnover 0.81, leverage 4.6) against the plain window volatility (1.31, 13.3%, 1.34, 6.9), which is a trading-cost question, outside these studies.

Next: BAHC (item 4) or long-only (item 6).

### 3.5 Why the fixes stall (2026-11-04)

A separate note, `docs/why_the_gap_persists.md`, collects the documented limits of every method above and a shuffled-days test on our panel. Short version: for the RIE at T = 1000, the 8.25% best possible risk becomes 13.3% through estimation noise (+18%, mostly eigenvectors, which no eigenvalue map touches), fat tails (+15%), calm and stormy spells (about +12%) and structural drift (about +5%). Only time had a tool pointed at it so far.

### 3.6 Fixing the directions: where the error sits and what others do (2026-11-05)

`docs/fixing_the_directions.md` reviews the methods that change eigenvectors (anchors and James–Stein, factor/sector/tree/sparse-inverse shapes, Kendall and Tyler, intraday ranges, dynamic correlation, weight-side regularisation) and measures where the RIE's direction error sits on our panel (`examples/sp500_eigenvector_error.py`, outputs in `output/sp500_eigenvector_error/`). Short version: the market direction is off by only 2–3 degrees, but its size makes that tilt spill risk onto quiet directions (a perfect market direction would close 13–18% of the RIE-to-truth gap); directions 8–20 blur into each other and fat tails make them much worse (a perfect top 20 would close 38–45%). On real data at our size, Kendall's directions are the strongest published gain (about 15%, one paper); a static factor prior gives exactly 0%; the James–Stein anchors are untested outside simulation. Tested here (`examples/sp500_james_stein_study.py`, `rie_js`): James–Stein on the market direction on top of the RIE makes risk slightly worse in every world and window (+0.3% to +2.5%), honesty slightly better; the market arrow's error is a leak into the sector arrows, not extra beta dispersion (fixing_the_directions.md §7).

## 4. Has anyone combined a robust co-movement measure with RMT cleaning? (checked 2026-11-01)

Short answer: nobody has published "Gerber statistic + RIE". The nearest published work uses Kendall-type rank
measures as the input to RMT cleaning, and it is recent (2024–2026). Open-source tools (skfolio) let you compose
`GerberCovariance` with `DenoiseCovariance`, so practitioners can do it, but I found no paper that evaluates it.
Below, each reference in plain words, grouped by the question it answers.

### 4.1 Robust input matrix, then RMT cleaning (closest to the idea)

- **Espana, Le Coz, Smerlak (2024), Kendall correlation coefficients for portfolio optimization.** Replace the Pearson
  correlation by Kendall's tau (and relatives built from an odd kernel such as tanh), then clean the eigenvalues with the
  usual RMT tools (clipping, RIE). On 500 liquid stocks per region, 2000–2024, with synthetic Student-t (ν = 3) data and
  real data. Two findings matter for us: (1) rank matrices have far fewer zero eigenvalues (they appear only when
  N ≈ T²/2, not when N > T), and their eigenvectors, especially the small-risk ones, are closer to the truth when data
  are scarce; (2) a minimum-risk portfolio built from the Kendall matrix beats RIE applied to Pearson out of sample. Their
  best recipe: Kendall eigenvectors with eigenvalues re-estimated by cross-validation (isotonic cross-validation
  covariance, ICVC). Lesson: use the robust matrix for *directions*, and moment-based eigenvalues for *sizes*.
  Bouchaud-adjacent group (CFM chair). https://arxiv.org/abs/2410.17366
- **Karamikabir, Arashi (2026), MENS: nonlinear shrinkage in nonparanormal models.** Convert each stock's returns to
  "normal scores" (rank them, then map the ranks onto a bell curve), compute that covariance, then apply Ledoit–Wolf
  analytic nonlinear shrinkage. This is "rank measure + RIE" with a Gaussian-copula justification. Minimum-variance
  backtest on 200 S&P 500 stocks, 2013–2018, one-year windows: realised vol 9.4% vs 11.0% for linear shrinkage, lower
  turnover, condition number three times better. Caveat stated by the authors: the advantage reverses when N ≥ T.
  https://arxiv.org/abs/2607.19825
- **Hediger, Näf, Wolf (2022), R-NL.** Tyler's estimator (strip each day's size, keep its direction) combined with
  nonlinear shrinkage. Robust to heavy tails and high dimension in elliptical models; proofs of convergence, simulations,
  one empirical application. Handles the shared-storm tail, not single-stock jumps. https://arxiv.org/abs/2210.14854
- **Yang, Couillet, McKay (2015), A robust statistics approach to minimum variance portfolio optimization.** Tyler +
  Ledoit–Wolf shrinkage, with RMT used to estimate the realised portfolio risk from in-sample data and tune the shrinkage
  online. Synthetic and real market data. https://arxiv.org/abs/1503.08013
- **Jay, Terreaux, Ovarlez, Pascal (2018; journal version with Soler, de Peretti, Chorro in *Soft Computing* 2020).**
  Robust M-estimator of the covariance plus an RMT rule for how many factors to keep, applied to maximum-variety and
  minimum-variance portfolios on real data. "Promising improvements"; signal-processing group.
  https://arxiv.org/abs/1804.00191 ; https://link.springer.com/article/10.1007/s00500-020-04840-9
- **Bongiorno, Berritta (2023), Optimal covariance cleaning for heavy-tailed distributions.** Shows that the RIE's
  least-squares optimality equals minimal information loss for Gaussian data; for Student-t data the two differ at finite
  N but the difference vanishes as N grows. Reassuring: the RIE formula stays approximately right under fat tails.
  https://arxiv.org/abs/2304.14098
- **Theory for rank matrices.** Bandeira, Lodhia, Rigollet (2017): the Kendall matrix of independent data follows a
  squeezed Marchenko–Pastur law, (2/3)·MP + 1/3, so its noise edge is lower than Pearson's. Bouchaud, Bousseyroux,
  Espana, Smerlak (2026): for returns with a shared daily volatility scale, the rank transform does *not* erase that
  scale; the Spearman bulk is a generalised MP law. Implication: if we clip a rank matrix, the plain MP edge is the
  wrong edge. https://arxiv.org/abs/1611.04505 ; https://arxiv.org/abs/2607.25486
- **Tools.** skfolio's `DenoiseCovariance`/`DetoneCovariance` accept any covariance estimator, including
  `GerberCovariance` (2019 and 2022 variants). https://skfolio.org/generated/skfolio.moments.GerberCovariance.html

### 4.2 The Gerber statistic itself: evidence and critique

- **Gerber, Markowitz, Ernst, Miao, Javid, Sargen (2022), *JPM*.** Nine assets, 1990–2020. Gerber beats the historical
  covariance and Ledoit–Wolf at almost all target-risk levels. Small universe: nothing about N ≈ T.
- **Gerber et al. (2023), Proofs that the Gerber statistic is positive semidefinite.** The 2019 form (divide by the
  number of days both stocks are outside the band) can fail to be PSD; the two 2022 forms, including the one that
  divides by the number of days *not both neutral*, are proven PSD. https://arxiv.org/abs/2305.05663
- **Flint, Polakow (2023), Deconstructing the Gerber statistic, *Finance Research Letters* 56.** Two-asset simulation
  across the three forms. Conclusion: the statistic is "an imperfect conditional dependence metric"; because it is
  conditional (only days beyond the band count) its values diverge from the full-sample correlation. Read: it measures
  something other than ρ, so feeding it to an optimizer that expects ρ is a modelling choice, not a free lunch.
  https://doi.org/10.1016/j.frl.2023.104144
- **Gerber, Ernst, Markowitz, Miao, Saini, Smyth, Cuttler, Sargen (2026), The Modified Gerber Statistic (SSRN).**
  A threshold-free version. Two out-of-sample studies 1990–2025; at a 9% risk target the annual returns are MGS 11.26%,
  original Gerber 11.04%, Ledoit–Wolf 10.92%, historical 11.05%. The margins are a few tenths of a percent a year.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7190778
- **Zhou (2024), Portfolio optimization with robust covariance and CVaR constraints.** Large-cap US stocks 2012–2022;
  compares Ledoit shrinkage variants and Gerber variants; Gerber with a MAD scale is the top performer, but "robust
  estimators do not manage tail risk well under extreme market conditions, for example the Covid-19 period".
  https://arxiv.org/abs/2406.00610

### 4.3 Staleness versus noise: which to fix first

- **Pafka, Potters, Kondor (2004).** Exponential weighting (recent days count more) plus RMT clipping with the
  correct edge for exponentially weighted matrices beats both exponential weighting alone and flat-window RMT alone.
  https://arxiv.org/abs/cond-mat/0402573
- **Engle, Ledoit, Wolf (2019), Large dynamic covariance matrices.** DCC-NL: divide each stock by its own GARCH
  volatility, nonlinear-shrink the correlation of the standardised returns, re-dress with forecast volatilities. Beats
  static nonlinear shrinkage for large N. The standardisation step is at once a staleness fix and a shared-storm fix.
- **Bongiorno, Challet (2023), Non-linear shrinkage of the price return covariance matrix is far from optimal for
  portfolio optimisation, *Finance Research Letters* 52.** The eigenvalues that are optimal for the *future* are not the
  in-sample oracle ones because the market changes. Shuffling past and future days (making the data stationary)
  removes most of the gap, so the gap *is* non-stationarity; long windows give "out-of-date eigenvector bases". This is
  exactly what our real window sweep showed. https://arxiv.org/abs/2112.07521
- **Bongiorno, Challet, Loeper (2021/2023), time-independent eigenvalues ("Average Oracle"); Bongiorno, Challet
  (2023), Average Oracle vs DCC-NLS.** A fixed eigenvalue profile, averaged over history, beats data-dependent cleaning
  in a changing market; DCC-NLS sometimes has the lowest vol but ten times the turnover and lower Sharpe.
  https://arxiv.org/abs/2111.13109 ; https://arxiv.org/abs/2309.17219
- **Bongiorno, Manolakis, Mantegna (2025), End-to-end neural minimum variance.** A network that learns the time
  weighting of past days, the per-stock volatility transform and the eigenvalue cleaning jointly; 2000–2024, lower
  realised vol than nonlinear shrinkage. Confirms the three levers are time weighting, volatility scaling and eigenvalue
  cleaning. https://arxiv.org/abs/2507.01918

### 4.4 Fat tails: what they look like and what robust statistics buys

- **Chicheportiche, Bouchaud (2012), The joint distribution of stock returns is not elliptical.** Large US and Japanese
  panels; Student copulas fit strongly correlated pairs but fail as correlation falls; the failure is traced to "the
  assumption of a single volatility mode for all stocks". Direct evidence that one-dial-per-day models (Student-t,
  Tyler) miss the single-stock tail. Follow-up: nested factor model (2015). https://arxiv.org/abs/1009.1100
- **Biroli, Bouchaud, Potters (2007), The Student ensemble of correlation matrices.** The Marchenko–Pastur law
  generalised to Student-t returns; fits financial spectra better than MP. The "Student edge" for clipping.
  https://arxiv.org/abs/0710.0802
- **Fan, Wu, Yang (2024), Shocks-adaptive robust minimum variance portfolio.** Robust PCA factor model (POET made
  robust) that handles both market-wide and single-stock shocks "without the need to distinguish them".
  https://arxiv.org/abs/2410.01826
- **Härdle, Klochkov, Petukhina, Zhivotovskiy (2022), Robustifying Markowitz.** Median-of-means gradient steps
  instead of a covariance inverse; lowest turnover among the methods compared, similar out-of-sample risk.
  https://arxiv.org/abs/2212.13996
- **DeMiguel, Nogales (2009), Portfolio selection with robust estimation, *Operations Research* 57.** M- and
  S-estimators; the gain is mostly lower turnover, with a slight out-of-sample improvement. Later studies find robust
  estimators help mainly when N is small.
- **Chen, Wiesel, Hero (2011), Robust shrinkage estimation of high-dimensional covariance matrices.** Tyler's
  iteration regularised with a shrinkage coefficient so it works when N > T. https://arxiv.org/abs/1009.5331

### 4.5 What this changes in the suggested order

1. Staleness before fat tails. The cost of staleness is measured in our own sweep and in Bongiorno–Challet; the cost of
   fat tails to the portfolio is not yet measured here. Dividing by recent volatility (the DCC step) treats staleness
   and the shared-storm tail together, so it is the first move either way.
2. "RIE + Gerber" is untested in print. The tested cousin is "Kendall + cleaning" (Espana et al.) and "normal scores +
   nonlinear shrinkage" (MENS). If we try a robust input, start with Kendall or normal scores, where the map back to a
   correlation is exact, and test Gerber beside it. Keep the robust matrix for eigenvectors and moment-based
   eigenvalues for sizes, as Espana et al. do.
3. Do not cap single-stock jumps before deciding what a jump is. Earnings recur quarterly, so a 21-day hold contains
   one about a third of the time; the jump is part of the stock's true variance even if it should not drive its
   correlations. Robust correlation, raw or exponentially weighted variance.

## 5. References

- Laloux, Cizeau, Bouchaud, Potters (1999), Noise dressing of financial correlation matrices, *PRL* 83, 1467. https://arxiv.org/abs/cond-mat/9810255
- Potters, Bouchaud, Laloux (2005), Financial applications of random matrix theory: old laces and new pieces, *Acta Phys. Pol. B* 36. https://arxiv.org/abs/physics/0507111
- Bun, Bouchaud, Potters (2017), Cleaning large correlation matrices: tools from random matrix theory, *Physics Reports* 666, 1–109. https://arxiv.org/abs/1610.08104 **[full text]**
- Bun, Allez, Bouchaud, Potters (2016), Rotational invariant estimator for general noisy matrices. https://arxiv.org/abs/1502.06736
- Ledoit, Péché (2011), Eigenvectors of some large sample covariance matrix ensembles, *PTRF* 151, 233–264. https://arxiv.org/abs/0911.3010
- Ledoit, Wolf (2004), A well-conditioned estimator for large-dimensional covariance matrices, *J. Multivariate Analysis* 88. (Repo's current shrinkage.)
- Ledoit, Wolf (2017), Nonlinear shrinkage of the covariance matrix for portfolio selection: Markowitz meets Goldilocks, *RFS* 30, 4349–4388. https://papers.ssrn.com/abstract=2383361
- Ledoit, Wolf (2020), Analytical nonlinear shrinkage of large-dimensional covariance matrices, *Annals of Statistics* 48, 3043–3065. https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-5/Analytical-nonlinear-shrinkage-of-large-dimensional-covariance-matrices/10.1214/19-AOS1921.full
- Ledoit, Wolf (2022), Quadratic shrinkage for large covariance matrices, *Bernoulli* 28, 1519–1547. MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/106240-covshrinkage
- Engle, Ledoit, Wolf (2019), Large dynamic covariance matrices, *JBES* 37, 363–375. https://econpapers.repec.org/RePEc:taf:jnlbes:v:37:y:2019:i:2:p:363-375
- Fan, Liao, Mincheva (2013), Large covariance estimation by thresholding principal orthogonal complements, *JRSS-B* 75, 603–680. https://arxiv.org/abs/1201.0175 ; R package https://cran.r-project.org/web/packages/POET/index.html
- Tumminello, Lillo, Mantegna (2007), Shrinkage and spectral filtering of correlation matrices: a comparison via the Kullback-Leibler distance, *Acta Phys. Pol. B* 38, 4079. https://arxiv.org/abs/0710.0576 ; companion *PRE* 76, 031123, https://arxiv.org/abs/0706.0168
- Bongiorno, Challet (2021), Covariance matrix filtering with bootstrapped hierarchies, *PLOS ONE*. https://pmc.ncbi.nlm.nih.gov/articles/PMC7808632
- Bongiorno, Challet (2022), Reactive global minimum variance portfolios with k-BAHC covariance cleaning, *European Journal of Finance* 28, 1344–1360. https://arxiv.org/abs/2005.08703
- Friedman, Hastie, Tibshirani (2008), Sparse inverse covariance estimation with the graphical lasso, *Biostatistics* 9. Torri, Giacometti, Paterlini (2019), Sparse precision matrices for minimum variance portfolios, *Computational Management Science*. https://dspace.vsb.cz/items/8e85b87f-c80b-4347-b4fb-70677d148199
- Pafka, Potters, Kondor (2004), Exponential weighting and random-matrix-theory-based filtering of financial covariance matrices for portfolio optimization. https://arxiv.org/abs/cond-mat/0402573
- Bouchaud, Mastromatteo, Potters, Tikhonov (2022), Excess out-of-sample risk and fleeting modes. https://arxiv.org/abs/2205.01012
- Biroli, Bouchaud, Potters (2007), The Student ensemble of correlation matrices, *Acta Phys. Pol. B* 38. https://arxiv.org/abs/0710.0802
- Yang, Couillet, McKay (2015), A robust statistics approach to minimum variance portfolio optimization. https://arxiv.org/abs/1503.08013
- Gerber, Markowitz, Ernst, Miao, Javid, Sargen (2022), The Gerber statistic: a robust co-movement measure for portfolio optimization, *JPM* 48(3), 87–102. https://www.hudsonbaycapital.com/documents/FG/hudsonbay/research/613040_The_Gerber_Statistic-_A_Robust_Co-Movement_Measure_for_Portfolio_Optimization.pdf
- Jagannathan, Ma (2003), Risk reduction in large portfolios: why imposing the wrong constraints helps, *Journal of Finance* 58. DeMiguel, Garlappi, Nogales, Uppal (2009), A generalized approach to portfolio optimization: improving performance by constraining portfolio norms, *Management Science* 55, 798–812. https://pubsonline.informs.org/doi/fpi/10.1287/mnsc.1080.0986
- Guhr, Kälber (2003), A new method to estimate the noise in financial correlation matrices, *J. Phys. A* 36, 3009. https://arxiv.org/abs/cond-mat/0206577
- López de Prado (2020), *Machine Learning for Asset Managers*, Cambridge; denoising/detoning as implemented in skfolio: https://skfolio.org/generated/skfolio.moments.DenoiseCovariance.html
- Python packages: `nonlinshrink` https://pypi.org/project/nonlinshrink ; `pyRMT` https://github.com/ggiecold/pyrmt ; skfolio API https://skfolio.org/api.html (EWCovariance, GerberCovariance, DenoiseCovariance, DetoneCovariance, LedoitWolf, OAS, GraphicalLassoCV, HierarchicalRiskParity, NestedClustersOptimization).
- Espana, Le Coz, Smerlak (2024), Kendall correlation coefficients for portfolio optimization. https://arxiv.org/abs/2410.17366
- Karamikabir, Arashi (2026), MENS: nonlinear shrinkage estimation in nonparanormal models for financial applications. https://arxiv.org/abs/2607.19825 ; code https://github.com/M-Arashi/Shrinkage
- Hediger, Näf, Wolf (2022), R-NL: covariance matrix estimation for elliptical distributions based on nonlinear shrinkage. https://arxiv.org/abs/2210.14854
- Jay, Terreaux, Ovarlez, Pascal (2018), Improving portfolios global performance with robust covariance matrix estimation, *EUSIPCO*. https://arxiv.org/abs/1804.00191 ; Jay, Soler, Terreaux, Ovarlez, Pascal, de Peretti, Chorro (2020), *Soft Computing*. https://link.springer.com/article/10.1007/s00500-020-04840-9
- Bongiorno, Berritta (2023), Optimal covariance cleaning for heavy-tailed distributions: insights from information theory. https://arxiv.org/abs/2304.14098
- Bandeira, Lodhia, Rigollet (2017), Marchenko–Pastur law for Kendall's tau, *Electron. Commun. Probab.* https://arxiv.org/abs/1611.04505
- Bouchaud, Bousseyroux, Espana, Smerlak (2026), Spectra of high-dimensional Spearman correlation matrices under scale-mixture dependence. https://arxiv.org/abs/2607.25486
- Gerber, Markowitz, Ernst, Miao, Javid, Sargen (2023), Proofs that the Gerber statistic is positive semidefinite. https://arxiv.org/abs/2305.05663
- Flint, Polakow (2023), Deconstructing the Gerber statistic, *Finance Research Letters* 56, 104144. https://doi.org/10.1016/j.frl.2023.104144
- Gerber, Ernst, Markowitz, Miao, Saini, Smyth, Cuttler, Sargen (2026), The modified Gerber statistic: a practical nonparametric covariance estimator for portfolio construction. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7190778
- Zhou (2024), Portfolio optimization with robust covariance and conditional value-at-risk constraints. https://arxiv.org/abs/2406.00610
- Bongiorno, Challet (2023), Non-linear shrinkage of the price return covariance matrix is far from optimal for portfolio optimisation, *Finance Research Letters* 52, 103383. https://arxiv.org/abs/2112.07521
- Bongiorno, Challet, Loeper (2021), Cleaning the covariance matrix of strongly nonstationary systems with time-independent eigenvalues. https://arxiv.org/abs/2111.13109
- Bongiorno, Challet (2023), Covariance matrix filtering and portfolio optimisation: the Average Oracle vs non-linear shrinkage and all the variants of DCC-NLS. https://arxiv.org/abs/2309.17219
- Bongiorno, Manolakis, Mantegna (2025), End-to-end large portfolio optimization for variance minimization with neural networks through covariance cleaning. https://arxiv.org/abs/2507.01918
- Chicheportiche, Bouchaud (2012), The joint distribution of stock returns is not elliptical, *IJTAF* 15. https://arxiv.org/abs/1009.1100 ; (2015) A nested factor model for non-linear dependencies in stock returns, *Quantitative Finance* 15. https://arxiv.org/abs/1309.3102
- Fan, Wu, Yang (2024), Shocks-adaptive robust minimum variance portfolio for a large universe of assets. https://arxiv.org/abs/2410.01826
- Härdle, Klochkov, Petukhina, Zhivotovskiy (2022), Robustifying Markowitz, *Journal of Econometrics* (2023). https://arxiv.org/abs/2212.13996
- DeMiguel, Nogales (2009), Portfolio selection with robust estimation, *Operations Research* 57, 560–577.
- Chen, Wiesel, Hero (2011), Robust shrinkage estimation of high-dimensional covariance matrices, *IEEE TSP*. https://arxiv.org/abs/1009.5331
