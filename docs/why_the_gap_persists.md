# Why the fixes stop working: limits of the methods, and the floor under the gap (2026-11-04)

The question: every cleaning method we built narrowed the gap between predicted and realised risk, then stalled. On the real market the best one, the RIE, still realises about 1.3 times the risk it predicts and 37% more risk than on Gaussian data from a fixed truth. Why can we not get the risk down further? This note combines a literature search with one new test on our own panel. The search read 25 primary sources and extracted 115 claims. The 25 most relevant claims were checked by three independent agents each: 23 confirmed, 2 refuted. Section 7 says which numbers went through that check.

## 0. The short answer, in plain words

The gap has three layers, and each method we tried could only reach the first.

1. **Noise in the sizes of the hidden directions.** A 1000-day window measures 456 × 456 numbers from too little data, and the optimiser bets hardest on whatever looks safest, so it amplifies the noise. This is the 1/(1 − N/T) law, and it is the layer every cleaning method attacks: clipping, Ledoit-Wolf and the RIE all repair the *eigenvalues*. In the Gaussian world the RIE brings honesty from 1.83 to 1.22. This layer is essentially solved.
2. **Noise in the directions themselves.** All of those methods keep the sample *eigenvectors*, because nobody knows how to improve them in general. The sample eigenvectors beyond the top twenty or so are mostly noise, and the minimum-variance portfolio is built almost entirely from the quietest directions, which are the noisiest ones. The literature proves this is a floor for the whole family: even the perfect eigenvalue map leaves a loss that does not shrink while N/T stays fixed. In our Gaussian world it is the distance from the RIE (1.22, 9.7%) to the oracle (0.99, 8.4%).
3. **Real days are not Gaussian days, and they do not come in a fixed order.** Two separate things, which our shuffle test (section 4) pulled apart. Fat tails: a handful of extreme days dominate a 1000-day window, so it holds fewer effective observations than its length says, and the estimate is noisier than random-matrix theory expects. Time: volatility comes in calm and stormy spells, and the correlation structure drifts, so the window describes a market that is already gone.

In numbers, for the RIE: knowing the true covariance in advance would give 8.25% risk. Estimation noise on Gaussian days adds 18%, fat tails 15%, calm-and-stormy spells about 12%, and structural drift only the last 5%, which brings us to the 13.3% we see. Noise, fat tails and time each cost roughly a sixth.

So the reason we cannot reduce the risk further with the tools we have is structural. Every tool we built repairs eigenvalues, or reacts to time. What is left lives in the eigenvectors, in the fat tails, and in time, and only the last of these had a tool pointed at it. The volatility experiments taught the same lesson from another side: the optimiser punishes noise in any input far more than it rewards signal, so any new information must come with very little noise.

## 1. Our own evidence, set side by side

Window T = 1000, 21-day holds, 93 holds from Dec 2018. "Oracle" is a look-ahead reference that uses the cleaned covariance of the whole panel at every date: it is the exact truth of the Gaussian panel, and on the real market it knows the average future.

| Estimator | Honesty, Gaussian | Honesty, real | Realised vol, Gaussian | Realised vol, real |
|---|---|---|---|---|
| Sample covariance | 1.83 | 1.97 | 11.4% | 14.8% |
| Ledoit-Wolf | 1.67 | 1.55 | 10.8% | 13.5% |
| MP clipped, fitted edge | 1.39 | 1.47 | 10.1% | 13.5% |
| RIE | 1.22 | 1.31 | 9.7% | 13.3% |
| RIE, exponential weights, half-life 252 | 1.28 | 1.29 | 10.1% | 13.0% |
| RIE, market volatility factor | 1.30 | 1.35 | 9.8% | 13.5% |
| Oracle (look-ahead) | 0.99 | 0.98 | 8.4% | 9.2% |
| Equal weight | 1.01 | 0.68 | 18.2% | 19.8% |

Three patterns in that table are the whole story:

- **Honesty moves a lot, realised risk moves little.** From the sample covariance to the RIE, honesty improves by a third, but real-market realised risk only falls from 14.8% to 13.3%. Cleaning mostly fixes the *prediction*; it changes the weights much less. The literature reports the same: eigenvalue cleaning cuts realised volatility by 5–12% over the sample covariance at our N/T, and the best cleaners differ by 1–3%.
- **Every estimator pays roughly the same real-market penalty in risk,** 24–37% above its Gaussian value. A penalty that is the same for every eigenvalue map is not something an eigenvalue map can remove.
- **The oracle is honest on both panels** and realises 9.2% on the real market. A covariance that knows the long-run average structure would beat every window estimate by four points. Part of that is look-ahead, since two thirds of the panel lies in the test period, but it shows the target is not unreachable in principle.

## 2. Why each method stops where it stops

| Method | What it fixes | Where it stops, in our runs | What the literature says |
|---|---|---|---|
| Marchenko-Pastur clipping | Flattens the noise bulk of eigenvalues | Fitted edge: honesty 1.47, keeps ~100 noisy "signal" modes; unit edge: buries real modes, permanent 13% bias | Gets the middle eigenvalues roughly right but leaves the large ones unshrunk and misses the curved shape of the optimal correction (Bun, Bouchaud, Potters 2017). On US data it trails the RIE by only 1–3% of realised volatility. |
| Ledoit-Wolf linear shrinkage | Pulls all eigenvalues toward their mean by one factor | Honesty 1.55; on the real market fat tails raise its shrinkage, which is why it looks more honest there than on Gaussian data | One number cannot match a curved optimal map for any intensity (Bun et al. 2017). Nonlinear shrinkage beats it by 2–5% of realised volatility, significant for N ≥ 50 (Ledoit, Wolf 2017). |
| Optimal RIE (Ledoit-Péché, inverse-Wishart lift) | The best possible eigenvalue for each sample direction | Gaussian 1.22 vs oracle 0.99; real 1.31 | Optimal only within "keep the sample eigenvectors" and only for a stationary process (Ledoit, Péché 2011; Bongiorno, Challet 2021). Even its exact oracle has a loss that does not shrink at fixed N/T (Ledoit, Wolf 2018). Ledoit and Wolf say they keep the eigenvectors "only because we do not know how to improve upon them". |
| Exponential weighting | Lets old days fade, to follow the market | Half-life 252 gains 0.3 points of risk; shorter half-lives lose badly | Same optimum in Pafka, Potters and Kondor (2004): about a year of memory, and exponential beats flat "only slightly". Short memories without cleaning are catastrophic because N/T_eff explodes; RiskMetrics-style covariances are the worst estimators in large minimum-variance tests (Engle, Ledoit, Wolf 2019). |
| Per-stock volatility standardisation | Follows each stock's calm and storm | Harmful: honesty 2.49 with a 20-day half-life | Published devolatilisation that helps uses GARCH(1,1) with long memory, or the whole-window standard deviation, not a 20-day EWMA (Hediger, Näf, Wolf 2022; Engle et al. 2019). No verified study tests per-stock forecasts on honesty. Our result that damage grows with the square of the forecast noise explains why the noisier forecaster loses. |
| Market volatility factor | One pooled storm gauge per day | Keeps honesty and risk, cuts leverage 6.9 → 4.6 and turnover 1.34 → 0.81 | The Bun-Bouchaud-Potters recipe. In their US test, cross-sectional standardisation leaves the RIE's realised volatility at 10.4% against 10.6% for clipping and 10.5% for Ledoit-Wolf: the cleaners barely differ once volatility is taken out. |
| Trust dial on each stock's own volatility | Believes 41% of each stock's own deviation | Every setting above zero makes the optimiser less honest | Consistent with the general result: the optimiser amplifies the noisiest input (Michaud 1989), and a forecast that is right on average is still a bad input if it is noisy. |

## 3. The three layers in the literature

### 3.1 Layer 1, eigenvalue noise: solved, and we match the theory

Pafka and Kondor (2003) derive that for the sample covariance the predicted risk of the minimum-variance portfolio is too low by √(1 − N/T) and its true risk too high by 1/√(1 − N/T), so honesty tends to 1/(1 − N/T). Their simulations at N = 500 hit the law exactly (T = 3000: 1.20; T = 1500: 1.49; T = 750: 3.00), and adding a market mode 25 times the noise edge does not change it. Our sample covariance follows the law at every window (11.13 against 11.36 at T = 500; 1.83 against 1.84 at T = 1000). Equal weights, which do not depend on the data, are honest at any N, so the amplification is entirely the optimiser's doing.

### 3.2 Layer 2, eigenvector noise: a proven floor for every method we built

Ledoit and Péché (2011) show the best estimator that keeps the sample eigenvectors sets each eigenvalue to the true variance along that sample direction, uᵢᵀ Σ uᵢ, which is what our RIE computes. That estimator still cannot converge to the truth at finite T/N; the best it can do is the projection of the truth onto the sample directions. Even that exact oracle keeps a loss of constant order, positive unless the truth is a multiple of the identity (Ledoit and Wolf 2018, as cited by Lin and Pan 2024). In plain words: once the eigenvalues are as good as they can be, the sample directions are the bottleneck.

Two facts make this worse for minimum variance specifically. The weights are w ∝ Σ⁻¹1 = Σᵢ vᵢ vᵢᵀ 1 / λᵢ, dominated by the smallest eigenvalues, and the small-eigenvalue sample eigenvectors are the least accurate. In their k-BAHC paper, Bongiorno and Challet (2020) find that the small-eigenvalue eigenvectors of the sample covariance are no more localised than those of shuffled returns. Goldberg, Papanicolaou and Shkolnik (2018) show the same mechanism at the top of the spectrum: the leading sample eigenvector is biased in a known direction, and for minimum variance it is this eigenvector error, not the eigenvalues, that drives the weight errors and the optimistic risk forecast. Our eigenvector study showed the same thing: modes beyond twenty overlap with their own value a month later at 0.25, and with the value two years later at the level of two random vectors.

### 3.3 Layer 3, the market changes: what the literature says about time order

The cleanest test comes from Pafka and Kondor (2003). On 400 US stocks they measured a sample-covariance honesty of 1.36 ± 0.07 where Gaussian noise predicts 1.19. When they put odd days in the estimation half and even days in the test half, which keeps every real return but destroys the time order, the ratio fell to 1.22 ± 0.06, back to the noise level. Bongiorno and Challet (2021, "Non-linear shrinkage of the price return covariance matrix is far from optimal") found the same with a shuffle: on real US data the gap between the eigenvalues nonlinear shrinkage aims for and the ones that minimise realised risk persists as the window grows, and shrinks substantially once days are shuffled so the two windows share one distribution. Bongiorno, Challet and Loeper (2021) measured it directly: consecutive real windows' eigenvectors overlap *less* than shuffled windows', so every constant-covariance method overestimates how well past directions carry forward.

The literature reads these tests as "time order, not fat tails, explains the real-market excess". That reading is partly the authors' inference: the alternate-day split also leaks clustered volatility between the two halves. Our own shuffle test (section 4) agrees that time order costs a lot of realised risk, but finds fat tails costing as much.

Two consequences matter for us. A longer window cannot fix this, because it averages over more structural change. Our own window sweep shows it: on the real market clipping's realised risk is lowest near T ≈ 500 and rises for longer windows, while in the Gaussian world it keeps falling. And Jagannathan and Ma (2003) already found that in-sample optimism persists for factor models and shrinkage estimators alike, concluding that sampling error alone cannot explain the gap.

## 4. Our own shuffle test

`examples/sp500_shuffle_test.py`, outputs in `output/sp500_shuffle/`. Two new panels, each run through the same backtest (T = 1000, 93 holds):

- **Real days, shuffled.** The 2,971 real sessions in a random order, three different orders. Each day keeps its 456 real returns, so fat tails, company jumps and the real cross-section survive, but a window and the month after it now come from one fixed distribution: no calm and stormy spells, no drift. Its exact truth is the raw full-sample covariance.
- **Gaussian, raw truth.** Gaussian draws from that raw covariance, two draws. The saved Gaussian panel was drawn from the *cleaned* covariance, which has a smoother spectrum, so this control checks whether a harder truth, rather than fat tails, explains the difference.

Realised risk of the RIE portfolio, one cause at a time (`risk_staircase.png`):

| Step | Realised volatility | Added |
|---|---|---|
| Best possible: the true covariance, known in advance | 8.25% | |
| Estimation noise on Gaussian days (raw truth) | 9.77% | +18% |
| Fat-tailed real days, order shuffled | 11.27% | +15% |
| Calm and stormy spells (estimated from the clocked panel) | 12.66% | +12% |
| Drift in the structure (the rest) | 13.30% | +5% |

The harder raw truth costs almost nothing (9.69% → 9.77%, honesty 1.22 → 1.25), so the cleaned truth was a fair stand-in. The best possible risk is 8.25% in both the raw Gaussian and the shuffled world: fat tails do not make the market harder to diversify, they make it harder to *measure*. A few extreme days dominate a 1000-day sum of squares, so the window holds fewer effective observations than its length says. Bun, Bouchaud and Potters found the same thing from the other side: on real data the RIE matches the empirical oracle only with an effective q 10–20% larger than N/T.

Honesty of the RIE (median of realised over predicted, and the ratio of mean realised to mean predicted risk):

| Panel | Median | Mean / mean |
|---|---|---|
| Gaussian | 1.22 | 1.22 |
| Gaussian, raw truth | 1.25 | 1.25 |
| Real days, shuffled | 1.33 | 1.41 |
| Real market | 1.31 | 1.51 |

On the median, shuffling the days leaves the honesty gap where it is: on our panel the excess dishonesty comes from fat tails, not from time order. The mean tells a little more: stormy holds, where the model under-predicts most, add a further 7% that the median does not see. This is a partial disagreement with the literature's reading of Pafka and Kondor's alternate-day test (section 3.3). Their test used the sample covariance at N/T = 1/6 on 1991–1996 data; ours uses the RIE at N/T = 0.46 on 2015–2026 data. For the sample covariance our split is fat tails +7% and order +2% in honesty, so the regimes differ less than the conclusions do. The safe statement is: for honesty, fat tails matter at least as much as time order on our panel; for realised risk, the two cost about the same.

The same split holds for every estimator, within a few percent: fat tails add 9–16% to realised risk and time order 13–18%, from the sample covariance to the RIE. The look-ahead oracle loses only 6% to time order (8.68% shuffled against 9.20% real), much less than the window estimators' 18%: a long-run average matrix carries through the drift better than a 1000-day window that over-weights its own stormy weeks.

**What this means.** The real market's extra cost is not one thing. Roughly one third is estimation noise that no eigenvalue map removes, one third is fat tails shrinking the effective sample, and one third is time: mostly calm and stormy spells, a little structural drift. Every tool we built aimed at eigenvalues (all of them) or at time (exponential weighting, standardisation). None aimed at the eigenvectors or at the fat tails, and those are two of the three pieces.

## 5. What the literature reports as reducing the gap, and how much

All numbers are realised annualised volatility of minimum-variance portfolios on US stocks. None of these studies reports honesty; only Pafka and Kondor do, and only for the sample covariance.

| Approach | What it changes | Reported gain | Caveats |
|---|---|---|---|
| One-factor preconditioning, then nonlinear shrinkage (NL-SF) | Eigenvectors, through a market-factor prior | N = 500, T = 250: 8.61% against 9.65% for nonlinear shrinkage alone, and 9.25% for POET (Ledoit, Wolf 2017) | Shown at N/T = 2, not tested for significance, data end in 2011, lead disappears long-only |
| BAHC and k-BAHC (bootstrapped hierarchical clustering) | Eigenvectors, especially the small-eigenvalue ones | Beats cross-validated eigenvalues only for windows under ~200–300 days with n = 100; at each method's best window the margin is under 1% | All evidence from the authors; with N = 456 that regime means windows far shorter than ours |
| Kendall-tau correlation with cross-validated eigenvalues | Eigenvectors, through a rank measure robust to fat tails | ~500 US stocks, T = 1000: about 15% lower risk than the RIE (Espana, Le Coz, Smerlak 2024) | Not verified in this pass. Advantage unclear at N/T = 0.2; gain partly lost when rescaled by sample volatilities; no turnover or costs reported |
| Tyler's robust estimator plus nonlinear shrinkage, on returns divided by whole-window volatility | Separates volatility from correlation, robust to tails | N = 1000, T = 252: 8.23% → 7.04% (Hediger, Näf, Wolf 2022) | Not verified in this pass. Gain mostly from the two-step scaling, not the robust step; robust step helps only at small N |
| DCC with nonlinear shrinkage | GARCH devolatilisation per stock, then shrinkage of the correlation | N = 1000, T = 1250: 8.16% against 13.54% sample and 8.81% plain nonlinear shrinkage; at N = 500: 9.55% against 9.75% (Zhao, Ledoit, Jiang 2023) | Not verified in this pass. Almost all the gain comes from shrinking the correlation of devolatilised returns: the constant-correlation version is within 0.1 points (Engle, Ledoit, Wolf 2019). Uses GARCH(1,1), long memory |
| Long-only constraint | Caps leverage, acts as implicit shrinkage | Helps the raw sample covariance a lot; makes every good cleaner perform like the sample covariance; raises the best cleaners' risk by 1–2 points (Pantaleo et al. 2011; Ledoit, Wolf 2017; Bongiorno, Manolakis, Mantegna 2026; also Jagannathan, Ma 2003; Tan, Zohren 2020; Dom, Jansen, Lohre 2025) | Verified across four sources. Constraints and shrinkage are substitutes, not additive. Moves the noise divergence from N/T = 1 to N/T = 2 but does not remove it (Kondor, Papp, Caccioli 2017) |
| Gross-exposure cap, about 130/30 to 150/50 | Bounds leverage without forcing long-only | U-shaped risk in the cap, minimum near 150/50 for 500 stocks; on a well-shrunk estimator the gain is small or negative (Zhao, Ledoit, Jiang 2023) | Not verified in this pass. Bounds the honesty gap by the cap squared times the largest entry error (Fan, Zhang, Yu 2012) |

Refuted in verification, so not used: that nonlinear shrinkage is near-optimal only with huge drift-free windows; and a full-universe k-BAHC table of Sharpe ratios and volatilities.

## 6. What this means for the build order

- **Another eigenvalue map will not help.** Swapping nonlinear shrinkage for cross-validated eigenvalues moves realised risk by 2–5% in published tests (Tan, Zohren 2020). We are already at the end of that road.
- **Only methods that change eigenvectors can get under layer 2.** The documented candidates are a factor prior (NL-SF, the closest relative of the market factor we already have), BAHC, and rank correlations. Of these, BAHC's published edge lives at windows much shorter than ours, while the factor prior's edge was shown at N/T = 2.
- **Fat tails are a third of the real-market cost, and nothing we built touches them.** The method aimed at exactly this, at exactly our size, is the Kendall-tau correlation with cross-validated eigenvalues: about 500 US stocks, T = 1000, about 15% lower realised risk than the RIE (Espana, Le Coz, Smerlak 2024). *[Update 2026-11-09: tested on our panel; it did not transfer (+2.5% risk against the RIE). See `fixing_the_directions.md` §7.3.]* It also changes the eigenvectors, so it attacks two of the three pieces at once. It did not survive this pass's verification for lack of a second source, so it should be tested, not trusted. Tyler's estimator helps less at large N (Hediger, Näf, Wolf 2022).
- **Layer 3 can only be reduced by modelling time, and the evidence says gently.** Long memory beats short: a one-year half-life, GARCH(1,1) rather than a 20-day EWMA, the market factor rather than per-stock forecasts.
- **Long-only will make the cleaners look alike.** Every study agrees on that. It will lower leverage and turnover, and probably raise realised risk relative to the best unconstrained portfolio. A cap near 150/50 is the published sweet spot.
- **Our honesty measure is almost unbenchmarked.** Every modern comparison reports only realised volatility. Our finding that cleaning fixes the prediction much more than the weights is visible only because we measure both.

## 7. Evidence quality

- **Checked by three independent agents (23 claims confirmed):** Pafka and Kondor 2003; Ledoit and Péché 2011; Lin and Pan 2024; Ledoit and Wolf 2017; Pantaleo et al. 2011; Bongiorno and Challet's BAHC (2021), k-BAHC (2020) and "far from optimal" (2021) papers; Bongiorno, Challet and Loeper 2021; Bongiorno, Manolakis and Mantegna 2026. The dispersion-bias result of Goldberg, Papanicolaou and Shkolnik was checked in the first, interrupted pass.
- **Read from the paper by one agent, not re-checked:** the numbers from Bun, Bouchaud and Potters 2017, Pafka, Potters and Kondor 2004, Kondor and coauthors 2007 and 2017, Jagannathan and Ma 2003, Fan, Zhang and Yu 2012, Engle, Ledoit and Wolf 2019, Zhao, Ledoit and Jiang 2023, Tan and Zohren 2020, Dom, Jansen and Lohre 2025, Hediger, Näf and Wolf 2022, Espana, Le Coz and Smerlak 2024, and Michaud 1989. Each came with a quote from the source, but nothing confirms the number independently.
- **Refuted, not used:** that nonlinear shrinkage is near-optimal only with huge drift-free windows (1–2), and a full-universe k-BAHC table of Sharpe ratios and volatilities (0–3).
- **High confidence:** the 1/(1 − N/T) law; the eigenvector floor of every rotationally invariant estimator; the shuffle and alternate-day tests; long-only collapsing the differences between cleaners. Each rests on two or more independent sources.
- **Medium:** the factor-prior and BAHC gains (one group each, few significance tests), and the finding that the eigenvalues nonlinear shrinkage aims for are not the risk-minimising ones under non-stationarity.
- **Thin or missing:** exponential weighting with RMT cleaning beyond Pafka, Potters and Kondor; per-stock volatility forecasts as optimiser inputs; Kendall and Tyler with RMT cleaning; isotonic cross-validation; resampled portfolios. Nothing found contradicts our own results on these; there is simply little independent evidence.
- **Test designs differ from ours.** Most studies use random subsets of 50–300 stocks, holds of 5–60 days, sample periods ending 2011–2024, and no transaction costs. BAHC, the Average Oracle and the learned cleaner all come from one research group; the nonlinear-shrinkage numbers come from Ledoit and Wolf. Only Pafka and Kondor report realised over predicted risk.
- **Our shuffle test** has three shuffles and two Gaussian draws. The honesty ratio moves by about 0.03 between shuffles, so differences under about 3% are noise. The clustering share in the staircase borrows the clocked panel's ratio, so the last two steps are an estimate.

## 8. Sources

- Michaud (1989), The Markowitz optimization enigma: is "optimized" optimal? *Financial Analysts Journal*. https://rpc.cfainstitute.org/research/financial-analysts-journal/1989/the-markowitz-optimization-enigma-is-optimized-optimal
- Pafka, Kondor (2002), Noisy covariance matrices and portfolio optimization, *Eur. Phys. J. B* 27. https://arxiv.org/abs/cond-mat/0111503
- Pafka, Kondor (2003), Noisy covariance matrices and portfolio optimization II, *Physica A* 319. https://arxiv.org/abs/cond-mat/0205119
- Pafka, Potters, Kondor (2004), Exponential weighting and random-matrix-theory-based filtering of financial covariance matrices for portfolio optimization. https://arxiv.org/abs/cond-mat/0402573
- Kondor, Pafka, Nagy (2007), Noise sensitivity of portfolio selection under various risk measures, *J. Banking & Finance* 31. https://arxiv.org/abs/physics/0611027
- Kondor, Papp, Caccioli (2017), Analytic solution to variance optimization with no short-selling, *J. Stat. Mech.* https://arxiv.org/abs/1612.07067
- Ledoit, Péché (2011), Eigenvectors of some large sample covariance matrix ensembles, *Probab. Theory Relat. Fields* 151. https://arxiv.org/abs/0911.3010
- Bun, Bouchaud, Potters (2017), Cleaning large correlation matrices: tools from random matrix theory, *Physics Reports* 666. https://arxiv.org/abs/1610.08104
- Ledoit, Wolf (2017), Nonlinear shrinkage of the covariance matrix for portfolio selection: Markowitz meets Goldilocks, *Rev. Financial Studies* 30. https://papers.ssrn.com/abstract=2383361
- Goldberg, Papanicolaou, Shkolnik (2018), The dispersion bias, later *SIAM J. Financial Math.* https://arxiv.org/abs/1711.05360
- Lin, Pan (2024), Eigenvector overlaps in large sample covariance matrices and nonlinear shrinkage estimators, *Annals of Statistics*. https://arxiv.org/abs/2404.18173
- Engle, Ledoit, Wolf (2019), Large dynamic covariance matrices, *J. Business & Economic Statistics* 37. https://www.econ.uzh.ch/dam/jcr:28fa9939-753e-4f5c-932d-945872f30cfd/jbes_2019.pdf
- Zhao, Ledoit, Jiang (2023), Risk reduction and efficiency increase in large portfolios: gross-exposure constraints and shrinkage of the covariance matrix, *J. Financial Econometrics* 21(1), 73–105. https://academic.oup.com/jfec/article/21/1/73/6151211
- Jagannathan, Ma (2003), Risk reduction in large portfolios: why imposing the wrong constraints helps, *J. Finance* 58. https://www.nber.org/papers/w8922.pdf
- Fan, Zhang, Yu (2012), Vast portfolio selection with gross-exposure constraints, *J. American Statistical Association*. https://pmc.ncbi.nlm.nih.gov/articles/PMC3535429
- Pantaleo, Tumminello, Lillo, Mantegna (2011), When do improved covariance matrix estimators enhance portfolio optimization? An empirical comparative study of nine estimators, *Quantitative Finance*. https://arxiv.org/abs/1004.4272
- Bongiorno, Challet (2021), Covariance matrix filtering with bootstrapped hierarchies, *PLoS ONE*. https://pmc.ncbi.nlm.nih.gov/articles/PMC7808632 ; https://arxiv.org/abs/2003.05807
- Bongiorno, Challet (2020), Reactive global minimum variance portfolios with k-BAHC covariance cleaning. https://arxiv.org/abs/2005.08703
- Bongiorno, Challet (2021), Non-linear shrinkage of the price return covariance matrix is far from optimal for portfolio optimisation. https://arxiv.org/abs/2112.07521
- Bongiorno, Challet, Loeper (2021), Cleaning the covariance matrix of strongly nonstationary systems with time-independent eigenvalues. https://arxiv.org/abs/2111.13109
- Bongiorno, Manolakis, Mantegna (2026), End-to-end large portfolio optimization for variance minimization with neural networks through covariance cleaning, *J. Finance and Data Science* 12. https://arxiv.org/abs/2507.01918
- Tan, Zohren (2020), Estimation of large financial covariances: a cross-validation approach. https://arxiv.org/abs/2012.05757
- Dom, Jansen, Lohre (2025), Beyond GMV: the relevance of covariance matrix estimation for risk-based portfolio construction, *Quantitative Finance* 25(3). https://eprints.lancs.ac.uk/228677/1/beyondgmv_nonblind_main.pdf
- Espana, Le Coz, Smerlak (2024), Kendall correlation coefficients for portfolio optimization. https://arxiv.org/abs/2410.17366
- Hediger, Näf, Wolf (2022), R-NL: covariance matrix estimation for elliptical distributions based on nonlinear shrinkage, *IEEE Trans. Signal Processing*. https://arxiv.org/abs/2210.14854
