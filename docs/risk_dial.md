# The customer's risk dial

*Written 2026-11-17. Code: `src/market_analyzer/risk_dial.py`; tests `tests/test_risk_dial.py`; study script `examples/sp500_risk_dial_study.py`; results in `output/sp500_risk_dial/`. Written for a reader new to portfolio management: every term is explained where it first appears.*

## 1. What this piece is for

Everything built before this answered one question: **which stocks**, and in what proportions, make the calmest basket. That answer is the same for every customer.

The dial answers the second question: **how much** of a customer's money goes into that basket, and how much stays in cash. This is where the customer's own goal enters. A cautious customer and a bold customer hold the same basket. They hold different amounts of it.

```
 prices ──> risk model ──> optimiser ──> stock basket ──┐
                                                        ├──> the customer's portfolio
              customer's risk target ──> THE DIAL ──────┘      (part basket, part cash)
```

## 2. Word list

| Word | Meaning |
|---|---|
| **Volatility** | How much a portfolio's value swings, written as a yearly percentage. A portfolio with 8% volatility typically ends a year within about 8% of where it was expected to end, and now and then twice that. |
| **Basket** | The mix of stocks the optimiser chose. |
| **Cash** | Money kept out of the stock market. It does not swing. In this test it earns nothing. |
| **Target** | The volatility the customer asks for. This is their risk goal. |
| **Forecast** | The dial's guess of how much the basket will swing next. |
| **Exposure** | The share of the customer's money that is in the basket. |
| **Rebalance** | To trade the portfolio back to the intended mix. |
| **Long-only** | A basket that only buys stocks and never bets against any. |
| **Half-life** | For an average that counts recent days more: the age at which a day counts half as much as yesterday. |

## 3. How the dial works

One line of arithmetic:

```
share in the basket = target volatility ÷ forecast volatility      (never above 100%)
```

If the basket is expected to swing 16% a year and the customer wants 8%, half the money goes in the basket and half stays in cash. Half of a 16% swing is 8%.

The arithmetic is exact. So the dial is precisely as good as its **forecast**, and the whole study is about which forecast to trust. Three were tried:

| Forecast | What it looks at | Character |
|---|---|---|
| **Risk model** | The cleaned covariance matrix that chose the basket, built from the last 1000 trading days, about four years | Slow and steady |
| **Recent behaviour** | How this exact basket would have moved lately. Either the last quarter with every day equal, or the last year with recent days counting more (half-life 20 days) | Fast |
| **Higher of the two** | Whichever of the above is larger, so the dial only ever leans cautious | Slow in calm times, fast in a storm |

The second choice is **how often the dial is reset**: monthly together with the stocks, weekly, or daily. Resetting the dial is cheap compared with reshuffling stocks. It is one trade that scales the whole basket up or down.

## 4. The test

The promise to check is simple: *the customer asked for X% swings. What did they get?*

- **Baskets:** the ones saved by the gross-cap study. 456 S&P 500 stocks, rebuilt every 21 trading days, from December 2018 to October 2026. This stretch holds the 2020 crash, the 2022 bear market and the April 2025 shock.
- **Score:** volatility delivered ÷ volatility asked for. **1.00 is a promise kept.** 1.50 means the customer felt one and a half times the swings they signed up for.
- **Control:** the same dials run in a lab world, made of random days drawn from one fixed covariance matrix. Risk never changes there. If the dial fails in the lab, the arithmetic is wrong. If it fails only in the real market, the market is the cause.
- **Point in time:** a split that applies on a given day uses only returns from before that day. A unit test enforces this.

## 5. Results

### 5.1 Which forecast keeps the promise

Long-only cleaned basket, real market, no limit on the basket share so the forecast is judged alone.

| Dial | Whole history | Calmest year | Wildest year (2020) | Dial trades per year |
|---|---|---|---|---|
| Risk model, reset monthly | **1.55** | 0.73 | **3.49** | 0.2 |
| Last quarter, reset monthly | 1.53 | 0.98 | 3.13 | 1.4 |
| Last quarter, reset weekly | 1.16 | 0.96 | 1.62 | 2.4 |
| Recent-weighted, reset monthly | 1.39 | 0.99 | 2.62 | 1.4 |
| Recent-weighted, reset weekly | 1.10 | 0.95 | 1.35 | 2.9 |
| Recent-weighted, reset daily | 1.08 | 0.94 | 1.23 | 4.8 |
| **Higher of the two, reset weekly** | **0.96** | 0.73 | **1.32** | 1.3 |
| Higher of the two, reset daily | 0.94 | 0.73 | 1.19 | 1.8 |

"Dial trades per year" counts how many times the dial alone turns over the stock holding in a year. For scale, reshuffling the stocks inside the basket trades 1.6 times the holding a year.

Figure: `output/sp500_risk_dial/promise.png`.

### 5.2 The same dials in the lab

| Dial | Lab, whole history | Lab, wildest year |
|---|---|---|
| Risk model, reset monthly | 1.02 | 1.09 |
| Recent-weighted, reset weekly | 1.05 | 1.08 |
| Higher of the two, reset weekly | 0.99 | 1.04 |

When risk does not change, every dial keeps its promise to within a few percent, and the slow risk model is as good as any. The failure in 5.1 comes from the market, not from the dial.

### 5.3 What a customer would have lived through

Long-only cleaned basket, no borrowing, so at most 100% in the basket.

**Slow dial** (risk model, reset monthly):

| Asked | Got | Wildest year | Average in basket | Worst day | Largest fall |
|---|---|---|---|---|---|
| 4% | 6.2% | 13.9% | 36% | −5.6% | −20.4% |
| 8% | 11.9% | 26.3% | 72% | −10.4% | −35.3% |
| 12% | 14.1% | 28.6% | 95% | −10.4% | −35.3% |

**Fast dial** (higher of the two, reset weekly):

| Asked | Got | Wildest year | Average in basket | Lowest in basket | Worst day | Largest fall |
|---|---|---|---|---|---|---|
| 4% | 3.9% | 5.3% | 33% | 6% | −1.9% | −7.4% |
| 6% | 5.8% | 7.9% | 50% | 10% | −2.9% | −10.9% |
| 8% | 7.7% | 10.5% | 66% | 13% | −3.9% | −14.4% |
| 10% | 9.4% | 12.7% | 80% | 16% | −4.8% | −17.7% |
| 12% | 10.6% | 14.5% | 90% | 19% | −5.0% | −21.0% |

The worst day and the largest fall are **flattered**. The stock list holds only companies that survived to today, so real losses would have been somewhat larger. The comparison between the two dials still stands, because both ran on the same stocks.

Between 19 February and 23 March 2020 the basket fell 35.3%. The 8% customer on the slow dial was fully invested and lost all of that. The same customer on the fast dial lost 14.0%, because the dial had cut the basket share from 100% to 13% as the storm built.

Figure: `output/sp500_risk_dial/through_time.png`.

## 6. What we learned

1. **A risk number that is right on a typical month can still break the promise.** Earlier studies found the long-only model's forecast honest, even slightly cautious, in the median month. Yet a dial built on it delivered 1.55 times the target, because nearly all of the miss sits in a few weeks of 2020. Without 2020 the slow dial scores 0.97. Customers do not experience the median month. They experience the worst one.
2. **Speed matters more than cleverness.** Switching from the four-year model to the last quarter barely helps while the dial is still reset monthly, 1.55 against 1.53. Resetting weekly is what helps, 1.53 down to 1.16. The 2020 crash took about a month from top to bottom, so a dial that looks once a month cannot react in time whatever it looks at.
3. **Counting recent days more helps again.** With weekly resets, the recent-weighted forecast scores 1.10 against 1.16 for the plain last quarter, and 1.35 against 1.62 in the wildest year.
4. **Taking the higher of slow and fast gave the best promise, on the cautious side.** It scored 0.96 overall and traded less than half as much as the fast forecast alone. The price is that it under-delivers in calm years, about 0.75. The four-year model kept remembering the 2020 crash until 28 March 2024, the day the crash dropped out of its window, and the basket share jumped 14 points that day.
5. **The basket's own optimism leaks into the dial.** The fast forecast alone overshoots by about 5% even in the lab, and by more for baskets that short stocks: 12% for the 150/50 basket and 34% for the unconstrained one. The basket was chosen to look calm on the very history the forecast reads. This is the same honesty gap measured in the earlier studies, and the long-only basket has the least of it.
6. **No dial kept the promise inside the crash itself.** Over the worst three months the fast dial delivered 18% against the 8% asked, and the slow dial delivered 51%. A dial can only react to a storm it has already seen the start of.

## 7. Limits of this test

- **One big storm.** The gain of the fast dial comes almost entirely from 2020. In the calmer second half the slow and fast dials score about the same, 0.92 and 0.87. The mechanism is well known in the literature: volatility clusters, so recent swings predict near-term swings. Barroso and Santa-Clara (2015), covered in `docs/return_prediction_review.md`, use it to tame momentum. Our evidence agrees with it but rests on a single crash.
- **Settings chosen after seeing the data.** The 20-day half-life and the weekly reset were picked on this same history. Neighbouring settings give similar scores, which is reassuring, but it is not an independent test.
- **No trading costs yet.** The fast dial adds 1.3 turns of the holding a year on top of the basket's own 1.6. That cost is not charged here.
- **Cash earns nothing here.** The test is about swings only. The return numbers cannot be trusted on this stock list in any case, as `docs/return_signals.md` explains.
- **Volatility is not the same as loss.** A customer who asks for 8% swings can still see a fall of 14% across several bad weeks. A page that offers the dial should say so.
- **Time horizon is not in the dial.** Turning "I need this money in three years" into a target needs an assumption about expected returns, which the project does not have yet.

## 8. Using it

```python
from market_analyzer.risk_dial import DialConfig, basket_volatility, exposure_for_target, split_with_cash, dial_backtest

# A basket chosen today (weights: pd.Series, returns: date × ticker daily returns)
recent = basket_volatility(weights, returns, lookback=252, halflife=20)   # how it has been swinging
forecast = max(recent, model_volatility)                                  # the cautious "higher" rule
share = exposure_for_target(0.08, forecast)                               # customer asked for 8%
portfolio = split_with_cash(weights, share)                               # stock weights plus a CASH line

# Replaying the dial over a finished backtest
dialled = dial_backtest(result, "longonly_rie", returns, DialConfig(0.08, forecast="higher"))
dialled.summary()      # asked, delivered, average share in the basket, worst year
dialled.by_year()
```

`DialConfig` defaults to the fast settings: recent-weighted behaviour with a 20-day half-life over the last 252 days, reset every 5 days. `forecast="higher"` needs the risk model's number as well. `max_exposure=1` means never borrow. `cash_rate` sets what cash earns.

On the web page the same two calls sit behind section 4, "How much of your money goes in", and the replay under section 5. `dial_basket` and `dial_history` in `src/market_analyzer/basket.py` wrap them for a basket picked on the page; `docs/webpage.md` describes the layout.

Rerun the study with:

```
python examples/sp500_risk_dial_study.py
```

It takes a few seconds, because it reuses the saved baskets.
