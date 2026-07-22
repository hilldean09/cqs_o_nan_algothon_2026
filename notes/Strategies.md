## Moving Average Crossover

### Description
This strategy compares a short-term moving average with a long-term moving average.

- Buy when the short-term moving average crosses above the long-term moving average.
- Sell or short when the short-term moving average crosses below the long-term moving average.
- Close or reduce the position when the moving averages cross in the opposite direction.

### Pros
- Easy to understand
- Easy to code and backtest
- Can identify sustained price trends
- Uses clear entry and exit rules
- Can be applied to all assets

### Cons
- Performs poorly in sideways markets
- Signals usually occur after the price has already started moving
- Can create frequent false buy and sell signals
- Results depend heavily on the selected moving-average periods
- May react too slowly to sudden market reversals

---

## Time-Series Momentum

### Description
This strategy assumes that an asset which has recently been rising may continue rising, while an asset which has recently been falling may continue falling.

- Calculate each asset’s return over a chosen lookback period.
- Buy assets with strong positive momentum.
- Short assets with strong negative momentum.
- Hold no position when momentum is weak or close to zero.

### Pros
- Simple to understand and implement
- Can capture large and sustained trends
- Works for both long and short positions
- Can be applied across all assets
- Easy to test using different lookback periods
- Can be combined with volatility-based position sizing

### Cons
- Performs poorly when prices move sideways
- Can enter a trade just before the trend reverses
- May react slowly to new market conditions
- Performance depends on the chosen lookback period
- Can produce several losses during choppy markets

---

## Mean Reversion

### Description
This strategy assumes that when an asset moves unusually far away from its recent average, it may eventually return toward that average.

- Calculate a rolling average price.
- Measure how far the current price is from that average.
- Buy when the price is unusually far below the average.
- Short when the price is unusually far above the average.
- Close the position when the price moves back toward the average.

### Pros
- Can perform well in sideways or stable markets
- Provides clear entry and exit rules
- Can generate frequent trading opportunities
- Can complement momentum strategies
- Can be applied using prices, returns or spreads

### Cons
- A price may move away from its average because a genuine new trend has started
- Can continue buying an asset while it keeps falling
- Can continue shorting an asset while it keeps rising
- Sensitive to the selected rolling-window length
- Can suffer large losses during strong trends
- The historical average may stop being relevant

---

## Cross-Sectional Momentum

### Description
This strategy compares the performance of all assets against each other.

- Calculate momentum for every asset.
- Rank the assets from strongest to weakest.
- Buy the strongest-performing assets.
- Short the weakest-performing assets.
- Rebalance the portfolio regularly.

### Pros
- Uses information from the entire group of assets
- Can create a diversified portfolio
- Can remain approximately market-neutral
- Does not rely on the whole market moving in one direction
- May produce more stable returns than trading one asset
- Can work well with the competition’s Sharpe-based scoring system

### Cons
- Rankings may change frequently
- Strong recent performers may suddenly reverse
- Weak recent performers may suddenly recover
- Equal position sizes may give too much weight to volatile assets
- Performance depends on the number of assets selected
- Assets may be highly correlated, reducing diversification

---

## Pairs Trading

### Description
This strategy searches for two assets that usually move together.

- Identify pairs with a stable historical relationship.
- Measure the difference or spread between their prices.
- Buy the relatively undervalued asset.
- Short the relatively overvalued asset.
- Close both positions when the relationship returns toward normal.

### Pros
- Can be approximately market-neutral
- Does not require predicting whether the entire market will rise or fall
- Can produce stable returns when genuine relationships exist
- Can work in rising, falling or sideways markets
- Can diversify risk across several independent pairs

### Cons
- Two assets may stop moving together
- A temporary difference may become permanent
- Correlation alone does not prove a reliable relationship
- Requires more statistical testing than simple strategies
- Several pairs may create hidden concentration in the same asset
- Can be sensitive to the selected lookback period

---

## Principal Component Analysis Residual Strategy

### Description
This strategy identifies common movements across many assets and then trades movements that cannot be explained by those common factors.

- Use principal component analysis to identify major market factors.
- Estimate how strongly each asset is affected by those factors.
- Calculate the unexplained movement, called the residual.
- Buy assets with unusually negative residuals.
- Short assets with unusually positive residuals.
- Close positions when residuals return toward normal.

### Pros
- Uses information from the entire dataset
- Can remove broad market movements
- May identify hidden relationships between anonymous assets
- Can create a diversified and approximately market-neutral portfolio
- More flexible than simple pairs trading
- May detect relative mispricing across many assets

### Cons
- More difficult to understand and implement
- Principal components may not have a clear economic meaning
- Relationships may change over time
- Highly vulnerable to overfitting
- Requires enough historical data
- Results depend on the number of components used
- Can accidentally use future information if not implemented carefully

---

## Volatility Breakout

### Description
This strategy trades when an asset moves outside its recent normal price range.

- Calculate the recent highest and lowest prices.
- Buy when the price breaks above the recent high.
- Short when the price breaks below the recent low.
- Exit when the breakout fails or the trend reverses.
- A trailing stop can be used to protect profits.

### Pros
- Can capture large price moves early
- Uses clear and objective trading rules
- Can perform well when quiet markets are followed by strong trends
- Can produce large gains from successful trades
- Easy to combine with volatility-based position sizing

### Cons
- False breakouts are common
- Can produce many small losses
- Performs poorly in sideways markets
- Daily profit and loss may be inconsistent
- Sudden reversals can cause large losses
- Results depend heavily on the breakout window

---

## ALGO Lead-Lag Strategy

### Description
This strategy tests whether movements in ALGO predict future movements in the other assets, or whether movements in the other assets predict ALGO.

- Measure the relationship between today’s ALGO return and future returns of other assets.
- Test whether other assets react to ALGO with a delay.
- Buy or short assets based on lagged ALGO signals.
- Test whether a group of other assets can predict ALGO.
- Treat ALGO separately because it has a larger position limit.

### Pros
- Takes advantage of the competition’s unusual ALGO structure
- May uncover an intentionally designed relationship in the data
- Can generate signals across several assets
- May help identify whether ALGO acts like a market index
- Could be highly profitable if the lead-lag relationship is strong

### Cons
- The larger ALGO position limit does not guarantee predictability
- Testing many assets and time lags creates overfitting risk
- Relationships may disappear in later datasets
- Large ALGO positions may create unstable daily profit and loss
- A few extreme ALGO movements could dominate results

---

## Multi-Strategy Portfolio

### Description
This approach combines several trading strategies instead of relying on only one.

- Generate signals from momentum, mean reversion, pairs trading and other strategies.
- Assign a weight to each strategy.
- Combine the signals into one final position.
- Reduce the weight of strategies that become highly correlated or perform poorly.
- Use portfolio-level risk controls.

### Pros
- Reduces dependence on one market condition
- Can combine strategies that perform well at different times
- May create more stable daily profit and loss
- Can improve diversification
- May produce a higher Sharpe ratio
- Allows weaker but uncorrelated strategies to add value

### Cons
- More complicated to build and manage
- Poorly chosen strategies can cancel each other out
- Weight selection can lead to overfitting
- Correlations between strategies may increase during stressful periods
- Harder to identify which strategy caused profits or losses
- Requires stronger portfolio-level risk controls
