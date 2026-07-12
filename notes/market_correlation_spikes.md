# Market Correlation Spikes
As we have observed the synthetic universe we are provided has sporadic spikes of market wide correlation (averages exceeding over 0.7). I consider these spikes to occur for correlation matrix averages above 0.5. Given the frequency and consistency of these spikes I believe finding a correlated (or anti correlated) relationship between market correlation and another metric to be a possible strategy backbones).

Note that we can also consider the market return average as possibly tangential metric.

# Possible Exploitations
If we can determine a directional relationship between a given signal and the average market correlation then a possible exploitation is investing as much as possible is a number of stocks (as reduce the chance of concentrating investment in a handful of market anti-correlated stocks (which are observed). The consistency of the underlying signal used to predict high mean market correlation will determine how aggressively we invest.

# Possible Metric to Measure Correlation Against
We'd like to focus on lead-lag relationships between different metrics and the average market correlation and average log market return. However, no lagging correlations can also be beneficial as the market correlation spike does occur over multiple days.

Additionally, if a consistent leading metric can be found in advance via excessive brute force checking then we can hard code said metric to be an indicator, we can then confirm the consistency of this metric against the unseen data.

Possible metrics :
- Individual asset returns
- Market average returns
- Individual asset moving average returns
- Market moving average returns
- Individual asset volatility
- Market average volatility

Given our uncertainty in this relationship structure, we'd like to investigate both linear and non-linear lead-lag relationships using both the C1 and C2 metrics for lead-lag signals.

# Steps

We'd like to first implement functions to return the metrics we are interested in investigating. We'd like to then implement a general function to detect lead-lag relationships between two arbitrary 1-dimensional numpy arrays of equal size. We then perform brute force analysis.
