import numpy as np

"""
NOTE: Variable Name Prefixes - Dean

This can help a lot sometimes:
    g_... : Denotes a global variable
    m_... : Denotes a member variable of a class
    c_... : Denotes a constant


NOTE: Style - Dean

I am currently keeping with the camel case variable
naming of the original code (e.g. fizzBizz) but with
underscores as I personally find they help 
(e.g. fizz_Buzz).

Putting spaces between parenthesises and arguments (as
seen below) can help a lot. It's that they encouraged in
the UNIX and C unit at the least 

Adding whitespace (blanks lines) between sections of code
is also incredibly helpful for readibility.


NOTE: Notes and ToDo's

You may notice I have repeatedly used "NOTE:" in these
comments. The use of "NOTE:" and "TODO:" is common 
practice in code comments to denotes important notes 
and things that need to be done. Many IDEs have functions
to automatically search for these tags so they can be very
useful.


I am mcuh more familar with C/C++ and it's 
naming/documentation conventions so please let me know if
things are different in Python.

Please do let everyone know if you have any comments or 
feedback.
"""

# Big ToDo List :
#   TODO: Write basic trading strategy (e.g. moving average
#   crossover). Focus on writing reusable functions for future
#   more competitive strategies.
#   TODO: Implement realized volatility functions. (WIP - Dean)


##### Code Start #####

g_number_Of_Instruments = 51
current_Position = np.zeros( g_number_Of_Instruments )

# Strategy Enumeration :
#   0 : main strategy (reserved)
#   1 : moving average crossover
g_strategy_Selection_Enum = 1

# NOTE: We cannot change the argument variable name from
# prcSoFar
def getMyPosition( prcSoFar ):
    global current_Position
    global g_strategy_Selection_Enum

    ( number_Of_Instruments, number_Of_Timesteps ) = prcSoFar.shape

    if( number_Of_Timesteps < 2 ):
        return np.zeros( number_Of_Instruments )

    if( g_strategy_Selection_Enum == 1 ):
        return_Position = runMovingAverageCrossoverStrategy( prcSoFar )

    current_Position = return_Position

    return current_Position 


##### Logging #####

# NOTE: Logging verbosity
#   0 : No logging
#   1 : Errors only
#   2 : Errors and warnings
#   3 : Errors, warnings, and general info

g_logging_Verbosity = 3

# Error Logging #
def logErrorHeader( function_Name_String, error_String ):
    global g_logging_Verbosity
    if( g_logging_Verbosity >= 1 ):
        print( "[O(NaN)] [Error] " + function_Name_String + " : " + error_String )

def logErrorValue( variable_Name_String, variable_Value ):
    global g_logging_Verbosity
    if( g_logging_Verbosity >= 1 ):
        print( "\t" + variable_Name_String + " : " + str( variable_Value ) )

def logErrorInfo( leading_String, tailing_String ):
    global g_logging_Verbosity
    if( g_logging_Verbosity >= 1 ):
        print( "\t" + leading_String + " : " + tailing_String )

# Warning Logging #
def logWarningHeader( function_Name_String, warning_String ):
    global g_logging_Verbosity
    if( g_logging_Verbosity >= 2 ):
        print( "[O(NaN)] [Error] " + function_Name_String + " : " + warning_String )

def logWarningValue( variable_Name_String, variable_Value ):
    global g_logging_Verbosity
    if( g_logging_Verbosity >= 2 ):
        print( "\t" + variable_Name_String + " : " + str( variable_Value ) )

def logWarningInfo( leading_String, tailing_String ):
    global g_logging_Verbosity
    if( g_logging_Verbosity >= 2 ):
        print( "\t" + leading_String + " : " + tailing_String )

# General Info Logging #
def logGeneralHeader( function_Name_String, string ):
    global g_logging_Verbosity
    if( g_logging_Verbosity >= 3 ):
        print( "[O(NaN)] [Error] " + function_Name_String + " : " + string )

def logGeneralValue( variable_Name_String, variable_Value ):
    global g_logging_Verbosity
    if( g_logging_Verbosity >= 3 ):
        print( "\t" + variable_Name_String + " : " + str( variable_Value ) )

def logGeneralInfo( leading_String, tailing_String ):
    global g_logging_Verbosity
    if( g_logging_Verbosity >= 3 ):
        print( "\t" + leading_String + " : " + tailing_String )


##### Suuporting #####

# Pearson Correlation Matrix #
def getSeriesPairPearsonCorrelationValue( first_Series, second_Series ):
    number_Of_Values = first_Series.size

    # Getting needed values
    first_Mean = np.mean( first_Series )
    second_Mean = np.mean( second_Series )

    first_Sum_Of_Squares = np.sum( first_Series ** 2 )
    second_Sum_Of_Squares = np.sum( second_Series ** 2 )

    sum_Of_Products = np.sum( first_Series * second_Series )

    correlation_Value = ( sum_Of_Products - number_Of_Values * first_Mean * second_Mean ) / ( np.sqrt( first_Sum_Of_Squares - number_Of_Values * first_Mean * first_Mean ) * np.sqrt( second_Sum_Of_Squares - number_Of_Values * second_Mean * second_Mean ) )

    #Error detection
    if( correlation_Value > 1.1 or correlation_Value < -1.1 ):
        logErrorHeader( "getSeriesPairPearsonCorrelationValue", "Correlation value outside of expected range" )
        logErrorValue( "correlation_Value", correlation_Value )

    return correlation_Value


def getAssetPairPearsonCorrelationValue( prices_So_Far, latest_Day, window_Size, first_Asset_Idx, second_Asset_Idx ):
    window_Start_Day = latest_Day - window_Size + 1

    first_Series = prices_So_Far[ first_Asset_Idx ][ window_Start_Day : latest_Day + 1 : 1 ]
    second_Series = prices_So_Far[ second_Asset_Idx ][ window_Start_Day : latest_Day + 1 : 1 ]

    pearson_Correlation_Value = getSeriesPairPearsonCorrelationValue( first_Series, second_Series )
    return pearson_Correlation_Value

def getPearsonCorrelationMatrix( prices_So_Far, desired_Latest_Day, desired_Window_Size ):
    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape
    
    latest_Day = desired_Latest_Day
    window_Size = desired_Window_Size

    # Setting latest day to most recent
    # day if inputted latest day exceeds 
    # number of available timesteps
    if( latest_Day > number_Of_Timesteps - 1 ):
        latest_Day = number_Of_Timesteps - 1

    # Setting the window to the maximum 
    # available size if the entire desired
    # window size is not available
    if( latest_Day - window_Size + 1 < 0 ):
        window_Size = latest_Day + 1

    # This can be significantly optimised
    # if need be
    correlation_Matrix = np.zeros( ( number_Of_Instruments, number_Of_Instruments ) )

    for first_Asset_Idx in range( number_Of_Instruments ):
        for second_Asset_Idx in range( number_Of_Instruments ):
            correlation_Matrix[ first_Asset_Idx ][ second_Asset_Idx ] = getAssetPairPearsonCorrelationValue( prices_So_Far, latest_Day, window_Size, first_Asset_Idx, second_Asset_Idx )

    return correlation_Matrix

def getCorrelationMatrixAndMeanCorrelation( prices_So_Far, desired_Latest_Day, desired_Window_Size ):
    correlation_Matrix = getPearsonCorrelationMatrix( prices_So_Far, desired_Latest_Day, desired_Window_Size )
    correlation_Mean = np.mean( correlation_Matrix )

    return correlation_Matrix, correlation_Mean 

# Moving Average #
def getAssetMovingAverage( prices_So_Far, asset_Idx, desired_Latest_Day, desired_Window_Size ):
    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape
    
    latest_Day = min( desired_Latest_Day, number_Of_Timesteps - 1 )

    window_Size = desired_Window_Size
    # Setting the window to the maximum 
    # available size if the entire desired
    # window size is not available
    if( latest_Day - window_Size + 1 < 0 ):
        window_Size = latest_Day + 1

    window_Start_Day = int( latest_Day - window_Size + 1 )

    moving_Average = np.mean( prices_So_Far[ asset_Idx ][ window_Start_Day : latest_Day + 1 : 1 ] )

    return moving_Average


    

# Realizaed Variance and Volatility #

def getAssetRealizedVariance( prices_So_Far, asset_Idx : int, desired_Latest_Day : int, desired_Window_Size ):
    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape
    
    latest_Day = min( desired_Latest_Day, number_Of_Timesteps - 1 )

    window_Size = desired_Window_Size
    # Setting the window to the maximum 
    # available size if the entire desired
    # window size is not available
    if( latest_Day - window_Size + 1 < 0 ):
        window_Size = latest_Day + 1

    window_Start_Day = int( latest_Day - window_Size + 1 )

    # Calculating
    series_Mean = np.mean( prices_So_Far[ asset_Idx ][ window_Start_Day : latest_Day + 1 : 1 ] )
    series_Mean_Of_Squared = np.mean( prices_So_Far[ asset_Idx ][ window_Start_Day : latest_Day + 1 : 1 ] ** 2 )

    realized_Variance = series_Mean_Of_Squared - ( series_Mean * series_Mean )

    # Returning
    if( realized_Variance < 0 ):
        logErrorHeader( "getAssetRealizedVariance", "Returning negative value" )
        logErrorValue( "realized_Variance", realized_Variance )

    return realized_Variance

def getAssetRealizedVolatility( prices_So_Far, asset_Idx, desired_Latest_Day, desired_Window_Size ):
    realized_Variance = getAssetRealizedVariance( prices_So_Far, asset_Idx, desired_Latest_Day, desired_Window_Size )

    return np.sqrt( realized_Variance )


"""
Gets the realized volatility of an asset as a
decimal relative to a moving average of the
asset
"""
def getAssetMARelativeRealizedVolatility( prices_So_Far, asset_Idx, desired_Latest_Day, moving_Average_Window_Size, realized_Volatility_Window_Size ):
    realized_Volatility = getAssetRealizedVolatility( prices_So_Far, asset_Idx, desired_Latest_Day, realized_Volatility_Window_Size )
    moving_Average = getAssetMovingAverage( prices_So_Far, asset_Idx, desired_Latest_Day, moving_Average_Window_Size )

    return ( realized_Volatility / moving_Average )

# Appreciation #
def getAssetLogMovementSeries( prices_So_Far, asset_Idx, desired_Latest_Day, desired_Window_Size ):
    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape
    
    latest_Day = min( desired_Latest_Day, number_Of_Timesteps - 1 )

    window_Size = desired_Window_Size
    # Setting the window to the maximum 
    # available size if the entire desired
    # window size is not available
    if( latest_Day - window_Size + 1 < 0 ):
        window_Size = latest_Day + 1

    window_Start_Day = int( latest_Day - window_Size + 1 )

    log_Movement_Series = np.zeros( ( number_Of_Instruments, window_Size ) )

    for start_Day_Offset in range( window_Size - 1 ):
        log_Movement_Series[ start_Day_Offset ] = np.log( prices_So_Far[ asset_Idx ][ window_Start_Day + start_Day_Offset + 1 ] / prices_So_Far[ asset_Idx ][ window_Start_Day + start_Day_Offset ] )

    return log_Movement_Series

def getAssetLogDrift( prices_So_Far, asset_Idx, desired_Latest_Day, desired_Window_Size ):
    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape
    
    latest_Day = min( desired_Latest_Day, number_Of_Timesteps - 1 )

    window_Size = desired_Window_Size
    # Setting the window to the maximum 
    # available size if the entire desired
    # window size is not available
    if( latest_Day - window_Size + 1 < 0 ):
        window_Size = latest_Day + 1

    log_Movement_Series = getAssetLogMovementSeries( prices_So_Far, asset_Idx, latest_Day, window_Size )
    log_Drift = np.mean( log_Movement_Series )

    return log_Drift

def getAssetLogMovementVolatility( prices_So_Far, asset_Idx, desired_Latest_Day, desired_Window_Size ):
    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape
    
    latest_Day = min( desired_Latest_Day, number_Of_Timesteps - 1 )

    window_Size = desired_Window_Size
    # Setting the window to the maximum 
    # available size if the entire desired
    # window size is not available
    if( latest_Day - window_Size + 1 < 0 ):
        window_Size = latest_Day + 1

    log_Movement_Series = getAssetLogMovementSeries( prices_So_Far, asset_Idx, latest_Day, window_Size )
    log_Movement_Volatility = np.sqrt( np.mean( log_Movement_Series** 2 ) - ( np.mean( log_Movement_Series ) ) ** 2 )

    return log_Movement_Volatility

def getAssetArithmeticDrift( prices_So_Far, asset_Idx, desired_Latest_Day, drift_Window_Size ):
    log_Drift = getAssetLogDrift( prices_So_Far, asset_Idx, desired_Latest_Day, drift_Window_Size )
    log_Movement_Volatility = getAssetLogMovementVolatility( prices_So_Far, asset_Idx, desired_Latest_Day, drift_Window_Size )

    arithmetic_Drift = log_Drift + ( log_Movement_Volatility ** 2 ) / 2

    return arithmetic_Drift

def getAssetMARelativeArithmeticDrift( prices_So_Far, asset_Idx, desired_Latest_Day, moving_Average_Window_Size, realized_Volatility_Window_Size, drift_Window_Size ):
    arithmetic_Drift = getAssetArithmeticDrift( prices_So_Far, asset_Idx, desired_Latest_Day, drift_Window_Size, realized_Volatility_Window_Size )
    moving_Average = getAssetMovingAverage( prices_So_Far, asset_Idx, desired_Latest_Day, moving_Average_Window_Size )

    ma_Relative_Arithmetic_Drift = arithmetic_Drift / moving_Average

    return ma_Relative_Arithmetic_Drift

# Maret Average Returns #

def getMarketMeanLogReturns( prices_So_Far, day_Num ):
    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape
    day_Num = min( day_Num, number_Of_Timesteps - 1 )

    sum_Of_Log_Returns = 0

    for asset_Idx in range( number_Of_Instruments ):
        sum_Of_Log_Returns += np.log( prices_So_Far[ asset_Idx ][ day_Num ] / prices_So_Far[ asset_Idx ][ day_Num - 1 ] )

    return sum_Of_Log_Returns / number_Of_Instruments

def getMarketMovingMeanLogReturns( prices_So_Far, desired_Latest_Day, desired_Window_Size ):
    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape
    
    latest_Day = min( desired_Latest_Day, number_Of_Timesteps - 1 )

    window_Size = desired_Window_Size
    # Setting the window to the maximum 
    # available size if the entire desired
    # window size is not available
    if( latest_Day - window_Size + 1 < 0 ):
        window_Size = latest_Day + 1

    window_Start_Day = int( latest_Day - window_Size + 1 )

    sum_Of_Mean_Log_Returns = 0.0
    
    for day_Offset in range( window_Size ):
        sum_Of_Mean_Log_Returns += getMarketMeanLogReturns( prices_So_Far, window_Start_Day + day_Offset )

    mean_Of_Mean_Log_Retursn = sum_Of_Mean_Log_Returns / window_Size

    if( mean_Of_Mean_Log_Retursn == 0.0 ):
        logWarningHeader( "getMarketMovingMeanLogReturns",  "Returning 0" )

    return mean_Of_Mean_Log_Retursn

# Lead-Lag Correlation #
# Reference : https://financialnoob.substack.com/p/statistical-arbitrage-with-lead-lag
#           : and mentioned paper

def getSeriesLaggedCoefficient( first_Series, second_Series, lag ):
    correlation = 0.0

    if lag > 0:
        correlation = np.corrcoef( first_Series[ lag: ], second_Series[ :-lag ] )[ 0 ][ 1 ]
    elif lag == 0:
        correlation = np.corrcoef( first_Series, second_Series )[ 0 ][ 1 ]
    elif lag < 0:
        correlation = np.corrcoef( first_Series[ :lag ], second_Series[ -lag: ] )[ 0 ][ 1 ]

    return correlation

def getSeriesC1Coefficient( first_Series, second_Series, max_Lag ):
    # Defining a correlation dictionary comprehension
    correlations = { lag: np.abs( getSeriesLaggedCoefficient( first_Series, second_Series, lag ) ) for lag in range( -max_Lag, max_Lag + 1, 1 ) }

    c1_Coefficient = max( correlations, key = correlations.get )

    return c1_Coefficient

def getAssetToMarketMeanCorrelationC1LeadLag( prices_So_Far, asset_Idx, desired_Latest_Day, desired_Window_Size, desired_Correlation_Window_Size, max_Lag ):
    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape
    
    latest_Day = min( desired_Latest_Day, number_Of_Timesteps - 1 )

    window_Size = desired_Window_Size
    # Setting the window to the maximum 
    # available size if the entire desired
    # window size is not available
    if( latest_Day - window_Size + 1 < 0 ):
        window_Size = latest_Day + 1

    window_Start_Day = int( latest_Day - window_Size + 1 )
    
    market_Mean_Correlation_Series = np.zeros( window_Size )
    asset_Log_Returns_Series = np.zeros( window_Size )

    for day_Offset in range( window_Size ):
        market_Mean_Correlation_Series[ day_Offset ] = np.mean( getPearsonCorrelationMatrix( prices_So_Far, latest_Day, desired_Correlation_Window_Size ) )
        asset_Log_Returns_Series = getAssetLogMovementSeries( prices_So_Far, asset_Idx, latest_Day, window_Size )

    c1_Coefficient = getSeriesC1Coefficient( asset_Log_Returns_Series, market_Mean_Correlation_Series, max_Lag )
    c1_Coefficient_Correlation = getSeriesLaggedCoefficient( asset_Log_Returns_Series, market_Mean_Correlation_Series, c1_Coefficient )

    return c1_Coefficient, c1_Coefficient_Correlation


##### Strategies #####


# Moving Average Crossover Parameters (Optimised - Dean)
g_ma_Crossover_Short_Window_Size = 1
g_ma_Crossover_Long_Window_Size = 33

# TODO: Write a little documentation 
# explaining formulas
def runMovingAverageCrossoverStrategy( prices_So_Far ):
    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape

    global g_ma_Crossover_Short_Window_Size
    global g_ma_Crossover_Long_Window_Size

    # TODO: make these global parameters 
    # (so they can be moified and optimised in
    # the future)

    positions = np.zeros( number_Of_Instruments)

    # Dean - Note trading for 50 of the 250 days might be
    # problematic. Note that my functions will still run
    # even if the window size is greater than the available
    # timesteps
    if number_Of_Timesteps < g_ma_Crossover_Long_Window_Size:
        return positions

    for asset_Idx in range( number_Of_Instruments ):
        short_MA = getAssetMovingAverage( prices_So_Far, asset_Idx, number_Of_Timesteps - 1, g_ma_Crossover_Short_Window_Size)
        long_MA = getAssetMovingAverage( prices_So_Far, asset_Idx, number_Of_Timesteps - 1, g_ma_Crossover_Long_Window_Size)


        moving_Average_Signal = - ( short_MA - long_MA )/ long_MA

        if( asset_Idx == 0 ):
            dollar_Position_Limit = 100000
        else:
            dollar_Position_Limit = 10000

        # Dean - Consider a constant so we don't run
        # the chance of betting the entire dollar
        # limit at once in case the signal is high 
        # eneough as the signal can be greater than 1
        # (difference in short_MA - long_Ma > long_Ma
        positions[ asset_Idx ] = int( dollar_Position_Limit * moving_Average_Signal / prices_So_Far[ asset_Idx ][ -1 ])
        
    return positions

##### Lead-lag Correlation Strategy #####

"""Candidate submission: online factor-regime ensemble.

This is intentionally self-contained and uses NumPy only.  It is designed to
be copied to the required submission filename after the team has completed its
own validation.
"""

import numpy as np


# The values match the published limits and commission asymmetry.  We only use
# commission in the model-selection proxy; evaluation itself applies the true
# fees after this function returns the desired position.
_EPS = 1e-12
_VOL_WINDOW = 60
_PERFORMANCE_WINDOW = 30
_LOOKBACKS = tuple(range(1, 11))
# Do not trade when even the best expert has weak recent evidence.  This guards
# against forcing a forecast during a regime transition.
_MIN_INFORMATION_RATIO = 0.30


def _rolling_sum(values, window):
    """Trailing sums with shorter windows at the beginning of the history."""
    cumulative = np.cumsum(values)
    answer = cumulative.copy()
    if window < len(values):
        answer[window:] -= cumulative[:-window]
    return answer


def _cross_sectional_factor(returns):
    """Past-only, volatility-normalised estimate of the common market move."""
    n_days, n_inst = returns.shape
    factor = np.empty(n_days)
    for day in range(n_days):
        recent = returns[max(0, day - _VOL_WINDOW + 1) : day + 1]
        scale = np.std(recent, axis=0) + _EPS
        factor[day] = np.mean(returns[day] / scale)
    return factor


def _expert_directions(returns):
    """Return directional forecasts, one per row, for every observed day.

    Each value at index t is a forecast for the return following t.  The pool
    includes reversal and momentum at 1--10 day horizons for two independent
    factor estimates (the broad standardised factor and ALGO itself), plus
    permanent long/short fallbacks.  The selectors below decide which expert is
    currently credible; no outcome after t is used to form that day's forecast.
    """
    common_factor = _cross_sectional_factor(returns)
    factors = (common_factor, returns[:, 0])
    experts = []
    for factor in factors:
        for window in _LOOKBACKS:
            reversal = -np.sign(_rolling_sum(factor, window))
            experts.append(reversal)
            experts.append(-reversal)  # momentum counterpart
    experts.append(np.ones(len(returns)))
    experts.append(-np.ones(len(returns)))
    return np.asarray(experts)


def _choose_direction(returns, dollar_limits):
    """Choose the best recent expert by realised, risk-adjusted PnL.

    At the latest return index L-1, expert forecasts through L-2 already have
    known outcomes.  The latest forecast is therefore selected without any
    look-ahead.  We use a winner-take-all rule rather than averaging conflicting
    experts, which retains useful exposure when the current regime is clear.
    """
    n_days = len(returns)
    if n_days < _PERFORMANCE_WINDOW + 2:
        return 0.0

    experts = _expert_directions(returns)

    # A proxy for the PnL earned by holding every instrument at its dollar cap
    # over each next day.  It weights expert performance in the same direction
    # as the official objective, including ALGO's deliberately larger limit.
    full_long_pnl = np.sum(dollar_limits * np.expm1(returns), axis=1)

    # Forecast k was made after return k and earns the known return k+1.
    known_end = n_days - 1  # exclusive expert index; ends at n_days - 2
    known_start = max(0, known_end - _PERFORMANCE_WINDOW)
    pnl = experts[:, known_start:known_end] * full_long_pnl[known_start + 1 : known_end + 1]
    if pnl.shape[1] < 10:
        return 0.0
    risk_adjusted = np.mean(pnl, axis=1) / (np.std(pnl, axis=1) + _EPS)
    chosen = int(np.argmax(risk_adjusted))
    if risk_adjusted[chosen] < _MIN_INFORMATION_RATIO:
        return 0.0
    return float(experts[chosen, -1])


def runOnlineFactorRegimeEnsembleStrategy(prcSoFar):
    """Return desired integer share holdings for the current close."""
    n_inst, n_times = prcSoFar.shape
    if n_inst != 51 or n_times < _PERFORMANCE_WINDOW + 3:
        return np.zeros(n_inst, dtype=int)

    returns = np.diff(np.log(prcSoFar), axis=1).T
    dollar_limits = np.full(n_inst, 10_000.0)
    dollar_limits[0] = 100_000.0
    direction = _choose_direction(returns, dollar_limits)

    target_shares = direction * dollar_limits / prcSoFar[:, -1]
    return target_shares.astype(int)


##### External #####

def setGlobalVariable( name, value ):
    globals()[ name ] = value
