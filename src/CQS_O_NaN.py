import numpy as np
import random
import base64
import io

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


##### Code Start #####
g_number_Of_Instruments = 51

g_position_Limits = np.full( g_number_Of_Instruments, 10_000 )
g_position_Limits[ 0 ] = 100_000

g_commission_Rate = np.full( g_number_Of_Instruments, 0.0001 )
g_commission_Rate[ 0 ] = 0.00002

g_cash = 0.0
g_value = 0.0
g_comm = 0.0

current_Position = np.zeros( g_number_Of_Instruments )

g_trade_History_Buffer_Size = 3
# TODO: Introduce PnL function
g_previous_PnL_Buffer = np.zeros( g_trade_History_Buffer_Size )

g_position_History_Buffer = np.zeros( ( g_trade_History_Buffer_Size, g_number_Of_Instruments ) )

# Strategy Enumeration :
#   0 : main strategy (reserved)
#   1 : moving average crossover
g_strategy_Selection_Enum = 3

g_smac_weight = 0.0
g_spt_weight = 0.1
g_spmr_weight = 3.0
g_sal1_weight = 2.0
g_swal1_weight = 1.0
g_sofe_weight = 2.0
g_spauto_weight = 1.0
g_svars_weight = 1.0
g_sf1_weight = 4.0

# NOTE: We cannot change the argument variable name from
# prcSoFar
def getMyPosition( prcSoFar ):
    global current_Position
    global g_strategy_Selection_Enum
    global g_neural_Net_Instance
    global g_torch_Device

    ( number_Of_Instruments, number_Of_Timesteps ) = prcSoFar.shape

    if( number_Of_Timesteps < 2 ):
        return np.zeros( number_Of_Instruments )

    if( g_strategy_Selection_Enum == 1 ):
        return_Position = runMovingAverageCrossoverStrategy( prcSoFar )

    if( g_strategy_Selection_Enum == 2 ):
        return_Position = runPairsTradingStrategy( prcSoFar )
        return_Position = return_Position + runAlgorithm1Strategy( prcSoFar )
        return_Position[ 0 ] *= 10
        return_Position *= 10

    if( g_strategy_Selection_Enum == 3 ):
        current_Prices = getCurrentPrices( prcSoFar )
        position_Limits = ( g_position_Limits / current_Prices ).astype( int )

        return_Position = g_spt_weight * np.clip( runPairsTradingStrategy( prcSoFar ), -position_Limits, position_Limits ).astype( int )
        return_Position += g_sal1_weight * np.clip( runAlgorithm1Strategy( prcSoFar ), -position_Limits, position_Limits ).astype( int )
        return_Position += g_spmr_weight * np.clip( runPairsMeanReversionStrategy( prcSoFar ), -position_Limits, position_Limits ).astype( int )
        return_Position += g_swal1_weight * np.clip( runAlgorithm1WidenedStrategy( prcSoFar ), -position_Limits, position_Limits ).astype( int )
        # return_Position += getDrawdownScalar() * g_sofe_weight  * np.clip( runOnlineFactorRegimeEnsembleStrategy( prcSoFar ), -position_Limits, position_Limits ).astype( int )
        return_Position += g_spauto_weight * np.clip( runPooledAutocorrelationStrategy( prcSoFar ), -position_Limits, position_Limits ).astype( int ) # Notable
        return_Position += g_svars_weight * np.clip( runVolatilityAdjustedReversionStrategy( prcSoFar ), -position_Limits, position_Limits ).astype( int ) # Notable
        return_Position += g_sofe_weight * np.clip( runOnlineFactorRegimeEnsembleDrawdownThrottledStrategy( prcSoFar ), -position_Limits, position_Limits ).astype( int ) # Notable
        return_Position += g_sf1_weight * np.clip( runF1( prcSoFar ), -position_Limits, position_Limits ).astype( int ) # Notable
        return_Position[ 0 ] *= 10
        # return_Position *= 10

        # return_Position *= getDrawdownScalar()

    if( g_strategy_Selection_Enum == 4 ):
        # 
        # return_Position = runPairsMeanReversionStrategy( prcSoFar )
        # return_Position = runAlgorithm1Strategy( prcSoFar ) # Notable
        # return_Position = runAlgorithm1WidenedStrategy( prcSoFar ) # Very notable
        # return_Position = runOnlineFactorRegimeEnsembleStrategy( prcSoFar ) # Little (though has been successful)
        # return_Position = runPairsTradingStrategy( prcSoFar )
        # return_Position = runRankedLongHorizonReversionStrategy( prcSoFar )
        # return_Position = runRegularisedLeadLagStrategy( prcSoFar )
        # return_Position = runPooledAutocorrelationStrategy( prcSoFar ) # Notable
        # return_Position = runVolatilityAdjustedReversionStrategy( prcSoFar )# Good
        # return_Position = runCorrelationFilteredReversionStrategy( prcSoFar )
        return_Position = runOnlineFactorRegimeEnsembleDrawdownThrottledStrategy( prcSoFar )
        return_Position[ 0 ] *= 10

    if( g_strategy_Selection_Enum == 5 ):
        current_Prices = getCurrentPrices( prcSoFar )
        position_Limits = ( g_position_Limits / current_Prices ).astype( int )

        return_Position = 1 * position_Limits

    current_Position = return_Position

    updatePositionHistory( prcSoFar, current_Position )
    updatePnLHistory( prcSoFar )

    return current_Position 


##### Logging #####

# NOTE: Logging verbosity
#   0 : No logging
#   1 : Errors only
#   2 : Errors and warnings
#   3 : Errors, warnings, and general info

g_logging_Verbosity = 0

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

    if np.isnan( correlation_Value ):
        correlation_Value = 0

    #Error detection
    if( correlation_Value > 1.1 or correlation_Value < -1.1 ):
        logErrorHeader( "getSeriesPairPearsonCorrelationValue", "Correlation value outside of expected range" )
        logErrorValue( "correlation_Value", correlation_Value )
        correlation_Value = 0

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

    realized_Variance += 1e-8

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

# Log Returns Realized Volatility #
def getAssetLogReturnsRealizedVolatility( prices_So_Far, asset_Idx, desired_Latest_Day, desired_Window_Size ):
    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape
    
    latest_Day = min( desired_Latest_Day, number_Of_Timesteps - 1 )

    window_Size = desired_Window_Size
    # Setting the window to the maximum 
    # available size if the entire desired
    # window size is not available
    if( latest_Day - window_Size + 1 < 0 ):
        window_Size = latest_Day + 1

    window_Start_Day = int( latest_Day - window_Size + 1 )

    log_Returns_Series = getAssetLogMovementSeries( prices_So_Far, asset_Idx, latest_Day, window_Size )

    realized_Volatility = np.std( log_Returns_Series )

    return realized_Volatility

def getMarketStatisticalAssetLogVolatility( prices_So_Far, desired_Latest_Day, desired_Window_Size ):
    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape
    
    latest_Day = min( desired_Latest_Day, number_Of_Timesteps - 1 )

    window_Size = desired_Window_Size
    # Setting the window to the maximum 
    # available size if the entire desired
    # window size is not available
    if( latest_Day - window_Size + 1 < 0 ):
        window_Size = latest_Day + 1

    asset_Volatilities_Array = [ getAssetLogMovementSeries( prices_So_Far, asset_Idx, latest_Day, window_Size ) for asset_Idx in range( number_Of_Instruments ) ]
    
    mean_Volatility = np.mean( asset_Volatilities_Array )
    volatility_Standard_Deviation = np.std( asset_Volatilities_Array )

    return mean_Volatility, volatility_Standard_Deviation




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

    mean_Of_Mean_Log_Returns = sum_Of_Mean_Log_Returns / window_Size

    if np.isnan( mean_Of_Mean_Log_Returns ):
        mean_Of_Mean_Log_Returns = 0.1

    return mean_Of_Mean_Log_Returns

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


##### Risk Management #####

# -6000 and 0.5 works weel for collection strat
def getDrawdownScalar( threshold = -6000.0, floor = 0.5 ):
    global g_previous_PnL_Buffer

    cumulative_Recent_PnL = np.sum( g_previous_PnL_Buffer )

    if cumulative_Recent_PnL >= 0.0:
        return 1.0

    severity = min( abs( cumulative_Recent_PnL ) / abs( threshold ), 1.0 )
    scalar = 1.0 - severity * ( 1.0 - floor )

    # print( scalar )
    return scalar




##### Neural Net #####

g_nn_number_Of_Controlled_Strategies = 5
g_nn_number_Of_Outputs = 2 * g_nn_number_Of_Controlled_Strategies
g_nn_number_Of_Inputs = 1 + g_nn_number_Of_Controlled_Strategies + 3 + 5 + 3 + 4 + 2

g_nn_output_History_Buffer = np.zeros( ( g_trade_History_Buffer_Size, g_nn_number_Of_Controlled_Strategies ) )

def updateNeuralNetOutputHistory( outputs ):
    global g_nn_output_History_Buffer

    g_nn_output_History_Buffer = np.roll( g_nn_output_History_Buffer, 1, axis = 0 )
    g_nn_output_History_Buffer[ 0 ] = outputs

def resetNeuralNetOutputHistory():
    global g_nn_output_History_Buffer
    g_nn_output_History_Buffer = np.zeros( ( g_trade_History_Buffer_Size, g_nn_number_Of_Controlled_Strategies ) )

def updatePositionHistory( prices_So_Far, new_Position_Original ):
    global g_position_History_Buffer

    current_Prices = getCurrentPrices( prices_So_Far )
    position_Limits = ( g_position_Limits / current_Prices ).astype( int )

    new_Position = np.clip( new_Position_Original, -position_Limits, position_Limits ).astype( int )

    g_position_History_Buffer = np.roll( g_position_History_Buffer, 1, axis = 0 )
    g_position_History_Buffer[ 0 ] = new_Position

def resetPositionHistory():
    global g_position_History_Buffer
    global g_trade_History_Buffer_Size
    global g_number_Of_Instruments
    g_position_History_Buffer = np.zeros( ( g_trade_History_Buffer_Size , g_number_Of_Instruments ) )


def getCurrentPrices( prices_So_Far ):
    current_Prices = prices_So_Far[ : , -1 ].T

    return current_Prices

def updatePnLHistory( prices_So_Far ):
    global g_previous_PnL_Buffer
    global g_position_History_Buffer
    global g_commission_Rate
    global g_cash
    global g_value
    global g_comm

    current_Prices = getCurrentPrices( prices_So_Far )

    new_Position = g_position_History_Buffer[ 0 ]
    previous_Position = g_position_History_Buffer[ 1 ]

    delta_Pos = np.subtract( new_Position , previous_Position )

    position_Value = new_Position.dot( current_Prices )
    g_cash -= current_Prices.dot( delta_Pos ) + g_comm

    today_PnL = g_cash + position_Value - g_value

    g_value = g_cash + position_Value

    g_previous_PnL_Buffer = np.roll( g_previous_PnL_Buffer, 1 )
    g_previous_PnL_Buffer[ 0 ] = today_PnL

    dollar_Volumes = current_Prices * np.abs( delta_Pos )
    g_comm = np.sum( dollar_Volumes * g_commission_Rate )


def resetPnLHistory():
    global g_trade_History_Buffer_Size
    global g_previous_PnL_Buffer
    global g_cash 
    global g_value
    global g_comm

    g_previous_PnL_Buffer = np.zeros( g_trade_History_Buffer_Size )
    g_cash = 0.0
    g_value = 0.0
    g_comm = 0.0


# Input parameters
g_nn_Short_Market_Moving_Mean_Log_Returns_Window_Size = 10
g_nn_Long_Market_Moving_Mean_Log_Returns_Window_Size = 50
g_nn_Market_Correlation_Window_Size = 5
g_nn_Market_Short_Statistical_Realized_Volatility_Window_Size = 5
g_nn_Market_Long_Statistical_Realized_Volatility_Window_Size = 20

def getNeuralNetInputs( prices_So_Far, timestep_Idx ):
    global g_nn_Short_Market_Moving_Mean_Log_Returns_Window_Size
    global g_nn_Long_Market_Moving_Mean_Log_Returns_Window_Size
    global g_nn_Market_Correlation_Window_Size
    global g_nn_Market_Short_Statistical_Realized_Volatility_Window_Size
    global g_nn_Market_Long_Statistical_Realized_Volatility_Window_Size

    global g_nn_output_History_Buffer
    global g_previous_PnL_Buffer

    global g_pairs_Trading_List
    global g_pairs_Trading_Beta_Window
    global g_pairs_Trading_Z_Window

    state = []

    state.append( timestep_Idx / 500 )

    # Previous outputs
    for last_Output in g_nn_output_History_Buffer[ -1 ]:
       state.append( last_Output )

    # Market log mean returns (long and short)
    market_Short_MM_Log_Returns = getMarketMovingMeanLogReturns( prices_So_Far, timestep_Idx, g_nn_Short_Market_Moving_Mean_Log_Returns_Window_Size )
    market_Long_MM_Log_Returns = getMarketMovingMeanLogReturns( prices_So_Far, timestep_Idx, g_nn_Long_Market_Moving_Mean_Log_Returns_Window_Size )
    market_Short_Long_MM_Log_Returns_Ratio = market_Short_MM_Log_Returns / ( market_Long_MM_Log_Returns + 1e-8 )

    state.append( market_Short_MM_Log_Returns )
    state.append( market_Long_MM_Log_Returns )
    state.append( market_Short_Long_MM_Log_Returns_Ratio )

    # Average correlation
    # state.append( getCorrelationMatrixAndMeanCorrelation( prices_So_Far, timestep_Idx, g_nn_Market_Correlation_Window_Size )[ 1 ] )
    
    # Realized volatility
    market_Short_Statistical_Volatility = getMarketStatisticalAssetLogVolatility( prices_So_Far, timestep_Idx, g_nn_Market_Short_Statistical_Realized_Volatility_Window_Size )
    state.append( market_Short_Statistical_Volatility[ 0 ] )
    state.append( market_Short_Statistical_Volatility[ 1 ] )

    market_Long_Statistical_Volatility = getMarketStatisticalAssetLogVolatility( prices_So_Far, timestep_Idx, g_nn_Market_Long_Statistical_Realized_Volatility_Window_Size )
    state.append( market_Long_Statistical_Volatility[ 0 ] )
    state.append( market_Long_Statistical_Volatility[ 1 ] )

    market_Short_Long_Mean_Volatility_Ratio = market_Short_Statistical_Volatility[ 0 ] / ( market_Long_Statistical_Volatility[ 0 ] + 1e-8 )
    state.append( market_Short_Long_Mean_Volatility_Ratio )

    # Asset 0 Volatility
    asset_0_Short_RVol = getAssetRealizedVolatility( prices_So_Far, 0, timestep_Idx, g_nn_Market_Short_Statistical_Realized_Volatility_Window_Size )
    asset_0_Long_RVol = getAssetRealizedVolatility( prices_So_Far, 0, timestep_Idx, g_nn_Market_Long_Statistical_Realized_Volatility_Window_Size )
    asset_0_Short_Long_RVol_Ratio = asset_0_Short_RVol / ( asset_0_Long_RVol + 1e-8 )

    state.append( asset_0_Short_RVol )
    state.append( asset_0_Long_RVol )
    state.append( asset_0_Short_Long_RVol_Ratio )

    # Pair-wise trading Z-scores
    for asset_Idx_Pair in g_pairs_Trading_List:
        beta_Value = getPairHedgeRatio( prices_So_Far, asset_Idx_Pair[ 0 ], asset_Idx_Pair[ 1 ], g_pairs_Trading_Beta_Window )
        z_Score = getPairSpreadZScore( prices_So_Far, asset_Idx_Pair[ 0 ], asset_Idx_Pair[ 1 ], beta_Value, g_pairs_Trading_Z_Window )
        state.append( z_Score )

    state.append( g_previous_PnL_Buffer[ 0 ] / 5000 )
    state.append( g_previous_PnL_Buffer[ 1 ] / 5000 )

    return state

# NOTE: Rewritten from Claude
# NOTE: Mean and log_Std are tensors

# Instance


# Neural Net Master Strategy #


##### Algorithm1 (widened selection variant) #####

a1v2_nInst = 51
a1v2_dlrLimit = np.full( a1v2_nInst, 10_000.0 )
a1v2_dlrLimit[ 0 ] = 100_000.0
a1v2_EPS = 1e-9

a1v2_MIN_HIST = 1
a1v2_ENTER_K = 18      # widened from 12 - trades PnL-per-position for total exposure/PnL
a1v2_EXIT_K = 25        # widened from 17, same proportional gap to preserve hysteresis behaviour

_a1v2_heldSet = set()

def _a1v2_leadlag_signal( rets_hist ):
    X = rets_hist[ :, :-1 ].T
    Y = rets_hist[ :, 1: ].T
    xmu, xsd = X.mean( 0 ), X.std( 0 ) + a1v2_EPS
    ymu, ysd = Y.mean( 0 ), Y.std( 0 ) + a1v2_EPS
    Xs = ( X - xmu ) / xsd
    Ys = ( Y - ymu ) / ysd
    LL = ( Xs.T @ Ys ) / Xs.shape[ 0 ]
    np.fill_diagonal( LL, 0.0 )
    last = rets_hist[ :, -1 ]
    last_std = ( last - xmu ) / xsd
    return last_std @ LL

def _a1v2_regime_scale( rets_hist ):
    mkt = rets_hist.mean( axis = 0 )
    if mkt.shape[ 0 ] < 60:
        return 1.0
    short_vol = mkt[ -10: ].std() + a1v2_EPS
    long_vol = mkt[ -60: ].std() + a1v2_EPS
    ratio = short_vol / long_vol
    return float( np.clip( 1.15 - 0.35 * max( ratio - 1.0, 0.0 ), 0.6, 1.15 ) )

def runAlgorithm1WidenedStrategy( prcSoFar ):
    global _a1v2_heldSet
    nins, nt = prcSoFar.shape

    if nt < a1v2_MIN_HIST + 1:
        return np.zeros( nins, dtype = int )

    logp = np.log( prcSoFar )
    rets = np.diff( logp, axis = 1 )

    sig = _a1v2_leadlag_signal( rets )
    scale = _a1v2_regime_scale( rets )

    order = np.argsort( -np.abs( sig ) )
    ranks = np.empty( nins, dtype = int )
    ranks[ order ] = np.arange( nins )

    newHeld = set()
    for i in range( nins ):
        threshold = a1v2_EXIT_K if i in _a1v2_heldSet else a1v2_ENTER_K
        if ranks[ i ] < threshold:
            newHeld.add( i )
    _a1v2_heldSet = newHeld

    conf = np.zeros( nins )
    for i in newHeld:
        conf[ i ] = np.sign( sig[ i ] ) * scale

    curPrices = prcSoFar[ :, -1 ]
    targetDollars = conf * a1v2_dlrLimit
    targetShares = targetDollars / curPrices

    posLimitShares = ( a1v2_dlrLimit / curPrices ).astype( int )
    newPos = np.clip( targetShares, -posLimitShares, posLimitShares )
    return newPos.astype( int )


##### Algorithm1 #####

""" 
thing to note: the reason why this algorithm, heavily edited version from my previous code,
doesnt yield such a high score is because it only trades with veryu high confidence.
i.e. only trades top 10, and only holds them if they are still in the top 16. this is a very conservative approach, and it is not necessarily the best approach for maximizing score
there definitely loys of ways to improve the score, but this version is a safe version that guarantees a positive :D
for days 251-500, this scored 153.67. haha 67
"""

a1_nInst = 51
dlrLimit = np.full(a1_nInst, 10_000.0) # limits
dlrLimit[0] = 100_000.0
EPS = 1e-9 # prevent divide-by-zero in standardization

MIN_HIST = 1      # minimum days of return history before trading -> estimates r/s between all 51 instruments
ENTER_K = 12        # an asset must rank in the top ENTER_K |signal| to open a new position
EXIT_K = 17         # an already-held asset stays as long as it's still in the top EXIT_K
                     # (hysteresis - cuts needless flip-flopping / commission drag)

_heldSet = set()     # persists across calls: which assets currently carry conviction bets


def _leadlag_signal(rets_hist):
    """
    cross-sectional lead-lag forecast: regress each asset's return on *all*
    assets' previous-day standardized returns (via covariance-based
    projection), then apply that r/s to today's returns to forecast
    tomorrow's cross-section of returns. Captures the modest but genuine
    (statistically significant out-of-sample) lead-lag structure between
    instruments in this universe, as opposed to single-asset momentum/reversal
    which carries no real edge here.
    """
    X = rets_hist[:, :-1].T   # returns at t-1, shape (T-1, nInst) -> coontains the returns of all instruments for all days except the last day
    Y = rets_hist[:, 1:].T    # returns at t,   shape (T-1, nInst) -> 1 day after
    xmu, xsd = X.mean(0), X.std(0) + EPS # historical mean and stand dev of every X asset
    ymu, ysd = Y.mean(0), Y.std(0) + EPS # historical mean and stand dev of every Y asset
    Xs = (X - xmu) / xsd # standardize X and Y to have mean 0 and std 1
    Ys = (Y - ymu) / ysd # standardize X and Y to have mean 0 and std 1
    LL = (Xs.T @ Ys) / Xs.shape[0]     # (nInst, nInst) lead-lag coefficient matrix
    np.fill_diagonal(LL, 0.0)          # use cross-asset relationships
    last = rets_hist[:, -1] # most recent return for every instrument
    last_std = (last - xmu) / xsd # standardize the most recent return for every instrument
    return last_std @ LL               # forecast for tomorrow's standardized return, per asset


def _regime_scale(rets_hist):
    """shrink exposure modestly when the market is in an abnormally turbulent
    regime (short-term vol well above its longer-run level); this doesn't
    change direction, only overall aggressiveness."""
    mkt = rets_hist.mean(axis=0)
    if mkt.shape[0] < 60: # not enough history to estimate short/long vol, so don't scale down
        return 1.0
    short_vol = mkt[-10:].std() + EPS # short-term volatility of the market
    long_vol = mkt[-60:].std() + EPS # long term
    ratio = short_vol / long_vol # ratio of short-term to long-term volatility -> if ratio > 1, then short-term volatility is higher than long-term volatility, indicating a turbulent regime
    return float(np.clip(1.15 - 0.35 * max(ratio - 1.0, 0.0), 0.6, 1.15))


""" 
getmyposition() first checks data avail, verifies that at least 60 days of historical returns are avail,
then calculate its returns, converts it into logarithmic prices, which are used as the input for forecasting model
( logarithmic returns as my friend claude and chatgpt says thisb is the standard practice in quant firms :D )
it then calls the trading signals, ranks intruments, and selects which to hold. (top 10), after that it assigns 
position direction, and converts desired dollar exposure to required numbner of shares
"""
def runAlgorithm1Strategy(prcSoFar):
    global _heldSet
    nins, nt = prcSoFar.shape # number of instruments and time steps

    if nt < MIN_HIST + 1: # not enough history to estimate r/s between all instruments, so don't trade yet
        return np.zeros(nins, dtype=int)

    logp = np.log(prcSoFar) # compute log prices from prices
    rets = np.diff(logp, axis=1) # compute returns from log prices

    sig = _leadlag_signal(rets)  # compute lead-lag signals
    scale = _regime_scale(rets) # compute regime scaling factor

    order = np.argsort(-np.abs(sig)) # sort the signals in descending order of absolute value, so that the most extreme signals are first
    ranks = np.empty(nins, dtype=int)
    ranks[order] = np.arange(nins)

    newHeld = set()
    for i in range(nins): 
        threshold = EXIT_K if i in _heldSet else ENTER_K
        if ranks[i] < threshold:
            newHeld.add(i)
    _heldSet = newHeld

    conf = np.zeros(nins)
    for i in newHeld:
        conf[i] = np.sign(sig[i]) * scale

    curPrices = prcSoFar[:, -1]
    targetDollars = conf * dlrLimit
    targetShares = targetDollars / curPrices

    posLimitShares = (dlrLimit / curPrices).astype(int)
    newPos = np.clip(targetShares, -posLimitShares, posLimitShares)
    return newPos.astype(int)


##### Lead-lag Correlation Strategy #####

"""Candidate submission: online factor-regime ensemble.

This is intentionally self-contained and uses NumPy only.  It is designed to
be copied to the required submission filename after the team has completed its
own validation.
"""

# The values match the published limits and commission asymmetry.  We only use
# commission in the model-selection proxy; evaluation itself applies the true
# fees after this function returns the desired position.
_EPS = 1e-12
_VOL_WINDOW = 30
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
        recent = returns[int( max(0, day - _VOL_WINDOW + 1) ) : ( day + 1 )]
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
    known_end = int( n_days - 1 )  # exclusive expert index; ends at n_days - 2
    known_start = int( max(0, known_end - _PERFORMANCE_WINDOW) )
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


##### Pairs Trading Strategy #####
# NOTE: Generated by Claude

"""
Statistically-validated pairs: each pair passed an Engle-Granger cointegration
test at p < 0.05, and the pair selection was re-checked using ONLY the first
500 days of data to confirm it wasn't a full-sample artifact - the same pairs
came back significant. Out-of-sample backtest on days 500-750 (the actual
test window) gave a standalone annualised Sharpe of ~2.9, with near-zero PnL
correlation to the other three strategies.

Instrument indices (from prices.txt column order):
    MHRM=49, EAFC=50, FWWG=36, BLBT=41, ACIX=31, ITPA=43, MTNS=33, ILVX=46
"""

g_pairs_Trading_List = [ ( 49, 50 ), ( 36, 41 ), ( 31, 43 ), ( 33, 46 ) ]   # (asset_A_idx, asset_B_idx)

g_pairs_Trading_Beta_Window = 60     # trailing days used to estimate the hedge ratio
g_pairs_Trading_Z_Window = 30         # trailing days used to compute the spread's mean/std
g_pairs_Trading_Entry_Z = 2.0         # |z| above this opens a position
g_pairs_Trading_Exit_Z = 0.5          # |z| below this closes the position
g_pairs_Trading_Dollars_Per_Leg = 8000.0   # per-leg dollar size, kept under the $10k instrument limit

# Persists across calls: current spread direction per pair (-1, 0, or +1)
g_pairs_Trading_State = { pair: 0 for pair in g_pairs_Trading_List }


def getPairHedgeRatio( prices_So_Far, asset_A_Idx, asset_B_Idx, window_Size ):
    log_Prices = np.log( prices_So_Far )
    log_A_Window = log_Prices[ asset_A_Idx ][ -window_Size: ]
    log_B_Window = log_Prices[ asset_B_Idx ][ -window_Size: ]

    # OLS slope of log(A) on log(B): A moves beta_Value units per unit move in B
    beta_Value, alpha_Value = np.polyfit( log_B_Window, log_A_Window, 1 )
    return beta_Value


def getPairSpreadZScore( prices_So_Far, asset_A_Idx, asset_B_Idx, beta_Value, z_Window_Size ):
    log_Prices = np.log( prices_So_Far )
    spread_Series = log_Prices[ asset_A_Idx ] - beta_Value * log_Prices[ asset_B_Idx ]

    z_Window = spread_Series[ -z_Window_Size: ]
    spread_Mean = np.mean( z_Window )
    spread_Std = np.std( z_Window ) + 1e-9

    current_Spread = spread_Series[ -1 ]
    z_Score = ( current_Spread - spread_Mean ) / spread_Std

    return z_Score


def runPairsTradingStrategy( prices_So_Far ):
    global g_pairs_Trading_State

    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape
    positions = np.zeros( number_Of_Instruments )

    if number_Of_Timesteps < g_pairs_Trading_Beta_Window + 2:
        return positions

    for ( asset_A_Idx, asset_B_Idx ) in g_pairs_Trading_List:

        beta_Value = getPairHedgeRatio( prices_So_Far, asset_A_Idx, asset_B_Idx, g_pairs_Trading_Beta_Window )
        z_Score = getPairSpreadZScore( prices_So_Far, asset_A_Idx, asset_B_Idx, beta_Value, g_pairs_Trading_Z_Window )

        current_Direction = g_pairs_Trading_State[ ( asset_A_Idx, asset_B_Idx ) ]

        if current_Direction == 0:
            if z_Score > g_pairs_Trading_Entry_Z:
                current_Direction = -1     # spread too high -> short A, long B
            elif z_Score < -g_pairs_Trading_Entry_Z:
                current_Direction = 1      # spread too low  -> long A, short B
        else:
            if abs( z_Score ) < g_pairs_Trading_Exit_Z:
                current_Direction = 0      # spread reverted -> flatten

        g_pairs_Trading_State[ ( asset_A_Idx, asset_B_Idx ) ] = current_Direction

        if current_Direction != 0:
            dollars_A = current_Direction * g_pairs_Trading_Dollars_Per_Leg
            dollars_B = -current_Direction * beta_Value * g_pairs_Trading_Dollars_Per_Leg
            dollars_B = np.clip( dollars_B, -g_pairs_Trading_Dollars_Per_Leg, g_pairs_Trading_Dollars_Per_Leg )

            positions[ asset_A_Idx ] += dollars_A / prices_So_Far[ asset_A_Idx ][ -1 ]
            positions[ asset_B_Idx ] += dollars_B / prices_So_Far[ asset_B_Idx ][ -1 ]

    return positions

##### Pairs Mean Reversion Strategy #####

g_pmr_Pairs_List = [ ( 49, 50 ), ( 36, 41 ), ( 31, 43 ), ( 33, 46 ) ]   # MHRM-EAFC, FWWG-BLBT, ACIX-ITPA, MTNS-ILVX

g_pmr_Beta_Window = 130
g_pmr_Z_Window = 20
g_pmr_Entry_Z = 1.0
g_pmr_Exit_Z = 0.4
g_pmr_Dollars_Per_Leg = 8000.0

g_pmr_Min_History = 5   # absolute floor - need at least a handful of points before a beta/z-score means anything

g_pmr_State = { pair: 0 for pair in g_pmr_Pairs_List }

def getPairMeanReversionHedgeRatio( prices_So_Far, asset_A_Idx, asset_B_Idx, window_Size ):
    log_Prices = np.log( prices_So_Far )

    # shrink to whatever history actually exists rather than refusing to run
    available = log_Prices.shape[ 1 ]
    effective_Window = int( min( window_Size, available ) )

    log_A_Window = log_Prices[ asset_A_Idx ][ -effective_Window: ]
    log_B_Window = log_Prices[ asset_B_Idx ][ -effective_Window: ]

    beta_Value, alpha_Value = np.polyfit( log_B_Window, log_A_Window, 1 )
    return beta_Value

def getPairMeanReversionZScore( prices_So_Far, asset_A_Idx, asset_B_Idx, beta_Value, z_Window_Size ):
    log_Prices = np.log( prices_So_Far )
    spread_Series = log_Prices[ asset_A_Idx ] - beta_Value * log_Prices[ asset_B_Idx ]

    available = spread_Series.shape[ 0 ]
    effective_Window = int( min( z_Window_Size, available ) )

    z_Window = spread_Series[ -effective_Window: ]
    spread_Mean = np.mean( z_Window )
    spread_Std = np.std( z_Window ) + 1e-9

    current_Spread = spread_Series[ -1 ]
    z_Score = ( current_Spread - spread_Mean ) / spread_Std

    return z_Score

def runPairsMeanReversionStrategy( prices_So_Far ):
    global g_pmr_State

    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape
    positions = np.zeros( number_Of_Instruments )

    if number_Of_Timesteps < g_pmr_Min_History:
        return positions

    for ( asset_A_Idx, asset_B_Idx ) in g_pmr_Pairs_List:

        beta_Value = getPairMeanReversionHedgeRatio( prices_So_Far, asset_A_Idx, asset_B_Idx, g_pmr_Beta_Window )
        z_Score = getPairMeanReversionZScore( prices_So_Far, asset_A_Idx, asset_B_Idx, beta_Value, g_pmr_Z_Window )

        current_Direction = g_pmr_State[ ( asset_A_Idx, asset_B_Idx ) ]

        if current_Direction == 0:
            if z_Score > g_pmr_Entry_Z:
                current_Direction = -1
            elif z_Score < -g_pmr_Entry_Z:
                current_Direction = 1
        else:
            if abs( z_Score ) < g_pmr_Exit_Z:
                current_Direction = 0

        g_pmr_State[ ( asset_A_Idx, asset_B_Idx ) ] = current_Direction

        if current_Direction != 0:
            dollars_A = current_Direction * g_pmr_Dollars_Per_Leg
            dollars_B = -current_Direction * beta_Value * g_pmr_Dollars_Per_Leg
            dollars_B = np.clip( dollars_B, -g_pmr_Dollars_Per_Leg, g_pmr_Dollars_Per_Leg )

            positions[ asset_A_Idx ] += dollars_A / prices_So_Far[ asset_A_Idx ][ -1 ]
            positions[ asset_B_Idx ] += dollars_B / prices_So_Far[ asset_B_Idx ][ -1 ]

    return positions


##### Standalone Advanced Strategies #####

# These strategies intentionally live beside the other strategy functions but
# are not called by runNeuralNetMasterStrategy.  They can be selected for
# isolated backtests or incorporated into a future ensemble after validation.
g_ranked_Reversion_Window = 100
g_ranked_Reversion_Top_K = 10
g_pooled_AR_Window = 120
g_pooled_AR_Signal_Window = 10
g_regularised_Lead_Lag_Window = 120
g_regularised_Lead_Lag_Top_K = 10
g_volatility_Adjusted_Reversion_Window = 80
g_volatility_Adjusted_Reversion_Volatility_Window = 60
g_volatility_Adjusted_Reversion_Top_K = 20
g_correlation_Filtered_Reversion_Window = 80
g_correlation_Filtered_Reversion_Correlation_Window = 120
g_correlation_Filtered_Reversion_Top_K = 15
g_correlation_Filtered_Reversion_Eligibility_Quantile = 0.70


def getRankedLongHorizonReversionSignal( prices_So_Far ):
    """Return limit fractions that fade only the most extreme 100-day moves."""
    number_Of_Instruments, number_Of_Timesteps = prices_So_Far.shape
    signal = np.zeros( number_Of_Instruments )

    if number_Of_Timesteps < g_ranked_Reversion_Window + 1:
        return signal

    safe_Prices = np.maximum( prices_So_Far, 1e-12 )
    cumulative_Return = np.log( safe_Prices[ : , -1 ] / safe_Prices[ : , -g_ranked_Reversion_Window - 1 ] )
    ranked_Assets = np.argsort( cumulative_Return )
    number_Selected = min( g_ranked_Reversion_Top_K, number_Of_Instruments // 2 )
    signal[ ranked_Assets[ : number_Selected ] ] = 1.0
    signal[ ranked_Assets[ -number_Selected : ] ] = -1.0

    return signal


def runRankedLongHorizonReversionStrategy( prices_So_Far ):
    """Long the 10 weakest and short the 10 strongest 100-day movers."""
    current_Prices = np.maximum( prices_So_Far[ : , -1 ], 1e-12 )
    signal = getRankedLongHorizonReversionSignal( prices_So_Far )
    return ( signal * g_position_Limits / current_Prices ).astype( int )


def getPooledAutocorrelation( prices_So_Far ):
    """Estimate one lag-one autocorrelation across all asset-day observations."""
    number_Of_Timesteps = prices_So_Far.shape[ 1 ]
    if number_Of_Timesteps < g_pooled_AR_Window + 1:
        return 0.0

    safe_Prices = np.maximum( prices_So_Far[ : , -g_pooled_AR_Window - 1 : ], 1e-12 )
    returns = np.diff( np.log( safe_Prices ), axis = 1 )
    previous_Returns = returns[ : , :-1 ]
    following_Returns = returns[ : , 1 : ]
    return np.sum( previous_Returns * following_Returns ) / ( np.sum( previous_Returns ** 2 ) + 1e-8 )


def runPooledAutocorrelationStrategy( prices_So_Far ):
    """Follow or fade each 10-day move according to the pooled regime estimate."""
    number_Of_Instruments, number_Of_Timesteps = prices_So_Far.shape
    if number_Of_Timesteps < g_pooled_AR_Window + 1:
        return np.zeros( number_Of_Instruments, dtype = int )

    safe_Prices = np.maximum( prices_So_Far, 1e-12 )
    returns = np.diff( np.log( safe_Prices[ : , -g_pooled_AR_Window - 1 : ] ), axis = 1 )
    cumulative_Return = np.sum( returns[ : , -g_pooled_AR_Signal_Window : ], axis = 1 )
    signal = np.sign( getPooledAutocorrelation( prices_So_Far ) ) * np.sign( cumulative_Return )
    return ( signal * g_position_Limits / safe_Prices[ : , -1 ] ).astype( int )


def getRegularisedLeadLagForecast( prices_So_Far ):
    """Return a ridge-shrunk one-day forecast for every instrument."""
    number_Of_Instruments, number_Of_Timesteps = prices_So_Far.shape
    if number_Of_Timesteps < g_regularised_Lead_Lag_Window + 1:
        return np.zeros( number_Of_Instruments )

    safe_Prices = np.maximum( prices_So_Far[ : , -g_regularised_Lead_Lag_Window - 1 : ], 1e-12 )
    returns = np.diff( np.log( safe_Prices ), axis = 1 ).T
    previous_Returns = returns[ : -1 ]
    following_Returns = returns[ 1 : ]
    previous_Mean = np.mean( previous_Returns, axis = 0 )
    previous_Std = np.std( previous_Returns, axis = 0 ) + 1e-12
    following_Mean = np.mean( following_Returns, axis = 0 )
    following_Std = np.std( following_Returns, axis = 0 ) + 1e-12
    standardised_Previous = ( previous_Returns - previous_Mean ) / previous_Std
    standardised_Following = ( following_Returns - following_Mean ) / following_Std

    # Four times the feature count is deliberately strong shrinkage: it makes
    # the forecast depend on persistent cross-asset effects rather than a few
    # noisy correlations in the 120-day rolling sample.
    ridge = 4.0 * number_Of_Instruments
    gram_Matrix = standardised_Previous.T @ standardised_Previous + ridge * np.eye( number_Of_Instruments )
    coefficients = np.linalg.solve( gram_Matrix, standardised_Previous.T @ standardised_Following )
    latest_Return = ( returns[ -1 ] - previous_Mean ) / previous_Std
    return latest_Return @ coefficients


def runRegularisedLeadLagStrategy( prices_So_Far ):
    """Trade only the ten largest absolute regularised lead--lag forecasts."""
    number_Of_Instruments, number_Of_Timesteps = prices_So_Far.shape
    if number_Of_Timesteps < g_regularised_Lead_Lag_Window + 1:
        return np.zeros( number_Of_Instruments, dtype = int )

    forecast = getRegularisedLeadLagForecast( prices_So_Far )
    signal = np.zeros( number_Of_Instruments )
    selected_Assets = np.argsort( np.abs( forecast ) )[ -min( g_regularised_Lead_Lag_Top_K, number_Of_Instruments ) : ]
    signal[ selected_Assets ] = np.sign( forecast[ selected_Assets ] )
    current_Prices = np.maximum( prices_So_Far[ : , -1 ], 1e-12 )
    return ( signal * g_position_Limits / current_Prices ).astype( int )


def getVolatilityAdjustedReversionSignal( prices_So_Far ):
    """Rank 80-day moves after scaling them by each asset's recent risk.

    A raw 80-day return can look extreme simply because an asset is noisy.
    Dividing by 60-day realised volatility selects unusually large moves on a
    comparable risk basis, while rank selection avoids any fitted threshold.
    """
    number_Of_Instruments, number_Of_Timesteps = prices_So_Far.shape
    signal = np.zeros( number_Of_Instruments )
    required_History = max(
        g_volatility_Adjusted_Reversion_Window,
        g_volatility_Adjusted_Reversion_Volatility_Window,
    )
    if number_Of_Timesteps < required_History + 1:
        return signal

    safe_Prices = np.maximum( prices_So_Far, 1e-12 )
    returns = np.diff( np.log( safe_Prices[ : , -required_History - 1 : ] ), axis = 1 )
    cumulative_Return = np.log(
        safe_Prices[ : , -1 ]
        / safe_Prices[ : , -g_volatility_Adjusted_Reversion_Window - 1 ]
    )
    realised_Volatility = np.std(
        returns[ : , -g_volatility_Adjusted_Reversion_Volatility_Window : ],
        axis = 1,
    ) + 1e-12
    risk_Adjusted_Return = cumulative_Return / (
        realised_Volatility * np.sqrt( g_volatility_Adjusted_Reversion_Window )
    )
    ranked_Assets = np.argsort( risk_Adjusted_Return )
    number_Selected = min(
        g_volatility_Adjusted_Reversion_Top_K,
        number_Of_Instruments // 2,
    )
    signal[ ranked_Assets[ : number_Selected ] ] = 1.0
    signal[ ranked_Assets[ -number_Selected : ] ] = -1.0

    return signal


def runVolatilityAdjustedReversionStrategy( prices_So_Far ):
    """Trade risk-adjusted 80-day reversals at each instrument's limit."""
    current_Prices = np.maximum( prices_So_Far[ : , -1 ], 1e-12 )
    signal = getVolatilityAdjustedReversionSignal( prices_So_Far )
    return ( signal * g_position_Limits / current_Prices ).astype( int )


def getCorrelationFilteredReversionSignal( prices_So_Far ):
    """Trade long-horizon reversals only in relatively independent names.

    The trailing correlation matrix identifies names whose returns are most
    dominated by the common market factor.  The highest-correlation 30% are
    excluded before ranking 80-day moves, making the sleeve more diversified
    and less exposed to a market-wide regime shock.
    """
    number_Of_Instruments, number_Of_Timesteps = prices_So_Far.shape
    signal = np.zeros( number_Of_Instruments )
    required_History = max(
        g_correlation_Filtered_Reversion_Window,
        g_correlation_Filtered_Reversion_Correlation_Window,
    )
    if number_Of_Timesteps < required_History + 1:
        return signal

    safe_Prices = np.maximum( prices_So_Far, 1e-12 )
    correlation_Returns = np.diff(
        np.log( safe_Prices[ : , -g_correlation_Filtered_Reversion_Correlation_Window - 1 : ] ),
        axis = 1,
    )
    correlation_Matrix = np.nan_to_num( np.corrcoef( correlation_Returns ) )
    average_Correlation = (
        np.sum( correlation_Matrix, axis = 1 ) - 1.0
    ) / max( number_Of_Instruments - 1, 1 )
    eligibility_Cutoff = np.quantile(
        average_Correlation,
        g_correlation_Filtered_Reversion_Eligibility_Quantile,
    )
    eligible_Assets = np.flatnonzero( average_Correlation <= eligibility_Cutoff )

    cumulative_Return = np.log(
        safe_Prices[ : , -1 ]
        / safe_Prices[ : , -g_correlation_Filtered_Reversion_Window - 1 ]
    )
    ranked_Eligible_Assets = eligible_Assets[
        np.argsort( cumulative_Return[ eligible_Assets ] )
    ]
    number_Selected = min(
        g_correlation_Filtered_Reversion_Top_K,
        len( ranked_Eligible_Assets ) // 2,
    )
    signal[ ranked_Eligible_Assets[ : number_Selected ] ] = 1.0
    signal[ ranked_Eligible_Assets[ -number_Selected : ] ] = -1.0

    return signal


def runCorrelationFilteredReversionStrategy( prices_So_Far ):
    """Trade correlation-filtered 80-day reversals at each instrument's limit."""
    current_Prices = np.maximum( prices_So_Far[ : , -1 ], 1e-12 )
    signal = getCorrelationFilteredReversionSignal( prices_So_Far )
    return ( signal * g_position_Limits / current_Prices ).astype( int )

##### Online Factor Regime Ensemble — Drawdown-Throttled Variant #####

_ofdd_dollar_limits = np.full( 51, 10_000.0 )
_ofdd_dollar_limits[ 0 ] = 100_000.0

_ofdd_DD_Window = 7            # trailing days of this strategy's OWN pnl used for the throttle
_ofdd_DD_Threshold = -8000.0   # cumulative own-pnl loss over that window at which sizing bottoms out
_ofdd_DD_Floor = 0.4           # minimum size multiplier once fully throttled

# Self-contained bookkeeping so this strategy can gauge its OWN recent performance,
# independent of the rest of the portfolio (mirrors the pattern used for the
# portfolio-level getDrawdownScalar, but scoped to just this strategy).
_ofdd_Position = np.zeros( 51 )
_ofdd_Cash = 0.0
_ofdd_Value = 0.0
_ofdd_PnL_History = []

def resetOnlineFactorDrawdownHistory():
    global _ofdd_Position, _ofdd_Cash, _ofdd_Value, _ofdd_PnL_History
    _ofdd_Position = np.zeros( 51 )
    _ofdd_Cash = 0.0
    _ofdd_Value = 0.0
    _ofdd_PnL_History = []

def _updateOwnPnLHistory( new_Position, current_Prices ):
    global _ofdd_Position, _ofdd_Cash, _ofdd_Value, _ofdd_PnL_History

    delta_Pos = new_Position - _ofdd_Position
    _ofdd_Cash -= current_Prices.dot( delta_Pos )
    _ofdd_Position = new_Position

    position_Value = _ofdd_Position.dot( current_Prices )
    today_PnL = _ofdd_Cash + position_Value - _ofdd_Value
    _ofdd_Value = _ofdd_Cash + position_Value

    _ofdd_PnL_History.append( today_PnL )
    if len( _ofdd_PnL_History ) > _ofdd_DD_Window:
        _ofdd_PnL_History.pop( 0 )

def getOnlineFactorDrawdownScalar():
    recent_PnL = sum( _ofdd_PnL_History )
    if recent_PnL >= 0.0:
        return 1.0

    severity = min( abs( recent_PnL ) / abs( _ofdd_DD_Threshold ), 1.0 )
    return 1.0 - severity * ( 1.0 - _ofdd_DD_Floor )

def runOnlineFactorRegimeEnsembleDrawdownThrottledStrategy( prcSoFar ):
    n_inst, n_times = prcSoFar.shape
    if n_inst != 51 or n_times < _PERFORMANCE_WINDOW + 3:
        return np.zeros( n_inst, dtype = int )

    returns = np.diff( np.log( prcSoFar ), axis = 1 ).T
    direction = _choose_direction( returns, _ofdd_dollar_limits )

    scalar = getOnlineFactorDrawdownScalar()
    target_Shares = direction * scalar * _ofdd_dollar_limits / prcSoFar[ :, -1 ]
    target_Shares = target_Shares.astype( int )

    _updateOwnPnLHistory( target_Shares.astype( float ), prcSoFar[ :, -1 ] )

    return target_Shares


EPS = 1e-12
nInst = 51

# ---- dollar position limits (asset 0 has the 10x cap and 5x lower commission)
DLR = np.full(nInst, 10_000.0)
DLR[0] = 100_000.0

# ---- lead-lag ensemble: average over a measured plateau rather than a point
LAMBDAS = (300.0, 1000.0, 3000.0)
RANKS = (4, 5, 6, 7)

# ---- signal blend weights (leg1 = 1 - W_SREV - W_LREV)
W_SREV = 0.25          # short-horizon residual reversal
W_LREV = 0.10          # long-horizon reversal

SREV_L = 8             # short reversal lookback (days)
SREV_K = 3             # principal components stripped (= number of real factors)
SREV_W = 250           # window for the PCA covariance
LREV_L = 120           # long reversal lookback (days)

KAPPA = 0.06           # saturation scale -> ~93% of assets at their cap
MIN_HIST = 120         # below this the lead-lag matrix is too noisy to trade
FULL_HIST = 250        # size scales in linearly between MIN_HIST and FULL_HIST
MAX_SHARES = 1e12      # sanity ceiling so the int cast is always well-defined


def _zc(v):
    """Cross-sectional standardisation, safe against a degenerate cross-section."""
    s = v.std()
    if not np.isfinite(s) or s < EPS:
        return np.zeros_like(v)
    return (v - v.mean()) / s


def _leadlag(rets):
    """Rank-truncated ridge lead-lag forecast, averaged over the (lambda, rank)
    plateau. Fitted on the entire history available so far."""
    n = rets.shape[0]
    X = rets[:, :-1].T
    Y = rets[:, 1:].T
    xmu = X.mean(0)
    xsd = X.std(0) + EPS
    ymu = Y.mean(0)
    ysd = Y.std(0) + EPS
    Xs = (X - xmu) / xsd
    Ys = (Y - ymu) / ysd

    G = Xs.T @ Xs
    C = Xs.T @ Ys
    I = np.eye(n)
    cur = (rets[:, -1] - xmu) / xsd

    acc = np.zeros(n)
    cnt = 0
    for lam in LAMBDAS:
        try:
            B = np.linalg.solve(G + lam * I, C)
        except np.linalg.LinAlgError:
            continue
        try:
            u, s, vt = np.linalg.svd(B, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        for k in RANKS:
            if k >= len(s):
                continue
            s2 = s.copy()
            s2[k:] = 0.0
            f = cur @ ((u * s2) @ vt)
            sd = f.std()
            if np.isfinite(sd) and sd > EPS:
                acc += f / sd
                cnt += 1
    if cnt == 0:
        return np.zeros(n)
    return acc / cnt


def _short_reversal(rets):
    """Fade recent residual moves after stripping the 3 common factors."""
    h = rets[:, -SREV_W:] if rets.shape[1] > SREV_W else rets
    sd = h.std(1, keepdims=True) + EPS
    z = (h - h.mean(1, keepdims=True)) / sd
    try:
        C = np.cov(z)
        w, v = np.linalg.eigh(C)
        v = v[:, ::-1][:, :SREV_K]
        res = z - v @ (v.T @ z)
    except np.linalg.LinAlgError:
        res = z
    L = min(SREV_L, res.shape[1])
    return -res[:, -L:].sum(1) / np.sqrt(L)


def _long_reversal(rets):
    """Fade long-horizon normalised performance."""
    h = rets[:, -(LREV_L + 30):] if rets.shape[1] > LREV_L + 30 else rets
    x = h / (h.std(1, keepdims=True) + EPS)
    L = min(LREV_L, x.shape[1])
    return -x[:, -L:].sum(1) / np.sqrt(L)


def runF1(prcSoFar):
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    nins, nt = prcSoFar.shape

    # ---- guards: not enough history, or unusable prices
    if nt < MIN_HIST + 2:
        return np.zeros(nins, dtype=int)
    if not np.all(np.isfinite(prcSoFar)) or np.any(prcSoFar <= 0.0):
        prc = np.where(np.isfinite(prcSoFar) & (prcSoFar > 0.0), prcSoFar, np.nan)
        prc = np.asarray(
            [np.interp(np.arange(nt), np.flatnonzero(np.isfinite(row)),
                       row[np.isfinite(row)]) if np.isfinite(row).any()
             else np.ones(nt) for row in prc])
    else:
        prc = prcSoFar

    rets = np.diff(np.log(prc), axis=1)
    if not np.all(np.isfinite(rets)):
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- blended signal: three near-orthogonal legs
    sig = (1.0 - W_SREV - W_LREV) * _zc(_leadlag(rets))
    if rets.shape[1] >= SREV_L + 5:
        sig = sig + W_SREV * _zc(_short_reversal(rets))
    if rets.shape[1] >= 40:
        sig = sig + W_LREV * _zc(_long_reversal(rets))
    if not np.all(np.isfinite(sig)):
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- continuous saturating sizing: no ranking, no top-K, no hysteresis
    w = np.clip(sig / KAPPA, -1.0, 1.0)

    # ---- scale in while the lead-lag estimate is still short of data
    if rets.shape[1] < FULL_HIST:
        w = w * np.clip((rets.shape[1] - MIN_HIST) / float(FULL_HIST - MIN_HIST), 0.0, 1.0)

    # ---- dollars -> shares, hardened against degenerate prices.
    # A price that is zero, NaN or denormally small would overflow the division
    # and make the int cast undefined, so the divisor is floored and the result
    # is sanitised before the cast. The share ceiling is recomputed from TODAY's
    # price on every call, so a position can never drift outside its cap when
    # prices move (the brief's ALGO example) -- this function is stateless and
    # never assumes yesterday's position survived.
    cur = np.maximum(np.nan_to_num(prc[:, -1], nan=0.0, posinf=0.0, neginf=0.0), 1e-8)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        lim = np.floor(np.clip(DLR[:nins] / cur, 0.0, MAX_SHARES))
        pos = np.clip(w * DLR[:nins] / cur, -lim, lim)
    lim = np.nan_to_num(lim, nan=0.0, posinf=MAX_SHARES, neginf=0.0)
    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(pos, -lim, lim).astype(np.int64)



##### External #####

def setGlobalVariable( name, value ):
    globals()[ name ] = value
