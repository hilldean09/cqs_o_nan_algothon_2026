import numpy as np
import torch
from torch import nn

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
current_Position = np.zeros( g_number_Of_Instruments )

g_trade_History_Buffer_Size = 3
# TODO: Introduce PnL function
g_previous_PnL_Buffer = np.zeros( g_trade_History_Buffer_Size )

g_position_History_Buffer = np.zeros( ( g_number_Of_Instruments, g_trade_History_Buffer_Size ) )

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

    mean_Of_Mean_Log_Retursn = sum_Of_Mean_Log_Returns / window_Size

    if( mean_Of_Mean_Log_Retursn == 0.0 ):
        logWarningHeader( "getMarketMovingMeanLogReturns",  "Returning 0" )

    return mean_Of_Mean_Log_Retursn


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



##### Neural Net #####

g_nn_number_Of_Outputs = 3
g_nn_output_History_Buffer = np.zeros( ( g_trade_History_Buffer_Size, g_nn_number_Of_Outputs ) )

def updateNeuralNetOutputHistory( outputs ):
    global g_nn_output_History_Buffer

    g_nn_output_History_Buffer = np.roll( g_nn_output_History_Buffer )
    g_nn_output_History_Buffer[ 0 ] = outputs

# Input parameters
g_nn_Short_Market_Moving_Mean_Log_Returns_Window_Size = 50
g_nn_Long_Market_Moving_Mean_Log_Returns_Window_Size = 10
g_nn_Market_Correlation_Window_Size = 5
g_nn_Market_Statistical_Realized_Volatility_Window_Size = 5

def getNeuralNetInputs( prices_So_Far, timestep_Idx ):
    global g_nn_Short_Market_Moving_Mean_Log_Returns_Window_Size
    global g_nn_Long_Market_Moving_Mean_Log_Returns_Window_Size
    global g_nn_Market_Correlation_Window_Size
    global g_nn_Market_Statistical_Realized_Volatility_Window_Size
    global g_nn_output_History_Buffer

    state = []

    state.append( timestep_Idx )

    # Previous outputs
    for last_Output in g_nn_number_Of_Outputs:
        state.append( last_Output )

    # Market log mean returns (long and short)
    state.append( getMarketMovingMeanLogReturns( prices_So_Far, timestep_Idx, g_nn_Short_Market_Moving_Mean_Log_Returns_Window_Size ) )
    state.append( getMarketMovingMeanLogReturns( prices_So_Far, timestep_Idx, g_nn_Long_Market_Moving_Mean_Log_Returns_Window_Size ) )

    # Average correlation
    state.append( getCorrelationMatrixAndMeanCorrelation( prices_So_Far, timestep_Idx, g_nn_Market_Correlation_Window_Size )[ 1 ] )
    
    # Realized volatility
    market_Statistical_Volatility = getMarketStatisticalAssetLogVolatility( prices_So_Far, timestep_Idx, g_nn_Market_Statistical_Realized_Volatility_Window_Size )
    state.append( market_Statistical_Volatility[ 0 ] )
    state.append( market_Statistical_Volatility[ 1 ] )

    return state


# Getting acclerator
g_torch_Device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class MasterNeuralNet( nn.Module ):
    def __init__( self, number_Of_Inputs, number_Of_Outputs ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear( number_Of_Inputs, int( number_Of_Inputs * 1.5 )  ),
            nn.ReLU(),
            nn.Linear( int( number_Of_Inputs * 1.5 ), int( number_Of_Inputs * 1.5 ) ),
            nn.ReLU(),
            nn.Linear( int( number_Of_Inputs * 1.5 ), int( number_Of_Outputs * 1.5 ) ),
            nn.ReLU(),
            nn.Linear( int( number_Of_Outputs * 1.5 ), number_Of_Outputs )
        )

    def forward( self, x ):
        x = self.flatten( x )
        logits = self.linear_relu_stack( x )
        return logtts


##### External #####

def setGlobalVariable( name, value ):
    globals()[ name ] = value

