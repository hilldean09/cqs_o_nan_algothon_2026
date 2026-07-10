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
g_strategy_Selection_Enum = 0

# NOTE: We cannot change the argument variable name from
# prcSoFar
def getMyPosition( prcSoFar ):
    global current_Position
    ( number_Of_Instruments, number_Of_Timesteps ) = prcSoFar.shape

    if( number_Of_Timesteps < 2 ):
        return np.zeros( number_Of_Instruments )

    last_Return  = np.log( prcSoFar[ :, -1 ] / prcSoFar[ :, -2 ] )
    last_Return_Norm = np.sqrt( last_Return.dot( last_Return ) )

    last_Return /= last_Return_Norm
    return_Position = np.array( [ int( x ) for x in 5000 * last_Return / prcSoFar[ :, -1 ] ] )
    current_Position = np.array( [ int( x ) for x in current_Position + return_Position ] )

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
def getAssetLogDrift( prices_So_Far, asset_Idx, desired_Latest_Day, desired_Window_Size ):
    ( number_Of_Instruments, number_Of_Timesteps ) = prices_So_Far.shape
    
    latest_Day = min( desired_Latest_Day, number_Of_Timesteps - 1 )

    window_Size = desired_Window_Size
    # Setting the window to the maximum 
    # available size if the entire desired
    # window size is not available
    if( latest_Day - window_Size + 1 < 0 ):
        window_Size = latest_Day + 1

    window_Start_Day = int( latest_Day - window_Size + 1 )

    sum_Of_Log_Rate = 0
    for start_Day_Offset in range( window_Size - 1 ):
        sum_Of_Log_Rate += np.log( prices_So_Far[ asset_Idx ][ window_Start_Day + start_Day_Offset ] / prices_So_Far[ asset_Idx ][ window_Start_Day + start_Day_Offset + 1 ] )

    mean_Of_Log_Rate = sum_Of_Log_Rate / ( window_Size + 1 )

    return mean_Of_Log_Rate

def getAssetArithmeticDrift( prices_So_Far, asset_Idx, desired_Latest_Day, drift_Window_Size, realized_Volatility_Window_Size ):
    log_Drift = getAssetLogDrift( prices_So_Far, asset_Idx, desired_Latest_Day, drift_Window_Size )
    realized_Volatility = getAssetRealizedVolatility( prices_So_Far, asset_Idx, desired_Latest_Day, realized_Volatility_Window_Size )

    arithmetic_Drift = log_Drift + ( realized_Volatility ** 2 ) / 2

    return arithmetic_Drift

def getMARelativeAssetArithmeticDrift( prices_So_Far, asset_Idx, desired_Latest_Day, moving_Average_Window_Size, drift_Window_Size, realized_Volatility_Window_Size ):
    arithmetic_Drift = getAssetArithmeticDrift( prices_So_Far, asset_Idx, desired_Latest_Day, drift_Window_Size, realized_Volatility_Window_Size )
    moving_Average = getAssetMovingAverage( prices_So_Far, asset_Idx, desired_Latest_Day, moving_Average_Window_Size )

    ma_Relative_Arithmetic_Drift = arithmetic_Drift / moving_Average

    return ma_Relative_Arithmetic_Drift








##### Strategies #####




