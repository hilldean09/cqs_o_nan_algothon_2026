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
#   TODO: Write logging functions


##### Code Start #####

g_number_Of_Instruments = 51
current_Position = np.zeros( g_number_Of_Instruments )

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


# TODO: Implement
def getPairPearsonCorrelationValue( prices_So_Far, latest_Day, window_Size, first_Asset_Idx, second_Asset_Idx ):
    return 0.0

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
            correlation_Matrix[ first_Asset_Idx ][ second_Asset_Idx ] = getPairPearsonCorrelationValue( prices_So_Far, latest_Day, widnows_Size, first_Asset_Idx, second_Asset_Idx )

    return correlation_Matrix








