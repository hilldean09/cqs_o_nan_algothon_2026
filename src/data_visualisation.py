
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import CQS_O_NaN as onan

g_data_File_Name = "prices.txt"

def runDataVisualisationMenu():
    print( "##### Data Visualisation Menu #####" )
    print( " " )
    print( "Options : " )
    print( "\t0 : All Data Overlayed" )
    print( "\t1 : Pearson Correlation Matrix" )
    print( "\t2 : Asset with Multiple Moving Averages" )
    print( "\t3 : Moving Average Relaive Realized Volatility" )
    print( "\t4 : Arithmetic Drift" )
    print( " " )
    print( "Enter choice : ", end="" )

    user_Visualisation_Choice_String = input()
    user_Visualisation_Choice_Int = int( user_Visualisation_Choice_String )

    print( " " )

    if( user_Visualisation_Choice_Int == 0 ):
        runAllDataOverlayedVisualisation()
    elif( user_Visualisation_Choice_Int == 1 ):
        runPearsonCorrelationCoefficientVisualisation()
    elif( user_Visualisation_Choice_Int == 2 ):
        runAssetWithMultipleMovingAveragesVisualisation()
    elif( user_Visualisation_Choice_Int == 3 ):
        runMovingAverageRelativeVolatilityVisualisation()
    elif( user_Visualisation_Choice_Int == 4 ):
        runArithmeticDriftVisualisation()



def runAllDataOverlayedVisualisation():
    global g_data_File_Name
    prices_Data = pd.read_csv( g_data_File_Name, sep=r"\s+", header=0, index_col=None )
    prices_Values = np.asarray( ( prices_Data.values ).T )

    ( number_Of_Instruments, number_Of_Timesteps ) = prices_Values.shape

    day_Number_Vector = range( number_Of_Timesteps )

    number_Of_Watched_Assets_String = input( "Enter number of assets to watch (or all): " )
    number_Of_Watched_Assets = 0

    if( number_Of_Watched_Assets_String == "all" ):
        # All case
        number_Of_Watched_Assets = number_Of_Instruments
        asset_Idx_Array = np.asarray( range( number_Of_Watched_Assets ) )
    else:
        # Integer case
        number_Of_Watched_Assets = int( number_Of_Watched_Assets_String )
        asset_Idx_Array = np.zeros( number_Of_Watched_Assets, dtype=int )

        for asset_Idx_Selection_Idx in range( number_Of_Watched_Assets ):
            asset_Idx_Array[ asset_Idx_Selection_Idx ] = int( input( "Enter asset index for asset selection " + str( asset_Idx_Selection_Idx ) + " : " ) )

    plt.suptitle( "Raw Asset Prices" )

    for asset_Idx_Array_Idx in range( number_Of_Watched_Assets ):
        asset_Idx = asset_Idx_Array[ asset_Idx_Array_Idx ]
        plt.plot( day_Number_Vector, prices_Values[ asset_Idx ], label = "Asset " + str( asset_Idx_Array[ asset_Idx_Array_Idx ] ) )

    if( number_Of_Watched_Assets <= 10 ):
        plt.legend()

    plt.show()

def runPearsonCorrelationCoefficientVisualisation():
    global g_data_File_Name
    prices_Data = pd.read_csv( g_data_File_Name, sep=r"\s+", header=0, index_col=None )
    prices_Values = ( prices_Data.values ).T

    ( number_Of_Instruments, number_Of_Timesteps ) = prices_Values.shape

    day_Number_Vector = range( number_Of_Timesteps )

    window_Size = int( input( "Select window size : " ) )
    print( " " )

    colour_Map = "cool"

    correlation_Matrix = np.zeros( ( number_Of_Instruments, number_Of_Instruments ) )

    plt.suptitle( "Pearson Correlation Matrix" )
    plt.title( "window_Size = " + str( window_Size ) + " - day_Num = " + str( 0 ) )
    plt.imshow( correlation_Matrix, cmap = colour_Map )
    plt.show(block=False)

    for day_Num in day_Number_Vector:
        correlation_Matrix = onan.getPearsonCorrelationMatrix( prices_Values, day_Num, window_Size )
        normalised_Correlation_Matrix = ( correlation_Matrix / 2 ) + 0.5

        # Updating plot
        plt.suptitle( "Pearson Correlation Matrix" )
        plt.title( "window_Size = " + str( window_Size ) + " - day_Num = " + str( day_Num ) )
        plt.imshow( normalised_Correlation_Matrix, cmap = colour_Map )

        plt.draw()
        plt.pause( 0.001 )
        plt.clf()

    plt.suptitle( "Pearson Correlation Matrix" )
    plt.title( "window_Size = " + str( window_Size ) + " - day_Num = " + str( day_Num ) )
    plt.imshow( normalised_Correlation_Matrix, cmap = colour_Map  )
    plt.show()

def runAssetWithMultipleMovingAveragesVisualisation():
    global g_data_File_Name
    prices_Data = pd.read_csv( g_data_File_Name, sep=r"\s+", header=0, index_col=None )
    prices_Values = ( prices_Data.values ).T

    ( number_Of_Instruments, number_Of_Timesteps ) = prices_Values.shape

    day_Number_Vector = range( number_Of_Timesteps )

    # Getting inputs
    asset_Idx = int( input( "Enter desired asset index : " ) )
    number_Of_Moving_Averages = int( input( "Enter desired number of moving averages : " ) )

    moving_Averages_Window_Size_Array = np.zeros( number_Of_Moving_Averages )

    for moving_Average_Idx in range( number_Of_Moving_Averages ):
        moving_Averages_Window_Size_Array[ moving_Average_Idx ] = int( input( "Enter window size for moving average " + str( moving_Average_Idx ) + " : " ) )

    print( " " )

    # Calculating moving average series
    moving_Average_Series_Array = np.zeros( ( number_Of_Moving_Averages, number_Of_Timesteps ) )

    for moving_Average_Idx in range( number_Of_Moving_Averages ):
        for timestep_Idx in range( number_Of_Timesteps ):
            moving_Average_Series_Array[ moving_Average_Idx ][ timestep_Idx ] = onan.getAssetMovingAverage( prices_Values, asset_Idx, timestep_Idx, moving_Averages_Window_Size_Array[ moving_Average_Idx ] )

    # Plotting
    plt.suptitle( "Moving Avergages of Asset " + str( asset_Idx ) )
    plt.plot( day_Number_Vector, prices_Values[ asset_Idx ], label = "Trading price" )

    for moving_Average_Idx in range( number_Of_Moving_Averages ):
        plt.plot( day_Number_Vector, moving_Average_Series_Array[ moving_Average_Idx ], label = "Window size = " + str( moving_Averages_Window_Size_Array[ moving_Average_Idx ] ) )

    plt.legend()

    plt.show()

def runMovingAverageRelativeVolatilityVisualisation():
    global g_data_File_Name
    prices_Data = pd.read_csv( g_data_File_Name, sep=r"\s+", header=0, index_col=None )
    prices_Values = ( prices_Data.values ).T

    ( number_Of_Instruments, number_Of_Timesteps ) = prices_Values.shape

    day_Number_Vector = range( number_Of_Timesteps )

    # Getting inputs
    number_Of_Watched_Assets_String = input( "Enter number of assets to watch (or all): " )
    number_Of_Watched_Assets = 0

    if( number_Of_Watched_Assets_String == "all" ):
        # All case
        number_Of_Watched_Assets = number_Of_Instruments
        asset_Idx_Array = np.asarray( range( number_Of_Watched_Assets ) )
    else:
        # Integer case
        number_Of_Watched_Assets = int( number_Of_Watched_Assets_String )
        asset_Idx_Array = np.zeros( number_Of_Watched_Assets, dtype=int )

        for asset_Idx_Selection_Idx in range( number_Of_Watched_Assets ):
            asset_Idx_Array[ asset_Idx_Selection_Idx ] = int( input( "Enter asset index for asset selection " + str( asset_Idx_Selection_Idx ) + " : " ) )

    moving_Average_Window_Size = int( input( "Enter moving average window size : " ) )
    realized_Volatility_Window_Size = int( input( "Enter realized volatility window size : " ) )

    # Evaluating
    ma_Relative_RVol_Series_Array = np.zeros( ( number_Of_Watched_Assets, number_Of_Timesteps ) )

    for asset_Idx_Array_Idx in range( number_Of_Watched_Assets ):
        for timestep_Idx in range( number_Of_Timesteps ):
            ma_Relative_RVol_Series_Array[ asset_Idx_Array_Idx ][ timestep_Idx ] = onan.getAssetMARelativeRealizedVolatility( prices_Values, asset_Idx_Array[ asset_Idx_Array_Idx ], timestep_Idx, moving_Average_Window_Size, realized_Volatility_Window_Size )

    # Visualising
    plt.suptitle( "Moving Average Relative Realized Volatility" )
    plt.title( "moving_Average_Window_Size = " + str( moving_Average_Window_Size ) + " - realized_Volatility_Window_Size = " + str( realized_Volatility_Window_Size ) )
    plt.ylim( 0, 0.4 )

    for asset_Idx_Array_Idx in range( number_Of_Watched_Assets ):
        plt.plot( day_Number_Vector, ma_Relative_RVol_Series_Array[ asset_Idx_Array_Idx ], label = "Asset " + str( asset_Idx_Array[ asset_Idx_Array_Idx ] ) )

    if( number_Of_Watched_Assets <= 10 ):
        plt.legend()

    plt.show()

def runArithmeticDriftVisualisation():
    global g_data_File_Name
    prices_Data = pd.read_csv( g_data_File_Name, sep=r"\s+", header=0, index_col=None )
    prices_Values = ( prices_Data.values ).T

    ( number_Of_Instruments, number_Of_Timesteps ) = prices_Values.shape

    day_Number_Vector = range( number_Of_Timesteps )

    number_Of_Watched_Assets_String = input( "Enter number of assets to watch (or all): " )
    number_Of_Watched_Assets = 0

    if( number_Of_Watched_Assets_String == "all" ):
        # All case
        number_Of_Watched_Assets = number_Of_Instruments
        asset_Idx_Array = np.asarray( range( number_Of_Watched_Assets ) )
    else:
        # Integer case
        number_Of_Watched_Assets = int( number_Of_Watched_Assets_String )
        asset_Idx_Array = np.zeros( number_Of_Watched_Assets, dtype=int )

        for asset_Idx_Selection_Idx in range( number_Of_Watched_Assets ):
            asset_Idx_Array[ asset_Idx_Selection_Idx ] = int( input( "Enter asset index for asset selection " + str( asset_Idx_Selection_Idx ) + " : " ) )

    moving_Average_Window_Size = int( input( "Enter moving average window size : " ) )
    realized_Volatility_Window_Size = int( input( "Enter realized volatility window size : " ) )
    arithmetic_Drift_Window_Size = int( input( "Enter arithmetic drift window size : " ) )

    # Evaluating
    arithmetic_Drift_Series_Array = np.zeros( ( number_Of_Watched_Assets, number_Of_Timesteps ) )

    for asset_Idx_Array_Idx in range( number_Of_Watched_Assets ):
        for timestep_Idx in range( number_Of_Timesteps ):
            arithmetic_Drift_Series_Array [ asset_Idx_Array_Idx ][ timestep_Idx ] = onan.getAssetArithmeticDrift( prices_Values, asset_Idx_Array[ asset_Idx_Array_Idx ], timestep_Idx,arithmetic_Drift_Window_Size, realized_Volatility_Window_Size )

    # Visualising
    plt.suptitle( "Arithmetic Drift" )
    plt.title( "realized_Volatility_Window_Size = " + str( realized_Volatility_Window_Size ) + " - arithmetic_Drift_Window_Size = " + str( arithmetic_Drift_Window_Size ) )

    for asset_Idx_Array_Idx in range( number_Of_Watched_Assets ):
        plt.plot( day_Number_Vector, arithmetic_Drift_Series_Array[ asset_Idx_Array_Idx ], label = "Asset " + str( asset_Idx_Array[ asset_Idx_Array_Idx ] ) )

    if( number_Of_Watched_Assets <= 10 ):
        plt.legend()

    plt.show()


if __name__ == "__main__":
    runDataVisualisationMenu()


