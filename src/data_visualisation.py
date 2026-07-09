
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


def runAllDataOverlayedVisualisation():
    global g_data_File_Name
    prices_Data = pd.read_csv( g_data_File_Name, sep=r"\s+", header=0, index_col=None )

    day_Number_Vector = np.linspace( 0, 499, 500 )

    for instrument_Num in range( 0, 50, 1):
        plt.plot( day_Number_Vector, prices_Data.iloc[ 0:500:1 , instrument_Num ], label=prices_Data.columns[ instrument_Num ] )
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
    plt.suptitle( "Moving Avergages of Asset " + "asset_Idx" )
    plt.plot( day_Number_Vector, prices_Values[ asset_Idx ], label = "Trading price" )

    for moving_Average_Idx in range( number_Of_Moving_Averages ):
        plt.plot( day_Number_Vector, moving_Average_Series_Array[ moving_Average_Idx ], label = "Window size = " + str( moving_Averages_Window_Size_Array[ moving_Average_Idx ] ) )

    plt.legend()

    plt.show()




if __name__ == "__main__":
    runDataVisualisationMenu()


