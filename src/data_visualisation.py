
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
    print( " " )
    print( "Enter choice : ", end="" )

    user_Visualisation_Choice_String = input()
    user_Visualisation_Choice_Int = int( user_Visualisation_Choice_String )

    print( " " )

    if( user_Visualisation_Choice_Int == 0 ):
        runAllDataOverlayedVisualisation()
    elif( user_Visualisation_Choice_Int == 1 ):
        runPearsonCorrelationCoefficientVisualisation()


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

    correlation_Matrix = np.zeros( ( number_Of_Instruments, number_Of_Instruments ) )

    plt.suptitle( "Pearson Correlation Matrix" )
    plt.title( "window_Size = " + str( window_Size ) + " - day_Num = " + str( 0 ) )
    plt.imshow( correlation_Matrix, cmap="cool" )
    plt.show(block=False)

    for day_Num in day_Number_Vector:
        correlation_Matrix = onan.getPearsonCorrelationMatrix( prices_Values, day_Num, window_Size )
        normalised_Correlation_Matrix = ( correlation_Matrix / 2 ) + 0.5

        # Updating plot
        plt.suptitle( "Pearson Correlation Matrix" )
        plt.title( "window_Size = " + str( window_Size ) + " - day_Num = " + str( day_Num ) )
        plt.imshow( normalised_Correlation_Matrix, cmap="cool" )

        plt.draw()
        plt.pause( 0.001 )
        plt.clf()

    plt.suptitle( "Pearson Correlation Matrix" )
    plt.title( "window_Size = " + str( window_Size ) + " - day_Num = " + str( day_Num ) )
    plt.imshow( normalised_Correlation_Matrix, cmap="cool" )
    plt.show()


if __name__ == "__main__":
    runDataVisualisationMenu()


