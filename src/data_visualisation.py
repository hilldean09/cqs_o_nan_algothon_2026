
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

g_data_File_Name = "prices.txt"

def run_Data_Visualisation_Menu():
    print( "##### Data Visualisation Menu #####" )
    print( " " )
    print( "Options : " )
    print( "\t0 : All Data Overlayed" )
    print( " " )
    print( "Enter choice : ", end="" )

    user_Visualisation_Choice_String = input()
    user_Visualisation_Choice_Int = int( user_Visualisation_Choice_String )

    if( user_Visualisation_Choice_Int == 0 ):
        run_All_Data_Overlayed_Visualisation()


def run_All_Data_Overlayed_Visualisation():
    global g_data_File_Name
    prices_Data = pd.read_csv( g_data_File_Name, sep=" " )

    day_Number_Vector = np.linspace( 0, 499, 500 )

    for instrument_Num in range( 0, 50, 1):
        plt.plot( day_Number_Vector, prices_Data.iloc[ 0:500:1 , instrument_Num ], label=prices_Data.columns[ instrument_Num ] )
    plt.show()


if __name__ == "__main__":
    run_Data_Visualisation_Menu()


