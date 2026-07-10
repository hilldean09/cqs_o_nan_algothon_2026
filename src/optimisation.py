import numpy as np
import torch
import itertools

import eval
import CQS_O_NaN as onan


# Note to self : use globals()[ name ]
def optimiseParametersForScore( parameter_Name_List, parameter_Value_Range_List, visualise = False ):
    number_Of_Parameters = len( parameter_Name_List )

    scoreDefaultParam = 1.0
    prcAll = eval.loadPrices(pricesFile)

    parameter_Combination_Array_Dim = np.array( number_Of_Parameters, dtype = int )
    for parameter_Idx in range( number_Of_Parameters ):
        parameter_Combination_Array_Dim[ parameter_Idx ] = len( parameter_Value_Range_List[ parameter_Idx ] )

    if visualise == True:
        score_Array = np.zeros( list( parameter_Combination_Array_Dim ) )

    parameter_Combination_Array = list( itertools.product( *parameter_Value_Range_List ) )
    
    # Max score initial
    max_Score = -9999
    max_Score_Parameters = np.zeros( number_Of_Parameters )

    parmaeter_Combination = np.zeros( number_Of_Parameters )

    for parameter_Combination_Index in np.ndindex( parameter_Combination_Array_Dim ):
        parameter_Combination = parameter_Combination_Array[ *parameter_Combination_Index ]

        # Updating global variables
        for parameter_Idx in range( number_Of_Parameters ):
            globals()[ parameter_Name_List[ parameter_Idx ] ] = parameter_Combination[ parameter_Idx ]

        # Copied from eval.py
        meanpl, ret, plstd, sharpe, dvol = eval.calcPL(prcAll, numTestDays)
        score = eval.score(meanpl, plstd, scoreDefaultParam)

        if visualise == True:
            score_Array[ parameter_Combination_Index ] = score

        if( score > max_Score ):
            max_Score_Parameters = parameter_Combination

    # TODO: Implemenet visual

    for parameter_Idx in range( number_Of_Parameters ):
        globals()[ parameter_Name_List[ parameter_Idx ] ] = max_Score_Parameters[ parameter_Idx ]

    return max_Score_Parameters


# Change to run
if __name__ == "__main__":
    number_Of_Parameters = 2

    parameter_Name_List = [ "g_ma_Crossover_Short_Window_Size", "g_ma_Crossover_Long_Window_Size" ]
    parameter_Value_Range_List = [ range( 0, 50, 1 ), range( 0, 75, 1 ) ]

    print( optimiseParametersForScore( parameter_Name_List, parameter_Value_Range_List ) )


