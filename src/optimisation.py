import numpy as np
import itertools

import eval
import CQS_O_NaN as onan


# Note to self : use globals()[ name ]
def optimiseParametersForScore( parameter_Name_List, parameter_Value_Range_List, visualise = False ):
    number_Of_Parameters = len( parameter_Name_List )

    # Copied from eval
    pricesFile = "./prices.txt"
    numTestDays = 500
    scoreDefaultParam = 1.0
    prcAll = eval.loadPrices(pricesFile)
    prcAll = prcAll[ :, :750 ]
    print( prcAll.shape )

    parameter_Combination_Array_Dim = np.zeros( number_Of_Parameters, dtype = int )
    for parameter_Idx in range( number_Of_Parameters ):
        parameter_Combination_Array_Dim[ parameter_Idx ] = len( parameter_Value_Range_List[ parameter_Idx ] )

    if visualise == True:
        score_Array = np.zeros( list( parameter_Combination_Array_Dim ) )

    parameter_Combination_Array = np.asarray( list( itertools.product( *parameter_Value_Range_List ) ) )
    
    # Max score initial
    max_Score = -9999

    parameter_Combination = np.zeros( number_Of_Parameters )
    max_Score_Parameters = np.zeros( number_Of_Parameters )

    was_Updated = False

    for iterative_Index, parameter_Combination_Index in enumerate( np.ndindex( *parameter_Combination_Array_Dim ) ):
        parameter_Combination = parameter_Combination_Array[ iterative_Index ]

        # Updating global variables
        for parameter_Idx in range( number_Of_Parameters ):
            onan.setGlobalVariable( parameter_Name_List[ parameter_Idx ], parameter_Combination[ parameter_Idx ] )

        # Copied from eval.py
        meanpl, ret, plstd, sharpe, dvol = eval.calcPL(prcAll, numTestDays)
        score = eval.score(meanpl, plstd, scoreDefaultParam)

        if visualise == True:
            score_Array[ parameter_Combination_Index ] = score

        if( score > max_Score ):
            max_Score = score

            for parameter_Idx in range( number_Of_Parameters ):
                max_Score_Parameters[ parameter_Idx ] = parameter_Combination[ parameter_Idx ]

            print( str( max_Score_Parameters ) + " : " + str( max_Score ) )
            was_Updated = True

    # TODO: Implemenet visual

    for parameter_Idx in range( number_Of_Parameters ):
        onan.setGlobalVariable( parameter_Name_List[ parameter_Idx ], max_Score_Parameters[ parameter_Idx ] )

    print( max_Score )
    print( max_Score_Parameters )
    return max_Score_Parameters


# Change to run
if __name__ == "__main__":
    number_Of_Parameters = 2

    parameter_Name_List = [ "g_smac_weight", "g_spt_weight", "g_spmr_weight", "g_sal1_weight", "g_swal1_weight", "g_sofe_weight", "g_spauto_weight", "g_svars_weight" ]
    parameter_Value_Range_List = [ np.arange( 0.0, 1.0, 0.4 ), np.arange( 0.1, 3.2, 1.0 ), np.arange( 0.1, 3.2, 1.0 ), np.arange( 2.0, 5.0, 1.0 ), np.arange( 2.0, 5.0, 1.0 ), np.arange( 0.0, 5.0, 1.0 ), np.arange( 0.0, 3.0, 1.0 ), np.arange( 0.0, 3.0, 1.0 ) ]

    optimiseParametersForScore( parameter_Name_List, parameter_Value_Range_List, visualise = False )


