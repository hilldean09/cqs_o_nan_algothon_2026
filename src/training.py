# Training the master neural
# network

import random
import numpy as np
import pandas as pd
import maptlotlib.pyplot as plt
import torch

import CQS_O_NaN as onan

class Training_Environment():

    def _setInitialConditions( self ):
        self.episode_Size = random.randint( 50, 500 )
        self.episode_Start = random.randint( 0, self.number_Of_Timesteps - self.episode_Size )
        self.prices_Values = self.all_Prices_Values[ : , self.episode_Start : self.episode_Start + self.episode_Size + 1 : 1 ]
        self.timestep_Idx = 0
        self.previous_Position = np.zeros( self.number_Of_Instruments )

        self.cash = 0
        self.value = 0

    def _setTimestepPricesSoFar( self, timestep_Idx ):
        self.prices_So_Far = self.prices_Values[ : , : timesteps_Idx + 1 ]

    def __init__( self, file_Name ):
        prices_Data = pd.read_csv( file_Name, sep=r"\s+", header=0, index_col=None )
        self.all_Prices_Values = np.asarray( ( prices_Data.values ).T )

        ( self.number_Of_Instruments, self.number_Of_Timesteps ) = self.all_Prices_Values.shape
        
        self._setInitialConditions()


    # TODO: Write implementation
    def _calculateStepReward( self, new_Position ):
        # TODO: Add position limit clipping
        current_Prices = self.prices_So_Far[ -1 ]
        delta_Pos = new_Position - self.previous_Position

        position_Value = new_Position.dot( current_Prices )

        self.cash -= current_Prices.dot( delta_Pos )

        today_PnL = cash + position_Value - self.value
        self.value = self.cash + position_Value

        return today_PnL


    def step( self, logits ):
        # returns a state vector corresponding to 
        # the neural net inputs
        return_Position = onan.runNeuralNetMasterStrategy( self.prices_So_Far, logits )

        reward = self._calculateReward( return_Position )

        self.previous_Position = return_Position
        self.timestep += 1

        if( self.timestep_Idx < self.episode_Size ):
            self._setTimestepPricesSoFar( self.timestep_Idx )
            output_State = getNeuralNetInputs( self.prices_So_Far, self.timestep_Idx )
            output_End = False
        else:
            output_End = True

        return output_State, reward, output_End


    def reset( self ):
        self._setInitialConditions()
        self._setTimestepPricesSoFar( self.timestep_Idx )


