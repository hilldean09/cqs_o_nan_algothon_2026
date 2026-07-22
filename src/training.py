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

    def _setTimestepPricesSoFar( self, timestep_Idx ):
        self.prices_So_Far = self.prices_Values[ : , : timesteps_Idx + 1 ]

    def __init__( self, file_Name ):
        prices_Data = pd.read_csv( file_Name, sep=r"\s+", header=0, index_col=None )
        self.all_Prices_Values = np.asarray( ( prices_Data.values ).T )

        ( self.number_Of_Instruments, self.number_Of_Timesteps ) = self.all_Prices_Values.shape
        
        self._setInitialConditions()


    # TODO: Write implementation
    def _calculateReward( self, new_Position ):
        pass

    def step( self, logits ):
        # returns a state vector corresponding to 
        # the neural net inputs
        self._setTimestepPricesSoFar( self.timestep_Idx )
        return_Position = onan.runNeuralNetMasterStrategy( self.prices_So_Far, logits )

        reward = self._calculateReward

        self.timestep += 1

    # TODO: Write implementation
    def reset( self ):
        pass


