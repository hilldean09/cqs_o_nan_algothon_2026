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
        
        self.dollar_Position_Limit = np.full( self.number_Of_Instruments, 10_000 )
        self.dollar_Position_Limit[ 0 ] = 100_000

        self.reset()


    # TODO: Write implementation
    def _calculateStepReward( self, new_Position_Original ):
        current_Prices = self.prices_So_Far[ -1 ]

        position_Limits = ( self.dollar_Position_Limit / current_Prices ).astype( int )
        new_Position = np.clip( new_Position_Original, -position_Limits, position_Limits ).astype( int )

        delta_Pos = new_Position - self.previous_Position

        position_Value = new_Position.dot( current_Prices )
        self.cash -= current_Prices.dot( delta_Pos )

        today_PnL = cash + position_Value - self.value

        self.value = self.cash + position_Value

        reward = today_PnL

        return reward, today_PnL


    def step( self, logits ):
        # returns a state vector corresponding to 
        # the neural net inputs
        return_Position, log_Prob = onan.runNeuralNetMasterStrategy( self.prices_So_Far, logits )

        reward, today_PnL = self._calculateReward( return_Position )

        self.previous_Position = return_Position
        self.timestep += 1

        if( self.timestep_Idx < self.episode_Size ):
            self._setTimestepPricesSoFar( self.timestep_Idx )
            output_State = getNeuralNetInputs( self.prices_So_Far, self.timestep_Idx )
            output_End = False
        else:
            output_End = True

        return output_State, reward, output_End, log_Prob, today_PnL


    def reset( self ):
        self._setInitialConditions()
        self._setTimestepPricesSoFar( self.timestep_Idx )

        output_State = getNeuralNetInputs( self.prices_So_Far, self.timestep_Idx )

        return output_State, 0, False

def runEpisode( environement, policy, device, do_Print = False ):
    state, _ = environement.reset()

    log_Prob_Array = []
    reward_Array = []

    daily_PnL_Array = []

    done = False

    while not done:
        state_tensor = torch.as_tensor( state, dtype=torch.float32, device = device )

        logits = policy( state_tensor )

        next_State, reward, done, log_Prob, today_PnL = environement.step( logits )
        daily_PnL_Array.append( today_PnL )

        log_Prob_Array.append( log_Prob )
        reward_Array.append( reward )

        state = next_State

    if do_Print:
        mean_PnL = np.mean( daily_PnL_Array )
        std_PnL = np.std( daily_PnL_Array )

        sharpe_Ratio = mean_PnL / std_PnL

        print( "Perforamnce : " )
        print( "\tmean_PnL : " + str( mean_PnL ) )
        print( "\tstd_PnL : " + str( std_PnL ) )
        print( "\tSR : " + str( sharpe_Ratio ) )

    return log_Prob_Array, reward_Array

def computeReturns( reward_Array, gamma = 0.99 ):
    return_Array = []
    G = 0.0

    for reward in reversed( reward_Array ):
        G = reward + gamma * G
        return_Array.insert( 0, G )

    return_Array = torch.tensor( return_Array, dtype=torch.float32 )

    return_Array = ( return_Array - return_Array.mean() ) / return_Array.std() + 1e-8 )

    return return_Array

def computeLoss( log_Prob_Array, return_Array ):
    loss = 0
    for log_Prob, G in zip( log_Prob_Array, return_Array ):
        loss += -log_Prob * G

    return loss




