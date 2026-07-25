# Training the master neural
# network

import copy
import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import eval

import CQS_O_NaN as onan

class Training_Environment():

    def _setInitialConditions( self ):
        # self.episode_Size = random.randint( 50, 500 )
        self.episode_Size = 250
        self.episode_Start = random.randint( 0, self.number_Of_Timesteps - self.episode_Size - 50 )

        self.prices_Values = self.all_Prices_Values[ : ,self.episode_Start : self.episode_Start + self.episode_Size + 1 : 1 ]

        self.timestep_Idx = 0
        self.previous_Position = np.zeros( self.number_Of_Instruments )

        self.cash = 0.0
        self.value = 0.0

    def _setTimestepPricesSoFar( self, timestep_Idx ):
        self.prices_So_Far = self.prices_Values[ : , 0 : timestep_Idx + 1 : 1 ]


    def reset( self ):
        self._setInitialConditions()
        self._setTimestepPricesSoFar( self.timestep_Idx )

        output_State = onan.getNeuralNetInputs( self.prices_So_Far, self.timestep_Idx )
        output_State = np.add( output_State, 1e-2 )

        return output_State, 0, False

    def __init__( self, file_Name = "prices.txt" ):
        prices_Data = pd.read_csv( file_Name, sep=r"\s+", header=0, index_col=None )
        self.all_Prices_Values = np.asarray( ( prices_Data.values ).T )

        ( self.number_Of_Instruments, self.number_Of_Timesteps ) = self.all_Prices_Values.shape
        
        self.dollar_Position_Limit = np.full( self.number_Of_Instruments, 10_000 )
        self.dollar_Position_Limit[ 0 ] = 100_000

        self.reset()

    def _calculateStepReward( self, new_Position_Original ):
        current_Prices = self.prices_So_Far[ : , -1 ].T

        position_Limits = ( self.dollar_Position_Limit / current_Prices ).astype( int )
        new_Position = np.clip( new_Position_Original, -position_Limits, position_Limits ).astype( int )

        delta_Pos = np.subtract( new_Position , self.previous_Position )

        position_Value = new_Position.dot( current_Prices )
        self.cash -= current_Prices.dot( delta_Pos )

        today_PnL = self.cash + position_Value - self.value

        """
        print( "_calculateStepReward : " )
        print( "\ttoday_PnL : " + str( today_PnL ) )
        print( "\tposition_Value : " + str( position_Value ) )
        print( "\tself.cash : " + str( self.cash ) )
        print( "\tself.value : " + str( self.value ) )
            """

        self.value = self.cash + position_Value
        self.previous_Position = new_Position

        reward = today_PnL

        return reward, today_PnL


    def step( self, logits ):
        # returns a state vector corresponding to 
        # the neural net inputs
        return_Position, log_Prob = onan.runNeuralNetMasterStrategy( self.prices_So_Far, logits )

        return_Position = return_Position.astype( int )

        reward, today_PnL = self._calculateStepReward( return_Position )

        self.timestep_Idx += 1

        if( self.timestep_Idx < self.episode_Size ):
            self._setTimestepPricesSoFar( self.timestep_Idx )
            output_State = onan.getNeuralNetInputs( self.prices_So_Far, self.timestep_Idx )
            output_End = False
        else:
            output_State = onan.getNeuralNetInputs( self.prices_So_Far, self.timestep_Idx )
            output_End = True

        return output_State, reward, output_End, log_Prob, today_PnL



def runEpisode( environement, policy, device, do_Print = False ):
    state, _, _ = environement.reset()

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


    mean_PnL = np.mean( daily_PnL_Array )
    std_PnL = np.std( daily_PnL_Array )

    if do_Print:
        sharpe_Ratio = mean_PnL / std_PnL

        print( "Perforamnce : " )
        print( "\tmean_PnL : " + str( mean_PnL ) )
        print( "\tstd_PnL : " + str( std_PnL ) )
        print( "\tSR : " + str( sharpe_Ratio ) )

    # reward_Array = np.divide( np.asarray( reward_Array ), np.sqrt( std_PnL ) + 5 )
    reward_Array = np.array( np.asarray( reward_Array ) / 1000 )

    return log_Prob_Array, reward_Array

def computeReturns( reward_Array, gamma = 0.99 ):
    return_Array = []
    G = 0.0

    for reward in reversed( reward_Array ):
        G = reward + gamma * G
        return_Array.insert( 0, G )

    return_Array = torch.tensor( return_Array, dtype=torch.float32 )

    return_Array = ( return_Array - return_Array.mean() ) / ( return_Array.std() + 1e-8 )

    return return_Array

def computeLoss( log_Prob_Array, return_Array ):
    loss = 0
    for log_Prob, G in zip( log_Prob_Array, return_Array ):
        loss += -log_Prob * G

    return loss

def runTrainingLoop( policy, number_Of_Episodes = 500, gamme = 0.99, lr = 1e-3 ):
    environement = Training_Environment()


    optimiser = optim.Adam( policy.parameters(), lr = lr )

    # Last known-good snapshot - this is what we roll back to
    best_Model_State = copy.deepcopy( policy.state_dict() )
    best_Optim_State = copy.deepcopy( optimiser.state_dict() )

    episode_Rewards = []

    for episode in range( number_Of_Episodes ):
        print( "Episode : " + str( episode ) )
        if episode % 2 == 0:
            do_Print = True

            print( "Average episode reward : " + str( np.mean( episode_Rewards ) ) )
        else:
            do_Print = False

        log_Prob_Array, reward_Array = runEpisode( environement, policy, device, do_Print = do_Print )

        for index in range( len( reward_Array ) ):
            if np.isnan( reward_Array[ index ] ):
                reward_Array[ index ] = 1e-8

        return_Array = computeReturns( reward_Array, gamme ).to( device )
        loss = computeLoss( log_Prob_Array, return_Array )

        optimiser.zero_grad()

        if torch.isnan( loss ).any() or torch.isinf( loss ).any():
            print( "DEBUG: NaN/Inf loss detected - rolling back to last good checkpoint" )
            policy.load_state_dict( best_Model_State )
            optimiser.load_state_dict( best_Optim_State )
            continue

        loss.backward()

        grads_Are_Finite = all(
            torch.isfinite( p.grad ).all()
            for p in policy.parameters() if p.grad is not None
        )

        if not grads_Are_Finite:
            print( "DEBUG: NaN/Inf gradient detected - rolling back to last good checkpoint" )
            policy.load_state_dict( best_Model_State )
            optimiser.load_state_dict( best_Optim_State )
            continue

        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimiser.step()

        # Final safety net: verify the step itself didn't produce NaN weights
        weights_Are_Finite = all( torch.isfinite( p ).all() for p in policy.parameters() )

        if not weights_Are_Finite:
            print( "DEBUG: NaN weights after step - rolling back to last good checkpoint" )
            policy.load_state_dict( best_Model_State )
            optimiser.load_state_dict( best_Optim_State )
            continue

        # This episode was clean - it becomes the new checkpoint
        best_Model_State = copy.deepcopy( policy.state_dict() )
        best_Optim_State = copy.deepcopy( optimiser.state_dict() )

        total_Reward = sum( reward_Array )

        episode_Rewards.append( total_Reward )

    return policy, episode_Rewards


if __name__ == "__main__":
    print( "TRAINING" )
    print( "" )

    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    policy = onan.MasterNeuralNet().to( device )

    # policy.load_state_dict(torch.load( "./model_save_2026-07-24 23:00:38.909668", weights_only=True))

    trained_Policy, episode_Rewards = runTrainingLoop(policy, number_Of_Episodes = 600)

    onan.setGlobalVariable( "g_neural_Net_Instance", trained_Policy )
    onan.setGlobalVariable( "g_strategy_Selection_Enum", 0 )

    # Copied from eval
    pricesFile = "./prices.txt"
    numTestDays = 250
    scoreDefaultParam = 1.0
    prcAll = eval.loadPrices(pricesFile)

    meanpl, ret, plstd, sharpe, dvol = eval.calcPL(prcAll, numTestDays)
    score = eval.score(meanpl, plstd, scoreDefaultParam)

    print( "" )
    print( "Score : " + str( score ) )

    date_Time_String = str( datetime.datetime.now() )
    torch.save( trained_Policy.state_dict(), "./model_save_" + date_Time_String )





