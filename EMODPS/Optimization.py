from sys import *
import math
from borg import *
import numpy as np
import pyswmm.toolkitapi as tkai
from pyswmm.simulation import Simulation
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import random
from SWMM import Swmm 
from Swmm_Model import Swmm_Model

random.seed(23)

# Read forecast data

df_dvlpd1 = pd.read_csv('/scratch/hk3sku/OptimizationVsRL/RL/Two_Objectives/forecast_24.csv')


train_dvlpd = Swmm(config="/scratch/hk3sku/OptimizationVsRL/StormWater/Borg-1.8/Config/Config_Events2/Train_ref.yaml", action_params = [0,0])
train_dvlpd.run_simulation()
df_dvlpd_closed = train_dvlpd.export_df()



filtered_forecast1 = df_dvlpd1['norfolk_airport_total_precip'][(df_dvlpd1['time'].values == str(df_dvlpd_closed["time"][0]))]
filtered_forecast2 = df_dvlpd1['norfolk_airport_total_precip'][(df_dvlpd1['time'].values == str(df_dvlpd_closed["time"][len(df_dvlpd_closed)-1]))]

start_index = filtered_forecast1.index[0]
end_index = filtered_forecast2.index[0]


rain_max = np.max(df_dvlpd1['norfolk_airport_total_precip'][start_index:end_index].values)


rain_forecast = (df_dvlpd1['norfolk_airport_total_precip'][start_index-1:end_index].values)

Max_states = [rain_max, 6, 6]

num_episodes = 1

#%% optimization
M = 3 # number of states
K = 2 # number of actions
#N = M + K + 1 # number of RBFs
N = 5
nvars = N*(2*M + K) + K # convex RBFs
nobjs = 2
# bounds on decision variables
bounds = [[0,1]]*K # constant for each action
bounds2 = [[-1,1],[0,1]]*M # center, radius for each state
bounds2.extend([[0,1]]*K) # weight for each action
bounds2 = bounds2*N # repeat for each RBF
bounds.extend(bounds2)


def comp_objectives(*vars):


    A = vars[0:K] # first K constants for non-convex RBFs
    vars2 = vars[K::] # remove first K constants
    C = np.zeros([M,N])
    R = np.zeros([M,N])
    W = np.zeros([K,N])
    for n in range(N):
        for m in range(M):
            C[m,n] = vars2[(2*M+K)*n + 2*m]
            R[m,n] = vars2[(2*M+K)*n + 2*m + 1]
        for k in range(K):
            W[k,n] = vars2[(2*M+K)*n + 2*M + k]
    

    Max_states = [rain_max, 6, 6]
    train_dvlpd = Swmm_Model(config="/scratch/hk3sku/OptimizationVsRL/StormWater/Borg-1.8/Config/Config_Events2/Train_ref.yaml", W=W, C=C, R=R, A=A, max_states = Max_states, forecast = rain_forecast)
    train_dvlpd.run_simulation()
    df_dvlpd = train_dvlpd.export_df()


    Obj1 = np.mean(np.mean(df_dvlpd["P1J_flooding"].values) + np.mean(df_dvlpd["P2J_flooding"].values))
    
    
    Obj2 = np.mean(np.mean(df_dvlpd["OvF1_flow"].values) + np.mean(df_dvlpd["OvF2_flow"].values))
        

    
    objs_values = [Obj1, Obj2]
    return objs_values
    

main_output_file_dir = '/scratch/hk3sku/OptimizationVsRL/StormWater/out/'
output_location = '/scratch/hk3sku/OptimizationVsRL/StormWater/out/'


Configuration.startMPI()

borg = Borg(nvars, nobjs, 0, comp_objectives)
borg.setBounds(*bounds)
borg.setEpsilons(*[0.001]*nobjs)

result = borg.solveMPI(maxTime=12, runtime='SWMM_EMODPS_ncrbfs_2obj_upvsdown_N5.runtime', frequency=100)

if result:
          
          
          f = open(main_output_file_dir + "/OptimizationSWMM_NCRBF_N5" + ".set", 'w')
          f.write("#Borg Optimization Results")
          f.write("#First " + str(nvars) + " are the decision variables, " \
          "last " + str(nobjs) + " are the objective values\n")
          for solution in result:
            line = ''
            for i in range(len(solution.getVariables())):
              line = line + (str(solution.getVariables()[i])) + ' '
          
            for i in range(len(solution.getObjectives())):
              line = line + (str(solution.getObjectives()[i])) + ' '
          
            f.write(line[0:-1] + '\n')
      
          f.write("#")
          f.close()
