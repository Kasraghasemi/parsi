"""
@author: kasra
"""
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *
try:
    import pypolycontain as pp
except:
    print('Error: pypolycontain package is not installed correctly') 
try:
    import parsi
except:
    raise ModuleNotFoundError("parsi package is not installed properly")



number_of_subsystems = 3
horizon = 10

landa = 0.0001
controller_magnitude = 4
n=2*number_of_subsystems
m=1*number_of_subsystems

A=np.ones((n,n)) * landa
B=np.zeros((n,m))

for i in range(number_of_subsystems): 
    A[2*i:2*(i+1),2*i:2*(i+1)]= np.array([[1,1],[0,1]]) 
    B[2*i:2*(i+1),i]= np.array([0,1])
A_ltv = [A]* horizon
B_ltv = [B]* horizon

X_i=pp.zonotope(G=np.eye(2)*10,x=np.zeros(2),color='red')
U_i=pp.zonotope(G=np.eye(1)*controller_magnitude,x=np.zeros(1))
W_i=pp.zonotope(G=np.eye(2)*0.1,x=np.zeros(2))

W=W_i
for _ in range(number_of_subsystems-1):
    W=W**W_i

X = pp.zonotope(G=np.eye(n)*10,x=np.zeros(n),color='red')
U = pp.zonotope(G=np.eye(m)*controller_magnitude,x=np.zeros(m))

W_ltv = [ W ] * horizon
X_ltv = [ X ] * horizon
U_ltv = [ U ] * horizon

system=parsi.Linear_system(A_ltv,B_ltv,W=W_ltv,X=X_ltv,U=U_ltv)

sub_sys=parsi.sub_systems_LTV(
    system,
    partition_A=[2]*number_of_subsystems,
    partition_B=[1]*number_of_subsystems,
    disturbance=[ [W_i for j in range(number_of_subsystems)] for t in range(horizon)], 
    admissible_x=[ [X_i for j in range(number_of_subsystems)] for t in range(horizon)] , 
    admissible_u=[ [U_i for j in range(number_of_subsystems)] for t in range(horizon)]
)


# GOAL SET
goal_set = pp.zonotope( x=[5,3], G = [[0.5,0],[0,0.5]])
# INITIAL STATE
for i in range(number_of_subsystems):
    sub_sys[i].X.append(goal_set)
    sub_sys[i].initial_state = np.array([1,-1])

omega , theta , alfa_x , alfa_u  , alpha_center_x , alpha_center_u = parsi.decentralized_viable_centralized_synthesis(sub_sys, size='min', order_max=30, algorithm='slow', horizon=None)


# Initialization of parameterized sets and alpha_x and alpha_u
for sys_index in range(number_of_subsystems):
    sub_sys[sys_index].parameterized_set_initialization()
    for step in range(horizon):
        sub_sys[sys_index].param_set_X[step].x = alpha_center_x[sys_index][step]
        sub_sys[sys_index].param_set_U[step].x =  alpha_center_u[sys_index][step]

    sub_sys[sys_index].alpha_x= alfa_x[sys_index]
    sub_sys[sys_index].alpha_u= alfa_u[sys_index]

# Output for one specific subsystem
for sys_index in range(number_of_subsystems):

    output=parsi.potential_function_synthesis(sub_sys, sys_index, 6, reduced_order=1, include_validity=True, horizon=None, algorithm='slow')
    assert abs( output['obj']) <= 10**(-5)
