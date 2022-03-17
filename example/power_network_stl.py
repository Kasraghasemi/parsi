"""
@author: kasra
"""
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
try:
    import pypolycontain as pp
except:
    print('Error: pypolycontain package is not installed correctly') 
try:
    import parsi
except:
    raise ModuleNotFoundError("parsi package is not installed properly")
from scipy.linalg import block_diag 

horizon = 9
delta_t = 0.1
disturbance = 0.001
controller_magnitude = 0.1
admissable_state_magnitude = 0.3
number_of_subsystems = 4
Kp_i = [110] * number_of_subsystems
Tp_i = [25] * number_of_subsystems
n=2*number_of_subsystems
m=1*number_of_subsystems

Ks_ij=np.zeros((number_of_subsystems,number_of_subsystems))


for i in range(number_of_subsystems):

    for j in range(i+1,number_of_subsystems):
        
        Ks_ij[i,j] = 0.5
        Ks_ij[j,i] = Ks_ij[i,j]

# Ks_ij[0,2]=0
# Ks_ij[2,0]=0

B_i = [
        [ 0 ] ,
        [ delta_t * Kp_i[i] / Tp_i[i] ]
        ]

B = block_diag( *[B_i]*number_of_subsystems )

A= np.zeros((2*number_of_subsystems , 2*number_of_subsystems))
for i in range(number_of_subsystems):
    for j in range(number_of_subsystems):
        if i==j:
            A[2*i:2*(i+1) , 2*i:2*(i+1) ] = np.array( [
        [ 1 , 2 * np.pi * delta_t ] ,
        [  -1 * delta_t * Kp_i[i] * sum(Ks_ij[i]) / ( 2 * np.pi * Tp_i[i] )   , 1 - delta_t / Tp_i[i] ]
        ])
        else:
            if Ks_ij[i][j]!=0:
                A[2*i:2*(i+1) , 2*j:2*(j+1)] = np.array(
                    [[0 , 0],
                    [ delta_t*Kp_i[i]*Ks_ij[i][j]/(2*np.pi*Tp_i[i]) , 0]]
                )



A_ltv = [A]* horizon
B_ltv = [B]* horizon

X_i=pp.zonotope(x=np.zeros(2),G=np.eye(2)*admissable_state_magnitude,color='red')
U_i=pp.zonotope(G=np.eye(1)*controller_magnitude,x=np.zeros(1))
W_i=pp.zonotope(G=np.eye(2)*disturbance,x=np.zeros(2))

W=W_i
for _ in range(number_of_subsystems-1):
    W=W**W_i

X = pp.zonotope(G=np.eye(n)*admissable_state_magnitude,x=np.zeros(n),color='red')
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
goal_set = pp.zonotope( x=[0,0], G = [[0.05,0],[0,0.01]])
# INITIAL STATE
for i in range(number_of_subsystems):
    sub_sys[i].X.append(goal_set)
    sub_sys[i].initial_state = np.array([0.1,0.1])

omega , theta , alpha_x , _  , center , _ = parsi.decentralized_viable_centralized_synthesis(sub_sys, size='min', order_max=30, algorithm='slow', horizon=None)


for sys in sub_sys:
    sys.state = sub_sys[i].initial_state
system.state = np.array( [ sub_sys[i].initial_state for i in range(number_of_subsystems)]).reshape(-1)

path = np.zeros(( horizon+1 , 2 * number_of_subsystems ))
path[0,:] = np.array([ sub_sys[i].initial_state for i in range(number_of_subsystems)]).reshape(-1)

for step in range(horizon):

    print("step: %i"%step)
    #Finding the controller
    zeta_optimal=[]
    u = [parsi.find_controller( sub_sys[i].omega[step] , sub_sys[i].theta[step] , sub_sys[i].state) for i in range(number_of_subsystems) ]  
    state = system.simulate( np.array( [u[i] for i in range(number_of_subsystems)] ).reshape(-1)  , step=step)

    for sys in range(number_of_subsystems):
        sub_sys[sys].state = state[ 2*sys : 2*sys+2 ]
    
    path[step+1,:] = state

    for i in range(number_of_subsystems):
        assert parsi.is_in_set( sub_sys[i].omega[step+1] , sub_sys[i].state ) == True
        assert parsi.is_in_set( sub_sys[i].theta[step] , u[i] ) == True


for sys in range(number_of_subsystems):
    param_sets = [ pp.pca_order_reduction( pp.zonotope( x = center[sys][step] , G = np.dot( sub_sys[sys].param_set_X[step].G, np.diag(alpha_x[sys][step]) )) ,desired_order=6) for step in range(1, horizon)]
    for sets in param_sets:
        sets.color = 'red'
    omega_reduced_order = [ pp.pca_order_reduction( sub_sys[sys].omega[step] ,desired_order=6) for step in range(1,horizon+1)]
    # pp.visualize( param_sets + omega_reduced_order )
    pp.visualize( omega_reduced_order )
    plt.plot(path[:, 2*sys] , path[:,2*sys+1 ] , '-*')
    plt.show()

