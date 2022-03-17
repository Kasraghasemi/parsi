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
controller_magnitude = 1
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

omega , theta , alpha_x , _  , center , _ = parsi.decentralized_viable_centralized_synthesis(sub_sys, size='min', order_max=30, algorithm='slow', horizon=None)


# check the equality constraints in the viable set constraints
for sys in range(number_of_subsystems):
    for step in range(horizon):
        predicated_generator = np.hstack( ( np.dot(sub_sys[sys].A[step] , sub_sys[sys].omega[step].G) + np.dot(sub_sys[sys].B[step] , sub_sys[sys].theta[step].G) ,sub_sys[sys].W[step].G ))
        predicated_center = np.dot(sub_sys[sys].A[step] , sub_sys[sys].omega[step].x) + np.dot(sub_sys[sys].B[step] , sub_sys[sys].theta[step].x) + sub_sys[sys].W[step].x
        for sys_neighbour in range(number_of_subsystems):
            if sys_neighbour in sub_sys[sys].A_ij[step]:
                predicated_generator = np.hstack(( predicated_generator,  np.dot( sub_sys[sys].A_ij[step][sys_neighbour] , sub_sys[sys_neighbour].param_set_X[step].G ) ) )
                predicated_center = predicated_center + np.dot( sub_sys[sys].A_ij[step][sys_neighbour] , sub_sys[sys_neighbour].param_set_X[step].x )
        next_generator = sub_sys[sys].omega[step+1].G
        next_center = sub_sys[sys].omega[step+1].x
        generator_offset = predicated_generator - next_generator
        center_offset = predicated_center - next_center


        assert (np.abs(generator_offset) < 10**(-9)).any()
        assert (np.abs(center_offset) < 10**(-9)).any()


# check the correctness
for sys in range(number_of_subsystems):
    for step in range(horizon):
        m = Model()
        pp.zonotope_subset(m, sub_sys[sys].omega[step] , sub_sys[sys].param_set_X[step] ,solver='gurobi' )
        pp.zonotope_subset(m, sub_sys[sys].theta[step] , sub_sys[sys].param_set_U[step] ,solver='gurobi' )

        m.setParam("OutputFlag",False)
        m.optimize()

        assert m.Status == 2
        del m

# check the validity
for sys in range(number_of_subsystems):
    for step in range(horizon):
        m = Model()
        pp.zonotope_subset(m, sub_sys[sys].omega[step] , sub_sys[sys].X[step] ,solver='gurobi' )
        pp.zonotope_subset(m, sub_sys[sys].theta[step] , sub_sys[sys].U[step] ,solver='gurobi' )

        m.setParam("OutputFlag",False)
        m.optimize()

        assert m.Status == 2
        del m


# running smaple trajectories

for run_number in range(200):
    print('run_number: ',run_number)

    for sys in sub_sys:
        sys.state = sub_sys[i].initial_state
    system.state = np.array( [ sub_sys[i].initial_state for i in range(number_of_subsystems)]).reshape(-1)


    for step in range(horizon):    
        print("step: %i"%step)
        #Finding the controller
        zeta_optimal=[]
        u = [parsi.find_controller( sub_sys[i].omega[step] , sub_sys[i].theta[step] , sub_sys[i].state) for i in range(number_of_subsystems) ]  

        state = system.simulate( np.array( [u[i] for i in range(number_of_subsystems)] ).reshape(-1)  , step=step)

        for sys in range(number_of_subsystems):
            sub_sys[sys].state = state[ 2*sys : 2*sys+2 ]
        # state= np.array([sub_sys[i].simulate(u[i],step=step) for i in range(number_of_subsystems)])

        for i in range(number_of_subsystems):
            assert parsi.is_in_set( sub_sys[i].omega[step+1] , sub_sys[i].state ) == True
            assert parsi.is_in_set( sub_sys[i].theta[step] , u[i] ) == True


    # for sys in range(number_of_subsystems):
    #     param_sets = [ pp.pca_order_reduction( pp.zonotope( x = center[sys][step] , G = np.dot( sub_sys[sys].param_set_X[step].G, np.diag(alpha_x[sys][step]) )) ,desired_order=6) for step in range(1, horizon)]
    #     for sets in param_sets:
    #         sets.color = 'red'
    #     omega_reduced_order = [ pp.pca_order_reduction( sub_sys[sys].omega[step] ,desired_order=6) for step in range(1,horizon+1)]
    #     pp.visualize( param_sets + omega_reduced_order )
    #     plt.show()


# TODO: write now it does not work when the subsystems are LTI

# system=parsi.Linear_system(A,B,W=W,X=X,U=U)

# sub_sys=parsi.sub_systems(
#     system,
#     partition_A=[2]*number_of_subsystems,
#     partition_B=[1]*number_of_subsystems,
#     disturbance=[ W_i for j in range(number_of_subsystems)] , 
#     admissible_x=[ X_i for j in range(number_of_subsystems)] , 
#     admissible_u=[ U_i for j in range(number_of_subsystems)] 
# )

# for i in range(number_of_subsystems):
#     sub_sys[i].X = [X_i]*horizon
#     sub_sys[i].X.append(goal_set)


# omega , theta , _ , _ = parsi.decentralized_viable_centralized_synthesis(sub_sys, size='min', order_max=30, algorithm='slow', horizon=None)