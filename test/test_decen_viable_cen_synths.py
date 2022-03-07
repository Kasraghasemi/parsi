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


landa = 0.0001
number_of_subsystems = 3
horizon = 10

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
U_i=pp.zonotope(G=np.eye(1),x=np.zeros(1))
W_i=pp.zonotope(G=np.eye(2)*0.1,x=np.zeros(2))

W=W_i
for _ in range(number_of_subsystems-1):
    W=W**W_i

X = pp.zonotope(G=np.eye(n)*10,x=np.zeros(n),color='red')
U = pp.zonotope(G=np.eye(m),x=np.zeros(m))

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

goal_set = pp.zonotope( x=[6,5], G = [[0.5,0],[0,0.5]])
for i in range(number_of_subsystems):
    sub_sys[i].X.append(goal_set)

omega , theta , _ , _ = parsi.decentralized_viable_centralized_synthesis(sub_sys, size='min', order_max=30, algorithm='slow', horizon=None)

for sys in sub_sys:
    sys.state = parsi.sample(sys.omega[0])


for step in range(horizon):

    #Finding the controller
    zeta_optimal=[]
    u = [parsi.find_controller( sub_sys[i].omega[step] , sub_sys[i].theta[step] , sub_sys[i].state) for i in range(number_of_subsystems) ]        
    state= np.array([sub_sys[i].simulate(u[i]) for i in range(number_of_subsystems)])

    for i in range(number_of_subsystems):
        assert parsi.is_in_set( sub_sys[i].omega[step+1] , state[i] ) == True
        assert parsi.is_in_set( sub_sys[i].theta[step] , u[i] ) == True


# for sys in range(number_of_subsystems):
#     omega_reduced_order = [ pp.pca_order_reduction( sub_sys[i].omega[step] ,desired_order=6) for step in range(1,horizon+1)]
#     pp.visualize(omega_reduced_order)
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