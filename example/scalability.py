"""
@author: kasra
"""
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
try:
    import pypolycontain as pp
except:
    raise ModuleNotFoundError("pypolycontain package is not installed correctly")
try:
    import parsi
except:
    raise ModuleNotFoundError("parsi package is not installed properly")



N =  50              # number of points
raduis = 100             # threshold for being neighbor
landa = 0.1             # affects couplings between subsystems
delta_t = 1            # time step for discretizing the dynamics
disturbance = 0.2               # inherent disturbance of each subsystems
control_input = 2               # size of the admissible control input
state_input = 2               # size of the admissible state space    

points = np.zeros((N,2))
np.random.seed(3)
points[:,0]= np.random.uniform(0,100,N)
points[:,1]= np.random.uniform(0,100,N)


distances= np.zeros((N , N))

for i in range(N):
    for j in range(i+1 , N):
        distances[i,j]= np.sqrt( (points[i,0] - points[j,0])**2 + (points[i,1] - points[j,1])**2 )
        distances[j,i] = distances[i,j]
        
neighbours={}
for i in range(N):
    neighbours[i]=[]
    for j in range(N):  
        if distances[i,j] < raduis and i!=j:
            neighbours[i].append(j)




d_bar = np.array([ 0 , 0 ])
G_d = np.eye(2) * disturbance
W = pp.zonotope( x = d_bar , G = G_d) 

x_bar = np.array([0 , 0])
G_x = np.eye(2) * state_input
X = pp.zonotope( x = x_bar , G = G_x )

u_bar = np.array([0])    
G_u = np.array([[1]]) * control_input
U = pp.zonotope( x = u_bar , G = G_u )

A={}
B={}
sub_sys = []

for i in range(N):
    
    for j in range(N):
        if i ==j:
            A[i,j] = np.array([ [1,1] , [0,1] ] ) * delta_t
        elif j in neighbours[i]:
            A[i,j] = np.ones((2,2)) * landa / (1+distances[i,j])
                    
    B[i] = np.array( [[0] , [1]] ) * delta_t
    
    sub_sys.append(parsi.Linear_system(A[i,i],B[i],W=W,X=X,U=U))

    for j in range(N):
        if j in neighbours[i]:
            sub_sys[i].A_ij[j] = A[i,j]
    

#####################################################


# omega,theta=parsi.decentralized_rci(sub_sys,method='centralized',initial_guess='nominal',size='min',solver='gurobi',order_max=100)

omega,theta=parsi.compositional_decentralized_rci(sub_sys,initial_guess='nominal',initial_order=2,step_size=0.1,alpha_0='random',order_max=100)

