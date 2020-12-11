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



N = 40              # number of points
raduis = 10             # threshold for being neighbor
landa = 0.1             # affects couplings between subsystems
delta_t = 0.2            # time step for discretizing the dynamics
disturbance = 0.2               # inherent disturbance of each subsystems
control_input = 5               # size of the admissible control input
state_input = 10               # size of the admissible state space    
REPEAT = 10             # Averaging the time for REPEAT number of examples


time_centralized_decentralized = np.zeros(REPEAT)
time_compositional = np.zeros(REPEAT)

for repeat in range(REPEAT):

    ########################################################
    ######### Definging the random coupled system ##########
    ########################################################

    points = np.zeros((N,2))
    # np.random.seed(3)
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

                A[i,j] = np.array([ 
                    [ 1 , delta_t ] ,
                    [ 0 , 1 ] 
                    ] ) 

            elif j in neighbours[i]:
                A[i,j] = np.ones((2,2)) * landa / (1+distances[i,j])
                        
        B[i] = np.array( [
            [0] ,
            [delta_t]
            ] ) 
        
        sub_sys.append(parsi.Linear_system(A[i,i],B[i],W=W,X=X,U=U))

        for j in range(N):
            if j in neighbours[i]:
                sub_sys[i].A_ij[j] = A[i,j]
        


    ##########################################################################################################
    ####################### Centralized computation of Decentralized RCI sets ################################
    ##########################################################################################################

    omega,theta=parsi.decentralized_rci(sub_sys,method='centralized',initial_guess='nominal',size='min',solver='gurobi',order_max=100)

    time_centralized_decentralized[repeat] =  sum( parsi.Monitor['time_centralized_decentralized'] ) 



    ##########################################################################################################
    ####################### Compozitional computation of Decentralized RCI sets ##############################
    ##########################################################################################################

    # omega,theta=parsi.compositional_decentralized_rci(sub_sys,initial_guess='nominal',initial_order=2,step_size=0.1,alpha_0='random',order_max=100)

    # time_compositional[repeat] = sum( [ sum(parsi.Monitor['time_compositional'][i]) for i in range(len(parsi.Monitor['time_compositional'])) ] )




    ##########################################################################################################
    ######################### Centralized computation of Centralized RCI sets ################################
    ##########################################################################################################





# print('time_centralized_centralized',  np.mean(time_centralized_centralized) )
print('time_centralized_decentralized', np.mean(time_centralized_decentralized))
print('time_compositional', np.mean(time_compositional))


