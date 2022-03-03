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
from timeit import default_timer as timer


N = 10              # number of points
raduis = 10             # threshold for being neighbor
landa = 0.01             # affects couplings between subsystems
delta_t = 0.2            # time step for discretizing the dynamics
disturbance = 0.00001               # inherent disturbance of each subsystems
control_input = 5               # size of the admissible control input
state_input = 10               # size of the admissible state space    
REPEAT = 10             # Averaging the time for REPEAT number of examples


time_centralized_decentralized = np.zeros(REPEAT)
time_total_cen_dec = np.zeros(REPEAT)

time_compositional = np.zeros(REPEAT)
time_total_com = np.zeros(REPEAT)

time_centralized_centralized = np.zeros(REPEAT)
time_total_cen_cen = np.zeros(REPEAT)
number_iterations = np.zeros(REPEAT)

for repeat in range(REPEAT):
    print('repeat',repeat)
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
    # time_1 = timer()
    
    # omega,theta=parsi.decentralized_rci(sub_sys,method='centralized',solver='gurobi',order_max=100)
    
    # time_2 = timer()
    # time_total_cen_dec[repeat] = time_2 - time_1

    # time_centralized_decentralized[repeat] =  sum( parsi.Monitor['time_centralized_decentralized'] ) 



    ##########################################################################################################
    ####################### Compositional computation of Decentralized RCI sets ##############################
    ##########################################################################################################

    time_3 = timer()

    omega,theta=parsi.compositional_decentralized_rci(sub_sys , initial_order = 4 , step_size = 1 , iteration_max = 100 , order_max=100 )

    time_4 = timer()
    time_total_com[repeat] = time_4 - time_3

    time_compositional[repeat] = sum( [ sum(parsi.Monitor['time_compositional'][i]) for i in range(len(parsi.Monitor['time_compositional'])) ] )
    number_iterations[repeat] = parsi.Monitor['num_iterations']



    ##########################################################################################################
    ######################### Centralized computation of Centralized RCI sets ################################
    ##########################################################################################################

    # from scipy.linalg import block_diag

    # A_total = np.zeros( (2*N , 2*N) )

    # for i in range(N):

    #     for j in range(N):
            
    #         if type( A.get((i,j)) ) == type( None ):

    #             A[i,j] = np.zeros((2,2))
            
    #         A_total[2*i:2*(i+1),2*j:2*(j+1)] = A[i,j]

    # B_total =  block_diag( *[ B[i] for i in range(N) ] )

    # X_total = X
    # U_total = U
    # W_total = W

    # for i in range(N-1):
        
    #     X_total = X_total ** X
    #     U_total = U_total ** U
    #     W_total = W_total ** W

    # system = parsi.Linear_system( A_total , B_total , W = W_total , X = X_total , U = U_total )

    # time_5 = timer()

    # omega,theta = system.rci()

    # time_6 = timer()
    # time_total_cen_cen[repeat] = time_6 - time_5

    # time_centralized_centralized[repeat] = parsi.Monitor['time_centralized_centralized']

    # with open('result.txt','w') as file:
    #     file.write(" time_centralized_centralized {} , {} , \n  time_centralized_decentralized {} , {} \n time_compositional {} , {} \n number_iterations {} \n number_iterations {}" .format( 
    #     np.mean(time_centralized_centralized) , 
    #     np.mean(time_total_cen_cen),
    #     np.mean(time_centralized_decentralized) , 
    #     np.mean(time_total_cen_dec),
    #     np.mean(time_compositional) , 
    #     np.mean(time_total_com),
        # np.mean(number_iterations) ,
        # number_iterations ) )

    with open('result.txt','w') as file:
        file.write(" time_centralized_decentralized {} , {} \n time_compositional {} , {} \n number_iterations {} \n number_iterations {}" .format( 
        np.mean(time_centralized_decentralized) , 
        np.mean(time_total_cen_dec) ,
        time_compositional , 
        time_total_com,
        np.mean(number_iterations) ,
        number_iterations ) )

print('time_centralized_centralized',  time_centralized_centralized  )
print('time_centralized_decentralized', time_centralized_decentralized )
print('time_compositional', time_compositional )
print('number_iterations' , np.mean(number_iterations) )
print('number_iterations' , number_iterations )


