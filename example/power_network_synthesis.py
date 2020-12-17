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

delta_t = 0.5
disturbance = 0.02
number_of_subsystems = 4
Kp_i = [110] * number_of_subsystems
Tp_i = [30] * number_of_subsystems

Ks_ij=np.zeros((number_of_subsystems,number_of_subsystems))

for i in range(number_of_subsystems):

    for j in range(i+1,number_of_subsystems):
        
        Ks_ij[i,j] = np.random.randint(2) * 0.1
        Ks_ij[j,i] = Ks_ij[i,j]

Ks_ij = np.array([
    [ 0 , 1 , 1 , 1 ], 
    [ 1 , 0 , 1 , 1 ], 
    [ 1 , 1 , 0 , 1 ], 
    [ 1 , 1 , 1 , 0 ]
]) * 0.01


sub_sys = []
A,B = {},{}

for i in range(number_of_subsystems):

    A[i] = np.array([
        [ 1 , 2 * np.pi * delta_t ] ,
        [  -1 * delta_t * Kp_i[i] * sum(Ks_ij[i]) / ( 2 * np.pi * Tp_i[i] )   , 1 - delta_t / Tp_i[i] ]
        ])

    B[i] = np.array([
        [ 0 ] ,
        [ delta_t * Kp_i[i] / Tp_i[i] ]
        ])

    X = pp.zonotope( x=np.zeros(2),G=np.array([[0.3,0],[0,0.3]])  )

    U = pp.zonotope( x=np.zeros(1),G=np.array([[1]]))

    W = pp.zonotope(x=np.zeros(2),G=np.array([[ 0.000000000000000000000000000000000000000000000000000000000001 , 0 ],[ 0 , -1 * delta_t * Kp_i[i] / Tp_i[i] ]]) * disturbance )

    sub_sys.append( parsi.Linear_system( A[i] , B[i] , W = W , X = X , U = U ) )

    for j in range(number_of_subsystems):
        if Ks_ij[i][j]!=0:
            sub_sys[i].A_ij[j] = np.array([[0 , 0],[ delta_t*Kp_i[i]*Ks_ij[i][j]/(2*np.pi*Tp_i[i]) ,0]])
     


# omega,theta=parsi.decentralized_rci(sub_sys,method='centralized',initial_guess='nominal',size='min',solver='gurobi',order_max=100)

_ , _ = parsi.compositional_decentralized_rci(sub_sys,initial_guess='nominal',initial_order=8,step_size=100,alpha_0='random',order_max=20)

for i in range(number_of_subsystems):
    # sub_sys[i].omega=omega[i]
    # sub_sys[i].theta=theta[i]

    sub_sys[i].state=parsi.sample(sub_sys[i].omega)

path= np.array( [sub_sys[i].state for i in range(number_of_subsystems)] ).reshape(-1,1)


fig, axs = plt.subplots(number_of_subsystems)
for i in range(number_of_subsystems):
    sub_sys[i].X.color='red'

    pp.visualize([sub_sys[i].X,sub_sys[i].omega], ax = axs[i],fig=fig, title='',equal_axis=True)
    
for step in range(50):
    #Finding the controller
    u=[parsi.mpc(sub_sys[i],horizon=1,x_desired='origin') for i in range(number_of_subsystems)]
    state= np.array([sub_sys[i].simulate(u[i]) for i in range(number_of_subsystems)])
    path=np.concatenate((path,state.reshape(-1,1)) ,axis=1)
    for i in range(number_of_subsystems):
        axs[i].plot(path[2*i,:],path[2*i+1,:],color='b')
    plt.pause(0.02)

plt.tight_layout()
plt.show()  


################################
print('====================================================================================')
###########################################################################################################################
# Testing compistional potential function for mpc 

horizon = 5
for sys in sub_sys:

    # Initializing x_nominal and u_nominal
    sys.state = np.array([0,0.2])
    sys.state = np.array([0.2,0.4])


    sys.x_nominal = np.array([ np.random.rand(2) for step in range(horizon)])
    sys.x_nominal[0] = sys.state
    sys.u_nominal = np.array([ np.random.rand(1) for step in range(horizon)])

    # sys.x_nominal = [ np.zeros(n) for step in range(horizon)]
    # sys.u_nominal = [ np.zeros(m) for step in range(horizon)]

    sys.alpha_x = [ np.ones( sys.omega.G.shape[1] ) for step in range(horizon-1)]
    sys.alpha_u = [ np.ones( sys.theta.G.shape[1] ) for step in range(horizon-1)]


# potential_result = parsi.potential_function_mpc(sub_sys, 0 , T_order=10, reduced_order=1,algorithm='fast')

parsi.compositional_synthesis( sub_sys , horizon , initial_order = 4 , step_size=0.01 , order_max=100 , algorithm='slow')


# Plotting the results

fig, axs = plt.subplots(number_of_subsystems)

for i in range(number_of_subsystems):
    sub_sys[i].omega.color='red'
    sub_sys[i].viable[-1].color='pink'
    path = np.array( [ sub_sys[i].viable[step].x for step in range(horizon)] )
    path = np.concatenate( ( sub_sys[i].x_nominal[0].reshape(1,2) , path ) ,axis=0)

    # drawing the parameterized sets
    for step in range(1,horizon):

        # sub_sys[i].viable[step] = pp.pca_order_reduction( sub_sys[i].viable[step] , desired_order=6 )

        # assump_set = pp.pca_order_reduction( pp.zonotope( G= np.dot( sub_sys[i].omega.G , np.diag( sub_sys[i].alpha_x[step-1]) ) ,  x= sub_sys[i].x_nominal[step] , color = 'yellow') , desired_order=6 )

        pp.visualize( [ pp.zonotope( G= np.dot( sub_sys[i].omega.G , np.diag( sub_sys[i].alpha_x[step-1]) ) ,  x= sub_sys[i].x_nominal[step] , color = 'yellow') ] 
                        , ax = axs[i] , title='' , equal_axis=True
                    )

    pp.visualize([sub_sys[i].omega,*sub_sys[i].viable], ax = axs[i],fig=fig, title='',equal_axis=True)

    axs[i].plot( path[:,0], path[:,1] ,color='b')

plt.show()

