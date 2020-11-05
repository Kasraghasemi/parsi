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

landa = 0.01                # coupling strength
delta = 1               # time step
number_of_subsystems= 3             # number of subsystems
disturb = 0.1
control_size = 2

n=2*number_of_subsystems
m=1*number_of_subsystems

np.random.seed(seed=2)
A=np.random.rand(n,n)* landa
# B=np.random.rand(n,m)* landa
# A=np.zeros((n,n))
B=np.zeros((n,m))

for i in range(number_of_subsystems): 
    A[2*i:2*(i+1),2*i:2*(i+1)]= np.array([[1,1],[0,1]]) * delta
    B[2*i:2*(i+1),i]= np.array([0,1]) * delta

w_i=pp.zonotope(G=np.eye(2)*disturb , x=np.array([0,0]))
x_i=pp.zonotope(G=np.array([[1,0],[0,1]]) , x=np.array([0,0]))
u_i=pp.zonotope(G=np.array([[ control_size ]]) , x=np.array([0]))

# W=w_i
# for i in range(number_of_subsystems-1):
#     W=W**w_i

X = pp.zonotope( G=np.eye(n) , x=np.zeros(n) )
U = pp.zonotope( G=np.eye(m) * control_size , x=np.zeros(m) )
W = pp.zonotope( G=np.eye(n) * disturb  , x=np.zeros(n) )

system=parsi.Linear_system(A,B,W=W,X=X,U=U)

sub_sys=parsi.sub_systems(system,partition_A=[2]*number_of_subsystems,partition_B=[1]*number_of_subsystems , \
    disturbance = [ w_i for j in range(number_of_subsystems) ] , \
        admissible_x = [ x_i for j in range(number_of_subsystems)] , \
            admissible_u= [u_i for j in range(number_of_subsystems)] )


###########################################################################################################################
# Finding decentralized rci sets

omega,theta=parsi.compositional_decentralized_rci(sub_sys,initial_guess='nominal',size='min',initial_order=4,step_size=0.1,alpha_0='random',order_max=100)


################################
print('====================================================================================')
###########################################################################################################################
# Testing compistional potential function for mpc 

horizon = 10
for sys in sub_sys:

    # Initializing x_nominal and u_nominal
    sys.state = np.array([5,2])

    sys.x_nominal = np.array([ np.random.rand(2) for step in range(horizon)])
    sys.x_nominal[0] = sys.state
    sys.u_nominal = np.array([ np.random.rand(1) for step in range(horizon)])

    # sys.x_nominal = [ np.zeros(n) for step in range(horizon)]
    # sys.u_nominal = [ np.zeros(m) for step in range(horizon)]

    sys.alpha_x = [ np.random.rand( sys.omega.G.shape[1] ) for step in range(horizon-1)]
    sys.alpha_u = [ np.random.rand( sys.theta.G.shape[1] ) for step in range(horizon-1)]


# potential_result = parsi.potential_function_mpc(sub_sys, 0 , T_order=10, reduced_order=1,algorithm='fast')

parsi.compositional_synthesis(sub_sys,horizon,initial_order=4,step_size=0.1 ,order_max=100,algorithm='fast')


# Plotting the results

fig, axs = plt.subplots(number_of_subsystems)




for i in range(number_of_subsystems):
    sub_sys[i].omega.color='red'
    sub_sys[i].viable[-1].color='pink'
    path = np.array( [ sub_sys[i].viable[step].x for step in range(horizon)] )
    path = np.concatenate( ( sub_sys[i].x_nominal[0].reshape(1,2) , path ) ,axis=0)

    pp.visualize([sub_sys[i].omega,*sub_sys[i].viable], ax = axs[i],fig=fig, title='',equal_axis=True)

    axs[i].plot( path[:,0], path[:,1] ,color='b')

    # drawing the parameterized sets
    for step in range(1,horizon):
        pp.visualize( [ pp.zonotope( G= np.dot( sub_sys[i].omega.G , np.diag( sub_sys[i].alpha_x[step-1]) ) ,  x= sub_sys[i].x_nominal[step] , color = 'yellow') ] , ax = axs[i] ,title='' , equal_axis=True)


# Showing an actual tokken path

system.state= np.array( [sub_sys[i].state for i in range(number_of_subsystems)] ).flatten()
path_actual=system.state.reshape(-1,1)

for step in range(horizon):
    #Finding the controller
    u = np.array( [ sub_sys[i].u_nominal[step] for i in range(number_of_subsystems)]).flatten()
    state = system.simulate(u)
    path_actual = np.concatenate((path_actual,state.reshape(-1,1)) ,axis=1)

    for i in range(number_of_subsystems):
        sub_sys[i].state=system.state[2*i:2*(i+1)]
        axs[i].plot(path_actual[2*i,:],path_actual[2*i+1,:],color='black')
    plt.pause(0.2)



plt.show()



parsi.compositional_mpc_initialization(sub_sys)

parsi.compositional_synthesis(sub_sys,horizon,initial_order=5,step_size=0.3 ,order_max=100,algorithm='fast')