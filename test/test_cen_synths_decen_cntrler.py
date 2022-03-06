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

landa=0.01
delta=1
number_of_subsystems= 10
n=2*number_of_subsystems
m=1*number_of_subsystems

A=np.ones((n,n))* landa
B=np.ones((n,m))* landa

for i in range(number_of_subsystems): 
    A[2*i:2*(i+1),2*i:2*(i+1)]= np.array([[1,1],[0,1]]) * delta
    B[2*i:2*(i+1),i]= np.array([0,1]) * delta

X_i=pp.zonotope(G=np.eye(2),x=np.zeros(2),color='red')
U_i=pp.zonotope(G=np.eye(1),x=np.zeros(1))
W_i=pp.zonotope(G=np.eye(2)*0.1,x=np.zeros(2))

W=W_i
for _ in range(number_of_subsystems-1):
    W=W**W_i
X=pp.zonotope(G=np.eye(n),x=np.zeros(n),color='red')
U=pp.zonotope(G=np.eye(m),x=np.zeros(m))

system=parsi.Linear_system(A,B,W=W,X=X,U=U)

sub_sys=parsi.sub_systems(
    system,partition_A=[2]*number_of_subsystems,
    partition_B=[1]*number_of_subsystems,
    disturbance=[W_i for j in range(number_of_subsystems)], 
    admissible_x=[X_i for j in range(number_of_subsystems)] , 
    admissible_u=[U_i for j in range(number_of_subsystems)]
)

omega , theta , alfa_x , alfa_u=parsi.decentralized_rci_centralized_synthesis(sub_sys,size='min',order_max=30)


# Plotting

# path= np.array( [sub_sys[i].state for i in range(number_of_subsystems)] ).reshape(-1,1)

# cols=5
# fig, axs = plt.subplots(int(ceil(number_of_subsystems / cols)),cols)
# for i in range(number_of_subsystems):
#     sub_sys[i].X.color='red'
#     r=i//cols
#     c=i%cols
#     pp.visualize([sub_sys[i].X,omega[i]], ax = axs[r,c],fig=fig, title='',equal_axis=True)

for step in range(100):

    #Finding the controller
    zeta_optimal=[]
    u = [parsi.find_controller( sub_sys[i].omega , sub_sys[i].theta , sub_sys[i].state) for i in range(number_of_subsystems) ]        
    state= np.array([sub_sys[i].simulate(u[i]) for i in range(number_of_subsystems)])

    for i in range(number_of_subsystems):
        assert parsi.is_in_set( sub_sys[i].omega , state[i] ) == True

# Plotting

#     path=np.concatenate((path,state.reshape(-1,1)) ,axis=1)
#     for i in range(number_of_subsystems):
#         r=i//cols
#         c=i%cols
#         axs[r,c].plot(path[2*i,:],path[2*i+1,:],color='b')
#     plt.pause(0.02)

# plt.tight_layout()
# plt.show()  