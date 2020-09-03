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
landa=0.001
delta=0.1
number_of_subsystems=2
n=2*number_of_subsystems
m=1*number_of_subsystems

A=np.random.rand(n,n)* landa
#B=np.random.rand(n,m)* landa
#A=np.zeros((n,n))
B=np.zeros((n,m))

for i in range(number_of_subsystems): 
    A[2*i:2*(i+1),2*i:2*(i+1)]= np.array([[1,1],[0,1]]) * delta
    B[2*i:2*(i+1),i]= np.array([0,1]) * delta

W=pp.zonotope(G=np.eye(n)*0.1,x=np.zeros(n))
X=pp.zonotope(G=np.eye(n),x=np.zeros(n),color='red')
U=pp.zonotope(G=np.eye(m),x=np.zeros(m))

system=parsi.Linear_system(A,B,W=W,X=X,U=U)
sub_sys=parsi.sub_systems(system,partition_A=[2]*number_of_subsystems,partition_B=[1]*number_of_subsystems)

for i in range(number_of_subsystems):
    sub_sys[i].U.G=np.array([sub_sys[i].U.G])

omega,theta = parsi.decentralized_rci(sub_sys,size='min')

for i in range(number_of_subsystems):
    sub_sys[i].omega=omega[i]
    sub_sys[i].theta=theta[i]

    sub_sys[i].state=parsi.sample(sub_sys[i].omega)

system.state= np.array( [sub_sys[i].state for i in range(number_of_subsystems)] ).flatten()
path=system.state.reshape(-1,1)

# fig, axs = plt.subplots(ceil((number_of_subsystems)**0.5),ceil((number_of_subsystems)**0.5))
fig, axs = plt.subplots(1,number_of_subsystems)
for i in range(number_of_subsystems):
    sub_sys[i].X.color='red'
    pp.visualize([sub_sys[i].X,omega[i]], ax = axs[i],fig=fig, title='')

for step in range(50):
    #Finding the controller
    u=np.array([parsi.mpc(sub_sys[i],horizon=1,x_desired='origin') for i in range(number_of_subsystems)]).flatten()
    state= system.simulate(u)
    path=np.concatenate((path,state.reshape(-1,1)) ,axis=1)
    for i in range(number_of_subsystems):
        sub_sys[i].state=system.state[2*i:2*(i+1)]
        axs[i].plot(path[2*i,:],path[2*i+1,:],color='b')
    plt.pause(0.09)
plt.show()  
