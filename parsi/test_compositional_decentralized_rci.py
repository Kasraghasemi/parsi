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
delta=0.1
number_of_subsystems= 10
n=2*number_of_subsystems
m=1*number_of_subsystems

np.random.seed(seed=2)
A=np.random.rand(n,n)* landa
B=np.random.rand(n,m)* landa
#A=np.zeros((n,n))
#B=np.zeros((n,m))

for i in range(number_of_subsystems): 
    A[2*i:2*(i+1),2*i:2*(i+1)]= np.array([[1,1],[0,1]]) * delta
    B[2*i:2*(i+1),i]= np.array([0,1]) * delta

w_i=pp.zonotope(G=[[0.05,0],[0,0.05]] , x=[0,0])
W=w_i
for i in range(number_of_subsystems-1):
    W=W**w_i
# W=pp.zonotope(G=np.eye(n)*0.3,x=np.zeros(n))
X=pp.zonotope(G=np.eye(n),x=np.zeros(n),color='red')
U=pp.zonotope(G=np.eye(m),x=np.zeros(m))

system=parsi.Linear_system(A,B,W=W,X=X,U=U)
sub_sys=parsi.sub_systems(system,partition_A=[2]*number_of_subsystems,partition_B=[1]*number_of_subsystems,disturbance=[w_i for i in range(number_of_subsystems)])
for i in range(number_of_subsystems):
    sub_sys[i].U.G=np.array([sub_sys[i].U.G])


omega,theta=parsi.compositional_decentralized_rci(sub_sys,initial_guess='nominal',size='min',initial_order=4,step_size=0.1,alpha_0='random')

for i in range(number_of_subsystems):
    sub_sys[i].state=parsi.sample(sub_sys[i].omega)

system.state= np.array( [sub_sys[i].state for i in range(number_of_subsystems)] ).flatten()
path=system.state.reshape(-1,1)

cols=5
fig, axs = plt.subplots(int(ceil(number_of_subsystems / cols)),cols)
for i in range(number_of_subsystems):
    sub_sys[i].X.color='red'
    r=i//cols
    c=i%cols
    pp.visualize([sub_sys[i].X,omega[i]], ax = axs[r,c],fig=fig, title='',equal_axis=True)

for step in range(50):
    #Finding the controller
    u=np.array([parsi.mpc(sub_sys[i],horizon=1,x_desired='origin') for i in range(number_of_subsystems)]).flatten()
    state= system.simulate(u)
    path=np.concatenate((path,state.reshape(-1,1)) ,axis=1)
    for i in range(number_of_subsystems):
        sub_sys[i].state=system.state[2*i:2*(i+1)]
        r=i//cols
        c=i%cols
        axs[r,c].plot(path[2*i,:],path[2*i+1,:],color='b')
    plt.pause(0.02)

plt.show()  