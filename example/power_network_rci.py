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

delta_t=2
disturbance=0.002
number_of_subsystems= 10
Kp_i=[115]*number_of_subsystems
Tp_i=[22]*number_of_subsystems
Ks_ij=np.zeros((number_of_subsystems,number_of_subsystems))
for i in range(number_of_subsystems):
    for j in range(i+1,number_of_subsystems):
        Ks_ij[i,j]=np.random.randint(2)
        Ks_ij[j,i]=Ks_ij[i,j]

sub_sys=[]
A,A_ij,B={},{},{}
for i in range(number_of_subsystems):
    A[i]=np.array([[1 , 2* np.pi* delta_t],[ (-delta_t * Kp_i[i]/(2*np.pi*Tp_i[i]))*sum(Ks_ij[i]) , 1-delta_t/Tp_i[i] ]])
    B[i]=np.array([[0],[delta_t*Kp_i[i]/Tp_i[i]]])
    for j in range(number_of_subsystems):
        if Ks_ij[i][j]!=0:
            A_ij[j]= np.array([[0 , 0],[ delta_t*Kp_i[i]*Ks_ij[i][j]/(2*np.pi*Tp_i[i]) ,0]])
    X=pp.zonotope(x=np.zeros(2),G=np.array([[0.3,0],[0,0.3]]))
    U=pp.zonotope(x=np.zeros(1),G=np.array([[1]]))
    W=pp.zonotope(x=np.zeros(2),G=np.array([[0.000000000000000000000001 ,0 ],[0 ,-delta_t*Kp_i[i]/Tp_i[i]]])*disturbance )
    sub_sys.append(parsi.Linear_system(A[i],B[i],W=W,X=X,U=U))

#omega,theta = parsi.decentralized_rci(sub_sys,size='min')
omega,theta=parsi.decentralized_rci(sub_sys,method='centralized',initial_guess='nominal',size='min',solver='gurobi',order_max=100)
#omega,theta=parsi.decentralized_rci(sub_sys,method='centralized',initial_guess='nominal',size='min',solver='drake',order_max=100)
# omega,theta=parsi.compositional_decentralized_rci(sub_sys,initial_guess='nominal',size='min',initial_order=4,step_size=0.1,alpha_0='random',order_max=100)

for i in range(number_of_subsystems):
    sub_sys[i].omega=omega[i]
    sub_sys[i].theta=theta[i]

    sub_sys[i].state=parsi.sample(sub_sys[i].omega)

path= np.array( [sub_sys[i].state for i in range(number_of_subsystems)] ).reshape(-1,1)

cols=5
fig, axs = plt.subplots(int(ceil(number_of_subsystems / cols)),cols)
for i in range(number_of_subsystems):
    sub_sys[i].X.color='red'
    r=i//cols
    c=i%cols
    pp.visualize([sub_sys[i].X,omega[i]], ax = axs[r,c],fig=fig, title='',equal_axis=True)

for step in range(50):
    #Finding the controller
    u=[parsi.mpc(sub_sys[i],horizon=1,x_desired='origin') for i in range(number_of_subsystems)]
    state= np.array([sub_sys[i].simulate(u[i]) for i in range(number_of_subsystems)])
    path=np.concatenate((path,state.reshape(-1,1)) ,axis=1)
    for i in range(number_of_subsystems):
        r=i//cols
        c=i%cols
        axs[r,c].plot(path[2*i,:],path[2*i+1,:],color='b')
    plt.pause(0.02)

plt.tight_layout()
plt.show()  
