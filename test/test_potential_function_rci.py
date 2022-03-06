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

landa=0.000001
delta=0.1
disturbance = 0.3
number_of_subsystems= 5
n=2*number_of_subsystems
m=1*number_of_subsystems

A=np.ones((n,n))* landa
B=np.ones((n,m))* landa

for i in range(number_of_subsystems): 
    A[2*i:2*(i+1),2*i:2*(i+1)]= np.array([[1,1],[0,1]]) * delta
    B[2*i:2*(i+1),i]= np.array([0,1]) * delta

W=pp.zonotope(G=np.eye(n)*disturbance,x=np.zeros(n))
X=pp.zonotope(G=np.eye(n),x=np.zeros(n),color='red')
U=pp.zonotope(G=np.eye(m),x=np.zeros(m))

system=parsi.Linear_system(A,B,W=W,X=X,U=U)
sub_sys=parsi.sub_systems(
    system,
    partition_A=[2]*number_of_subsystems,
    partition_B=[1]*number_of_subsystems,
    disturbance= [pp.zonotope(G=np.eye(2)*disturbance ,x=np.zeros(2)) for _ in range(number_of_subsystems)],
    admissible_x= [pp.zonotope(G=np.eye(2),x=np.zeros(2)) for _ in range(number_of_subsystems)], 
    admissible_u= [pp.zonotope(G=np.eye(1),x=np.zeros(1)) for _ in range(number_of_subsystems)]
    )

omega , theta , alfa_x , alfa_u=parsi.decentralized_rci_centralized_synthesis(sub_sys,size='min',order_max=30)

# Initializarion of parameterized sets and alpha_x and alpha_u
for sys_index in range(number_of_subsystems):
    sub_sys[sys_index].parameterized_set_initialization()
    # sub_sys[sys_index].set_alpha_max({'x': i.param_set_X, 'u':i.param_set_U})

    sub_sys[sys_index].alpha_x= alfa_x[sys_index]
    sub_sys[sys_index].alpha_u= alfa_u[sys_index]

# Output for one specific subsystem
for sys_index in range(number_of_subsystems):
    output=parsi.potential_function_rci(sub_sys,sys_index , 6, reduced_order=1)
    assert abs( output['obj']) <= 10**(-5)