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
number_of_subsystems= 5
n=2*number_of_subsystems
m=1*number_of_subsystems

#np.random.seed(seed=2)
A=np.random.rand(n,n)* landa
B=np.random.rand(n,m)* landa
#A=np.zeros((n,n))
#B=np.zeros((n,m))

for i in range(number_of_subsystems): 
    A[2*i:2*(i+1),2*i:2*(i+1)]= np.array([[1,1],[0,1]]) * delta
    B[2*i:2*(i+1),i]= np.array([0,1]) * delta

W=pp.zonotope(G=np.eye(n)*0.3,x=np.zeros(n))
X=pp.zonotope(G=np.eye(n),x=np.zeros(n),color='red')
U=pp.zonotope(G=np.eye(m),x=np.zeros(m))

system=parsi.Linear_system(A,B,W=W,X=X,U=U)
sub_sys=parsi.sub_systems(system,partition_A=[2]*number_of_subsystems,partition_B=[1]*number_of_subsystems)

for i in range(number_of_subsystems):
    sub_sys[i].U.G=np.array([sub_sys[i].U.G])

# Initiali parameterized sets


# Initializing alpha_x and alpha_u
for i in sub_sys:
    i.parameterized_set_initialization()
    i.set_alpha_max({'x': i.param_set_X, 'u':i.param_set_U})

    i.alpha_x=np.random.rand(len(i.alpha_x_max))
    i.alpha_u=np.random.rand(len(i.alpha_u_max))

    # i.alpha_x=np.array(i.alpha_x_max)*0.5
    # i.alpha_u=np.array(i.alpha_u_max)*0.5

# Output for one specific sub-system
output=parsi.potential_function(sub_sys, 0, T_order=10, reduced_order=1)

print('x',output['alpha_x_grad'])
print('u',output['alpha_u_grad'])
print('obj',output['obj'])