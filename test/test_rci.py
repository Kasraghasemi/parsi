"""
@author: kasra
"""
import numpy as np
import matplotlib.pyplot as plt
try:
    import pypolycontain as pp
except:
    print('Error: pypolycontain package is not installed correctly') 
try:
    import parsi
except:
    raise ModuleNotFoundError("parsi package is not installed properly")
from gurobipy import *

A=np.array([[1,1],[0,1]])
B=np.array([[0],[1]])
W=pp.zonotope(G=np.eye(2),x=[0,0])*0.4
X=pp.zonotope(G=np.array([[1,-1,2,0.5],[0,1,-3,0.5]]) ,x=[0.3,0.4],color='red')
#X=pp.zonotope(G=np.eye(2)*3,x=[0,0],color='red')
U=pp.zonotope(G=np.eye(1)*0.8,x=[0])

sys=parsi.Linear_system(A,B,W=W,X=X,U=U)
#omega,theta = parsi.rci(sys,order_max=10)
omega,theta = sys.rci(order_max=10)

sys.state = parsi.sample(omega)
state = sys.state
for step in range(200):

    u = parsi.find_controller( omega , theta , sys.state)
    sys.simulate(u)

    assert parsi.is_in_set(omega,sys.state)==True


# pp.visualize([X,omega])
# plt.show()


