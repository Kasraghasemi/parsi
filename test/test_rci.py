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

A=np.array([[1,1],[0,1]])
B=np.array([[0],[1]])
W=pp.zonotope(G=np.eye(2),x=[0,0])*0.1
X=pp.zonotope(G=np.random.rand(2,4)*3,x=[0,0],color='red')
#X=pp.zonotope(G=np.eye(2)*3,x=[0,0],color='red')
U=pp.zonotope(G=np.eye(1),x=[0])

sys=parsi.Linear_system(A,B,W=W,X=X,U=U)
sys.beta=0.2
sys.E=False
#omega,theta = parsi.rci(sys,order_max=10)
omega,theta = sys.rci(order_max=10)

print('Generator is ',omega.G)
print('center is ',omega.x)
pp.visualize([X,omega])
plt.show()


