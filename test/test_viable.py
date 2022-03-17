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


def sample_trajectory(horizon = 100):
    for _ in range(horizon):
        sys.state = parsi.sample(omega[0])
        path = [sys.state]
        for step in range(number_of_steps):
            u = parsi.find_controller(omega[step] , theta[step] , sys.state )
            sys.simulate(u,step = step)

            path.append(sys.state)

            assert parsi.is_in_set(omega[step+1],sys.state)==True


number_of_steps=5
A=[np.array([[1+0.001*t,1],[0,1-0.001*t]]) for t in range(number_of_steps)]
B=[np.array([[0],[1+0.002*t]]) for t in range(number_of_steps)]
W=[pp.zonotope(G=np.eye(2),x=[0,0])*0.2 for t in range(number_of_steps)]
X=[pp.zonotope(G=np.eye(2)*3,x=[0,0],color='red') for t in range(number_of_steps+1)]
U=[pp.zonotope(G=np.eye(1),x=[0]) for t in range(number_of_steps)]

sys=parsi.Linear_system(A,B,W=W,X=X,U=U)

omega,theta=parsi.viable_limited_time(sys,horizon = number_of_steps,order_max=10,algorithm='slow')
sample_trajectory(1000)

omega,theta=parsi.viable_limited_time(sys,horizon = number_of_steps,order_max=10,algorithm='fast')
sample_trajectory()


# Plotting

# shift=10
# path = np.array(path)
# for i in range(number_of_steps):
#     omega[i].x=omega[i].x+10*i*np.array([1,0])
#     X[i].x=X[i].x+10*i*np.array([1,0])
#     path[i,0] = path[i,0] + 10 * i

# pp.visualize([*X,*omega])
# print('path',path)
# plt.plot(np.array(path)[:,0],np.array(path)[:,1], '*')
# plt.show()