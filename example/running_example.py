"""
@author: kasra
"""
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy.spatial import ConvexHull
try:
    import pypolycontain as pp
except:
    print('Error: pypolycontain package is not installed correctly') 
try:
    import parsi
except:
    raise ModuleNotFoundError("parsi package is not installed properly")

landa = 0.2
delta = 0.1
disturb= 0.3
control_size = 1
number_of_subsystems= 3
n=2*number_of_subsystems
m=1*number_of_subsystems


A = np.array( [
    [1 , 1 , 0.5 , 0.1 , 0.2 , -0.1 ], 
    [0 , 1 , -0.5 , 0.3 , 0 , -0.2 ], 
    [-0.4 , 0 , 1 , 1 , 0.2 , 0.5 ],
    [0.1 , -0.3 , 0 , 1 , 0.4 , 0 ],
    [0 , 0 , 0.2 , 0.1 , 1 , 1 ], 
    [0 , 0 , 0.1 , 0.5 , 0 , 1 ]
]) * landa
np.random.seed(seed= 44 )
# A=np.random.rand(n,n)* landa
# B=np.random.rand(n,m)* landa
# A=np.zeros((n,n))
B=np.zeros((n,m))

for i in range(number_of_subsystems): 
    A[2*i:2*(i+1),2*i:2*(i+1)]= np.array([[1,1],[0,1]]) * delta
    # A[2*i:2*(i+1),2*i:2*(i+1)]= np.array([[1,delta],[0,1]]) 
    B[2*i:2*(i+1),i]= np.array([0,1]) * delta


W=pp.zonotope(G=np.eye(n)*disturb,x=np.zeros(n))
X=pp.zonotope(G=np.array([
    [ 1 , 0.1 , 0.2 , 0 , 0 , 0 ], 
    [ 0.1 , 1 , 0.02 , 0 , 0 , 0.1 ], 
    [ 0 , 0.01 , 1 , 0 , 0.1 , 0.1 ], 
    [ 0.2 , 0 , 0.03 , 1 , 0 , 0.1 ], 
    [ 0 , 0.1 , 0.1 , 0 , 1 , 0.2 ], 
    [ -0.1 , -0.02 , 0.1 , 0 , 0 , 1 ]
]),x=np.zeros(n),color='red')

U=pp.zonotope(G=np.eye(m)*control_size,x=np.zeros(m))

w_i=pp.zonotope(G=np.eye(2)*disturb , x=np.array([0,0]))
u_i=pp.zonotope(G=np.array([[ control_size ]]) , x=np.array([0]))
system=parsi.Linear_system(A,B,W=W,X=X,U=U)
sub_sys = parsi.sub_systems( system , partition_A = [2]*number_of_subsystems , partition_B = [1]*number_of_subsystems , \
    disturbance = [ w_i for j in range(number_of_subsystems) ] , \
        admissible_u= [u_i for j in range(number_of_subsystems)] )

# for i in range(number_of_subsystems):
#     sub_sys[i].U.G=np.array([sub_sys[i].U.G])







# omega,theta = parsi.decentralized_rci(sub_sys,method='centralized',initial_guess='nominal',size='min',solver='gurobi',order_max=300)
# print('centralizedcentralizedcentralizedcentralizedcentralizedcentralizedcentralizedcentralizedcentralizedcentralized')

# # omega,theta=parsi.compositional_decentralized_rci(sub_sys,initial_guess='nominal',initial_order=4,step_size=1000,alpha_0='random',order_max=20)
# # print('compositionalcompositionalcompositionalcompositionalcompositionalcompositionalcompositionalcompositionalcompositional')



# for i in range(number_of_subsystems):
#     sub_sys[i].omega=omega[i]
#     sub_sys[i].theta=theta[i]

#     sub_sys[i].state=parsi.sample(sub_sys[i].omega)

# system.state= np.array( [sub_sys[i].state for i in range(number_of_subsystems)] ).flatten()
# path=system.state.reshape(-1,1)


# fig, axs = plt.subplots(3)
# for i in range(number_of_subsystems):
#     sub_sys[i].X.color='red'
#     pp.visualize([sub_sys[i].X,omega[i]], ax = axs[i],fig=fig, title='',equal_axis=True)

# for step in range(50):
#     #Finding the controller
#     u=np.array([parsi.mpc(sub_sys[i],horizon=1,x_desired='origin') for i in range(number_of_subsystems)]).flatten()
#     state= system.simulate(u)
#     path=np.concatenate((path,state.reshape(-1,1)) ,axis=1)
#     for i in range(number_of_subsystems):
#         sub_sys[i].state=system.state[2*i:2*(i+1)]

#         axs[i].plot(path[2*i,:],path[2*i+1,:],color='b')
#     plt.pause(0.02)
# plt.show()














################################################
####### Drawing the X decomposition ############
################################################

circumbody_1 = pp.zonotope( X.G[:2,:] , X.x[0:2] ,color='green')
circumbody_2 = pp.zonotope( X.G[2:4,:] , X.x[2:4] ,color='green')
circumbody_3 = pp.zonotope( X.G[4:6,:] , X.x[4:6] ,color='green')

for i in range(number_of_subsystems):
    sub_sys[i].X.color = 'red' 

fig1, axs1 = plt.subplots(1,3)

pp.visualize([circumbody_1, sub_sys[0].X], ax = axs1[0],fig=fig1, title='')
axs1[0].axis('equal')
axs1[0].set_xlabel(r'$x_1$',FontSize=15)
axs1[0].set_ylabel(r'$x_2$',FontSize=15,rotation=0)
axs1[0].set_title(r'$x_1 - x_2$'  'plane')
axs1[0].set_xlim([-2,2])
# axs1[0].set_ylim([-2,2])

pp.visualize([circumbody_2, sub_sys[1].X], ax = axs1[1],fig=fig1, title='')
axs1[1].axis('equal')
axs1[1].set_xlabel(r'$x_3$',FontSize=15)
axs1[1].set_ylabel(r'$x_4$',FontSize=15,rotation=0)
axs1[1].set_title(r'$x_3 - x_4$'  'plane')
axs1[1].set_xlim([-2,2])
# axs1[1].set_ylim([-2,2])

pp.visualize([circumbody_3, sub_sys[2].X], ax = axs1[2],fig=fig1, title='')
axs1[2].axis('equal')
axs1[2].set_xlabel(r'$x_5$',FontSize=15)
axs1[2].set_ylabel(r'$x_6$',FontSize=15,rotation=0)
axs1[2].set_title(r'$x_5 - x_6$'  'plane')
axs1[2].set_xlim([-2,2])
# axs1[2].set_ylim([-2,2])

plt.tight_layout()
plt.show()



###################################################################################
#### Drawing Valid set of alphas and the trajectory (just the first subsystem) ####
###################################################################################


# finding valid set of alphas
transformation_degree = 10
degree = transformation_degree * np.pi / 180
transformation_matrix = np.array( [[ np.cos(degree) , -np.sin(degree) ], 
                                   [ np.sin(degree) , np.cos(degree)]])
c = np.array([1 , 0])
set_of_alpha = []
for i in range( int(360/transformation_degree +1 ) ):

    omega , theta , alfa_x , alfa_u = parsi.decentralized_rci_centralized_gurobi(sub_sys,initial_guess='nominal',size = c ,order_max=20)
    for sys in sub_sys:
        sys.omega = None
        sys.theta = None

    c = np.dot( transformation_matrix , c )
    print("i",i)
    if alfa_x != None:
        set_of_alpha.append( alfa_x[0][0:len(c)] )
set_of_alpha = np.array(set_of_alpha)
hull = ConvexHull( set_of_alpha )

print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')

# finding the trajectory toward the valid set of alpha
omega,theta=parsi.compositional_decentralized_rci(sub_sys,initial_guess='nominal',initial_order=4,step_size= 0.1,alpha_0='random',order_max=20)


# Drawing
fig2 = plt.figure(facecolor=(1, 1, 1))
ax2 = plt.axes()
# ax2.set_facecolor('xkcd:salmon')
# ax2.set_facecolor((1.0, 0.47, 0.42))

ax2.text(1.1, 1.0, r'Correct Composition $\mathcal{V}=0$', fontsize=18)

ax2.fill( set_of_alpha[hull.vertices,0] , set_of_alpha[hull.vertices,1] , 'green' , alpha = 0.7)
# plt.xlim(0,2)
# plt.ylim(0,2)
plt.xlabel(r'$\alpha_1^x[1]$',FontSize=17 )
plt.ylabel(r'$\alpha_1^x[2]$',FontSize=17,rotation=0)

alpha_trajectory = np.array( parsi.Monitor['alpha_x'][0] )
print('trajectory', alpha_trajectory)
ax2.plot( alpha_trajectory[:,0]  , alpha_trajectory[:,1] , 'blue')

ax2.arrow(alpha_trajectory[0][0] , alpha_trajectory[0][1] , -1 *parsi.Monitor['gradient'][0][0] ,-1 * parsi.Monitor['gradient'][0][1] ,head_width=0.01, head_length=0.03, fc='k', ec='k' )
ax2.arrow(alpha_trajectory[0][0] , alpha_trajectory[0][1] , - 1.5* parsi.Monitor['gradient'][1][0] ,- 1.5* parsi.Monitor['gradient'][1][1] ,head_width=0.01, head_length=0.03, fc='k', ec='k' )
ax2.arrow(alpha_trajectory[0][0] , alpha_trajectory[0][1] , - 1.5* parsi.Monitor['gradient'][2][0] ,- 1.5* parsi.Monitor['gradient'][2][1] ,head_width=0.01, head_length=0.03, fc='k', ec='k' )

# plt.tight_layout()
plt.show()




###################################################################################
######## Drawing 3D of the potential function (just the first subsystem) ##########
###################################################################################

from itertools import product

alpha1 = np.arange(0,2,0.1)
alpha2 = np.arange(0,2,0.1)
alpha_1_2 = list( product( alpha1 , alpha2 ) )

# for i in number_of_subsystems:
#     sub_sys[i].alpha_x = parsi.Monitor['alpha_x'][i][-1]
#     sub_sys[i].alpha_u = 

potential = np.zeros( len(alpha_1_2))
for a in enumerate(alpha_1_2):

    for sys in sub_sys:
        sys.omega = None
        sys.theta = None

    sub_sys[0].alpha_x[0:2] = a[1] 

    subsystems_output = [ parsi.potential_function(sub_sys, system_index, T_order=parsi.Monitor['order'], reduced_order=1) for system_index in range(number_of_subsystems) ]
    potential[ a[0] ] = sum([subsystems_output[i]['obj'] for i in range(number_of_subsystems) ])



        
X, Y = np.meshgrid(alpha1, alpha2)
fig3 = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, eps, 100,cmap='viridis')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#ax.view_init(60, 45)
#ax.set_zlim(-0.01,0.6)
ax3 = plt.axes(projection='3d')
ax3.plot_surface(X, Y, potential.reshape( len(alpha2) , len(alpha1) ) , rstride=1, cstride=1,cmap='viridis', edgecolor='none')
ax3.set_xlabel(r'$\alpha_1^x[2]$' , FontSize=14)
ax3.set_ylabel(r'$\alpha_1^x[1]$' , FontSize=14)
ax3.set_zlabel( r'$\mathcal{V}$' ,  FontSize=14)
# ax3.set_title('potential function')
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax3.view_init(45, 25)
plt.show()




