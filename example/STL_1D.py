"""
@author: kasra
"""
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *
try:
    import pypolycontain as pp
except:
    print('Error: pypolycontain package is not installed correctly') 
try:
    import parsi
except:
    raise ModuleNotFoundError("parsi package is not installed properly")

from scipy.linalg import block_diag as blk

def be_in_set_gurobi( point , set , model):

    zeta = np.array( [model.addVar( lb=-1 , ub=1 ) for i in range( set.G.shape[1] ) ] )
    model.update()
    
    [model.addConstr( point[j] == ( set.x.reshape(-1) + np.dot( set.G , zeta.reshape(-1,1)).reshape(-1) )[j])  for j in range(len(set.x)) ]
    model.update()
    return model

initial_state = np.array([ 0,0,0,0 ])

delta_t = 0.5
disturbance = 0.05
state_hard = 3
control_hard = 5
landa = 0.01

n= 4
m= 2
n_i = 2
m_i = 1
W_i = pp.zonotope( x= np.zeros(2) ,G =  disturbance * np.eye(2)  )
X_i = pp.zonotope( x= np.zeros(2) ,G =  state_hard * np.eye(2) )
U_i = pp.zonotope( x= np.zeros(1) ,G =  control_hard * np.eye(1) )

A = np.array([
    [1 , delta_t , landa , landa],
    [0 , 1 , landa , landa],
    [landa , landa , 1 , delta_t],
    [landa , landa , 0 , 1]
])

B= np.array([
    [0 , 0], 
    [delta_t , 0], 
    [0 , 0], 
    [0, delta_t]
])

system = parsi.Linear_system( A , B , W = None , X= pp.zonotope( x=np.zeros(4) , G= state_hard * np.eye(4) ) , U = pp.zonotope( x=np.zeros(1) ,G =  control_hard * np.eye(1) )  )

sub_sys=parsi.sub_systems(system,partition_A=[2,2],partition_B=[1,1] , 
disturbance = [ W_i, W_i ] , 
admissible_x = [ X_i , X_i ] , 
admissible_u = [ U_i , U_i] )

# for i in range(2):
#     sub_sys[i].W = W_i
#     sub_sys[i].X = X_i
#     sub_sys[i].U = U_i

###########################################################################################
############ Centralized computation ######################################################
###########################################################################################

horizon = 12

model_cen = Model()

x = np.array( [ [model_cen.addVar( lb = -GRB.INFINITY , ub = GRB.INFINITY) for i in range(n) ] for j in range(horizon)] )
u = np.array( [ [model_cen.addVar( lb = -GRB.INFINITY , ub = GRB.INFINITY) for i in range(m) ] for j in range(horizon-1)] )
model_cen.update()

[model_cen.addConstr( x[0][j] == initial_state[j] ) for j in range(n) ]

model_cen.update()

for i in range(horizon-1):

    [model_cen.addConstr( x[i+1][j] == ( np.dot(A,x[i]) + np.dot(B,u[i]) )[j] ) for j in range(n) ]

    model_cen.addConstr( x[i][0] <= state_hard )
    model_cen.addConstr( x[i][0] >= -state_hard )

    model_cen.addConstr( x[i][1] <= state_hard )
    model_cen.addConstr( x[i][1] >= -state_hard )

    model_cen.addConstr( x[i][2] <= state_hard )
    model_cen.addConstr( x[i][2] >= -state_hard )

    model_cen.addConstr( x[i][3] <= state_hard )
    model_cen.addConstr( x[i][3] >= -state_hard )

    model_cen.addConstr( u[i][0] <= control_hard )
    model_cen.addConstr( u[i][1] <= control_hard )
    model_cen.addConstr( u[i][0] >= -control_hard )
    model_cen.addConstr( u[i][1] >= -control_hard )

    model_cen.update()

model_cen.addConstr( x[horizon-1][0] <= state_hard )
model_cen.addConstr( x[horizon-1][1] <= state_hard )
model_cen.addConstr( x[horizon-1][2] >= -state_hard )
model_cen.addConstr( x[horizon-1][3] >= -state_hard )

model_cen.update()


# state 0 :Adding binary variables

z_si_1 = model_cen.addVar( vtype=GRB.BINARY ) 
model_cen.addConstr(z_si_1==True)
model_cen.update()


z_si_1_t = [model_cen.addVar( vtype=GRB.BINARY ) for i in range( 5 )] 

model_cen.addConstr( z_si_1 <= sum(z_si_1_t) )

model_cen.update()

z_si_1_t_pred = [ [model_cen.addVar( vtype=GRB.BINARY ) for i in range(3)] for j in range(5) ]

for j in range(5):
    for i in range(3):
        model_cen.addConstr( z_si_1_t[j] <= z_si_1_t_pred[j][i] )
model_cen.update()


# state 2: Adding binary variables

z_si_3 = model_cen.addVar( vtype=GRB.BINARY ) 
model_cen.addConstr(z_si_3==True)
model_cen.update()


z_si_3_t = [model_cen.addVar( vtype=GRB.BINARY ) for i in range( 5 )] 

model_cen.addConstr( z_si_3 <= sum(z_si_3_t) )

model_cen.update()

z_si_3_t_pred = [ [model_cen.addVar( vtype=GRB.BINARY ) for i in range(3)] for j in range(5) ]

for j in range(5):
    for i in range(3):
        model_cen.addConstr( z_si_3_t[j] <= z_si_3_t_pred[j][i] )
model_cen.update()



# Big M
big_M = 100
rho = model_cen.addVar( lb = -GRB.INFINITY , ub = GRB.INFINITY)

for j in range(5):
    for i in range(3):
        model_cen.addConstr( ( x[i+j][0] - 2 ) + big_M * (1 - z_si_1_t_pred[j][i] ) >= rho)
        model_cen.addConstr( ( x[i+j][0] - 2 ) - big_M * z_si_1_t_pred[j][i]  <= rho )

        model_cen.addConstr( (x[i+j][2] - 2 ) + big_M * (1 - z_si_3_t_pred[j][i] ) >= rho)
        model_cen.addConstr( (x[i+j][2] - 2) - big_M * z_si_3_t_pred[j][i]  <= rho )

model_cen.update()








# state 5: Adding binary variables

z_si_5 = model_cen.addVar( vtype=GRB.BINARY ) 
model_cen.addConstr(z_si_5==True)
model_cen.update()


z_si_5_t = [model_cen.addVar( vtype=GRB.BINARY ) for i in range( 5 )] 

model_cen.addConstr( z_si_5 <= sum(z_si_5_t) )

model_cen.update()

z_si_5_t_pred = [ [model_cen.addVar( vtype=GRB.BINARY ) for i in range(3)] for j in range(5) ]

for j in range(5):
    for i in range(3):
        model_cen.addConstr( z_si_5_t[j] <= z_si_5_t_pred[j][i] )
model_cen.update()




# state 7: Adding binary variables

z_si_7 = model_cen.addVar( vtype=GRB.BINARY ) 
model_cen.addConstr(z_si_7==True)
model_cen.update()


z_si_7_t = [model_cen.addVar( vtype=GRB.BINARY ) for i in range( 5 )] 

model_cen.addConstr( z_si_7 <= sum(z_si_7_t) )

model_cen.update()

z_si_7_t_pred = [ [model_cen.addVar( vtype=GRB.BINARY ) for i in range(3)] for j in range(5) ]

for j in range(5):
    for i in range(3):
        model_cen.addConstr( z_si_7_t[j] <= z_si_7_t_pred[j][i] )
model_cen.update()

for j in range(5):
    for i in range(3):
        model_cen.addConstr( ( -x[5+i+j][0] - 2 ) + big_M * (1 - z_si_1_t_pred[j][i] ) >= rho)
        model_cen.addConstr( ( -x[5+i+j][0] - 2 ) - big_M * z_si_1_t_pred[j][i]  <= rho )

        model_cen.addConstr( (-x[5+i+j][2] - 2 ) + big_M * (1 - z_si_3_t_pred[j][i] ) >= rho)
        model_cen.addConstr( (-x[5+i+j][2] - 2 ) - big_M * z_si_3_t_pred[j][i]  <= rho )

model_cen.update()



model_cen.setObjective( rho , GRB.MAXIMIZE)
model_cen.optimize()


x_0_up= np.array([ x[i][0].X for i in range(12) ] ) >= 2

x_0_low= np.array([ x[i][0].X for i in range(12) ] ) <= -2

x_2_up= np.array([ x[i][2].X for i in range(12) ] ) >= 2

x_2_low= np.array([ x[i][2].X for i in range(12) ] ) <= -2



####################################################################################################
################################### Computing the bounds ###########################################
####################################################################################################


omega = [ sub_sys[0].rci()[0].G , sub_sys[1].rci()[0].G ]


# AGC
k = 10
horizon = 12

model = Model()


cbar = [ [ np.array([model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(2)]) for step in range(horizon)] for l in range(2)]
model.update()

T = [ [ np.array([ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(k)] for j in range(2)]) for step in range(horizon)] for l in range(2)]

M = [ [ np.array([ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(k)] for j in range(1)]) for step in range(horizon-1)] for l in range(2)]

xbar = [ [ np.array([model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(2)]) for step in range(horizon)] for l in range(2)]

ubar = [ [ np.array([model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(1)]) for step in range(horizon-1)] for l in range(2)]

model.update()

model.addConstr( xbar[0][0][0]== initial_state[0] )
model.addConstr( xbar[0][0][1]== initial_state[1] )
model.addConstr( xbar[1][0][0]== initial_state[2] )
model.addConstr( xbar[1][0][1]== initial_state[3] )

model.update()


alpha= [0,0]

for l in range(2):

    alpha[l] = [ pp.zonotope_subset( model , pp.zonotope( x= xbar[l][step] , G= T[l][step] ) , pp.zonotope( x = cbar[l][step].reshape(-1) , G = omega[l] ) , solver='gurobi' , alpha= 'vecotor')[2] for step in range(horizon) ]
    model.update()



dist= [0,0]

A_ij = landa * np.ones((2,2))


dist[0] = [ pp.zonotope( x =  np.dot( A_ij , cbar[1][step].reshape(-1,1) ).reshape(-1)  , G = np.dot( A_ij , np.dot( omega[1] , np.diag(alpha[1][step]) ) )  ) + W_i   for step in range(horizon-1)] 
dist[1] = [ pp.zonotope( x =  np.dot( A_ij , cbar[0][step].reshape(-1,1) ).reshape(-1)  , G = np.dot( A_ij , np.dot( omega[0] , np.diag(alpha[0][step]) )  ) ) + W_i   for step in range(horizon-1)] 



# Constraint [AT+BM,W]=[T] or [0,T]
for l in range(2):


    left_side = [np.concatenate ( ( np.dot(sub_sys[l].A , T[l][step]) + np.dot(sub_sys[l].B,M[l][step])  , dist[l][step].G) ,axis=1) for step in range(horizon-1)]
    right_side = [np.concatenate(   ( np.zeros(dist[l][step-1].G.shape) , T[l][step])  , axis=1) for step in range(1,horizon)]

    [model.addConstrs(   ( left_side[step][i,j] == right_side[step][i,j]  for i in range( 2 ) for j in range(len(right_side[step][0]))   )  )  for step in range(horizon-1)]              
    
    model.update()
    
    # Conditions for center: A* x_bar + B* u_bar + d_bar = x_bar 
    

    center = [np.dot( sub_sys[l].A , xbar[l][step].reshape(-1,1) ) + np.dot( sub_sys[l].B , ubar[l][step].reshape(-1,1) ) + dist[l][step].x.reshape(-1,1) for step in range(horizon-1)]

    
    model.addConstrs( (center[step][i][0] == xbar[l][step+1][i] for i in range(2) for step in range(horizon-1) ))
    
    model.update()

    [ pp.zonotope_subset( model , pp.zonotope( x= xbar[l][step] , G= T[l][step] ) , X_i , solver='gurobi') for step in range(horizon)]

    [ pp.zonotope_subset( model , pp.zonotope( x= ubar[l][step] , G= M[l][step] ) , U_i , solver='gurobi') for step in range(horizon-1) ]

    # [ pp.zonotope_subset( model , pp.zonotope( x= xbar[l][step] , G= T[l][step] ) , pp.zonotope( x = cbar[l][step] , G = np.dot( omega[l] , np.diag(alpha[l][step]) ) ) , solver='gurobi') for step in range(horizon)]

    model.update()


q_up1 = []
for i in range(horizon):
    if x_0_up[i]:
    
        q_up1.append( [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for _ in range(k) ] )
        model.update()
        model.addConstrs( (q_up1[-1][uu] >= T[0][i][0][uu] for uu in range(k) ))

        model.addConstrs( (q_up1[-1][uu] >= -1 * T[0][i][0][uu] for uu in range(k) ))

        model.addConstr( -xbar[0][i][0] + sum(q_up1[-1]) + 2 <= -rho.X ) 

        model.update()


q_up2 = []
for i in range(horizon):
    if x_2_up[i]:
    
        q_up2.append( [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for _ in range(k) ] )
        model.update()
        model.addConstrs( (q_up2[-1][uu] >= T[1][i][0][uu] for uu in range(k) ))

        model.addConstrs( (q_up2[-1][uu] >= -1 * T[1][i][0][uu] for uu in range(k) ))

        model.addConstr( -xbar[1][i][0] + sum(q_up2[-1]) + 2 <= -rho.X ) 

        model.update()

for l in range(2):

    pp.zonotope_subset( model , pp.zonotope( x= xbar[l][9] , G= T[l][9] ) , pp.zonotope( x = [-6,0] , G= [[4 ,0],[0,100]]  ) , solver='gurobi') 
    pp.zonotope_subset( model , pp.zonotope( x= xbar[l][10] , G= T[l][10] ) , pp.zonotope( x = [-6,0] , G= [[4 ,0],[0,100]]  ) , solver='gurobi') 
    pp.zonotope_subset( model , pp.zonotope( x= xbar[l][11] , G= T[l][11] ) , pp.zonotope( x = [-6,0] , G= [[4 ,0],[0,100]]  ) , solver='gurobi') 

# q_down1 =  []
# for i in range(5,12):
#     if x_0_low[i]:
        
#         q_down1.append( [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for _ in range(k) ] )
#         model.update()
#         model.addConstrs( (q_down1[-1][uu] >= T[0][i][0][uu] for uu in range(k) ))

#         model.addConstrs( (q_down1[-1][uu] >= -1 * T[0][i][0][uu] for uu in range(k) ))

#         model.addConstr( xbar[0][i][0] + sum(q_down1[-1]) + 2 <= -rho.X ) 

#         model.update()

# q_down2 =  []
# for i in range(5,12):
#     if x_2_low[i]:
        
#         q_down2.append( [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for _ in range(k) ] )
#         model.update()
#         model.addConstrs( (q_down2[-1][uu] >= T[1][i][0][uu] for uu in range(k) ))

#         model.addConstrs( (q_down2[-1][uu] >= -1 * T[1][i][0][uu] for uu in range(k) ))

#         model.addConstr( xbar[1][i][0] + sum(q_down2[-1]) + 2 <= -rho.X ) 

#         model.update()


model.update()
model.optimize()

print("xxxxxxxxxx",rho.x)

print("x_0_up" , x_0_up)

print("x_0_low" , x_0_low)

omega= [0,0]

color = ['blue' , 'red']
for i in range(2):
    omega[i] = [ pp.zonotope( x = np.array( [ xbar[i][step][j].X for j in range(2)] ).reshape(-1) , G = [ [ T[i][step][j][kk].X for kk in range(k) ] for j in range(2)] , color = color[i] )  for step in range(1,horizon)]
 

center_1 =  np.array( [[xbar[0][step][j].X for step in range(horizon)] for j in range(2)])

center_2 =  np.array( [[xbar[1][step][j].X for step in range(horizon)] for j in range(2)])



fig, axs = plt.subplots(2) 

pp.visualize( omega[0] , ax= axs[0] , fig= fig , title="")

pp.visualize( omega[1] ,ax = axs[1],fig=fig ,  title="")

# axs[0].set_title("Subsystem 1",FontSize=12)
# axs[1].set_title("Subsystem 2",FontSize=12)
axs[0].set_xlabel(r'$x_ %d[1]$'%(1),FontSize=12)
axs[0].set_ylabel(r'$x_ %d[2]$'%(1),FontSize=12 , rotation=0)
axs[1].set_xlabel(r'$x_ %d[1]$'%(2),FontSize=12)
axs[1].set_ylabel(r'$x_ %d[2]$'%(2),FontSize=12 , rotation=0)
# plt.tight_layout()

axs[0].plot( center_1[0,:] ,center_1[1,:] )
axs[1].plot( center_2[0,:] ,center_2[1,:] )
plt.show()

