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



delta_t = 0.2
disturbance = 0.00002

# sub_sys = []
# A,B = {},{}

# for i in range(3):

#     A[i] = np.array([
#         [ 1 , delta_t , 0 , 0 ] ,
#         [ 0 , 1 , 0 , 0  ],
#         [0 , 0 , 1 , delta_t ], 
#         [0 , 0 , 0 , 1 ]
#         ])

#     B[i] = np.array([
#         [ 0 , 0 ] ,
#         [ delta_t , 0 ] , 
#         [ 0 , 0], 
#         [0 , delta_t]
#         ])

#     X_i = pp.zonotope( x=np.zeros(4),G=np.eye(4)  )

#     U_i = pp.zonotope( x=np.zeros(2),G=np.eye(2) )

#     W_i = pp.zonotope(x=np.zeros(4),G= np.eye(4) * disturbance )

#     sub_sys.append( parsi.Linear_system( A[i] , B[i] , W = W_i , X = X_i , U = U_i ) )

#     for j in range(3):
        
#         sub_sys[i].A_ij[j] = 0.00001 * np.random.rand(4,4)


A_total = np.ones((12,12)) * 0.0000001
B_total = np.zeros((12,6))

A_total[0:4,0:4] = np.array([
        [ 1 , delta_t , 0 , 0 ] ,
        [ 0 , 1 , 0 , 0  ],
        [0 , 0 , 1 , delta_t ], 
        [0 , 0 , 0 , 1 ]
        ])
A_total[4:8,4:8] = np.array([
        [ 1 , delta_t , 0 , 0 ] ,
        [ 0 , 1 , 0 , 0  ],
        [0 , 0 , 1 , delta_t ], 
        [0 , 0 , 0 , 1 ]
        ])
A_total[8:12,8:12] = np.array([
        [ 1 , delta_t , 0 , 0 ] ,
        [ 0 , 1 , 0 , 0  ],
        [0 , 0 , 1 , delta_t ], 
        [0 , 0 , 0 , 1 ]
        ])

B_total[0:4,0:2] = np.array([
        [ 0 , 0 ] ,
        [ delta_t , 0 ] , 
        [ 0 , 0], 
        [0 , delta_t]
        ])
B_total[4:8,2:4] = np.array([
        [ 0 , 0 ] ,
        [ delta_t , 0 ] , 
        [ 0 , 0], 
        [0 , delta_t]
        ])
B_total[8:12,4:6] = np.array([
        [ 0 , 0 ] ,
        [ delta_t , 0 ] , 
        [ 0 , 0], 
        [0 , delta_t]
        ])

system = parsi.Linear_system( A_total , B_total , W=None , X= pp.zonotope( x=np.zeros(12) , G= 100*np.eye(12) ) , U = pp.zonotope( x=np.zeros(6) ,G =  10*np.eye(6) )  )

sub_sys=parsi.sub_systems(system,partition_A=[4,4,4],partition_B=[2,2,2] , 
disturbance=[pp.zonotope( x= np.zeros(4) ,G =  np.eye(4) * disturbance ), pp.zonotope( x= np.zeros(4) , G = np.eye(4) * disturbance ), pp.zonotope( x= np.zeros(4) ,G =  np.eye(4) * disturbance )] , 
admissible_x=[pp.zonotope( x= np.zeros(4) ,G =  10 * np.eye(4) ) , pp.zonotope( x= np.zeros(4) , G = 10 * np.eye(4) ) , pp.zonotope( x= np.zeros(4) , G = 10 * np.eye(4) )] , 
admissible_u=[pp.zonotope( x= np.zeros(2) , G = 10 * np.eye(2) ) , pp.zonotope( x= np.zeros(2) , G = 10 * np.eye(2) ), pp.zonotope( x= np.zeros(2) , G = 10 * np.eye(2) )] )



###########################################################################################
############ Centralized computation ######################################################
###########################################################################################

horizon = 30 

n = system.A.shape[0]               # Matrix A is n*n
m = system.B.shape[1]               # Matrix B is n*m
x_desired=np.array([
    [0.5],
    [0], 
    [0.5],
    [0],
    [0.5],
    [0], 
    [0.5],
    [0],
    [0.5],
    [0], 
    [0.5],
    [0],
]) * 5

model_cen = Model()

x = np.array( [ [model_cen.addVar( lb = -GRB.INFINITY , ub = GRB.INFINITY) for i in range(12) ] for j in range(horizon)] )
u = np.array( [ [model_cen.addVar( lb = -GRB.INFINITY , ub = GRB.INFINITY) for i in range(6) ] for j in range(horizon-1)] )
model_cen.update()

[model_cen.addConstr( x[0][j] == x_desired[j] ) for j in range(12) ]

model_cen.update()

for i in range(horizon-1):

    [model_cen.addConstr( x[i+1][j] == (np.dot(A_total , x[i]) + np.dot( B_total , u[i] ))[j] ) for j in range(12) ]

    be_in_set_gurobi(x[i] , system.X , model_cen )
    
    be_in_set_gurobi(u[i] , system.U , model_cen )

    model_cen.update()

be_in_set_gurobi(x[horizon-1] , system.X , model_cen )

model_cen.update()


# Adding binary variables

z_si_2 = model_cen.addVar( vtype=GRB.BINARY ) 
model_cen.addConstr(z_si_2==True)
model_cen.update()

z_si_2_t = [model_cen.addVar( vtype=GRB.BINARY ) for i in range(15)] 

model_cen.addConstr( z_si_2 <= sum(z_si_2_t) )

model_cen.update()

z_si_2_t_pred = [ [model_cen.addVar( vtype=GRB.BINARY ) for i in range(12)] for j in range(15) ]


for j in range(15):
    for i in range(12):
        model_cen.addConstr( z_si_2_t[j] <= z_si_2_t_pred[j][i] )
model_cen.update()

# Big M
big_M = 1000
print('x',x.shape)
rho = model_cen.addVar( lb = -GRB.INFINITY , ub = GRB.INFINITY)
for i in range(15):

    model_cen.addConstr( ( x[i+15][0] + 0.5 ) + big_M * (1 - z_si_2_t_pred[i][0] ) >= rho)
    model_cen.addConstr( ( x[i+15][0] + 0.5 ) - big_M * z_si_2_t_pred[i][0]  <= rho )

    model_cen.addConstr( (x[i+15][2] + 0.5) + big_M * (1 - z_si_2_t_pred[i][1] ) >= rho)
    model_cen.addConstr( (x[i+15][2] + 0.5) - big_M * z_si_2_t_pred[i][1]  <= rho )

    model_cen.addConstr( (0.5 - x[i+15][0]) + big_M * (1 - z_si_2_t_pred[i][2] ) >= rho)
    model_cen.addConstr( (0.5 - x[i+15][0]) - big_M * z_si_2_t_pred[i][2]  <= rho )

    model_cen.addConstr( (0.5 - x[i+15][2]) + big_M * (1 - z_si_2_t_pred[i][3] ) >= rho)
    model_cen.addConstr( (0.5 - x[i+15][2]) - big_M * z_si_2_t_pred[i][3]  <= rho )



    model_cen.addConstr( ( x[i+15][4] + 0.5 ) + big_M * (1 - z_si_2_t_pred[i][4] ) >= rho)
    model_cen.addConstr( ( x[i+15][4] + 0.5 ) - big_M * z_si_2_t_pred[i][4]  <= rho )

    model_cen.addConstr( ( x[i+15][6] + 0.5 ) + big_M * (1 - z_si_2_t_pred[i][5] ) >= rho)
    model_cen.addConstr( ( x[i+15][6] + 0.5 ) - big_M * z_si_2_t_pred[i][5]  <= rho )

    model_cen.addConstr( ( 0.5 - x[i+15][4] ) + big_M * (1 - z_si_2_t_pred[i][6] ) >= rho)
    model_cen.addConstr( ( 0.5 - x[i+15][4] ) - big_M * z_si_2_t_pred[i][6]  <= rho )

    model_cen.addConstr( ( 0.5 - x[i+15][6] ) + big_M * (1 - z_si_2_t_pred[i][7] ) >= rho)
    model_cen.addConstr( ( 0.5 - x[i+15][6] ) - big_M * z_si_2_t_pred[i][7]  <= rho )



    model_cen.addConstr( ( x[i+15][8] + 0.5 ) + big_M * (1 - z_si_2_t_pred[i][8] ) >= rho)
    model_cen.addConstr( ( x[i+15][8] + 0.5 ) - big_M * z_si_2_t_pred[i][8]  <= rho )

    model_cen.addConstr( ( x[i+15][10] + 0.5 ) + big_M * (1 - z_si_2_t_pred[i][9] ) >= rho)
    model_cen.addConstr( ( x[i+15][10] + 0.5 ) - big_M * z_si_2_t_pred[i][9]  <= rho )

    model_cen.addConstr( ( 0.5 - x[i+15][8] ) + big_M * (1 - z_si_2_t_pred[i][10] ) >= rho)
    model_cen.addConstr( ( 0.5 - x[i+15][8] ) - big_M * z_si_2_t_pred[i][10]  <= rho )

    model_cen.addConstr( ( 0.5 - x[i+15][10] ) + big_M * (1 - z_si_2_t_pred[i][11] ) >= rho)
    model_cen.addConstr( ( 0.5 - x[i+15][10] ) - big_M * z_si_2_t_pred[i][11]  <= rho )

#Objective
# obj=  0
# for i in range(horizon):
#     r = x[i].reshape(-1,1) - x_desired.reshape(-1,1)
#     obj = obj + np.dot( r.reshape(1,-1) , r ) 

model_cen.setObjective( rho , GRB.MAXIMIZE)
model_cen.optimize()
print(rho.X)



####################################################################################################
################################### Computing the bounds ###########################################
####################################################################################################



# AGC
k = 30
horizon = 30




model = Model()

cbar = [ [ np.array([model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(4)]) for step in range(horizon)] for l in range(3)]
model.update()


for i in range(3):
    sub_sys[i].rci()
    sub_sys[i].X_i = [ pp.zonotope( x= cbar[i][step] , G = sub_sys[i].omega.G ) for step in range(horizon) ]


T = [ [ np.array([ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(k)] for j in range(4)]) for step in range(horizon)] for l in range(3)]

M = [ [ np.array([ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(k)] for j in range(2)]) for step in range(horizon-1)] for l in range(3)]

xbar = [ [ np.array([model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(4)]) for step in range(horizon)] for l in range(3)]

ubar = [ [ np.array([model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(2)]) for step in range(horizon-1)] for l in range(3)]

model.update()

alpha= [0,0,0]

for l in range(3):
    alpha[l] = [ pp.zonotope_subset( model , pp.zonotope( x= xbar[l][step] , G= T[l][step] ) , sub_sys[l].X_i[step] , solver='gurobi' , alpha= 'vecotor')[2] for step in range(horizon) ]
    model.update()




dist= [0,0,0]

A_ij = sub_sys[0].A_ij[1]



dist[0] = [ pp.zonotope( x = np.dot( A_ij ,  cbar[1][step].reshape(-1,1) ).reshape(-1) , G =  np.dot( A_ij , np.dot( sub_sys[1].omega.G , np.diag( alpha[1][step]) ))  ) \
    + pp.zonotope( x = np.dot( A_ij , cbar[2][step].reshape(-1,1) ).reshape(-1) , G=  np.dot( A_ij ,  np.dot( sub_sys[2].omega.G , np.diag(alpha[2][step])  ) ) ) \
        +  sub_sys[0].W   for step in range(horizon-1)] 

dist[1] = [ pp.zonotope( x = np.dot( A_ij , cbar[0][step].reshape(-1,1) ).reshape(-1) , G = np.dot( A_ij , np.dot( sub_sys[0].omega.G , np.diag( alpha[0][step] ) ) ) ) \
    + pp.zonotope( x = np.dot( A_ij , cbar[2][step].reshape(-1,1) ).reshape(-1) , G= np.dot( A_ij , np.dot( sub_sys[2].omega.G , np.diag(alpha[2][step]) ) ) ) \
        + sub_sys[1].W   for step in range(horizon-1)] 

dist[2] = [ pp.zonotope( x = np.dot( A_ij , cbar[0][step].reshape(-1,1) ).reshape(-1) , G = np.dot( A_ij , np.dot( sub_sys[0].omega.G , np.diag( alpha[0][step] ) )) ) \
    + pp.zonotope( x = np.dot( A_ij , cbar[1][step].reshape(-1,1) ).reshape(-1) , G= np.dot( A_ij , np.dot( sub_sys[1].omega.G , np.diag(alpha[1][step]) ) ) ) \
        + sub_sys[2].W  for step in range(horizon-1)] 




# Constraint [AT+BM,W]=[T] or [0,T]
for l in range(3):

    left_side = [np.concatenate ( ( np.dot(sub_sys[l].A , T[l][step]) + np.dot(sub_sys[l].B,M[l][step])  , dist[l][step].G) ,axis=1) for step in range(horizon-1)]
    right_side = [np.concatenate(   ( np.zeros(dist[l][step-1].G.shape) , T[l][step])  , axis=1) for step in range(1,horizon)]

    [model.addConstrs(   ( left_side[step][i,j] == right_side[step][i,j]  for i in range( 4 ) for j in range(len(right_side[step][0]))   )  )  for step in range(horizon-1)]              
    
    model.update()
    
    # Conditions for center: A* x_bar + B* u_bar + d_bar = x_bar 
    

    center = [np.dot( sub_sys[l].A , xbar[l][step].reshape(-1,1) ) + np.dot( sub_sys[l].B , ubar[l][step].reshape(-1,1) ) + dist[l][step].x.reshape(-1,1) for step in range(horizon-1)]

    
    model.addConstrs( (center[step][i][0] == xbar[l][step+1][i] for i in range(4) for step in range(horizon-1) ))
    
    model.update()

    [ pp.zonotope_subset( model , pp.zonotope( x= xbar[l][step] , G= T[l][step] ) , sub_sys[l].X , solver='gurobi') for step in range(horizon)]

    [ pp.zonotope_subset( model , pp.zonotope( x= ubar[l][step] , G= M[l][step] ) , sub_sys[l].U , solver='gurobi') for step in range(horizon-1) ]

    model.update()

    
model.optimize()


omega= [0,0,0]


for i in range(3):

    omega[i] = [ pp.zonotope( x = [ xbar[i][step][j].X for j in range(4)] , G = [ [ T[i][step][j][kk].X for j in range(4) ] for kk in range(k)]  )  for step in range(horizon)]

                    