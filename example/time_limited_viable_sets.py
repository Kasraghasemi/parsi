import numpy as np
import matplotlib.pyplot as plt
try:
    import pypolycontain as pp
except:
    raise ModuleNotFoundError("pypolycontain package is not installed correctly")
from gurobipy import *


# %% zonotope containment constraint
"""
zon(x_bar,X) subset of zon(y_bar, alpha * Y)
x_bar and y_bar are the enters
X and Y are the matrices of generators
"model" is a gurobi optimization model

"""

def zon_subset(model,x_bar,X,y_bar,Y,Alpha = None):                                         
    
    if len(X) != len(Y) or len(x_bar)!= len(y_bar):

        raise Exception("Generators should have the same dimensions")                                   # If two zonotopes are in different dimensions, the code raises an error.
        
    Gamma = model.addVars(len(Y[0]) , len(X[0]) ,lb= -GRB.INFINITY, ub= GRB.INFINITY)
    beta = [model.addVar( lb = -GRB.INFINITY , ub = GRB.INFINITY) for i in range(len(Y[0])) ]
    Gamma_abs = model.addVars(len(Y[0]) , len(X[0]) ,lb= 0, ub= GRB.INFINITY)
    beta_abs = [model.addVar( lb = 0 , ub = GRB.INFINITY) for i in range(len(Y[0])) ]
    model.update()
    
    model.addConstrs(  X[i][j] == sum([ Y[i][k]*Gamma[k,j] for k in range(len(Y[0])) ])  for i in range(len(X)) for j in range(len(X[0]))  )
    model.addConstrs(   sum( [ Y[i][j] * beta[j]   for j in range(len(Y[0])) ] ) ==  y_bar[i][0] - x_bar[i][0]   for i in range(len(X))   )
    [model.addGenConstrAbs(Gamma_abs[i,j], Gamma[i,j] )   for i in range(len(Y[0])) for j in range(len(X[0])) ]
    [model.addGenConstrAbs(beta_abs[i] , beta[i] ) for i in range(len(Y[0]))]
    
    if Alpha == None:
        model.addConstrs(    sum( [ Gamma_abs[i,j] for j in range(len(X[0])) ]) + beta_abs[i]  <= 1   for i in range(len(Y[0])) )
        model.update()
        return model
    
    elif Alpha == 'scaler':
        Alpha = model.addVar(lb=0,ub=1)
        model.addConstrs(    sum( [ Gamma_abs[i,j] for j in range(len(X[0])) ]) + beta_abs[i]  <= Alpha   for i in range(len(Y[0])) )
    elif Alpha == 'diag':
        Alpha = [model.addVar(lb=0,ub=1) for i in range(len(Y[0]))]
        model.addConstrs(    sum( [ Gamma_abs[i,j] for j in range(len(X[0])) ]) + beta_abs[i]  <=  Alpha[i]   for i in range(len(Y[0])) )
    
    model.update()
    
    return model , Alpha


# %% Finding decentralized viable sets using a single optimizatin problem
    
def Viable_decentralized(A_partition,A_total,B,d_bar,G_d,E , x_bar = None,G_x = None,u_bar = None,G_u = None):
    

   
    sys_number = len(A_partition)
    horizon =len(A_total)
        
    A = {}
    
    for i in range(sys_number):
        for j in range(sys_number):
            A[i,j] = {}
#            B[i,j] = {}
            for time in range(horizon):   
                A[i,j][time]= A_total[time][sum(A_partition[:(i)]):sum(A_partition[:(i)])+A_partition[i],sum(A_partition[:(j)]):sum(A_partition[:(j)])+A_partition[j]]
                
#            for j in range(len(B_partition)):
#                B[time][i,j]= B_total[time][ sum(A_partition[:(i)]):A_partition[i],sum(B_partition[:(j)]):B_partition[j] ]

    
    for sys in range(sys_number):
        for step in range(horizon):
            
            G_d[sys][time] = np.dot(E[sys][time],G_d[sys][time])
    
    
    

    max_k =50
    k = 2
    n = A_partition
    m = []
    for sys in range(sys_number):
        
        m.append(len(B[sys][0][0]))

    
    p={}
    p_agregate={}
    
    for sys in range(sys_number):
        p[sys] = {}
        p_agregate[sys]=[0]
        
        Sum=0
        for i in range(horizon):
            sub_sum=0
           
            for sys2 in range(sys_number):
                if sys2 == sys:
                    continue
                
                p[sys][i]= len(G_x[sys2][i][0]) 
                sub_sum = sub_sum + p[sys][i]
                
            Sum = Sum + sub_sum + len(G_d[sys][i][0])
            p_agregate[sys].append(Sum)
                
    
    
    
    
    
    while k <= max_k:
    
            
        model = Model()
        
        xbar={}
        T={}
        ubar={}
        M={}
        alpha={}
            
        for sys in range(sys_number):        
            
            xbar[sys] = {}
            T[sys] = {}
            ubar[sys] = {}
            M[sys] = {}
            alpha[sys] = {}
            
            for step in range(horizon):
                
                T[sys][step] = [ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(k+p_agregate[sys][step])] for j in range(n[sys])] 
                xbar[sys][step] = [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(n[sys])]
                
                model,alpha[sys][step] = zon_subset (model, np.array([xbar[sys][step]]).T , T[sys][step] , x_bar[sys][step] , G_x[sys][step] ,Alpha = 'scaler')                       #state be a subset of state space
          
                if step == horizon-1:
                    break
                M[sys][step] = [ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(k+p_agregate[sys][step])] for j in range(m[sys])]
                ubar[sys][step] = [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(m[sys])]
                
                model = zon_subset (model, np.array([ubar[sys][step]]).T , M[sys][step] , u_bar[sys][step] , G_u[sys][step])                        #controller be a subset of control space
                
                
            model.update()
            
        G_new = {}
        d_bar_new = {}    
        
        for sys in range(sys_number):
            G_new[sys] = {}
            d_bar_new[sys] = {}
            
            for time in range(horizon):
                G_new[sys][time] = G_d[sys][time]
                d_bar_new[sys][time] = d_bar[sys][time]
                
                for j in range(sys_number):
                    if j == sys:
                        continue
                    G_new[sys][time] = np.concatenate(  (G_new[sys][time] , np.dot(A[sys,j][time] , np.array(alpha[sys][time]) * G_x[j][time] ) ), axis = 1 )
                    d_bar_new[sys][time] = np.dot(E[sys][time],d_bar_new[sys][time]) + np.dot(A[sys,j][time] , x_bar[j][time] )
         
            
      
        
        for sys in range(sys_number):
            
            for step in range(horizon-1):
                      
                        
                left_side = np.concatenate (( np.dot(A[sys,sys][step],T[sys][step]) + np.dot(B[sys][step],M[sys][step])  , G_new[sys][step]) ,axis=1)
                right_side =  np.array(T[sys][step+1])
                
                model.addConstrs(   ( left_side[i,j] == right_side[i,j]  for i in range(n[sys]) for j in range(p_agregate[sys][step+1]+k))     )    
                        
                # Conditions for center: A* x_bar + B* u_bar + E* d_bar = x_bar^+ 
                center = np.dot(A[sys,sys][step] ,np.array([xbar[sys][step]]).T ) + np.dot(B[sys][step] ,np.array([ubar[sys][step]]).T ) + d_bar_new[sys][step]
                model.addConstrs( center[i][0] == xbar[sys][step+1][i] for i in range(n[sys]))
                model.update()    
                        
    
    
    
        model.setObjective( sum([sum(alpha[i]) for i in range(sys_number)]) , GRB.MINIMIZE )
        model.update()
        model.setParam("OutputFlag",False)
        model.optimize()
                
        if model.Status==3:
            k=k+1
            del model
                    
            if k == max_k+1:
                return "Infeasible"
                    
            continue
                
        Genx = {}
        x_center = {}
        scaler = {}
        for sys in range(len(A_partition)):
            Genx[sys] = {}
            x_center[sys] = {}
            scaler[sys] = []
            for step in range(horizon):
                scaler[sys].append(alpha[sys][step].X)
                Genx[sys][step] = np.zeros((n[sys],k+p_agregate[sys][step]))
                x_center[sys][step] = np.zeros(n[sys])
                for i in range(n[sys]):
                    x_center[sys][step][i] = xbar[sys][step][i].X
                    for j in range(k+p_agregate[sys][step]):
                        Genx[sys][step][i][j] = T[sys][step][i][j].X
                    
        Genu = {}
        u_center = {}
        for sys in range(len(A_partition)):
            Genu[sys] = {}
            u_center[sys] = {}
            for step in range(horizon-1):
                Genu[sys][step] = np.zeros((m[sys],k+p_agregate[sys][step]))
                u_center[sys][step] = np.zeros(m[sys])
                for i in range(m[sys]):
                        
                    u_center[sys][step][i] = ubar[sys][step][i].X
                    for j in range(k+p_agregate[sys][step]):
                        Genu[sys][step][i][j] = M[sys][step][i][j].X
                    
                
        return x_center,Genx,u_center,Genu,scaler
            
                      
                
# %% Testing Viable_decentralized


# from pypolycontain.lib.zonotope import zonotope,zonotope_directed_distance
# from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes_ax as visZ
# from zonotope_order_reduction import PCA, Boxing

delta = 1
A_partition = [ 2 , 2 , 2 ,2 ]
sys_number = len(A_partition)
A_total = {}
B = {}
E={}
x_bar = {}
G_x = {}
u_bar = {}
G_u = {}
d_bar = {}
G_d = {}
horizon = 15
for time in range(horizon):
    A_total[time]  = np.array([
        [ delta * 0.01 * time , delta , 0.001 , 0.001 , 0 , 0 , -0.01* np.sin(time) , 0.01* time] ,
        [ 0 , delta , 0.001 , 0.001 ,  0 , 0 , 0.01* np.log(time+1) , 0.01* np.cos(time)] ,
        [ 0.001 , 0.001 , delta , delta , 0.001 , 0.001 , 0 , 0] , 
        [ 0.001 , 0.001 , 0 , delta , 0.001 , 0.001 , 0 , -0.01] , 
        [ 0 , 0 , 0.001 , 0.001 , delta , delta , 0 , 0.01] ,
        [ 0.001* time**2 , 0 , 0.001 , 0.001 , 0 , delta , 0 , 0] , 
        [ 0.01* time , 0 , 0 , 0 , 0 , 0 , delta , delta ] , 
        [ -0.01* time , 0.01 , 0 , 0.01 , 0 , 0 , 0 , delta* 0.01* time ]
        ])                

for sys in range(sys_number):
    B[sys] = {}
    E[sys] = {}
    x_bar[sys] = {}
    G_x[sys] = {}
    u_bar[sys] = {}
    G_u[sys] = {}
    d_bar[sys] = {}
    G_d[sys] = {}
    for time in range(horizon):
        B[sys][time] = np.array([[0],[ 1 ]]) * delta
        
        E[sys][time] = np.array([[1,0],[0,1]])
        x_bar[sys][time] = np.array([[0],[0]])
        #G_x[sys][time] = np.array([[3 + np.sin(time) , 1+ np.cos(time)],[1-  np.sin(time),5 + 3* np.cos(time)]])
        #G_x[sys][time] = np.array([[5 - time/5 , 0],[0,5 - 1.5* time/5]])
        if sys ==1:
            G_x[sys][time] = np.array([[5 - 2 * np.sin(np.pi/8 * time ) , 0],[0,6 - 5.5 * np.sin(np.pi/20 * time )]])
        elif sys == 0:
            G_x[sys][time] = np.array([[5 -  np.sin(np.pi/15 * time ) , 0],[0,6 - 5.5 * np.sin(np.pi/12 * time )]])
        elif sys == 2:
            G_x[sys][time] = np.array([[5 -  np.cos(np.pi/15 * time ) , 0],[0,6 - 5.5 * np.cos(np.pi/12 * time )]])
        else:
            G_x[sys][time] = np.array([[5 - time/5 , 0],[0, 5 - time/5 ]])
        u_bar[sys][time] = np.array([[0]])
        G_u[sys][time] = np.array([[10]])
        d_bar[sys][time] = np.array([[0],[0]])
        G_d[sys][time] = np.array([[0.4 ,0],[0,0.4]])
x_center,Genx,u_center,Genu,scaler = Viable_decentralized(A_partition,A_total,B,d_bar,G_d,E, x_bar,G_x ,u_bar,G_u )


zon={}


fig, ax = plt.subplots(sys_number)
for i in range(sys_number):
    zon[i] = {}
    for step in range(1,horizon):
        if i ==0:
            color = "green"
        elif i==1:
            color = "purple"
        elif i==2:
            color = "orange"
        else:
            color = "blue"
        zon[i][step] = pp.zonotope( x= step * np.array([[15],[0]])+np.array([x_center[i][step]]).T , G = Genx[i][step], color=color)
    

    pp.visualize([pp.zonotope( x= x_bar[i][step]+step * np.array([[15],[0]]), G = G_x[i][step],color="gray") for step in range(1,horizon)] + [zon[i][step] for step in range(1,horizon)] ,ax = ax[i],fig=fig)
    ax[i].set_title("Subsystem %d"%(i+1),FontSize=8)
    ax[i].set_xlabel(r'$x_ %d$'%(2*i+1),FontSize=8)
    ax[i].set_ylabel(r'$x_ %d$'%(2*i+2),FontSize=8 , rotation=0)
    ax[i].set_xticklabels([])
plt.tight_layout()
plt.show()
    # pp.visualize(ax,[pp.zonotope( x= x_bar[i][step]+step * np.array([[15],[0]]), G = G_x[i][step],color="gray") for step in range(1,horizon)]\
        # +[zon[i][step] for step in range(1,horizon)] , title="Sub-system %d"%(i+1))
    # ax.set_xlabel(r"$x$",FontSize=20)
    # ax.set_ylabel(r"$y$",FontSize=20)
    # ax.set_title(r"Subsystem %d"%(i+1),FontSize=20)
    # fig.savefig('Centralized_LTV_sub-system_%d.png' %(i+1), format='png', dpi=1000)  
    
#visZ([zon[0][step] for step in range(1,horizon)])
# print(x_center,Genx,u_center,Genu,scaler,zon)
# x_center,Genx,u_center,Genu,scaler,zon