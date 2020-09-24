"""
@author: kasra
"""
import numpy as np
try:
    from gurobipy import *
except:
    print('gurobi is not installed correctly') 
try:
    import pypolycontain as pp
except:
    print('Error: pypolycontain package is not installed correctly') 

try:
    import parsi
except:
    print('Error: parsi package is not installed correctly') 

def rci_constraints(program,system,T_order=3):
    """
    This function adds necessary constraints for RCI (robust control invariant) set for a LTI.
    """
    assert(system.sys=='LTI'), "The system has to be LTI (linear time invariant)"
    n = system.A.shape[0]               # Matrix A is n*n
    m = system.B.shape[1]               # Matrix B is n*m
    k = round(n*T_order)
    p = system.W.G.shape[1]
    
    E=system.E
    flag=0
    if E==True:
        e=program.NewContinuousVariables( n, p, 'E')                #E is n*p
        if (system.X.x==0).all() and (system.U.x==0).all() and (system.W.x==0).all():
            flag=1
    elif E==False:
        e=np.zeros((n,p))
        system.beta=0
    else:
        raise ValueError('E in the argument needs to be either False or True')      
    beta = system.beta

    #Defining Variables
    T=program.NewContinuousVariables( n, k, 'T')                #T is n*k
    M=program.NewContinuousVariables( m, k, 'M')                #M is m*k
    T_x=program.NewContinuousVariables( n,'T_center')               #T_x is the x_bar in the paper
    M_x=program.NewContinuousVariables( m,'M_center')               #M_x is the u_bar in the paper

    #Defining Constraints
    left_side = np.concatenate( (np.dot(system.A,T)+np.dot(system.B,M) , system.W.G) ,axis=1)               #[AT+BM,W]
    right_side = np.concatenate( (e,T),axis=1)              #[E,T]
    program.AddLinearConstraint(np.equal(left_side,right_side,dtype='object').flatten())             #[AT+BM,W]==[E,T]

    if flag==1:
        _,_,beta = pp.zonotope_subset(program, pp.zonotope(G=e,x=np.zeros(e.shape[0])) , pp.zonotope(G=system.W.G,x=np.zeros(system.W.G.shape[0])) , alpha='scalar')             #Z(0,E) \subset Z(0,beta*W)
        program.AddBoundingBoxConstraint(0,0.999,beta)              #CHECK IT, I NEED 0<=beta<1
        # program.AddLinearConstraint(beta < 1)
        # program.AddLinearConstraint(beta >= 0)
        program.AddLinearConstraint(np.equal(T_x, np.zeros(T.shape[0]),dtype='object').flatten())
        program.AddLinearConstraint(np.equal(M_x, np.zeros(M.shape[0]),dtype='object').flatten())
        if system.X!=None:
            _,_,beta1=pp.zonotope_subset(program, pp.zonotope(G=T,x=np.zeros(T.shape[0])) , system.X , alpha='scalar' )
            program.AddLinearConstraint(np.equal(beta1,(1-beta),dtype='object'))
        if system.U!=None:
            _,_,beta2=pp.zonotope_subset(program, pp.zonotope(G=M,x=np.zeros(M.shape[0])) , system.U , alpha='scalar' )
            program.AddLinearConstraint(np.equal(beta2,(1-beta),dtype='object'))
    elif flag==0:
        if E==True:
            pp.zonotope_subset(program, pp.zonotope(G=e,x=np.zeros(e.shape[0])) , pp.zonotope(G=beta *system.W.G,x=np.zeros(system.W.G.shape[0])) )
        if system.X!=None:
            pp.zonotope_subset(program, pp.zonotope(G=T/(1-beta),x=T_x) , system.X )
        if system.U!=None:
            pp.zonotope_subset(program, pp.zonotope(G=M/(1-beta),x=M_x) , system.U )
        center=np.equal( np.dot(system.A , T_x) + np.dot(system.B , M_x) + system.W.x , T_x ,dtype='object').flatten()
        program.AddLinearConstraint(center)               #A*x_bar+ B*u_bar +w_bar==x_bar       #IT WILL MAKE PROBLEM WHEN IT BECOMES TRUE!
    
    output={
        'T':T,
        'T_x': T_x,
        'M':M,
        'M_x': M_x
    }
    if flag==1:
        output['beta']=beta
    
    return output


def viable_constraints(program,system,T_order=3,algorithm='slow'):
    """
    This function is useful in finding time-limited viable sets
    """
    assert(system.sys=='LTV'), "The system has to be LTV (linear time variant)"
    from itertools import accumulate
    number_of_steps=len(system.A)
    n = system.A[0].shape[0]               # Matrix A is n*n
    m = system.B[0].shape[1]               # Matrix B is n*m
    k = round(n*T_order)
    dist_G_numberofcolumns = [system.W[i].G.shape[1] for i in range(number_of_steps)]
    if algorithm=='slow':
        dist_G_numberofcolumns.insert(0,k)
        p = list(accumulate(dist_G_numberofcolumns))
    
    #Defining Variables
    T=[program.NewContinuousVariables( n, p[i], 'T') for i in range(number_of_steps)] if algorithm=='slow' \
        else [program.NewContinuousVariables( n, k, 'T') for i in range(number_of_steps)]               #T is n*p if algorithm is defined as slow, otherwise n*k
    M=[program.NewContinuousVariables( m, p[i], 'M') for i in range(number_of_steps-1)] if algorithm=='slow' \
        else [program.NewContinuousVariables( m, k, 'T') for i in range(number_of_steps)]                #M is m*k
    T_x=[program.NewContinuousVariables( n,'T_center') for i in range(number_of_steps)]               #T_x is the x_bar in the paper
    M_x=[program.NewContinuousVariables( m,'M_center') for i in range(number_of_steps-1)]               #M_x is the u_bar in the paper

    #Defining Constraints
    left_side = [np.concatenate( (np.dot(system.A[i],T[i])+np.dot(system.B[i],M[i]) , system.W[i].G) ,axis=1) for i in range(number_of_steps-1)]               #[AT+BM,W]
    right_side = T[1:] if algorithm=='slow' else [np.concatenate( ( np.zeros((n,dist_G_numberofcolumns[i-1])) ,T[i] ),axis=1) for i in range(1,number_of_steps)]             #[T] if algorithm=='slow' , [0,T] if algorithm=='fast'
    [program.AddLinearConstraint(np.equal(left_side[i],right_side[i],dtype='object').flatten()) for i in range(number_of_steps-1)]             #[A(t)T(t)+B(t)M(t),W(t)]==[T(t+1)]

    #Implementing Hard Constraints over control input and state space
    if system.X!=None:
        [pp.zonotope_subset(program, pp.zonotope(G=T[i],x=T_x[i]) , system.X[i] ) for i in range(number_of_steps)]
    if system.U!=None:
        [pp.zonotope_subset(program, pp.zonotope(G=M[i],x=M_x[i]) , system.U[i] ) for i in range(number_of_steps-1)]
    
    #Constraints for the centers
    center=[np.equal( np.dot(system.A[i] , T_x[i]) + np.dot(system.B[i] , M_x[i]) + system.W[i].x , T_x[i+1] ,dtype='object').flatten() for i in range(number_of_steps-1)]
    [program.AddLinearConstraint(center[i]) for i in range(number_of_steps-1)]              #A*x_bar+ B*u_bar +w_bar==x_bar       #IT WILL MAKE PROBLEM WHEN IT BECOMES TRUE!

    output={
        'T':T,
        'T_x': T_x,
        'M':M,
        'M_x': M_x
    }    

    return output


def mpc_constraints(program,system,horizon=1,hard_constraints=True):

    if system.sys=='LTI':
        n = system.A.shape[0]               # Matrix A is n*n
        m = system.B.shape[1]               # Matrix B is n*m

        x=program.NewContinuousVariables( n,horizon,'x')
        x=np.concatenate((system.state.reshape(-1,1),x ), axis=1)
        u=program.NewContinuousVariables( m,horizon,'u')

        #Dynamics
        dynamics=np.equal( np.dot(system.A , x[:,:-1]) + np.dot(system.B , u) , x[:,1:] ,dtype='object').flatten()
        program.AddLinearConstraint(dynamics)

        #Hard constraints over state
        if hard_constraints==True:
            if system.X!=None:
                _=[pp.be_in_set(program,x[:,i],system.X) for i in range(1,horizon)]
            #Hard constraints over control input
            if system.U!=None:
                _=[pp.be_in_set(program,u[:,i],system.U) for i in range(horizon)]

        return x,u


def rci_decentralized_constraints_gurobi(model,list_system,T_order=3,initial_guess='nominal'):
    """
    """

    sys_number = len(list_system)
    n=[list_system[i].A.shape[0] for i in range(sys_number)]
    m=[list_system[i].B.shape[1] for i in range(sys_number)]
    k = [round(n[i]*T_order) for i in range(sys_number)] 

    # Initilizing the paramterize set
    if any([sys.omega==None for sys in list_system]) and any([sys.theta==None for sys in list_system]) :
        X_i,U_i = parsi.rci_decentralized_initialization(list_system,initial_guess='nominal',order_max=10,size='min',obj='include_center')
    else:
        X_i=[sys.omega for sys in list_system]
        U_i=[sys.theta for sys in list_system]

    # Adding Variables

    T= [ np.array([ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(k[sys])] for j in range(n[sys])] ) for sys in range(sys_number)]
    M= [ np.array([ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(k[sys])] for j in range(m[sys])] ) for sys in range(sys_number)]

    xbar=[ np.array([model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(n[sys])]) for sys in range(sys_number)]
    ubar=[ np.array([model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(m[sys])]) for sys in range(sys_number)]
    
    for sys in range(sys_number):
        # Adding hard constraints over state and control input spaces
        if list_system[sys].U != None:
            pp.zonotope_subset(model, pp.zonotope(x=np.array(ubar[sys]).T,G= M[sys]) , list_system[sys].U ,solver='gurobi')
        if list_system[sys].X != None:
            pp.zonotope_subset(model, pp.zonotope(x=np.array(xbar[sys]).T,G=T[sys]) , list_system[sys].X ,solver='gurobi' )

    # Adding Correctness constraints
    alpha_u=[ pp.zonotope_subset(model, pp.zonotope(x=np.array(ubar[sys]).T,G= M[sys]) , U_i[sys] ,alpha='vector',solver='gurobi')[2] for sys in range(sys_number)]
    [model.addConstrs((alpha_u[sys][i] >= 0 for i in range(len(alpha_u[sys]))) ) for sys in range(sys_number)]
    alpha_x=[ pp.zonotope_subset(model, pp.zonotope(x=np.array(xbar[sys]).T,G=T[sys]) , X_i[sys] ,alpha='vector',solver='gurobi' )[2] for sys in range(sys_number)]
    [model.addConstrs((alpha_x[sys][i] >= 0 for i in range(len(alpha_x[sys]))) ) for sys in range(sys_number)]

    model.update()

    # Computing the disturbance set
    disturb=[]
    for i in range(sys_number):
        disturb.append(list_system[i].W)
    for i in range(sys_number):
        for j in range(sys_number):
            if j in list_system[i].A_ij:
                w=pp.zonotope(G= np.dot(np.dot( list_system[i].A_ij[j], X_i[j].G), np.diag(alpha_x[j]) ) , x= np.dot( list_system[i].A_ij[j], X_i[j].x))
                disturb[i] = disturb[i]+ w
            if j in list_system[i].B_ij:
                w=pp.zonotope(G= np.dot(np.dot( list_system[i].B_ij[j], U_i[j].G), np.diag(alpha_u[j]) ) , x= np.dot( list_system[i].B_ij[j], U_i[j].x))
                disturb[i] = disturb[i]+ w

    # Adding main rci constraints: [AT+BM,W]=[0,T] and A x_bar + B u_bar + d_bar = x_bar
    for sys in range(sys_number):

        left_side = np.concatenate (( np.dot(list_system[sys].A,T[sys]) + np.dot(list_system[sys].B,M[sys])  , disturb[sys].G) ,axis=1)
        right_side = np.concatenate(   ( np.zeros(( list_system[sys].A.shape[0],disturb[sys].G.shape[1] )), T[sys])  , axis=1)
        model.addConstrs(   ( left_side[i,j] == right_side[i,j]  for i in range(n[sys]) for j in range(disturb[sys].G.shape[1]+k[sys]))     )  
        
        center = np.dot(list_system[sys].A ,np.array([xbar[sys]]).T ) + np.dot(list_system[sys].B ,np.array([ubar[sys]]).T ) + disturb[sys].x
        model.addConstrs( center[i][0] == xbar[sys][i] for i in range(n[sys]))
    model.update()
    
    return xbar,T,ubar,M, alpha_x,alpha_u


def hausdorff_distance_condition(model,zon1,zon2,alpha):
    """
    Inputs: two zonotope
    Outputs: d
    """
    d=model.addVar(lb= 0, ub= GRB.INFINITY)             # The radius in the directed hausdorff distance
    circumbody=pp.zonotope(x=zon2.x , G= np.concatenate( ( zon2.G,np.eye(zon2.G.shape[0])) , axis=1) )                # circumbody = zon2 \oplus zonotope(x=0, G= d* eye)
    _,_,scale=pp.zonotope_subset(model,zon1,circumbody,alpha='vector',solver='gurobi')

    alpha_i_constraints = [model.addConstr(scale[i] == alpha[i])  for i in range(zon2.G.shape[1])]

    model.addConstrs(scale[i] == d  for i in range(zon2.G.shape[1],circumbody.G.shape[1] ))
    
    # model.setObjective( d , GRB.MINIMIZE )              # minimizing d to find the smallest raduis
    model.update()
    
    return d,alpha_i_constraints


# the potential function for just a single system
def potential_function(list_system, system_index, T_order=3, reduced_order=1):
    """
    Inputs: 
            (1)the list of coupled linear systems: Note that alpha_x,alpha_u must have a value
            (2)the index of the system which you want to compute
            (3)the order of the candidate rci set
            (4)the order of the reduced disturbance set
    Outputs:
            (1)the directed hausdorf distance between rci set and the admissible state space 
            (2)the directed hausdorf distance between action set and the admissible control input
            (3)the gradients of (1) with respect to alpha_x
            (4)the gradients of (2) with respect to alpha_u
    """
    from itertools import accumulate
    sys_number = len(list_system)
    n=[list_system[i].A.shape[0] for i in range(sys_number)]
    m=[list_system[i].B.shape[1] for i in range(sys_number)]
    k = round(n[system_index]*T_order) 

    # Initilizing the paramterize set
    if any([sys.omega==None for sys in list_system]) and any([sys.theta==None for sys in list_system]) :
        X_i,U_i = parsi.rci_decentralized_initialization(list_system,initial_guess='nominal',order_max=10,size='min',obj='include_center')
    else:
        X_i=[sys.omega for sys in list_system]
        U_i=[sys.theta for sys in list_system]

    # Computing the disturbance set
    disturb=list_system[system_index].W
    for j in range(sys_number):
        if j in list_system[system_index].A_ij:
            w=pp.zonotope(G= np.dot(np.dot( list_system[system_index].A_ij[j], X_i[j].G), np.diag( list_system[j].alpha_x) ) , x= np.dot( list_system[system_index].A_ij[j], X_i[j].x))
            disturb = disturb+w
        if j in list_system[system_index].B_ij:
            w=pp.zonotope(G= np.dot(np.dot( list_system[system_index].B_ij[j], U_i[j].G), np.diag(list_system[j].alpha_u) ) , x= np.dot( list_system[system_index].B_ij[j], U_i[j].x))
            disturb= disturb+w
    
    # Using Zonotope order reduction
    disturb= pp.boxing_order_reduction(disturb,desired_order=reduced_order)      ##################################################################################

    # Creating the model
    model = Model()

    # Definging the new disturbance as a variable
    W_x = np.array([ model.addVar(lb = -GRB.INFINITY) for i in range(n[system_index]) ])
    W_G = np.array([[model.addVar(lb = -GRB.INFINITY) for i in range( disturb.G.shape[1] )] for j in range(n[system_index])])
    model.update()
    
    # Adding the constraint for disturbance: W_aug == disturb               ###################################################################################
    [model.addConstr(W_x[i] == disturb.x[i]) for i in range(n[system_index])]
    for i in range(n[system_index]):
        for j in range(disturb.G.shape[1]):
            if i!=j:
                model.addConstr(W_G[i][j] == disturb.G[i][j])
    alpha_j_constraints=[model.addConstr(W_G[i][i] == disturb.G[i][i]) for i in range(n[system_index]) ]                # JUST FOR reduced_order=1
    #[model.addConstr(W_G[i][j] == disturb.G[i][j] ) for i in range(n[system_index]) for j in range(disturb.G.shape[1]) ]
    model.update()
    
    # Defining the rci set (zonotope(x=xbar,G=T)) and action set (zonotope(x=ubar,G=M))
    T = [ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(k)] for j in range(n[system_index])] 
    M = [ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(k)] for j in range(m[system_index])]
    xbar = [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(n[system_index])]
    ubar = [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(m[system_index])]
    model.update()
    
    # Constraint [AT+BM,W]=[0,T]
    left_side = np.concatenate (( np.dot(list_system[system_index].A,T) + np.dot(list_system[system_index].B,M)  , W_G) ,axis=1)
    right_side = np.concatenate(   ( np.zeros(W_G.shape) , T)  , axis=1)

    model.addConstrs(   ( left_side[i,j] == right_side[i,j]  for i in range(n[system_index]) for j in range(W_G.shape[1]+k))     )                
    
    # Conditions for center: A* x_bar + B* u_bar + d_bar = x_bar 
    center = np.dot(list_system[system_index].A ,np.array(xbar) ) + np.dot(list_system[system_index].B ,np.array(ubar) ) + W_x
    model.addConstrs( (center[i] == xbar[i] for i in range(n[system_index])))
    
    model.update()

    # Setting 
    d_x,alpha_i_x_constraints=parsi.hausdorff_distance_condition(model, pp.zonotope(x=xbar,G=T) , X_i[system_index] ,list_system[system_index].alpha_x)
    d_u,alpha_i_u_constraints=parsi.hausdorff_distance_condition(model, pp.zonotope(x=ubar,G=M) , U_i[system_index] ,list_system[system_index].alpha_u)
    
    model.setObjective( d_x+d_u , GRB.MINIMIZE )
    model.update()

    model.setParam("OutputFlag",False)
    model.optimize()

    
    length_x_alpha=[i.alpha_x.shape[0] for i in list_system]
    length_x_alpha.insert(0,0)
    accum_length_x_alpha=list(accumulate(length_x_alpha))
    grad_x=np.zeros(accum_length_x_alpha[-1])             # you need to intialize alpha for all sub-systems before running this code!
    
    length_u_alpha=[i.alpha_u.shape[0] for i in list_system]
    length_u_alpha.insert(0,0)
    accum_length_u_alpha=list(accumulate(length_u_alpha))
    grad_u=np.zeros(accum_length_u_alpha[-1])             # you need to intialize alpha for all sub-systems before running this code!

    for j in range(sys_number):
        if j==system_index:
            grad_x[accum_length_x_alpha[j]:accum_length_x_alpha[j+1]]=np.array([alpha_i_x_constraints[i].pi for i in range(len(alpha_i_x_constraints))])
        elif not j in list_system[system_index].A_ij:
            continue
        else:
            grad_x[accum_length_x_alpha[j]:accum_length_x_alpha[j+1]]= \
                            np.array([sum([alpha_j_constraints[row].pi * (abs(np.dot( list_system[system_index].A_ij[j][row,:], X_i[j].G[:,i]))) \
                            for row in range(disturb.G.shape[0])]) \
                            for i in range(X_i[j].G.shape[1])])

    for j in range(sys_number):
        if j==system_index:
            grad_u[accum_length_u_alpha[j]:accum_length_u_alpha[j+1]]=np.array([alpha_i_u_constraints[i].pi for i in range(len(alpha_i_u_constraints))])
        elif not j in list_system[system_index].B_ij:
            continue
        else:
            grad_u[accum_length_u_alpha[j]:accum_length_u_alpha[j+1]]= \
                            np.array([sum([alpha_j_constraints[row].pi * (abs(np.dot( list_system[system_index].B_ij[j][row,:], U_i[j].G[:,i]))) \
                            for row in range(disturb.G.shape[0])]) \
                            for i in range(U_i[j].G.shape[1])])

    potential_output={
        'obj': model.objVal ,
        'obj_x': d_x.X,
        'obj_u':d_u.X,
        'alpha_x_grad':grad_x,
        'alpha_u_grad':grad_u,
        'grad':np.concatenate((grad_x,grad_u))
    }

    return potential_output
