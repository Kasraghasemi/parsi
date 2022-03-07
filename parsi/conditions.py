"""
@author: kasra
"""
from matplotlib.pyplot import step
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
from copy import deepcopy


def rci_constraints(model, system, T_order, general_version=True , include_hard_connstraints= True):
    """
    This function adds necessary constraints for RCI (robust control invariant) set for a LTI.
    Inputs: 
        program; which must be a Gurobi model
        system; must be a time invariant linear system
        T_order; is the order of T matrix
        general_version; True -> Considers beta as a variable in the algorithm
                         False -> Does NOT consider beta. More restricted version of the algorithm
    Outputs:
        a dictionaty containing:
            x_bar; Center of the rci
            T; Generator of the rci 
            u_bar; Center of the action set
            M; Generator of the action set
            beta; None -> if the general_version in argument is set to False
                  a positive gurobi variable between zero and one if the general_version in argument is set to True
    """
    
    assert(system.sys=='LTI'), "The system has to be LTI (linear time invariant)"
    n = system.A.shape[0]               # Matrix A is n*n
    m = system.B.shape[1]               # Matrix B is n*m
    k = int(round(n*T_order))
    p = system.W.G.shape[1]

    # Defining Variables
    T = np.array( [ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for ـ in range(k)] for ـ in range(n)] )
    x= np.array( [ model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for ـ in range(n)] )              

    M = np.array( [ [ model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for ـ in range(k)] for ـ in range(m)] )
    u= np.array( [ model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for ـ in range(m)] )       
    model.update()

    # Defining RCI Constraint 

    #   [AT+BM,W] == [E,T] (when general_version is False it is [AT+BM,W] == [0,T])
    left_side_rci_constraint = np.hstack(( np.dot(system.A,T) + np.dot(system.B,M) , system.W.G ))

    # Adding the constraint Z(0,E) \subseteq Z(0,beta * W.G ) if general_version is True
    if general_version == True:
        E = np.array( [ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for _ in range(left_side_rci_constraint.shape[1]-k )] for _ in range(n)] )
        model.update()

        _ , _ , beta = pp.zonotope_subset(model, pp.zonotope( x = np.zeros(n) , G = E ) , pp.zonotope( x = np.zeros(n) ,G = system.W.G ) ,alpha='scalar' ,solver='gurobi')
        beta.UB=0.999999
        beta.LB=0
        model.update()

    elif general_version == False:
        E = np.zeros((n,n))
        beta = None
    
    right_side_rci_constraint = np.hstack((  E, T ))

    model.addConstrs( (left_side_rci_constraint[i,j]==right_side_rci_constraint[i,j] for i in range(n) for j in range(n+k)) )
    model.update()

    # Ax+Bu+d=x
    model.addConstrs( ( (np.dot(system.A , x) + np.dot(system.B , u) + system.W.x)[i] ==x[i] for i in range(n) ) )
    model.update()

    # Adding the hard constraints over state and control inputs
    if include_hard_connstraints:
        if system.X is not None:
            if general_version == True:
                _ , _ , coeff_x = pp.zonotope_subset(model, pp.zonotope(G=T,x=x) , system.X ,alpha='scalar' ,solver='gurobi')
                model.update()
                model.addConstr( (1-beta) == coeff_x )
            elif general_version == False:
                pp.zonotope_subset(model, pp.zonotope(G=T,x=x) , system.X ,solver='gurobi')
        model.update()

        if system.U is not None:
            if general_version == True:
                _ , _ , coeff_u = pp.zonotope_subset(model, pp.zonotope(G=M,x=u) , system.U ,alpha='scalar' ,solver='gurobi')
                model.update()
                model.addConstr( (1-beta) == coeff_u )
                
            elif general_version == False:
                pp.zonotope_subset(model, pp.zonotope(G=M,x=u) , system.U ,solver='gurobi')
        model.update()
        
    output={
        'T':T,
        'x_bar': x,
        'M':M,
        'u_bar': u,
        'beta': beta
    }
    
    return output


def viable_constraints(model, system, T_order, horizon=None, algorithm='slow', initial_state=True):
    """
    It adds time limited viable set constraints.
    Inputs:
        model; which must be a gurobi model
        system; can be either a linear LTI ot LTV system
        horizon; the number steps to be considered. If it is None, it automattically sets it to the number of time steps in the LTV system.
        T_order; which is scalar and is the first viable set order
        algorithm; slow -> if we want the order of the viable sets grow by number of steps
                   fast -> if the order of viable sets remain fixed at T_order
    Outputs: a dictionary denoted as ourput
        output['T']; is a list where T[step][i = 0 , 1, ..., n][0, ... , k] , where step is one more
        output['x_bar']; is a list where x_bar[step][i = 0 , 1, ..., n] , where step is one more
        output['M']; is a list where M[step][i = 0 , 1, ..., m][0, ... , k] 
        output['u_bar']; is a list where u_bar[step][i = 0 , 1, ..., m]

    """

    from itertools import accumulate

    if system.sys == 'LTI':

        number_of_steps= horizon if horizon is not None else 'inf'
        
        if number_of_steps == 'inf':
            return rci_constraints(model, system, T_order, general_version=True)
        
        else:
            n = system.A.shape[0]
            m = system.B.shape[1]
            W = [system.W for i in range(number_of_steps)]
            A = [system.A for i in range(number_of_steps)]
            B = [system.B for i in range(number_of_steps)]
            X = [system.X for i in range(number_of_steps+1)]        # TODO: we can change it to number_of_steps and append it with a goal set
            U = [system.U for i in range(number_of_steps)]

    elif system.sys == 'LTV':

        number_of_steps= horizon if horizon is not None else len(system.A)
        n = system.A[0].shape[0]               # Matrix A is n*n
        m = system.B[0].shape[1]               # Matrix B is n*m    
        W = [system.W[i] for i in range(number_of_steps)]
        A = [system.A[i] for i in range(number_of_steps)]
        B = [system.B[i] for i in range(number_of_steps)]
        X = [system.X[i] for i in range(number_of_steps+1)]         # TODO: we can change it to number_of_steps and append it with a goal set
        U = [system.U[i] for i in range(number_of_steps)]
    
    k = int( round(n*T_order) )
    
    dist_G_numberofcolumns = [ W[i].G.shape[1] for i in range(number_of_steps) ]

    if algorithm =='slow':
        dist_G_numberofcolumns.insert(0,k)
        p = list(accumulate(dist_G_numberofcolumns))
    else: 
        p = [k]* (number_of_steps+1)
    
    # Defining Variables

    T = [np.array( [ [ model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for ـ in range(p[step]) ] for ـ in range(n)] ) for step in range(number_of_steps+1)]
    x_bar = [np.array( [ model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for ـ in range(n)] ) for step in range(number_of_steps+1)]

    M = [np.array( [ [ model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for ـ in range(p[step]) ] for ـ in range(m)] ) for step in range(number_of_steps)]
    u_bar = [np.array( [ model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for ـ in range(m)] ) for step in range(number_of_steps)]

    model.update()

    # Defining constraints [AT(t)+BM(t) ,W(t)] = T(t+1) and A x_bar(t) + B u_bar(t) + d_bar(t) = x_bar(t+1)

    # [AT(t)+BM(t) ,W(t)] = T(t+1)
    left_side_constraint = [ np.hstack( (np.dot( A[step] , T[step]) + np.dot( B[step] , M[step]) , W[step].G)) for step in range(number_of_steps) ]
    right_side_constraint = [T[step] for step in range(1, number_of_steps+1) ] if algorithm =='slow' \
                            else [ np.hstack(( np.zeros((n,dist_G_numberofcolumns[step-1])) , T[step] )) for step in range(1, number_of_steps+1) ]

    for step in range(number_of_steps):
        model.addConstrs( (left_side_constraint[step][i][j]==right_side_constraint[step][i][j] for i in range(n) for j in range(p[step+1])  ))

    # A x_bar(t) + B u_bar(t) + d_bar(t) = x_bar(t+1)
    model.addConstrs( ( (np.dot(A[step] , x_bar[step]) + np.dot(B[step] , u_bar[step]) + W[step].x)[i] == x_bar[step+1][i] for i in range(n) for step in range(number_of_steps) ) )
    model.update()

    #Implementing Hard Constraints over control input and state space
    [pp.zonotope_subset(model, pp.zonotope(G=T[step],x=x_bar[step]) , X[step] ,solver='gurobi') for step in range(number_of_steps+1) ]
    [pp.zonotope_subset(model, pp.zonotope(G=M[step],x=u_bar[step]) , U[step] ,solver='gurobi') for step in range(number_of_steps) ]
    
    #imposing the initial state
    if initial_state == True:
        # if it is not loaded with a value, we select the center of the admissible state set
        if system.initial_state is None:
            system.initial_state = X[0].x

        model.addConstrs( ( x_bar[0][i] == system.initial_state[i] for i in range(n)))

    model.update()

    output={
        'T':T,
        'x_bar': x_bar,
        'M':M,
        'u_bar': u_bar
    }    

    return output


def mpc_constraints(model,system,horizon=1,hard_constraints=True):
    """
    It adds mpc constraints to a gurobi model
    Inputs:
        model; a gurobi model
        system; either LTV or LTI model
        horizon; it is the mpc horizon
        hrad_constraints; True -> If you want to impose control and state hard constraints
                          False -> Otherwise
    Output:
        x; which is the predicated states, x[n * (horizon+1)]
        u; which is the predicated control inputs, u[m * (horizon)]
    """

    n = system.A.shape[0]               # Matrix A is n*n
    m = system.B.shape[1]               # Matrix B is n*m


    # Variables for predicated states and control inputs
    x = np.array( [ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for ـ in range(horizon+1)] for ـ in range(n)] )
    u = np.array( [ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for ـ in range(horizon)] for ـ in range(m)] )
    model.update()

    # imposing intial state
    model.addConstrs( (x[:,0][i] == system.state[i] for i in range(n) ))
    model.update()

    if system.sys=='LTI':
        # dynamics constraints
        model.addConstrs( ( (np.dot(system.A , x[:,step]) + np.dot(system.B , u[:,step]))[i]== x[i,step+1] for i in range(n) for step in range(horizon)) )

    elif system.sys=='LTV':
        # dynamics constraints
        model.addConstrs( (np.dot(system.A[step] , x[:,step]) + np.dot(system.B[step] , u[:,step])[i]== x[:,step+1] for i in range(n) for step in range(horizon)) )

    #Hard constraints over state
    if hard_constraints==True:
        if system.X!=None:
            if system.sys=='LTI':
                _=[parsi.be_in_set(model,system.X,x[:,i]) for i in range(horizon+1)]
            else:
                _=[parsi.be_in_set(model,system.X[i],x[:,i]) for i in range(horizon+1)]
        #Hard constraints over control input
        if system.U!=None:
            if system.sys=='LTI':
                _=[parsi.be_in_set(model,system.U,u[:,i]) for i in range(horizon)]
            else:
                _=[parsi.be_in_set(model,system.U[i],u[:,i]) for i in range(horizon)]

    return x,u


def rci_cen_synthesis_decen_controller_constraints(model,list_system,T_order):
    """
    # Note: general_version MUST be False, otherwise the probelm becomes non-convex
    This function will add required constraints for centralized synthesis of decentralized controllers for rci sets.
    Inputs:
        model; a gurobi model
        list_system; must be a list of linear LTI systems
        T_order; desired order for the RCI sets. For now, all subsystems have the same order
        initial_guess;
    Outputs:
        var; which is a dictionary containing the followings:
            x_bar; Center of the rci sets, x_bar[system][i=0 , ... , n]
            T; Generator of the rci, , T[system][i=0 , ... , n][0,...,k]
            u_bar; Center of the action set
            M; Generator of the action set, M[system][i=0 , ... , m][0,...,k]
            alpha_x; parameters for the parameterized sets in state space alpha_x[system][0,...,d]
            alpha_u; parameters for the parameterized sets in control space alpha_u[system][0,...,b]
    """

    sys_number = len(list_system)
    # n=[list_system[i].A.shape[0] for i in range(sys_number)]
    # m=[list_system[i].B.shape[1] for i in range(sys_number)]
    # k = [int(round(n[i]*T_order)) for i in range(sys_number)] 

    
    for sys in range(sys_number):

        # Setting the paamterized sets for each subsystem
        if list_system[sys].param_set_X is None or list_system[sys].param_set_U is None:
            list_system[sys].parameterized_set_initialization()

        # Defining the parameters
        dim_alpha_x = list_system[sys].param_set_X.G.shape[1]
        dim_alpha_u = list_system[sys].param_set_U.G.shape[1]

        list_system[sys].alpha_x = np.array([model.addVar(lb= 0, ub= GRB.INFINITY) for _ in range(dim_alpha_x)])
        list_system[sys].alpha_u = np.array([model.addVar(lb= 0, ub= GRB.INFINITY) for _ in range(dim_alpha_u)])
    
    model.update()

    # breaking the coupling among subsystems
    disturb = parsi.break_subsystems(list_system)
    
    real_disturbance_sets = [ deepcopy(sys.W) for sys in list_system ] 

    for sys in range(sys_number):
        list_system[sys].W = disturb[sys]

    # adding rci sonstraints; Satisfiability and Validity
    # var = parsi.concate_dictionaries( [ rci_constraints(model, list_system[sys], T_order, general_version=True) for sys in range(sys_number)] )
    var = [ rci_constraints(model, list_system[sys], T_order, general_version=False) for sys in range(sys_number)]      # Note: general_version MUST be False, otherwise the probelm becomes non-convex

    # adding Correctness constraints
    alpha_u=[ pp.zonotope_subset(model, pp.zonotope(x=np.array(var[sys]['u_bar']).T,G= var[sys]['M']) , list_system[sys].param_set_U ,alpha='vector',solver='gurobi')[2] for sys in range(sys_number)]
    [model.addConstrs((alpha_u[sys][i] == list_system[sys].alpha_u[i] for i in range(len(alpha_u[sys]))) ) for sys in range(sys_number)] #THIS MIGHT BE CONSTARINT I NEED O FINC THE DUAL

    alpha_x=[ pp.zonotope_subset(model, pp.zonotope(x=np.array(var[sys]['x_bar']).T,G=var[sys]['T']) , list_system[sys].param_set_X ,alpha='vector',solver='gurobi' )[2] for sys in range(sys_number)]
    [model.addConstrs((alpha_x[sys][i] == list_system[sys].alpha_x[i] for i in range(len(alpha_x[sys]))) ) for sys in range(sys_number)] #THIS MIGHT BE CONSTARINT I NEED O FINC THE DUAL

    model.update()

    for sys in range(sys_number):
        # By not chooing list_system[sys].alpha_x and list_system[sys].alpha_u, I am isolating the defined parameters
        var[sys]['alpha_x'] = alpha_x[sys] 
        var[sys]['alpha_u'] = alpha_u[sys]

    # returning the disturbance sets to their original value for all subsystems
    for sys in range(sys_number):
        list_system[sys].W = real_disturbance_sets[sys]
    
    return var


def hausdorff_distance_condition(model,zon1,zon2,alpha=None):
    """
    This function adds a variable to the optimization problem which if it is minimzed in the objective function it would be equal to directed hausdorff distance
    Inputs: 
        model; must be a gurobi model
        zon1; zonotopic set
        zon2; zonotopic set             Note: zon1 \subseteq zon2 \oplus z(0, d*I)
        alpha; -> None : when the zon2 set is not paramterized and has no variables, because variables cannot be on the right hand side of \subseteq
               -> the paramters of zon2 for the parameterized set zon2.G * Diag(alpha)
    Outputs: d
    """
    # circumbody = zon2 \oplus zonotope(x=0, G= d* eye) , where d is a variable which is the radius in the directed hausdorff distance

    # defining d, which the radius in the directed hausdorff distance
    d=model.addVar(lb= 0, ub= GRB.INFINITY)
    model.update()

    circumbody=pp.zonotope(x=zon2.x , G= np.concatenate( ( zon2.G,np.eye(zon2.G.shape[0])) , axis=1) )  

    # zon1 \subseteq circumbody * Diag(alpha)
    _,_,scale=pp.zonotope_subset(model,zon1,circumbody,alpha='vector',solver='gurobi')

    # if alpha is None, first number_of_columns_of_zon2 of scale must be equal to 1 
    # otherwise, it should be equal to alpha
    if alpha is not None:
        alpha_i_constraints = [model.addConstr(scale[i] == alpha[i])  for i in range(zon2.G.shape[1])]
    else:
        alpha_i_constraints = [model.addConstr(scale[i] == 1)  for i in range(zon2.G.shape[1])]
    
    # the rest of elements of scale must be equal to d
    model.addConstrs((scale[i] == d  for i in range(zon2.G.shape[1],circumbody.G.shape[1]) ))
    model.update()
    
    # d should be minimzed in the objective function to find the minimum possible value for it
    # model.setObjective( d , GRB.MINIMIZE )        
    # NOTE: if you add the objective funciton here, later on if you set the objective funciton again, it will overwrite this one.
    

    # TODO: we might not need to pass alpha_i_constraints

    return d,alpha_i_constraints


def potential_function_rci(list_system, system_index, T_order, reduced_order=1, include_validity=True):
    """
    Note: list_system[sys].param_set_X , list_system[sys].param_set_U, list_system[sys].alpha_x , list_system[sys].alpha_u MUST be already assigned before calling this function
    This function computes the component of the potenitial function for only one given subsystem,
    thus, for computing the potential fucniton you need to iterate it over subsystems and sum the outputs
    Inputs: 
        list_system; the list of coupled linear systems: Note that alpha_x,alpha_u must have a value
        system_index; the index of the system which you want to compute
        T_order; the order of the candidate rci set
        reduced_order; the order of the reduced disturbance set
    Outputs:
        potential_output; it is a dictionary containing the following keys:
            obj: which is the value for the potential function
            d_x_correctness: the directed hausdorf distance between rci set and the parameterized set on state space
            d_u_correctness: the directed hausdorf distance between action set and the parameterized set on control space
            alpha_x_grad: the gradients of the potential function with respect to alpha_x
            alpha_u_grad: the gradients of the potential function with respect to alpha_u
            x_bar: the center of the rci set
            T: the generator for the rci set
            u_bar: the center of the action set
            M: the generator for the action set
            d_x_valid: if the include_validity is True, it will be among the keys and 
                    shows the directed hausdorf distance between rci set and the admissible state set
            d_u_valid: if the include_validity is True, it will be among the keys and 
                    shows the directed hausdorf distance between action set and the admissible control set
            NOTE: the output rci and action sets are not necessary true. It is, only if the potential function is zero for all subsystems.
    """

    sys_number = len(list_system)
    n = list_system[system_index].A.shape[1]
    m = list_system[system_index].B.shape[1]

    # breaking the coupling among subsystems
    # list_system[sys].param_set_X , list_system[sys].param_set_U, list_system[sys].alpha_x , list_system[sys].alpha_u 
    # MUST be already assigned before calling this function parsi.break_subsystems()
    disturb = parsi.break_subsystems(list_system , subsystem_index=system_index)
    
    # Using Zonotope order reduction
    disturb= pp.boxing_order_reduction(disturb,desired_order=reduced_order)      # For now it just covers reduced_order equal to 1
    real_disturbance_sets = deepcopy(list_system[system_index].W)
    
    # Creating the model
    model = Model()

    ########################################################################################################################
    # NOTE: to make finding the gradient w.r.t alpha_j easy, a new set of variables is created fpr disturbance. 
    # This trick also forces us to set general_version to False, otherwise the problem becomes non-convex.
    # TODO: modify here later

    # Definging the new disturbance as a variable
    W_G = np.array([[model.addVar(lb = -GRB.INFINITY , ub = GRB.INFINITY) for _ in range( disturb.G.shape[1] )] for _ in range( disturb.G.shape[0] )])
    model.update()
    
    # Adding the constraint for disturbance: W_aug == disturb 
    for i in range( disturb.G.shape[0] ):
        for j in range(disturb.G.shape[1]):
            if i!=j:
                model.addConstr(W_G[i][j] == disturb.G[i][j])
    # NOTE: only for reduced_order=1, I am separating those constraints that contain alpha_j
    alpha_j_constraints=[model.addConstr(W_G[i][i] == disturb.G[i][i]) for i in range(n) ]

    model.update()

    # NOTE: should return to its original value at the end
    list_system[system_index].W = pp.zonotope(x = disturb.x , G = W_G) 
    ########################################################################################################################

    # adding rci sonstraints; Satisfiability and Validity
    # NOTE: general_version=False , because for now I wanted to make it easy to find the gradients of alpha_j
    var = rci_constraints(model, list_system[system_index], T_order, general_version=False , include_hard_connstraints=False)
    
    # Adding hausdorff distance terms for Correctness constraints
    d_x_correctness , alpha_i_x_constraints = parsi.hausdorff_distance_condition(model, pp.zonotope(x=var['x_bar'],G=var['T']) , list_system[system_index].param_set_X ,list_system[system_index].alpha_x)
    d_u_correctness , alpha_i_u_constraints = parsi.hausdorff_distance_condition(model, pp.zonotope(x=var['u_bar'],G=var['M']) , list_system[system_index].param_set_U ,list_system[system_index].alpha_u)

    # Adding hausdorff distance terms for Validity constraints
    if include_validity:
        d_x_valid , _ =parsi.hausdorff_distance_condition(model, pp.zonotope(x=var['x_bar'],G=var['T']) , list_system[system_index].X )
        d_u_valid , _ =parsi.hausdorff_distance_condition(model, pp.zonotope(x=var['u_bar'],G=var['M']) , list_system[system_index].U )
        model.setObjective( d_x_correctness + d_u_correctness + d_x_valid + d_u_valid, GRB.MINIMIZE )
    else:
        model.setObjective( d_x_correctness + d_u_correctness, GRB.MINIMIZE )
    model.update()

    model.setParam("OutputFlag",False)
    model.optimize()


    # Computing the gradient of this particular term of the potential function w.r.t alpha_i and alpha_j 
    grad_alpha_x=[]
    grad_alpha_u=[]

    # alpha_x
    for j in range(sys_number):
        
        # alpha_i_x
        if j==system_index:
            grad_alpha_x.append( np.array([alpha_i_x_constraints[i].pi for i in range(len(alpha_i_x_constraints))] ))
        
        # alpha_j_x does not affect A_ij
        elif not j in list_system[system_index].A_ij:
            grad_alpha_x.append( np.zeros(list_system[j].alpha_x.shape[0]))
        
        # alpha_j_x
        else:
            grad_alpha_x.append( np.array( [sum([alpha_j_constraints[row].pi * (abs(np.dot( list_system[system_index].A_ij[j][row,:], list_system[j].param_set_X.G[:,i]))) \
                            for row in range(disturb.G.shape[0])]) \
                            for i in range(list_system[j].param_set_X.G.shape[1])] ))

    # alpha_u
    for j in range(sys_number):

        # alpha_i_u
        if j==system_index:
            grad_alpha_u.append( np.array( [alpha_i_u_constraints[i].pi for i in range(len(alpha_i_u_constraints))] ))

        # alpha_j_u does not affect B_ij
        elif not j in list_system[system_index].B_ij:
            grad_alpha_u.append( np.zeros(list_system[j].alpha_u.shape[0]))
        
        # alpha_j_u
        else:
            grad_alpha_u.append( np.array( [sum([alpha_j_constraints[row].pi * (abs(np.dot( list_system[system_index].B_ij[j][row,:], list_system[j].param_set_U.G[:,i]))) \
                            for row in range(disturb.G.shape[0])]) \
                            for i in range(list_system[j].param_set_U.G.shape[1])]) )
    
    # Results
    k = var['T'].shape[1]
    T_result= np.array([ [ var['T'][i][j].X for j in range(k) ] for i in range(n) ] )
    x_bar_result = np.array( [ var['x_bar'][i].X for i in range(n) ] ) 

    M_result= np.array([ [ var['M'][i][j].X for j in range(k) ] for i in range(m) ] )
    u_bar_result = np.array( [ var['u_bar'][i].X for i in range(m) ] ) 

    potential_output={
        'obj': model.objVal ,
        'd_x_correctness': d_x_correctness.X,
        'd_u_correctness':d_u_correctness.X,
        'alpha_x_grad':grad_alpha_x,
        'alpha_u_grad':grad_alpha_u,
        'x_bar':x_bar_result,
        'T':T_result,
        'u_bar':u_bar_result,
        'M':M_result
    }
    if include_validity:
        var['d_x_valid'] = d_x_valid.X
        var['d_u_valid'] = d_u_valid.X
    
    del model
    
    # return the disturbance set to its original set
    list_system[system_index].W = real_disturbance_sets

    return potential_output


def viable_cen_synthesis_decen_controller_constraints(model, list_system, T_order, horizon=None, algorithm='slow'):
    """
    This function adds


    ??????


    This function adds the necessary constrainnts for viable set of an LTI system. It is written in Gurobi, directly.
    Inputs: 
        (1) model, needs to be a Gurobi model
        (2) horizon, which is the horizon, or in other words, number of viable sets
        (3) list_system, which is a set of systems. Note that each system.state needs to be initialized before calling this function.
        (4) T_order, which lead to number of columns of T (viable sets generator)
        (5) algoritm, it is either 'slow' or 'fast', which specifies the right hand side of the viable set constraint. [T] for 'slow' and [0,T] for 'fast'
    Outputs:
        T_x,T,M_x,M which specify the viables sets and action sets in zonotope for all systems.
    """



    sys_number = len(list_system)

    if horizon is None:
        horizon = len(list_system[0].X) - 1

    for sys in range(sys_number):

        # Setting the paamterized sets for each subsystem
        # does not matter if the system is LTI or LTV
        # the baseline set even for the LTV system, would be the rci set for the first time step system
        # NOTE: T order is increasing by steps (algorithm is assigned to slow right now)

        if list_system[sys].param_set_X is None or list_system[sys].param_set_U is None:
            list_system[sys].parameterized_set_initialization()
            print("parameterized set initialization PASSED for subsystem with index = %i"%sys )

        if list_system[sys].sys == 'LTI':
            rci_set = deepcopy(list_system[sys].param_set_X)
            action_set = deepcopy(list_system[sys].param_set_U)

            list_system[sys].param_set_X = [ rci_set ] * horizon
            list_system[sys].param_set_U = [action_set] * horizon

        # Defining the parameters
        # parameters are defined for length of horizon. there is no need to define parametric set on state space for t = horizon
        dim_alpha_x = [ list_system[sys].param_set_X[steps].G.shape[1] for steps in range(horizon) ]
        dim_alpha_u = [ list_system[sys].param_set_U[steps].G.shape[1] for steps in range(horizon) ]

        list_system[sys].alpha_x = [ np.array([model.addVar(lb= 0, ub= GRB.INFINITY) for _ in range(dim_alpha_x[steps])]) for steps in range(horizon) ]
        list_system[sys].alpha_u = [ np.array([model.addVar(lb= 0, ub= GRB.INFINITY) for _ in range(dim_alpha_u[steps])]) for steps in range(horizon) ]

        n = len( list_system[sys].param_set_X[0].x)
        m = len( list_system[sys].param_set_U[0].x)

        list_system[sys].alpha_center_x = [ np.array([model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for _ in range( n )]) for steps in range(horizon) ]
        list_system[sys].alpha_center_u = [ np.array([model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for _ in range( m )]) for steps in range(horizon) ]
        
        for steps in range(horizon):
            list_system[sys].param_set_X[steps].x = list_system[sys].alpha_center_x[steps]
            list_system[sys].param_set_U[steps].x = list_system[sys].alpha_center_u[steps]

    
    model.update()

    # breaking the coupling among subsystems
    disturb = parsi.break_subsystems(list_system)
    
    real_disturbance_sets = [ deepcopy(sys.W) for sys in list_system ] 

    for sys in range(sys_number):
        list_system[sys].W = disturb[sys]

    # adding viable constraints; Satisfiability and Validity
    # NOTE: before calling viable_constraints(), list_system[i].X[horizon+1] MUST be filled with a zonotpic set.
    var = [ viable_constraints(model, list_system[sys], T_order, horizon=None, algorithm=algorithm) for sys in range(sys_number)]

    # adding Correctness constraints
    alpha_u=[ [ pp.zonotope_subset(model, pp.zonotope(x=np.array(var[sys]['u_bar'][t]).T,G= var[sys]['M'][t]) , list_system[sys].param_set_U[t] ,alpha='vector',solver='gurobi')[2] for sys in range(sys_number)] for t in range(horizon) ]
    [ [ model.addConstrs(( alpha_u[t][sys][i] == list_system[sys].alpha_u[t][i] for i in range(len(alpha_u[t][sys])) ) ) for t in range(horizon) ] for sys in range(sys_number)] #THIS MIGHT BE CONSTARINT I NEED O FINC THE DUAL

    alpha_x=[ [ pp.zonotope_subset(model, pp.zonotope(x=np.array(var[sys]['x_bar'][t]).T,G=var[sys]['T'][t]) , list_system[sys].param_set_X[t] ,alpha='vector',solver='gurobi' )[2] for sys in range(sys_number)] for t in range(horizon) ]
    [ [ model.addConstrs(( alpha_x[t][sys][i] == list_system[sys].alpha_x[t][i] for i in range(len(alpha_x[t][sys])) ) ) for t in range(horizon) ]  for sys in range(sys_number)] #THIS MIGHT BE CONSTARINT I NEED O FINC THE DUAL

    model.update()

    for sys in range(sys_number):
        # By not choosing list_system[sys].alpha_x and list_system[sys].alpha_u, I am isolating the defined parameters
        var[sys]['alpha_x'] = [ alpha_x[t][sys] for t in range(horizon) ]
        var[sys]['alpha_u'] = [ alpha_u[t][sys] for t in range(horizon) ]

        var[sys]['alpha_center_x'] = [ list_system[sys].alpha_center_x[t] for t in range(horizon) ]
        var[sys]['alpha_center_u'] = [ list_system[sys].alpha_center_u[t] for t in range(horizon) ]

    # returning the disturbance sets to their original value for all subsystems
    for sys in range(sys_number):
        list_system[sys].W = real_disturbance_sets[sys]
    
    return var












# the potential function for coupled LTI subsystems. MPC strating from a given point and ending up inside rci set.
def potential_function_mpc(list_system, system_index, coefficient = 0 , T_order=3, reduced_order=1,algorithm='slow'):
    """
    The state (system.state) and rci set (system.omega) needs to be initialized prior to using this function.
    The same is true for alpha_x and alpha_u (system.alpha_x, system.alpha_u)
    Inputs: 
            (1)the list of coupled linear systems
            (2)the index of the system which you want to compute
            (3)horizon of the mpc
            (4)the order of the candidate viable sets
            (5)the order of the reduced disturbance set
            (6)algorithm: 'slow' represents [AT+BM,W]=[T] and 'fast' represents [AT+BM,W]=[0,T]
    Outputs:
            (1)the sum of directed hausdorf distances between viable sets and the sate paramterized sets (assumptions over state) 
            (2)the sum of directed hausdorf distances between action set and the control input paramterized sets (assumptions over control input) 
            (3)the gradients of (1) with respect to alpha_x , does not include time 0
            (4)the gradients of (2) with respect to alpha_u , does not include time 0
            (5)the gradients with respect to centers of assumptions for both state (does not include time 0) and control inputs (does include time 0)
    """
    from copy import copy,deepcopy
    from itertools import accumulate
    horizon = len(list_system[system_index].x_nominal)
    sys_number = len(list_system)
    n=[list_system[i].A.shape[0] for i in range(sys_number)]
    m=[list_system[i].B.shape[1] for i in range(sys_number)]
    k = round(n[system_index]*T_order) 

    # Initilizing the paramterize set
    # FOR NOW I ASSUME THEIR CENTER IS ZERO
    X_i=[sys.omega for sys in list_system]
    U_i=[sys.theta for sys in list_system]


    # CENTERS ARE GIVEN IN sytem.x_nominal and system.u_nominal: Both need to start from 0 to h (for u it will go up to h-1 )

    # # Center of the parameterized sets for state
    # X_i_x= [ [np.array([ model.addVar(lb = -GRB.INFINITY) for i in range(n[sys]) ]) for j in range(horizon)] for sys in range(sys_number)]
    # [X_i_x[sys].insert(0,list_system[system_index].state) for sys in range(sys_number)]

    # # THIS IS WRONG IT NEEDS TO BE GIVEN

    # # Center of the parameterized sets for control
    # U_i_x= [ [np.array([ model.addVar(lb = -GRB.INFINITY) for i in range(m[sys]) ]) for j in range(horizon)] for sys in range(sys_number)] 

    # # THIS IS WRONG IT NEEDS TO BE GIVEN


    # Creating the model
    model = Model()

    # For making it easy to find the dual variables for x_nominal and u_nominal, I am going to create some other variables and use them instead of system.x_nominal and system.u_nominal
    x_nominal = [ [ np.array([model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(n[sys])]) for step in range(horizon)] for sys in range(sys_number)]
    u_nominal = [ [ np.array([model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(m[sys])]) for step in range(horizon)] for sys in range(sys_number)]

    model.update()

    x_i_nominal_constraints = [[[model.addConstr( x_nominal[sys][step][i] == list_system[sys].x_nominal[step][i]) for i in range(n[sys])] for step in range(horizon)] for sys in range(sys_number)] 
    u_i_nominal_constraints = [[[model.addConstr( u_nominal[sys][step][i] == list_system[sys].u_nominal[step][i]) for i in range(m[sys])] for step in range(horizon)] for sys in range(sys_number)]
    
    model.update()

    # Computing the disturbance set
    disturb=[ deepcopy(list_system[system_index].W) for i in range(horizon)]
    for step in range(horizon):
        for j in range(sys_number):
            if j in list_system[system_index].A_ij:
                if step==0:
                    w= np.dot( list_system[system_index].A_ij[j], x_nominal[j][0])
                    disturb[0].x = disturb[0].x + w
                else:
                    w=pp.zonotope(G= np.dot(np.dot( list_system[system_index].A_ij[j], X_i[j].G), np.diag( list_system[j].alpha_x[step-1]) ) , x= np.dot( list_system[system_index].A_ij[j], x_nominal[j][step]))
                    disturb[step] = disturb[step]+w
            if j in list_system[system_index].B_ij:
                if step==0:
                    w= np.dot( list_system[system_index].B_ij[j], u_nominal[j][0])
                    disturb[0].x = disturb[0].x + w
                else:
                    w=pp.zonotope(G= np.dot(np.dot( list_system[system_index].B_ij[j], U_i[j].G), np.diag(list_system[j].alpha_u[step-1]) ) , x= np.dot( list_system[system_index].B_ij[j], u_nominal[j][step]))
                    disturb[step]= disturb[step]+w

        # Using Zonotope order reduction
        disturb[step]= pp.boxing_order_reduction(disturb[step],desired_order=reduced_order)      # For now it just covers reduced_order equal to 1
    # Note that alpha_x and alpha_u both need to have their first element be NONE


    # Definging the new disturbance as a variable
    W_x = [np.array([ model.addVar(lb = -GRB.INFINITY) for i in range(n[system_index]) ]) for step in range(horizon)]
    W_G = [np.array([[model.addVar(lb = -GRB.INFINITY) for i in range( disturb[step].G.shape[1] )] for j in range(n[system_index])]) for step in range(horizon)]
    model.update()
    
    # Adding the constraint for disturbance: W_aug == disturb   
    
    # Center
    # print('W_x',[W_x[step][i] for i in range(n[system_index]) for step in range(horizon)])
    # print('disturb[step].x[i]',[disturb[step].x[i] for i in range(n[system_index]) for step in range(horizon)])
    [[model.addConstr(W_x[step][i] == disturb[step].x[i]) for i in range(n[system_index])] for step in range(horizon)]
    
    # Genrator
    for step in range(horizon):
        for i in range(n[system_index]):
            for j in range(disturb[step].G.shape[1]):
                if i!=j:
                    model.addConstr(W_G[step][i][j] == disturb[step].G[i][j])
    alpha_j_constraints=[[model.addConstr(W_G[step][i][i] == disturb[step].G[i][i]) for i in range(n[system_index]) ]  for step in range(horizon)]              # JUST FOR reduced_order=1

    model.update()
    
    # Number of columns for T by time

    dist_G_numberofcolumns=[disturb[i].G.shape[1] for i in range(horizon)]
    if algorithm=='slow':
        dist_G_numberofcolumns.insert(0,k)
        p = list(accumulate(dist_G_numberofcolumns))

    # Defining the rci set (zonotope(x=xbar,G=T)) and action set (zonotope(x=ubar,G=M))
    # viable sets are zonotope(x=xbar[time],G=T[time])
    # action sets are zonotope(x=ubar[time],G=M[time])
    # They do not include initial position

    T = [ np.array([ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(k)] for j in range(n[system_index])]) for step in range(horizon)] if algorithm=='fast' \
        else [ np.array([ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(p[step])] for j in range(n[system_index])]) for step in range(horizon)]

    M = [ np.array([ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(k)] for j in range(m[system_index])]) for step in range(horizon-1)] if algorithm=='fast'\
        else [ np.array([ [model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(p[step])] for j in range(m[system_index])]) for step in range(horizon-1)]

    xbar = [ np.array([model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(n[system_index])]) for step in range(horizon)]

    ubar = [ np.array([model.addVar(lb= -GRB.INFINITY, ub= GRB.INFINITY) for i in range(m[system_index])]) for step in range(horizon-1)]

    model.update()
    
    # Constraint [AT+BM,W]=[T] or [0,T]

    left_side = [np.concatenate (( np.dot(list_system[system_index].A,T[step]) + np.dot(list_system[system_index].B,M[step])  , W_G[step+1]) ,axis=1) for step in range(horizon-1)]
    right_side = [np.concatenate(   ( np.zeros(W_G[step].shape) , T[step])  , axis=1) for step in range(1,horizon)] if algorithm=='fast' \
        else [ T[step] for step in range(1,horizon)]

    [model.addConstrs(   ( left_side[step][i,j] == right_side[step][i,j]  for i in range(n[system_index]) for j in range(len(right_side[step][0]))   )  )  for step in range(horizon-1)]              
    
    # Conditions for center: A* x_bar + B* u_bar + d_bar = x_bar 
    center = [np.dot(list_system[system_index].A , xbar[step] ) + np.dot(list_system[system_index].B , ubar[step] ) + W_x[step+1] for step in range(horizon-1)]
    [model.addConstrs( (center[step][i] == xbar[step+1][i] for i in range(n[system_index]))) for step in range(horizon-1)]
    
    model.update()

    # viable conditions for the first step

    # Genrator
    left_side_first_step = W_G[0]
    right_side_first_step = T[0]
    if k == disturb[0].G.shape[1]:
        model.addConstrs(   ( left_side_first_step[i,j] == right_side_first_step[i,j]  for i in range(n[system_index]) for j in range(k)   )  )
    elif k > disturb[0].G.shape[1]:
        model.addConstrs(   ( np.concatenate( (np.zeros( (n[system_index],k-disturb[0].G.shape[1]) ),left_side_first_step) ,axis=1) [i,j] == right_side_first_step[i,j]  for i in range(n[system_index]) for j in range(k)    ) )
    else:
        print('K is invalid')
        raise ValueError
    
    # center
    center_first_step= np.dot( list_system[system_index].A , x_nominal[system_index][0] ) + np.dot( list_system[system_index].B , u_nominal[system_index][0] ) + W_x[0]
    model.addConstrs( xbar[0][i] == center_first_step[i] for i in range(n[system_index]) ) 
    
    model.update()

    # Terminal constraint last_viable_set_i \subseteq omega_i (it is horizon-1 because the number of T is h and itsindex starts from 0 to h-1)
    ############### IT MAY MAKE THE WHOLE OPTIMIZATION PROBLEM INFEASIBLE
    # pp.zonotope_subset( model , pp.zonotope( x= xbar[horizon-1] , G= T[horizon-1] ) , list_system[system_index].omega , solver='gurobi')

    terminal_hausdorff_result = parsi.hausdorff_distance_condition(model, pp.zonotope(x=xbar[ horizon-1 ],G=T[ horizon-1 ]) , list_system[system_index].omega , np.ones( list_system[system_index].omega.G.shape[1] ) )
    model.update()

    # Hard constraint over control input
    # [ pp.zonotope_subset( model , pp.zonotope(x=ubar[step],G=M[step])  , list_system[system_index].U , solver='gurobi' ) for step in range(horizon-1)]
    
    be_in_set( model ,  list_system[system_index].U , u_nominal[system_index][0] )
    model.update()
    ######### for first u_nominal

    # Finding the Hausdurff Distances
    x_hausdorff_result = [ parsi.hausdorff_distance_condition(model, pp.zonotope(x=xbar[step],G=T[step]) , pp.zonotope(x=x_nominal[system_index][step+1] , G=X_i[system_index].G) ,list_system[system_index].alpha_x[step]) for step in range(horizon-1) ]
    d_x,alpha_i_x_constraints = [x_hausdorff_result[step][0] for step in range(horizon-1)] , [x_hausdorff_result[step][1] for step in range(horizon-1)]
    
    u_hausdorff_result = [ parsi.hausdorff_distance_condition(model, pp.zonotope(x=ubar[step],G=M[step]) , pp.zonotope(x=u_nominal[system_index][step+1] , G=U_i[system_index].G) ,list_system[system_index].alpha_u[step]) for step in range(horizon-1) ] 
    d_u,alpha_i_u_constraints = [u_hausdorff_result[step][0] for step in range(horizon-1)] , [u_hausdorff_result[step][1] for step in range(horizon-1)]


    # Objective

    # h is the sum of Hausdorff distances
    h = sum(d_x) + sum(d_u) + 10 * terminal_hausdorff_result[0]

    u_hausdorff_result  = [ parsi.hausdorff_distance_condition(model, pp.zonotope(x = ubar[ step ],G=M[ step ]) , list_system[system_index].U , np.ones( list_system[system_index].U.G.shape[1] ) )[0] for step in range(horizon-1)]
    
    model.update()

    h = h + sum ( u_hausdorff_result )





    # cost is x^TQx + u^TRu , where x and u are centers of the viable sets and action sets, respectively
    # cost  = np.dot( np.array(ubar).reshape(-1) , np.array(ubar).reshape(-1) ) + np.dot( np.array(xbar).reshape(-1) , np.array(xbar).reshape(-1) )
    cost  = np.dot( np.array(xbar).reshape(-1) , np.array(xbar).reshape(-1) )

    model.setObjective( h + coefficient * cost , GRB.MINIMIZE )

    model.update()

    model.setParam("OutputFlag",False)
    model.optimize()
    #print('MODEL',model.IsMIP)
    print('MODEL STATUS',model.Status)


    # Computing the gradients of alpha_x(1,h-1) , alpha_u(1,h-1) , x_nominal(1,h-1) , u_nominal(0,h-1) 


    grad_alpha_x_i=[]
    grad_alpha_u_i=[]
    grad_x_nominal=[]              # The gradient of the first step is not computed because the first step is considered to be x0 which is given
    grad_u_nominal=[]

    for step in range(horizon-1):
        grad_alpha_x_i.append([])
        grad_alpha_u_i.append([])
        grad_x_nominal.append([])
        grad_u_nominal.append([])

        for j in range(sys_number):
            
            grad_x_nominal[step].append( np.array([ x_i_nominal_constraints[j][step+1][i].pi for i in range(n[j]) ]) )

            if j==system_index:
                grad_alpha_x_i[step].append( np.array([alpha_i_x_constraints[step][i].pi for i in range(len(alpha_i_x_constraints[step]))] ))
            elif not j in list_system[system_index].A_ij:
                grad_alpha_x_i[step].append( np.zeros(list_system[j].alpha_x[step].shape[0]))
            else:
                grad_alpha_x_i[step].append( np.array( [sum([alpha_j_constraints[step+1][row].pi * (abs(np.dot( list_system[system_index].A_ij[j][row,:], X_i[j].G[:,i]))) \
                                for row in range(disturb[step+1].G.shape[0])]) \
                                for i in range(X_i[j].G.shape[1])] ))

        for j in range(sys_number):

            grad_u_nominal[step].append( np.array([ u_i_nominal_constraints[j][step][i].pi for i in range(m[j]) ]) )

            if j==system_index:
                grad_alpha_u_i[step].append( np.array( [alpha_i_u_constraints[step][i].pi for i in range(len(alpha_i_u_constraints[step]))] ))
            elif not j in list_system[system_index].B_ij:
                grad_alpha_u_i[step].append( np.zeros(list_system[j].alpha_u[step].shape[0]))
            else:
                grad_alpha_u_i[step].append( np.array( [sum([alpha_j_constraints[step+1][row].pi * (abs(np.dot( list_system[system_index].B_ij[j][row,:], U_i[j].G[:,i]))) \
                                for row in range(disturb[step+1].G.shape[0])]) \
                                for i in range(U_i[j].G.shape[1])]) )

    grad_u_nominal.append( [ np.array([ u_i_nominal_constraints[j][horizon-1][i].pi for i in range(m[j]) ])  for j in range(sys_number)])


    T_result = [ np.array([ [ T[step][i][j].X for j in range(k) ] for i in range(n[system_index]) ] ) for step in range(horizon)] if algorithm=='fast' \
        else [ np.array([ [ T[step][i][j].X for j in range(p[step]) ] for i in range(n[system_index]) ] ) for step in range(horizon)]
    T_x_result = [ np.array( [ xbar[step][i].X for i in range(n[system_index]) ] ) for step in range(horizon)] 

    M_result = [ np.array([ [ M[step][i][j].X for j in range(k) ] for i in range(m[system_index]) ] ) for step in range(horizon-1)] if algorithm=='fast' \
        else [ np.array([ [ M[step][i][j].X for j in range(p[step]) ] for i in range(m[system_index]) ] ) for step in range(horizon-1)]
    M_x_result = [ np.array( [ ubar[step][i].X for i in range(m[system_index]) ] ) for step in range(horizon-1)]
    

    potential_output = {
        'obj' : model.objVal ,
        'h' : sum( [i.X for i in d_x] + [i.X for i in d_u] ) + terminal_hausdorff_result[0].X + sum ( [i.X for i in u_hausdorff_result]  ), 
        'alpha_alpha_x_grad' : np.array(grad_alpha_x_i),
        'alpha_alpha_u_grad' : np.array(grad_alpha_u_i),
        'x_nominal_grad' : np.array(grad_x_nominal),
        'u_nominal_grad' : np.array(grad_u_nominal),
        'xbar' : T_x_result,
        'T' : T_result,
        'ubar' : M_x_result,
        'M' : M_result
    }
    del model

    return potential_output


def be_in_set( model , zonotope_set ,point):
    """
    given a "point", this function adds sufficient constraints to force point \in zonotope_set
    inputs: 
            point
            zonotope_set
    outputs: zeta
    """

    zeta = np.array( [ model.addVar( lb=-1, ub=1 ) for _ in range( zonotope_set.G.shape[1] )] )

    model.addConstrs( ( zonotope_set.x[i] + np.dot( zonotope_set.G[ i,: ] , zeta ) == point[i] ) for i in range( zonotope_set.G.shape[0] ) )
    model.update()

    return zeta

