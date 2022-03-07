"""
@author: kasra
"""
import warnings
import numpy as np
from timeit import default_timer as timer
try:
    from gurobipy import *
except:
    print('gurobi is not installed correctly') 
try:
    import pypolycontain as pp
except:
    warnings.warn("You don't have pypolycontain not properly installed.")
try:
    import pydrake.solvers.mathematicalprogram as MP
    import pydrake.solvers.gurobi as Gurobi_drake
    # use Gurobi solver
    global gurobi_solver, license
    gurobi_solver=Gurobi_drake.GurobiSolver()
    license = gurobi_solver.AcquireLicense()
except:
    warnings.warn("You don't have pydrake installed properly. Methods that rely on optimization may fail.")
try:
    import parsi
except:
    warnings.warn("You don't have parsi not properly installed.")

def rci(system,order_max=10, general_version=True,obj=True):
    """
    Given a LTI system, this function returns a rci set and its action set.
    Inputs:
        system; which must be a linear system
        order_max; the algorithm use order_max to iterate over k. It stops iterating over k after it reaches order_max
        general_version; True -> Considers beta as a variable in the algorithm
                         False -> Does NOT consider beta. More restricted version of the algorithm
        obj; True -> it assings mean square objective function to minimze the Euclidian distance between the rci set center and the admissable state
             False -> There is no objective function
    Outputs:
        omega; the rci set in for of a zontope
        theta; the action set in for of a zontope
    """

    n= system.A.shape[0]

    for order in np.arange(1, order_max, 1/n):
        
        model = Model()
        var = parsi.rci_constraints(model,system,order,general_version=general_version)

        # adding objective function to make the center of the rci set as close as possible tot he center of X, admission state set
        if obj == True:
            model.setObjective( np.dot(var['x_bar'] - system.X.x , var['x_bar'] - system.X.x) )

        #Result 
        model.update()
        model.setParam("OutputFlag",False)
        model.optimize() 

        if model.status == 2:
            k= var['T'].shape[1]
            m= var['M'].shape[0]

            T_result = np.array( [ [var['T'][i][j].x for j in range(k)] for i in range(n) ] )
            x_bar_result = np.array( [ var['x_bar'][i].x for i in range(n)] )
            M_result = np.array( [ [var['M'][i][j].x for j in range(k)] for i in range(m) ] )
            u_bar_result = np.array( [ var['u_bar'][i].x for i in range(m)] )
            beta = var['beta'].x if var['beta'] is not None else 0

            omega = pp.zonotope(x = x_bar_result , G = T_result / (1-beta))
            theta = pp.zonotope(x = u_bar_result , G = M_result / (1-beta))

            return omega , theta
        else:
            del model

    print('Not able to find a feasible solution for the RCI set. You can try again by increasing the order_max')
    return None , None


def viable_limited_time(system,horizon = None ,order_max=10,obj=True,algorithm='slow'):
    """
    Given a system, this function returns a limited time vaibale set and its action set.
    Inputs:
        system; which can be a LTI or LTV system
        horizon; the number steps to be considered. If it is None, it automattically sets it to the number of time steps in the LTV system.
        order_max; the algorithm use order_max to iterate over k. It stops iterating over k after it reaches order_max
        obj; True -> it assings mean square objective function to minimze the Euclidian distance between the viable sets' centers and the admissable states
             False -> There is no objective function
        algorithm; slow -> if we want the order of the viable sets grow by number of steps
                   fast -> if the order of viable sets remain fixed at T_order
    Outputs:
        omega; The sequence of viable sets, omega(0) , ... , omega(horizon+1)
        theta; The sequence of action sets, theta(0) , ... , theta(horizon)
    """
    if horizon == None:
        number_of_steps=len(system.A)
    else:
        number_of_steps=horizon
        
    n= system.A[0].shape[0]
    
    for order in np.arange(1, order_max, 1/n):

        model = Model()
        var=parsi.viable_constraints(model, system, order, horizon=horizon, algorithm=algorithm)

        #Defining the objective function
        #objective function for minimizing the distance between the RCI set and the set of admissible states
        if obj==True:
            if system.sys =='LTI':
                X = [system.X for i in range(number_of_steps+1)] 
            else:
                X = [system.X[i] for i in range(number_of_steps+1)]

            objective = [np.dot(var['x_bar'][step] - X[step].x , var['x_bar'][step] - X[step].x) for step in range(number_of_steps+1) ]
            model.setObjective( sum(objective) )

        #Result
        
        model.update()
        model.setParam("OutputFlag",False)
        model.optimize() 

        if model.status == 2:

            T_result = [ np.array( [ [var['T'][step][i][j].x for j in range( len(var['T'][step][0]) )] for i in range(len(var['T'][step])) ] ) for step in range(number_of_steps+1)]
            x_bar_result = [ np.array( [ var['x_bar'][step][i].x for i in range(len(var['T'][step])) ] ) for step in range(number_of_steps+1) ]

            M_result = [ np.array( [ [var['M'][step][i][j].x for j in range( len(var['M'][step][0])) ] for i in range(len(var['M'][step])) ] ) for step in range(number_of_steps) ]
            u_bar_result = [ np.array( [ var['u_bar'][step][i].x for i in range(len(var['M'][step]))] ) for step in range(number_of_steps)]

            omega=[pp.zonotope(G=T_result[i],x=x_bar_result[i]) for i in range(number_of_steps+1)]
            theta=[pp.zonotope(G=M_result[i],x=u_bar_result[i]) for i in range(number_of_steps)]

            return omega , theta

        else:
            del model

    print('Not able to find a feasible solution for the viable set. You can try again by increasing the order_max')
    return None , None


def sub_systems(system, partition_A, partition_B, disturbance=True, admissible_x=True, admissible_u=True ):
    """
    This function gets a single LTI system and returns a network of connected LTI systems
    Inputs:
        system; a LTI system
        partition_A; a list, partition over aggregated A matrix, for example: [2,3,1] creates A_{11}=2*2, A_{22}=3*3, and A_{33}=1*1
        partition_B; a list, partition over aggregated B matrix
        disturbance; 
            -> True: it over approximates the set and uses decomposition to decompose by subsystems. NOTE that it need Drake.
            -> list of zonotopic sets in order of subsystems
        admissible_x;
            -> True: uses decomposition to decompose by subsystems. NOTE that it need Drake.
            -> list of zonotopic sets in order of subsystems
        admissible_u
            -> True: uses decomposition to decompose by subsystems. NOTE that it need Drake.
            -> list of zonotopic sets in order of subsystems
    Outputs:
        sys; list of subsystems
    """
    
    from itertools import accumulate
    number_of_subsys=len(partition_A)
    par_accum_A=list(accumulate(partition_A))
    par_accum_B=list(accumulate(partition_B))
    par_accum_A.insert(0,0)
    par_accum_B.insert(0,0)

    A=[[ system.A[par_accum_A[i]:par_accum_A[i+1] , par_accum_A[j]:par_accum_A[j+1]] for i in range(number_of_subsys)] for j in range(number_of_subsys)]
    B=[[ system.B[par_accum_A[i]:par_accum_A[i+1] , par_accum_B[j]:par_accum_B[j+1]] for i in range(number_of_subsys)] for j in range(number_of_subsys)]

    if admissible_x==True:
        X=pp.decompose(system.X,partition_A)
    else:
        X = admissible_x
    
    if admissible_u==True:
        U=pp.decompose(system.U,partition_B)
    else:
        U = admissible_u

    #Right now, it just covers disturbances with order 1
    if disturbance==True:
        W = pp.boxing_order_reduction(system.W,desired_order=1)
        W=pp.decompose(W,partition_A)
    else:
        W = disturbance             # disturbance is a list of zonotopes

    sys=[ parsi.Linear_system(A[i][i] , B[i][i] , W=W[i],X=X[i] , U=U[i]) for i in range(number_of_subsys)]
    for i in range(number_of_subsys):
        for j in range(number_of_subsys):
            if i==j:
                continue
            if (A[i][j]==0).all()==False:
                sys[i].A_ij[j]= A[i][j]
            if (B[i][j]==0).all()==False:
                sys[i].B_ij[j]= B[i][j]

    return sys


def sub_systems_LTV(system, partition_A, partition_B, disturbance=True, admissible_x=True, admissible_u=True):
    """
    This function gets a single LTV system and returns a network of connected LTV systems.
    Inputs:
        system; a LTV system
        partition_A; a list, partition over aggregated A matrix, for example: [2,3,1] creates A_{11}=2*2, A_{22}=3*3, and A_{33}=1*1
        partition_B; a list partition over aggregated B matrix
        disturbance; 
            -> True: it over approximates the set and uses decomposition to decompose by subsystems. NOTE that it need Drake.
            -> list of list of zonotopic sets in order of subsystems, disturbance[subsystem][t]
        admissible_x;
            -> True: uses decomposition to decompose by subsystems. NOTE that it need Drake.
            -> list of list of zonotopic sets in order of subsystems, admissible_x[subsystem][t]
        admissible_u
            -> True: uses decomposition to decompose by subsystems. NOTE that it need Drake.
            -> list of list of zonotopic sets in order of subsystems, admissible_u[subsystem][t]
    Outputs:
        sys; list of LTV subsystems
    NOTE: 
    * The result sys[i].A_ij and sys[i].B_ij are LIST of DICTIONARIES, where in the index of the list is the time step. Also the keys of the dictionary is the index of the neighbouring subsystem
        where in case they are not neighbour, there is no key for that specific subsystem.
    * the length of all the lists would be equal to horizon, which means that final admissible state set is not considered and need be added manually: X[horizon+1]
        did this so that someone can add a goal set to a LTV system.
    * if you want to set disturbance, admissble_x, admissible_u yourself, you need to set it like: [[W_i for j in range(number_of_subsystems)] for t in range(horizon)]
    * For now, you have to set all above three yourself.
    """

    horizon = len( system.A )
    number_of_subsys = len(partition_A)

    sub_sys = []
    for t in range(horizon):
        system_t=parsi.Linear_system( system.A[t], system.B[t], W=system.W[t], X=system.X[t], U=system.U[t] )

        sub_sys.append( 
            parsi.sub_systems(
                system_t,
                partition_A=partition_A,
                partition_B=partition_B,
                disturbance=disturbance[t], 
                admissible_x=admissible_x[t] , 
                admissible_u=admissible_u[t]
            )
        )

    sys=[ parsi.Linear_system(
        [ sub_sys[t][i].A for t in range(horizon)] ,
        [ sub_sys[t][i].B for t in range(horizon)] ,
        [ sub_sys[t][i].W for t in range(horizon)] ,
        [ sub_sys[t][i].X for t in range(horizon)] ,
        [ sub_sys[t][i].U for t in range(horizon)] 
    ) for i in range(number_of_subsys)]

    # sys[i].A_ij and sys[i].B_ij is a list of dictionaries at different time steps
    # for example: sys[i].A_ij = [ {0:A_i0(0), 1:A_i1(0), 3:A_i3(0) } , {1:A_i1(1), 2:A_i2(1)} , {3:A_i3(2)} , {} ]
    for i in range(number_of_subsys):
        sys[i].A_ij=[]
        sys[i].B_ij=[]

        for t in range(horizon):
            sys[i].A_ij.append( sub_sys[t][i].A_ij )
            sys[i].B_ij.append( sub_sys[t][i].B_ij )

    return sys
    

def decentralized_rci_centralized_synthesis(list_system,size='min',order_max=30):
    """
    This function return a set of decentralized rci sets and their action sets. It computes everything in a centralized fashion using AGC
    Inputs:
        list_system; list of subsystems
        size; size of the viable sets
            min -> minimizing the rci sets by minimizng sum of the parameters on state space alpha_x
            max -> maximizing the rci sets by maximizing sum of the parameters on state space alpha_x
            nothing -> there will be no objective function
            numpy.array vector -> it is used to find the maximum/minimume paramters at certain directions. useful for drawing the correct set of parameters set only
        order_max; maximum allowed order for the zonotopic rci set
    Outputs:
        omega; the list of rci sets
        theta; the list of action sets
        alfa_x; the list of parameters on the state space
        alfa_u; the list of parameters on the control space
    It it could not find any solution it returns None for all outputs
    """

    number_of_subsys=len(list_system)
    n=[len(list_system[i].A) for i in range(number_of_subsys)]
    m=[list_system[i].B.shape[1] for i in range(number_of_subsys)]

    for order in np.arange(1, order_max, 1/max(n)):
        
        model= Model()

        var = parsi.rci_cen_synthesis_decen_controller_constraints(model,list_system,order)

        # Objective function
        if size == 'min' or size == 'max':

            obj = sum( [sum(var[sys]['alpha_x']) for sys in range(number_of_subsys)] )
            # NOTE: not sure if it works
            if size == 'max':
                obj = -1 * obj 

        # This will find the maximum alpha (it is for drawing)
        elif type(size) == np.ndarray:
            obj = np.dot( size , np.array(var[0]['alpha_x'][0: len(size)]) )

        else:
            obj = 1 
        
        model.setObjective( obj , GRB.MINIMIZE )
        model.update()

        # Result
        model.setParam("OutputFlag",False)
        model.optimize()
        
        if model.Status!=2:
            
            del model
            continue

        T_result= [ np.array( [ [ var[sys]['T'][i][j].X for j in range(len(var[sys]['T'][i])) ] for i in range(n[sys]) ] ) for sys in range(number_of_subsys) ]
        T_x_result = [ np.array( [ var[sys]['x_bar'][i].X for i in range(n[sys]) ] ) for sys in range(number_of_subsys) ]
        alfa_x = [ np.array( [ var[sys]['alpha_x'][i].X for i in range(len( var[sys]['alpha_x'])) ] ) for sys in range(number_of_subsys)]
        omega= [ pp.zonotope(x=T_x_result[sys] , G=T_result[sys]) for sys in range(number_of_subsys) ] 


        M_result= [ np.array( [ [ var[sys]['M'][i][j].X for j in range(len(var[sys]['M'][i])) ] for i in range(m[sys]) ] ) for sys in range(number_of_subsys) ]
        M_x_result= [ np.array( [ var[sys]['u_bar'][i].X for i in range(m[sys]) ] ) for sys in range(number_of_subsys) ]
        alfa_u = [ np.array( [ var[sys]['alpha_u'][i].X for i in range(len(var[sys]['alpha_u'])) ] ) for sys in range(number_of_subsys)]
        theta= [ pp.zonotope(x=M_x_result[sys] , G=M_result[sys]) for sys in range(number_of_subsys) ] 
        
        for i in range(number_of_subsys):
            list_system[i].omega=omega[i]
            list_system[i].theta=theta[i]

        return omega , theta , alfa_x , alfa_u
    
    print('Could not find any solution, you can increase order_max and try again.')
    return None , None , None , None


#MPC: right now it just covers point convergence
def mpc(system,horizon=1,x_desired='origin'):
    """
    MPC: Model Predictive Control
    Inputs:
        system; can be a LTI ot LTV system
        horizon; which is the MPC horizon
        x_desired; is the desired state
    Output:
        u
    """
    
    terminal_cost = 2
    cost_x_coeff = 0.1
    cost_u_coeff = 0.1

    n = system.A.shape[0]               # Matrix A is n*n
    m = system.B.shape[1]               # Matrix B is n*m
    if x_desired=='origin':
        x_desired=np.zeros(n)

    model = Model()
    # dynamical constraint and hard constraints
    x,u=parsi.mpc_constraints(model,system,horizon=horizon,hard_constraints=True)

    # Objective function
    cost_x = sum( [ cost_x_coeff * np.dot( (x[:,step] - x_desired) , (x[:,step] - x_desired) ) for step in range(horizon)])
    # Terminal Cost
    cost_terminal = terminal_cost * np.dot( (x[:,horizon] - x_desired) , (x[:,horizon] - x_desired) ) 
    # Energy Cost
    cost_u = sum( [ cost_u_coeff * np.dot( u[:,step] , u[:,step] ) for step in range(horizon)])

    model.setObjective( cost_x+cost_u+cost_terminal , GRB.MINIMIZE)

    model.setParam("OutputFlag",False)
    model.optimize() 
    print('mpc mode is ', model.Status)

    x_mpc = np.array( [ [x[i,j].x for j in range(horizon+1)] for i in range(n)] )
    u_mpc = np.array( [ [u[i,j].x for j in range(horizon)] for i in range(m)] )

    implementable_u = u_mpc[:,0]

    return implementable_u , x_mpc , u_mpc


def decentralized_rci_compositional_synthesis(list_system,initial_order=2,step_size=0.1,alpha_0='ones',order_max=100 , iteration_max=100 , VALUE_ZERO=10**(-6)):
    """
    This function computes a set of decentralized rci sets in a compisinal fashion.
    NOTE: 
        * the order is the same for all subsystems
        * validity is included in the potential function
    Input:
        list_system; list of LTI systems
        initial_order; order of the rci sets. it starts from initial_order and increases the order until potential funciton reaches zero for all subsystems
        step_size; step size for gradient descent
        alpha_0; initialization of parameters
            -> ones: all are set to zero (recommended)
            -> random: random values between [0,1)
        order_max; maximum allowable order for the zonotopic rci set
        iteration_max; maximum number of iterations for the same order. If potential fucntion does not reach zero, we will increase the order by one unit for all subsystems
    Output: 
        Omega; list of decentralized rci sets
        Theta; list of decentralized action sets
    if it fails to find a solution, it return the final potential function that it could get to.
    """
   
    order=initial_order
    num_sys = len(list_system)

    # Initilization of the parametric sets : list_system[sys].param_set_X , list_system[sys].param_set_U, list_system[sys].alpha_x , list_system[sys].alpha_u
    for sys in list_system:
        sys.parameterized_set_initialization()

    # assigning paramters
    # all ones
    if alpha_0=='ones':
        for sys in list_system:
            sys.alpha_x= np.ones(sys.param_set_X.G.shape[1])
            sys.alpha_u= np.ones(sys.param_set_U.G.shape[1])
    # random
    elif alpha_0=='random':
        for sys in list_system:
            sys.alpha_x= np.random.rand(sys.param_set_X.G.shape[1])
            sys.alpha_u= np.random.rand(sys.param_set_U.G.shape[1])
           
    objective_function=1
    objective_function_previous=2
    iteration=0

    while objective_function>0 or order==order_max:

        # finding all componenets of the potential function
        subsystems_output = [ parsi.potential_function_rci(list_system, system_index, order, reduced_order=1, include_validity=True) for system_index in range(num_sys) ]
     
        objective_function_previous=objective_function

        # computing the potential function
        objective_function = sum([subsystems_output[i]['obj'] for i in range(num_sys) ])

        print('potential function : ',objective_function)

        # if the potential funciton is smaller than VALUE_ZERO, the solution is found
        if objective_function <= VALUE_ZERO:
            for sys in range(len(list_system)):
                list_system[sys].omega=pp.zonotope(G=subsystems_output[sys]['T'],x=subsystems_output[sys]['x_bar'])
                list_system[sys].theta=pp.zonotope(G=subsystems_output[sys]['M'],x=subsystems_output[sys]['u_bar'])

            return [sys.omega for sys in list_system],[sys.theta for sys in list_system]

        else:
            for i in range(num_sys):

                # finding gradients and adding all the gradient for find the best direction
                grad_x= np.array(sum([subsystems_output[j]['alpha_x_grad'][i] for j in range(num_sys)]))
                grad_u= np.array(sum([subsystems_output[j]['alpha_u_grad'][i] for j in range(num_sys)]))


                # gradient descent

                # I am not normalizing the gradients because it prevent convergence when some of the parameters are assigned to zero after they go below zero
                
                # normalized gradient descent 
                # grad_norm_x = np.linalg.norm( grad_x ,ord=2)
                # grad_norm_x = 1 if grad_norm_x == 0 else grad_norm_x
                # grad_norm_u = np.linalg.norm( grad_u ,ord=2)
                # grad_norm_u = 1 if grad_norm_u == 0 else grad_norm_u
                # list_system[i].alpha_x = list_system[i].alpha_x - ( step_size / grad_norm_x ) * grad_x
                # list_system[i].alpha_u = list_system[i].alpha_u - ( step_size / grad_norm_u ) * grad_u 
                
                # non-normalized gradient descent 
                list_system[i].alpha_x = list_system[i].alpha_x - step_size * grad_x 
                list_system[i].alpha_u = list_system[i].alpha_u - step_size * grad_u 


                # gradient descent may cause the parameters to go below zero, which must not happen because the parameters have zero as their lower bound
                # so those elements that go below zero after updating are replaced with zero

                list_system[i].alpha_x[ list_system[i].alpha_x < 0 ] = 0
                list_system[i].alpha_u[ list_system[i].alpha_u < 0 ] = 0

                # for jj in range(len(list_system[i].alpha_x)):
                #     print('YYYYYYYYYYYYYYYYYYYYYYYY')
                #     if list_system[i].alpha_x[jj] < 0:
                #         list_system[i].alpha_x[jj]=0
                # for jj in range(len(list_system[i].alpha_u)):
                #     print('XXXXXXXXXXXXXXXXXXXXXXXX')
                #     if list_system[i].alpha_u[jj] < 0:
                #         list_system[i].alpha_u[jj]=0


        # if the potential funciton is increased compared to its previous value more than a threshold, it can be because the order of the sets is small
        if objective_function > ((objective_function_previous) + 10**(-2)):
            order = order + 1

        # if the potential funciton is not changing that mucg, it can because of a small step size
        elif abs(objective_function - objective_function_previous)< 10**(-3):

            step_size=step_size+0.1
            
            if iteration == iteration_max:
                order = order + 1
                iteration = 0

        iteration += 1

    return objective_function


def shrinking_rci(list_system,reduced_order=2,order_reduction_method='pca'):
    """
    The goal is shrinking the rci sets.
    """
    sys_number=len(list_system)

    while any([i.X!=i.omega for i in list_system]):
        
        # Computing the disturbance set
        disturb=[]
        for i in range(sys_number):
            disturb.append(list_system[i].W)
        for i in range(sys_number):
            for j in range(sys_number):
                if j in list_system[i].A_ij:
                    w=pp.zonotope(G= np.dot( list_system[i].A_ij[j], list_system[j].omega.G) , x= np.dot( list_system[i].A_ij[j], list_system[j].omega.x))
                    disturb[i] = disturb[i]+ w
                if j in list_system[i].B_ij:
                    w=pp.zonotope(G= np.dot( list_system[i].B_ij[j], list_system[j].theta.G) , x= np.dot( list_system[i].B_ij[j], list_system[j].theta.x))
                    disturb[i] = disturb[i]+ w
            
            # ADDING ORDER REDUCTION FOR DISTURBANCE
            if order_reduction_method=='pca':
                list_system[i].W= pp.pca_order_reduction(disturb[i],desired_order=reduced_order)
            elif order_reduction_method=='boxing':
                list_system[i].W= pp.boxing_order_reduction(disturb[i],desired_order=reduced_order)
            else:
                list_system[i].W=disturb[i]

        for system in list_system:  

            system.X=system.omega               # this will enforce the new rci set be subset of the previous one.
            system.omega,_ = rci(system,order_max=100,size='min',obj='include_center')
            
    return [i.omega for i in list_system]
















def decentralized_viable_centralized_synthesis(list_system, size='min', order_max=30, algorithm='slow', horizon=None ):
    """
    ??????????
    TODO: 
    * add the center of baseline sets to optimization variables
    * add the initial state to the constraints
    * add an objective function
    """

    number_of_subsys=len(list_system)
    
    if horizon is None:
        horizon = len(list_system[0].X) - 1

    n= [len(list_system[i].A[0]) for i in range(number_of_subsys)] if list_system[0].sys=='LTV' else [len(list_system[i].A) for i in range(number_of_subsys)]
    m= [list_system[i].B[0].shape[1] for i in range(number_of_subsys)] if list_system[0].sys=='LTV' else [list_system[i].B.shape[1] for i in range(number_of_subsys)]

    for order in np.arange(1, order_max, 1/max(n)):
        
        model= Model()

        # adding the required constraints for the finite time viable set
        var = parsi.viable_cen_synthesis_decen_controller_constraints(model, list_system, order, horizon=horizon, algorithm=algorithm )

        # Objective function
        if size == 'min' or size == 'max':
            
            obj = sum([ sum([ sum(var[sys]['alpha_x'][t]) for t in range(horizon)]) for sys in range(number_of_subsys)])
            # NOTE: not sure if it works
            if size == 'max':
                obj = -1 * obj 

        else:
            obj = 1 
        
        model.setObjective( obj , GRB.MINIMIZE )
        model.update()

        # Result
        model.setParam("OutputFlag",False)
        model.optimize()
        
        if model.Status!=2:
            
            del model
            continue

        T_result= [ [ np.array( [ [ var[sys]['T'][t][i][j].X for j in range(len(var[sys]['T'][t][i])) ] for i in range(n[sys]) ] ) for t in range(horizon+1)] for sys in range(number_of_subsys) ]
        T_x_result = [ [ np.array( [ var[sys]['x_bar'][t][i].X for i in range(n[sys]) ] ) for t in range(horizon+1)] for sys in range(number_of_subsys) ]
        alfa_x = [ [ np.array( [ var[sys]['alpha_x'][t][i].X for i in range(len( var[sys]['alpha_x'][t])) ] ) for t in range(horizon)] for sys in range(number_of_subsys)]
        alpha_center_x = [ [ np.array( [ var[sys]['alpha_center_x'][t][i].X for i in range(n[sys]) ] ) for t in range(horizon) ] for sys in range(number_of_subsys)]
        omega= [ [ pp.zonotope(x=T_x_result[sys][t] , G=T_result[sys][t]) for t in range(horizon+1)] for sys in range(number_of_subsys) ] 


        M_result= [ [ np.array( [ [ var[sys]['M'][t][i][j].X for j in range(len(var[sys]['M'][t][i])) ] for i in range(m[sys]) ] ) for t in range(horizon)] for sys in range(number_of_subsys) ]
        M_x_result= [ [ np.array( [ var[sys]['u_bar'][t][i].X for i in range(m[sys]) ] ) for t in range(horizon)] for sys in range(number_of_subsys) ]
        alfa_u = [ [ np.array( [ var[sys]['alpha_u'][t][i].X for i in range(len(var[sys]['alpha_u'][t])) ] ) for t in range(horizon)] for sys in range(number_of_subsys)]
        alpha_center_u = [ [ np.array( [ var[sys]['alpha_center_u'][t][i].X for i in range(m[sys]) ] ) for t in range(horizon) ] for sys in range(number_of_subsys)]
        theta= [ [ pp.zonotope(x=M_x_result[sys][t] , G=M_result[sys][t]) for t in range(horizon)] for sys in range(number_of_subsys) ] 
        
        for i in range(number_of_subsys):
            list_system[i].omega=omega[i]
            list_system[i].theta=theta[i]

        return omega , theta , alfa_x , alfa_u , alpha_center_x , alpha_center_u
    
    print('Could not find any solution, you can increase order_max and try again.')
    return None , None , None , None





















def compositional_synthesis( list_system , horizon , initial_order=2 , step_size=0.1 , order_max=100 , algorithm='slow' , delta_coefficient = 5 , iteration_max = 50):
    """
    this is a function for synthesis problem of a connected LTI systems.
    Inputs
    """

    # IT DOES NOT CONSIDER MAPING TO alpha_max REGION. THE SAME FOR CENTERS!
    
    # Initialization of alpha_x and alpha_x and x_nominal and u_nominal

    # if alpha_0=='random':
    #     for i in list_system:
    #         i.alpha_x=np.dot( np.random.rand(len(i.alpha_x_max)) , np.diag(i.alpha_x_max) )
    #         i.alpha_u=np.dot(np.random.rand(len(i.alpha_u_max)) , np.diag(i.alpha_u_max) )
    #         i.mapping_alpha_to_feasible_set()


    order = initial_order
    iteration = 0
    coefficient = 0
    flag = 0

    while order<=order_max:
        
        subsystems_output = [ parsi.potential_function_mpc(list_system, system_index, coefficient = coefficient ,T_order=order, reduced_order=1,algorithm=algorithm) for system_index in range(len(list_system)) ]

        # h is sum of Hausdorff distances
        h = sum([subsystems_output[i]['h'] for i in range(len(list_system)) ])

        # objective function
        objective_function = sum([subsystems_output[i]['obj'] for i in range(len(list_system)) ])







        # # GIF
        # number_of_subsystems = len(list_system)
        # if iteration % 5 ==0 :
        #     import matplotlib.pyplot as plt
        #     u_0 = [ i.u_nominal[0] for i in list_system ]
        #     viable = [ [ pp.zonotope(G=subsystems_output[i]['T'][step],x=subsystems_output[i]['xbar'][step]) for step in range(horizon)] for i in range(len(list_system)) ]
        #     action = [ [ pp.zonotope(G=subsystems_output[i]['M'][step],x=subsystems_output[i]['ubar'][step]) for step in range(horizon-1)] for i in range(len(list_system)) ]
        #     for i in range(len(list_system)):

        #         list_system[i].viable = viable[i]
        #         list_system[i].action = action[i]
        #         list_system[i].u_nominal[0] = u_0[i]

        #     fig, axs = plt.subplots( 2,2 )
        #     for i in range(number_of_subsystems):
        #         list_system[i].omega.color='red'
        #         # sub_sys[i].viable[-1].color='pink'
        #         path = np.array( [ list_system[i].viable[step].x for step in range(horizon)] )
        #         path = np.concatenate( ( list_system[i].x_nominal[0].reshape(1,2) , path ) ,axis=0)

        #         # drawing the parameterized sets

        #         if i == 0 :
        #             j = (0,0)
        #         elif i == 1 :
        #             j = (0,1)
        #         elif i == 2 :
        #             j= (1,0)
        #         elif i == 3 :
        #             j = (1,1) 

        #         # for step in range(1,horizon):

        #         #     # sub_sys[i].viable[step] = pp.pca_order_reduction( sub_sys[i].viable[step] , desired_order=6 )
                    
        #         #     # assump_set = pp.pca_order_reduction( pp.zonotope( G= np.dot( sub_sys[i].omega.G , np.diag( sub_sys[i].alpha_x[step-1]) ) ,  x= sub_sys[i].x_nominal[step] , color = 'yellow') , desired_order=6 )


        #         #     pp.visualize( [ pp.zonotope( G= np.dot( sub_sys[i].omega.G , np.diag( sub_sys[i].alpha_x[step-1]) ) ,  x= sub_sys[i].x_nominal[step] , color = 'yellow') ] 
        #         #                     , ax = axs[j] , title='' , equal_axis=True
        #         #                 )

        #         # pp.visualize([sub_sys[i].omega,*sub_sys[i].viable], ax = axs[j],fig=fig, title='',equal_axis=True)
        #         pp.visualize([list_system[i].X,*list_system[i].viable], ax = axs[j],fig=fig, title='',equal_axis=True)

        #         axs[j].plot( path[:,0], path[:,1] , 'r--')


        # axs[0,0].set_xlim( -0.22 , 0.02 )
        # axs[0,1].set_xlim( -0.01 , 0.03 )
        # axs[1,0].set_xlim( -0.01 , 0.06 )
        # axs[1,1].set_xlim( -0.04 , 0.005 )

        # axs[0,0].set_ylim( -0.01 , 0.12 )
        # axs[0,1].set_ylim( -0.025 , 0.012 )
        # axs[1,0].set_ylim( -0.06 , 0.02 )
        # axs[1,1].set_ylim( -0.012 , 0.03)


        # # for i, ax in enumerate(axs.flat):
        # #     ax.set_title(f'Area {i+1}')


        # axs[0,0].set_xlabel(r'$\delta_1$',FontSize=12 )
        # axs[0,0].set_ylabel(r'$f_1$',FontSize=12 , rotation=0)

        # axs[0,1].set_xlabel(r'$\delta_2$',FontSize=12 )
        # axs[0,1].set_ylabel(r'$f_2$',FontSize=12 , rotation=0)

        # axs[1,0].set_xlabel(r'$\delta_3$',FontSize=12 )
        # axs[1,0].set_ylabel(r'$f_3$',FontSize=12 , rotation=0)

        # axs[1,1].set_xlabel(r'$\delta_4$',FontSize=12 )
        # axs[1,1].set_ylabel(r'$f_4$',FontSize=12 , rotation=0)

        # plt.tight_layout() 
        # plt.pause(0.01)
        # plt.savefig('/Users/kasra/Desktop/prospectus_fig/pospectus_fig{}_{}.png'.format(order , iteration))

        # # plt.show()








        # if h<= 10**(-4) :
        if h<= 0.005:
            flag = 1
            print("HHHHHHHHEEEEEEEEEERRRRRRRREEEEEEEE")
            coefficient = coefficient + delta_coefficient

            u_0 = [ i.u_nominal[0] for i in list_system ]
            viable = [ [ pp.zonotope(G=subsystems_output[i]['T'][step],x=subsystems_output[i]['xbar'][step]) for step in range(horizon)] for i in range(len(list_system)) ]
            action = [ [ pp.zonotope(G=subsystems_output[i]['M'][step],x=subsystems_output[i]['ubar'][step]) for step in range(horizon-1)] for i in range(len(list_system)) ]
            
        elif h!=0 and flag==1 and order == order_max: 
            print("sssssssssssssssssss")
            for i in range(len(list_system)):

                list_system[i].viable = viable[i]
                list_system[i].action = action[i]
                list_system[i].u_nominal[0] = u_0[i]
                
            return [ i.u_nominal[0] for i in list_system] , [i.viable for i in list_system] , [i.action for i in list_system]

        else:

            for i in range(len(list_system)):

                # gradients

                grad_alphax = sum([subsystems_output[j]['alpha_alpha_x_grad'][:,i] for j in range(len(list_system))])
                grad_alphau = sum([subsystems_output[j]['alpha_alpha_u_grad'][:,i] for j in range(len(list_system))])

                grad_x = sum([subsystems_output[j]['x_nominal_grad'][:,i] for j in range(len(list_system))])
                grad_u = sum([subsystems_output[j]['u_nominal_grad'][:,i] for j in range(len(list_system))])

                # gradient descent 
                list_system[i].alpha_x = list_system[i].alpha_x - step_size * grad_alphax
                list_system[i].alpha_u = list_system[i].alpha_u - step_size * grad_alphau

                list_system[i].x_nominal[1:] = list_system[i].x_nominal[1:] - step_size * grad_x
                list_system[i].u_nominal = list_system[i].u_nominal - step_size * grad_u
                


                ######################################## Mapping the removed temperarly
                # list_system[i].mapping_alpha_to_feasible_set()            ##############################################?????
                
                # u_0 \in U

                list_system[i].u_nominal[0] = parsi.point_projection_on_set( list_system[i].U  , list_system[i].u_nominal[0] )

                # assumption \subseteq U

                # for step in range(horizon-1):
                    
                #     # control input
                #     list_system[i].alpha_u[step] , list_system[i].u_nominal[step+1] =  parsi.parameter_projection( list_system[i].U , 
                #                                                                 pp.zonotope( x = list_system[i].u_nominal[step+1] , G = list_system[i].theta.G) ,
                #                                                                 list_system[i].alpha_u[step] 
                #                                                                 , center_include = True )

                    # list_system[i].alpha_u[step]  =  parsi.parameter_projection( list_system[i].U , 
                    #                                                             pp.zonotope( x = list_system[i].u_nominal[step+1] , G = list_system[i].theta.G) ,
                    #                                                             list_system[i].alpha_u[step] 
                    #                                                             , center_include = False )

                    # State space FOR NOW IT IS COMMENT, BECAUSE I WANT TO MAKE RCI SET SMALL AND GET INTO IT!
                    # list_system[i].alpha_x[step] =  parsi.parameter_projection( list_system[i].X ,
                    #                                                             pp.zonotope( x = list_system[i].x_nominal[step+1] , G = list_system[i].omega.G) ,
                    #                                                             list_system[i].alpha_x[step]
                    #                                                            )

                

        # if abs(objective_function - objective_function_previous)< 10**(-2) and flag==0:
        if iteration==iteration_max :
            iteration = 0
            order=order+1 ################################################################################
            # step_size=step_size+0.1
            print('order',order)
            # print('step size',step_size)

        iteration += 1
        print('h', h )
        
    return objective_function



def compositional_mpc_initialization(list_system):
    """
    
    """
    for sys in list_system:

        sys.x_nominal[0] = sys.state
        sys.x_nominal = np.delete( sys.x_nominal , 1 ,axis=0)
        sys.x_nominal = np.concatenate( ( sys.x_nominal , sys.omega.x.reshape(1,-1) ) , axis = 0)

        sys.u_nominal = np.delete( sys.u_nominal , 0 ,axis=0)
        sys.u_nominal = np.concatenate( ( sys.u_nominal , sys.theta.x.reshape(1,-1) ) , axis = 0)


        sys.alpha_x = np.delete( sys.alpha_x , 0 , axis=0 )
        sys.alpha_x = np.concatenate( (sys.alpha_x , np.ones(sys.omega.G.shape[1]).reshape(1,-1) ) , axis=0)

        sys.alpha_u = np.delete( sys.alpha_u , 0 , axis=0 )
        sys.alpha_u = np.concatenate( (sys.alpha_u , np.ones(sys.theta.G.shape[1]).reshape(1,-1) ) , axis=0)



def point_projection_on_set( zonotope_set , x ):
    """
    this function returns x_p which is the projection of x in the given set.
    input: 
            zon_set (input set in a form of a zonotope)
            x (a vector)
    output: 
            x_p
    """
    dimension =  zonotope_set.G.shape[0]

    model_projection = Model()

    x_p = np.array( [ model_projection.addVar( lb=-float('inf') , ub= float('inf') ) for _ in range(dimension) ] )
    b = np.array( [ model_projection.addVar( lb=-1, ub=1 ) for _ in range( zonotope_set.G.shape[1] ) ] )

    model_projection.update()

    # x_p \in zonotope_set

    model_projection.addConstrs( ( np.dot( zonotope_set.G[k,:] , b ) + zonotope_set.x[k]   == x_p[k] ) for k in range( dimension ) )

    model_projection.update()
    
    # MSE error

    difference_vector = x - x_p
    
    mse = np.dot( difference_vector.T , difference_vector )

    model_projection.setParam("OutputFlag",False)
    model_projection.setObjective( mse , GRB.MINIMIZE) 
    model_projection.update()

    model_projection.optimize()

    return np.array( [ i.X for i in x_p ] )


def parameter_projection( circumbody_set, inbody_set, alpha , center_include = False ):
    """
    this function returns alpha_p which is the projection of alpha such that inbody_set * diag(alpha) \subseteq circumbody_set.
    input: 
            circumbody_set (in a form of a zonotope)
            inbody_set (in a form of a zonotope)
            alpha (a vector)
    output: 
            alpha_p
    """

    model_projection = Model()

    alpha_p = np.array( [ model_projection.addVar( lb=-float('inf') , ub= float('inf') ) for _ in range( inbody_set.G.shape[1] ) ] )

    # if type( inbody_set.x[0] ) == gurobipy.Var:
    if center_include == True:
        
        center = np.array( [ model_projection.addVar( lb=-float('inf') , ub= float('inf') ) for _ in range( inbody_set.G.shape[0] ) ] )

    else:
        center = inbody_set.x 

    model_projection.update()

    # inbody_set * diag(alpha) \subseteq circumbody_set
    inbody = pp.zonotope( x = center , G = np.dot( inbody_set.G , np.diag( alpha_p ) )  )
    pp.zonotope_subset( model_projection , inbody , circumbody_set , solver='gurobi') 

    model_projection.update()
    
    # MSE error

    # if type( inbody_set.x[0] ) == gurobipy.Var:
    if center_include == True:

        difference_vector = np.hstack( (alpha , inbody_set.x)  ) - np.hstack( ( alpha_p , center) )
    
    else:
        difference_vector = alpha - alpha_p
    
    mse = np.dot( difference_vector.T , difference_vector )

    model_projection.setObjective( mse , GRB.MINIMIZE) 
    model_projection.update()

    model_projection.setParam("OutputFlag",False)
    model_projection.optimize()

    alpha_p = np.array( [ i.X for i in alpha_p ] )

    # if type( inbody_set.x[0] ) == gurobipy.Var:
    if center_include == True:

        return alpha_p , np.array( [ i.X for i in center ] )
    else:
        
        return alpha_p

def is_in_set(zonotopic_set,point):
    """
    Given a zonotopic set, it checks if a point is in the set.
    Inputs:
        zonotopic_set; which must be zonotopic
        point; muct have the same dimension an be a one dimensional vector
    Output:
        True -> if the point is in the set
        model.status -> Otherwise
    """
    model = Model()
    zeta = parsi.be_in_set(model , zonotopic_set , point)
    model.update()
    model.setParam("OutputFlag",False)
    model.optimize()
    if model.status == 2:
        del model
        return True
    else:
        del model
        return model.status

def find_controller( omega , theta , state):
    """
    It return the controller which is in the form x = T \zeta and u = M \zeta
    Inputs:
        omega; which is the rci set, a zonotopic set in the form Z(x_bar,T)
        theta; which the action set in the form Z(u_bar , M)
        state;
    Outputs:
        u; Controller
    """

    model = Model()
    zeta = parsi.be_in_set(model , omega , state)

    model.setParam("OutputFlag",False)
    model.optimize()
    zeta_optimal = np.array([zeta[i].x for i in range(len(zeta))])
    del model

    u = theta.x + np.dot(theta.G , zeta_optimal)
    return u