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

def rci(system,order_max=10,size='min',obj='include_center'):
    """
    Given a LTI system, this function returns a rci set and its action set.
    """
    n= system.A.shape[0]
    parsi.Monitor['time_centralized_centralized'] = 0

    for order in np.arange(1, order_max, 1/n):
        
        prog=MP.MathematicalProgram()
        var=parsi.rci_constraints(prog,system,T_order=order)
        T=var['T'].flatten()

        #Defining the objective function
        #objective function for minimizing the size of the RCI set
        if size=='min':
            prog.AddQuadraticErrorCost(
            Q=np.eye(len(T)),
            x_desired=np.zeros(len(T)),
            vars=T)
        #objective function for minimizing the distance between the RCI set and the set of admissible states
        if obj=='include_center':
            prog.AddQuadraticErrorCost(
            Q=np.eye(len(var['T_x'])),
            x_desired=system.X.x,
            vars=var['T_x'])

        start_time = timer()

        #Result 
        result=gurobi_solver.Solve(prog)    

        end_time = timer()

        parsi.Monitor['time_centralized_centralized'] = parsi.Monitor['time_centralized_centralized'] + (end_time - start_time)

        #print('result',result.is_success())
        beta = system.beta if not 'beta' in var else result.GetSolution(var['beta'])

        if result.is_success():
            T_x=result.GetSolution(var['T_x'])
            M_x=result.GetSolution(var['M_x'])
            omega=pp.zonotope(G=result.GetSolution(var['T'])/(1-beta),x=T_x)
            theta=pp.zonotope(G=result.GetSolution(var['M'])/(1-beta),x=M_x)
            return omega,theta
        else:
            del prog
            continue    
    print("Infeasible:We couldn't find any RCI set. You can change the order_max or system.beta and try again.")
    return None,None


def viable_limited_time(system,order_max=10,size='min',obj='include_center',algorithm='slow'):
    """
    Given a LTV system, this function returns a limited time vaibale set and its action set.
    """
    number_of_steps=len(system.A)
    n= system.A[0].shape[0]
    
    for order in np.arange(1, order_max, 1/n):
        print('order',order)
        prog=MP.MathematicalProgram()
        var=parsi.viable_constraints(prog,system,T_order=order,algorithm=algorithm)
        T=[var['T'][i].flatten() for i in range(number_of_steps)]

        #Defining the objective function
        #objective function for minimizing the size of the RCI set
        if size=='min':
            [prog.AddQuadraticErrorCost(
            Q=np.eye(len(T[i])),
            x_desired=np.zeros(len(T[i])),
            vars=T[i]) for i in range(number_of_steps)]
        #objective function for minimizing the distance between the RCI set and the set of admissible states
        if obj=='include_center':
            [prog.AddQuadraticErrorCost(
            Q=np.eye(len(var['T_x'][i])),
            x_desired=system.X[i].x,
            vars=var['T_x'][i]) for i in range(number_of_steps)]

        #Result
        result=gurobi_solver.Solve(prog)    
        print('result',result.is_success())
        if result.is_success():
            T_x=[result.GetSolution(var['T_x'][i]) for i in range(number_of_steps)]
            M_x=[result.GetSolution(var['M_x'][i]) for i in range(number_of_steps-1)]
            omega=[pp.zonotope(G=result.GetSolution(var['T'][i]),x=T_x[i]) for i in range(number_of_steps)]
            theta=[pp.zonotope(G=result.GetSolution(var['M'][i]),x=M_x[i]) for i in range(number_of_steps-1)]
            return omega,theta
        else:
            del prog
            continue    
    print("Infeasible:We couldn't find any time_limited viable set. You can change the order_max and try again.")
    return None,None


def sub_systems(system,partition_A,partition_B,disturbance=True , admissible_x=True , admissible_u=True ):
    """
    The input is a large system and partition over system.A 
    example: [2,3,1] creates A_{11}=2*2, A_{22}=3*3, and A_{33}=1*1
    """
    assert(len(partition_A)==len(partition_B)), "length of vector partition_A has to be equal to the length of the vector partition_B"
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


def decentralized_rci(list_system,method='centralized',initial_guess='nominal',size='min',solver='drake',order_max=30):
    """
    Given a set of coupled linear sub-systems
    """
    if solver=='drake' and method=='centralized':
        omega,theta = decentralized_rci_centralized_drake(list_system,initial_guess=initial_guess,size=size,order_max=order_max)

    elif solver=='gurobi' and method=='centralized':
        omega,theta,_,_=decentralized_rci_centralized_gurobi(list_system,initial_guess=initial_guess,size=size,order_max=order_max)
    
    for i in range(len(list_system)):
        list_system[i].omega=omega[i]
        list_system[i].theta=theta[i]

    return omega,theta


def decentralized_rci_centralized_drake(sys,initial_guess='nominal',size='min',order_max=10):
    from copy import copy,deepcopy
    for i in sys:
        i.E=False               #Setting all E=False
    
    number_of_subsys= len(sys)

    if initial_guess=='nominal':
        X_i,U_i=[],[]
        for i in range(number_of_subsys):
            omega,theta = rci(sys[i])
            X_i.append(omega)
            U_i.append(theta)
        
        number_of_columns=[sys[i].W.G.shape[1] for i in range(number_of_subsys)]
        for i in range(number_of_subsys):
            for j in range(number_of_subsys):
                if j in sys[i].A_ij:
                    number_of_columns[i]=number_of_columns[i]+ X_i[j].G.shape[1]
                if j in sys[i].B_ij:
                    number_of_columns[i]=number_of_columns[i]+ U_i[j].G.shape[1]
        
    n=[len(sys[i].A) for i in range(number_of_subsys)]
    disturb=[]
    for i in range(number_of_subsys):
        disturb.append(sys[i].W)
    W_i=deepcopy(disturb)

    for order in np.arange(1, order_max, 1/max(n)):
        
        prog= MP.MathematicalProgram()
        
        #Disturbance over each sub-system
        d_aug=[ prog.NewContinuousVariables( n[i], number_of_columns[i], 'd') for i in range(number_of_subsys) ]
        d_aug_x=[prog.NewContinuousVariables( n[i],'d_center') for i in range(number_of_subsys) ]

        #setting the disturbance for each sub-system
        for i in range(number_of_subsys):                
            sys[i].W = pp.zonotope(G=d_aug[i],x=d_aug_x[i])

        #rci_constraints
        var= [parsi.rci_constraints(prog,sys[i],T_order=order) for i in range(number_of_subsys)]

        #Correctness Criteria
        alpha_x = [pp.zonotope_subset(prog,pp.zonotope(G= var[i]['T'], x=var[i]['T_x']), X_i[i] , alpha='vector')[2] for i in range(number_of_subsys)]
        [prog.AddBoundingBoxConstraint(0,np.inf,alpha_x[i]) for i in range(number_of_subsys)]
        alpha_u = [pp.zonotope_subset(prog,pp.zonotope(G= var[i]['M'], x=var[i]['M_x']), U_i[i] , alpha='vector')[2] for i in range(number_of_subsys)]
        [prog.AddBoundingBoxConstraint(0,np.inf,alpha_u[i]) for i in range(number_of_subsys)]

        #Computing the disturbance set
        
        for i in range(number_of_subsys):
            for j in range(number_of_subsys):
                if j in sys[i].A_ij:
                    w=pp.zonotope(G= np.dot(np.dot( sys[i].A_ij[j], X_i[j].G), np.diag(alpha_x[j]) ) , x= np.dot( sys[i].A_ij[j], X_i[j].x))
                    disturb[i] = disturb[i]+ w
                if j in sys[i].B_ij:
                    w=pp.zonotope(G= np.dot(np.dot( sys[i].B_ij[j], U_i[j].G), np.diag(alpha_u[j]) ) , x= np.dot( sys[i].B_ij[j], U_i[j].x))
                    disturb[i] = disturb[i]+ w
        print("THE ORDER IS=",order)
        
        #Disturbance
        [prog.AddLinearConstraint(np.equal(d_aug_x[i], disturb[i].x,dtype='object').flatten()) for i in range(number_of_subsys)]
        [prog.AddLinearConstraint(np.equal(d_aug[i], disturb[i].G,dtype='object').flatten()) for i in range(number_of_subsys)]

        #Objective function
        if size=='min':
            [prog.AddLinearCost(np.ones(len(alpha_x[i])), 0, alpha_x[i]) for i in range(number_of_subsys)]
        elif size=='max':
            [prog.AddLinearCost(-1 *np.ones(len(alpha_x[i])), 0, alpha_x[i]) for i in range(number_of_subsys)]

        #Result
        result=gurobi_solver.Solve(prog)    
        print('result',result.is_success())
        if result.is_success():
            T_x=[result.GetSolution(var[i]['T_x']) for i in range(number_of_subsys)]
            M_x=[result.GetSolution(var[i]['M_x']) for i in range(number_of_subsys)]
            omega=[pp.zonotope(G=result.GetSolution(var[i]['T']),x=T_x[i]) for i in range(number_of_subsys)]
            theta=[pp.zonotope(G=result.GetSolution(var[i]['M']),x=M_x[i]) for i in range(number_of_subsys)]

            for i in range(number_of_subsys):
                sys[i].W=W_i[i]

            return omega,theta
        else:
            del prog
            disturb=deepcopy(W_i)
            #print('W_i2',W_i[0].G.shape)
            continue    
    print("Infeasible:We couldn't find a set of decentralized rci sets. You can change the order_max and try again.")
    return None,None


def decentralized_rci_centralized_gurobi(list_system,initial_guess='nominal',size='min',order_max=30):

    number_of_subsys=len(list_system)
    n=[len(list_system[i].A) for i in range(number_of_subsys)]
    m=[list_system[i].B.shape[1] for i in range(number_of_subsys)]

    parsi.Monitor['time_centralized_decentralized'] = []

    for order in np.arange(1, order_max, 1/max(n)):
        
        model= Model()

        xbar,T,ubar,M, alpha_x,alpha_u= parsi.rci_decentralized_constraints_gurobi(model,list_system,T_order=order,initial_guess='nominal')

        # Objective function
        if size == 'min' or size == 'max':
            
            obj = sum( [sum(alpha_x[i]) for i in range(number_of_subsys)] )
            if size == 'max':
                obj = -1 * obj 

        elif type(size) == np.ndarray:
            # This will find the maximum alpha (it is for drawing)
            obj = np.dot( size , np.array(alpha_x[0][0: len(size)]) )

        else:
            obj = 1 

        model.setObjective( obj , GRB.MINIMIZE )
        model.update()

        # Result
        model.setParam("OutputFlag",False)
        model.optimize()
        
        parsi.Monitor['time_centralized_decentralized'].append( model.Runtime )
        
        if model.Status!=2:
            
            del model
            continue
        T_result= [ np.array( [ [ T[sys][i][j].X for j in range(len(T[sys][i])) ] for i in range(n[sys]) ] ) for sys in range(number_of_subsys) ]
        T_x_result = [ np.array( [ xbar[sys][i].X for i in range(n[sys]) ] ) for sys in range(number_of_subsys) ]
        alfa_x = [ np.array( [ alpha_x[sys][i].X for i in range(len(alpha_x[sys])) ] ) for sys in range(number_of_subsys)]
        omega= [ pp.zonotope(x=T_x_result[sys] , G=T_result[sys]) for sys in range(number_of_subsys) ] 


        M_result= [ np.array( [ [ M[sys][i][j].X for j in range(len(M[sys][i])) ] for i in range(m[sys]) ] ) for sys in range(number_of_subsys) ]
        M_x_result= [ np.array( [ ubar[sys][i].X for i in range(m[sys]) ] ) for sys in range(number_of_subsys) ]
        alfa_u = [ np.array( [ alpha_u[sys][i].X for i in range(len(alpha_u[sys])) ] ) for sys in range(number_of_subsys)]
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
    """
    landa_terminal=10                #Terminal cost coefficient
    landa_controller=1

    n = system.A.shape[0]               # Matrix A is n*n
    m = system.B.shape[1]               # Matrix B is n*m
    if x_desired=='origin':
            x_desired=np.zeros(n)

    if system.omega==None and system.theta==None:
        system.rci()
    omega,theta=system.omega , system.theta

    prog=MP.MathematicalProgram()
    x,u=parsi.mpc_constraints(prog,system,horizon=horizon,hard_constraints=False)

    #Controller is in a shape of x=T_x + T zeta, so u=M_x + M zeta
    zeta =np.array([pp.be_in_set(prog,x[:,i],omega) for i in range(horizon)]).T                #Last one does not count
    prog.AddLinearConstraint( np.equal( theta.x.reshape(-1,1)+np.dot(theta.G,zeta) , u ,dtype='object').flatten() )

    #Objective
    
    #Cost Over x
    prog.AddQuadraticErrorCost(
    Q=np.eye(n*(horizon-1)),
    x_desired=np.tile(x_desired,horizon-1),
    vars=x[:,1:-1].T.flatten())

    #Terminal Cost
    prog.AddQuadraticErrorCost(
    Q=landa_terminal*np.eye(n),
    x_desired=x_desired,
    vars=x[:,-1].flatten())

    #Energy Cost
    prog.AddQuadraticErrorCost(
    Q=landa_controller*np.eye(m*horizon),
    x_desired=np.zeros(m*horizon),
    vars=u.flatten())

    #Result
    result=gurobi_solver.Solve(prog)    

    if result.is_success():
        u_mpc=result.GetSolution(u[:,0])
        return u_mpc


def compositional_decentralized_rci(list_system,initial_guess='nominal',initial_order=2,step_size=0.1,alpha_0='random',order_max=100 , iteration_max=1000):
    """
    This function is for compositional computation of decentralized rci sets.
    Input:  list of LTI systems
            initial_guess for paramterized sets
            size= minimal approximation
            maximum order to try to find the T and M
    Output: Omega and Theta
    """
    order=initial_order

    for i in list_system:
        i.parameterized_set_initialization()
        i.set_alpha_max({'x': i.param_set_X, 'u':i.param_set_U})

    parsi.Monitor['time_compositional'] = [ [] for i in range( len(list_system) ) ] 
    
    ######################################################################################################################
    # import matplotlib.pyplot as plt
    # from math import ceil
    # number_of_subsystems=len(list_system)
    # cols=5
    # fig, axs = plt.subplots(int(ceil(number_of_subsystems / cols)),cols)
    # for i in range(number_of_subsystems):
    #     list_system[i].X.color='red'
    #     r=i//cols
    #     c=i%cols
    #     pp.visualize([list_system[i].X,list_system[i].omega], ax = axs[r,c],fig=fig, title='',equal_axis=True)
    # plt.show()
    ######################################################################################################################



    if alpha_0=='random':
        for i in list_system:
            i.alpha_x=np.dot( np.random.rand(len(i.alpha_x_max)) , np.diag(i.alpha_x_max) )
            i.alpha_u=np.dot(np.random.rand(len(i.alpha_u_max)) , np.diag(i.alpha_u_max) )
            i.mapping_alpha_to_feasible_set()
    elif alpha_0=='zero':
        for i in list_system:
            i.alpha_x=np.zeros(len(i.alpha_x_max))
            i.alpha_u=np.zeros(len(i.alpha_u_max))
           
    
    # for monitoring the potential function and alphas (drawing them)
    parsi.Monitor['alpha_x'] = [ [list_system[i].alpha_x] for i in range(len(list_system)) ]
    parsi.Monitor['potential'] = []
    parsi.Monitor['num_iterations'] = 0


    # alpha_x= np.array([i.alpha_x for i in list_system]).reshape(-1)
    # alpha_u= np.array([i.alpha_u for i in list_system]).reshape(-1)

    objective_function=1
    objective_function_previous=2
    iteration=0
    while objective_function>0 or order==order_max:
        
        subsystems_output = [ parsi.potential_function(list_system, system_index, T_order=order, reduced_order=1) for system_index in range(len(list_system)) ]
        objective_function_previous=objective_function
        objective_function=sum([subsystems_output[i]['obj'] for i in range(len(list_system)) ])

        parsi.Monitor['potential'] .append(objective_function)
        parsi.Monitor['num_iterations'] = parsi.Monitor['num_iterations']+1

        if objective_function==0:
            for i in range(len(list_system)):
                list_system[i].omega=pp.zonotope(G=subsystems_output[i]['T'],x=subsystems_output[i]['xbar'])
                list_system[i].theta=pp.zonotope(G=subsystems_output[i]['M'],x=subsystems_output[i]['ubar'])

            parsi.Monitor['order'] = order

            return [i.omega for i in list_system],[i.theta for i in list_system]

        else:
            for i in range(len(list_system)):


                # finding gradients
                grad_x= np.array(sum([subsystems_output[j]['alpha_x_grad'][i] for j in range(len(list_system))]))
                grad_u= np.array(sum([subsystems_output[j]['alpha_u_grad'][i] for j in range(len(list_system))]))


                # gradient descent

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



                # Projection to the valid set of aplha

                # projecting to the maximum allowed amount paramter-wise (conservative)
                # list_system[i].mapping_alpha_to_feasible_set()


                # projecting to the set of valid parameters
                list_system[i].alpha_u =  parsi.parameter_projection( list_system[i].U , list_system[i].theta , list_system[i].alpha_u )
                list_system[i].alpha_x =  parsi.parameter_projection( list_system[i].X , list_system[i].omega , list_system[i].alpha_x )
                

                # this is for tracking alpha and its gradient (for drawing it)    
                parsi.Monitor['alpha_x'][i].append(list_system[i].alpha_x)  
                if iteration == 0 and i == 0:
                    parsi.Monitor['gradient'] = [ subsystems_output[j]['alpha_x_grad'][i] for j in range(len(list_system)) ]

        if abs(objective_function - objective_function_previous)< 10**(-4):

            step_size=step_size+0.1
            
            if iteration == iteration_max:
                order = order + 1
                iteration = 0

        iteration += 1
        print('objective_function',objective_function)
    return objective_function


def shrinking_rci(list_system,reduced_order=2,order_reduction_method='pca'):
    """
    The goal is shrinking the rci sets. It 
    """
    sys_number=len(list_system)

    while any([i.X!=i.omega for i in list_system]):
        print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
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