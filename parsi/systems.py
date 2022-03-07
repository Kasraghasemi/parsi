"""
@author: kasra
Defining classes for linear systems both LTV (linear time variant) and LTI (linear time invariant)
"""
import numpy as np
try:
    import pypolycontain as pp
except:
    print('Error: pypolycontain package is not installed correctly') 
try:
    from gurobipy import *
except:
    print('gurobi is not installed correctly') 


Monitor = {}


class Linear_system:
    
    def __init__(self,A,B,W=None,X=None,U=None):
        """
        A and B are the matrices for the dynamics of the linear syst: x^+ = Ax+Bu+w
        W is the disturbance set: w \in W
        X is the state space set: x \in X
        U is the control input set: u \in U
        This class covers both LTI and LTV systems: if teh system is LTV the inputs need to be in the form of lists, otherwise they need to be numpy.
        """
        if type(A)==list and type(B)==list:
            self.sys = 'LTV'                #it's a linear time-variant system

        elif type(A)!=list and type(B)!=list:
            if W == None:
                self.sys = 'non-disturbed'              # For x^+ = Ax+Bu (without given disturbance)
            else:
                self.sys = 'LTI'                #it's a linear time-invariant system
                assert(len(A) == W.x.shape[0]), "The dimension of matrix A should be the same as dimension of W"
                assert(len(B) == W.x.shape[0]), "The dimension of matrix B should be the same as dimension of W"

            assert(len(A) == len(B)), "The number of rows in matrix A and B need to be the same"

        else:
            raise ValueError ('Input areguments need to be in the form of a list for LTV systems and np.ndarray for LTI systems')

        self.A = np.array(A)
        self.B = np.array(B)
        self.W = W              # disturbance
        self.X = X              # admissible state space
        self.U = U              # admissible control input
        self.state = np.zeros(len(A))               # Current state of the system
        self.beta=0.2 if self.sys=='LTI' else None               # for finding RCI
        self.E=True if self.sys=='LTI' else None             # for finding RCI
        self.omega=None             # RCI set
        self.theta=None             # Action set
        self.neighbours=None                # for coupled linear systems
        self.A_ij={}                # A_{ij} for the neighboring subsystems j of the subsystem i
        self.B_ij={}                # B_{ij} for the neighboring subsystems j of the subsystem i
        self.param_set_X=None               # a set for parameterized set for X 
        self.param_set_U=None               # a set for parameterized set for U
        self.alpha_x=None               # The parameters for the paramterized set in state space, which is useful in compositional approach in finding decentralized rci sets
        self.alpha_u=None               # The parameters for the paramterized set in control space, which is useful in compositional approach in finding decentralized rci sets
        self.alpha_center_x=None        # The parameters for the center of the baseline set in state space
        self.alpha_center_u=None        # The parameters for the center of the baseline set in control space
        self.alpha_x_max=None               # allowable maximum amount of alpha_x 
        self.alpha_u_max=None               # allowable maximum amount of alpha_u
        self.x_nominal=None             # The nominal trajectory in mpc 
        self.u_nominal=None             # The nominal control input in mpc
        self.viable=None                # The sequence of viable sets for mpc
        self.action=None                # The sequence of action sets for mpc


    def __repr__(self):
        return self.sys

    def simulate(self,u,step=0):
        """
        Given a control input, it returns the next step by the environment.
        Input: control input for one step
        Output: state of the next step
        """

        if self.sys == 'LTI':
            x_next = np.dot(self.A,self.state)+np.dot(self.B,u)+sample(self.W)
        else:
            x_next = np.dot(self.A[step],self.state)+np.dot(self.B[step],u)+sample(self.W[step])

        self.state = x_next
        return x_next

    def rci(self,order_max=10):
        """
        It finds the rci set and action set for an LTI system
        Inputs: Maximum order for our iterative approach to find an rci set (order_max)
                size='min' or 'max' , it specifies the size of the rci set. It can be either min or max
                obj='include_center' or anything else. It includes the offset from the centers of the admissible sets to the cost.
        Outputs: rci set(omega) and action set(theta)
        """
        import parsi
        omega,theta=parsi.rci(self,order_max=10)
        self.omega=omega
        self.theta=theta
        return omega,theta
    
    def set_alpha_max(self,parameterized_zonotope):
        """
        This function sets the maximum amount for alpha_x and alpha_u which is useful for 
        the compositional approach for finding decentralized rci sets.
        
        Inputs: parameterized_zonotope which has to have the following dictionary form:
                parameterized_zonotope={'x':  ,'u': }
        Outputs: No outputs. It changes self.alpha_x_max and self.alpha_u_max
        """
        if 'x' in parameterized_zonotope:
            self.alpha_x_max=alpha_max(parameterized_zonotope['x'],self.X,solver='gurobi')
        if 'u' in parameterized_zonotope:
            self.alpha_u_max=alpha_max(parameterized_zonotope['u'],self.U,solver='gurobi')

    def parameterized_set_initialization(self,order_max=10):
        """
        It puts parameterized sets in self.param_set_X and self.param_set_U
        """
        
        if self.sys == 'LTI':
            omega , theta = self.rci( order_max=order_max)  
            self.param_set_X = omega            
            self.param_set_U = theta 

        elif self.sys == 'LTV':
            import parsi
            omega , theta = parsi.viable_limited_time( self,horizon = None ,order_max=10,obj=True,algorithm='slow')
            self.param_set_X = omega            
            self.param_set_U = theta   

    def mapping_alpha_to_feasible_set(self):
        """
        This function replaces the the elements of alpha which are larger than alpha_max with their corresponding elements in alpha_max 
        """
        # COMMENT THESE TWO FOLLOWING LINE WHEN USING MPC
        self.alpha_x[self.alpha_x>self.alpha_x_max]=self.alpha_x_max[self.alpha_x>self.alpha_x_max]
        self.alpha_x[self.alpha_x<0]=0

        self.alpha_u[self.alpha_u>self.alpha_u_max]=self.alpha_u_max[self.alpha_u>self.alpha_u_max]
        self.alpha_u[self.alpha_u<0]=0

#sampling from a set represented in zonotope
def sample(zonotope):

    random_box=np.random.uniform(low=-1.0, high=1.0, size=zonotope.G.shape[1])
    output= zonotope.x + np.dot(zonotope.G,random_box)
    return output


# Initilizing the paramterize set in computing decentralized rci sets
def rci_decentralized_initialization(list_system,initial_guess='nominal',order_max=50):
    
    # The intitial guesses are the rci sets without considering the couplings
    if initial_guess=='nominal':
        X_i,U_i=[],[]
        for i in range(len(list_system)):
            omega,theta = list_system[i].rci(order_max=order_max)
            X_i.append(omega)
            U_i.append(theta)
    
    # The intitial guesses are the admissible state and control inputs for each sub-system
    elif initial_guess=='admissible':
        X_i=[sys.X for sys in list_system]
        U_i=[sys.U for sys in list_system]

    return X_i,U_i


# Given to zonotope inbody and circumbody, it finds the biggest alpha such that zonotope(inbody.x,inbody.G * diag(alpha) ) \subset zonotope(circumbody.x,circumbody.G)
def alpha_max(inbody,circumbody,solver='gurobi'):

    model=Model()
    alpha = np.array([model.addVar(lb=0,ub=GRB.INFINITY) for i in range(inbody.G.shape[1])])
    model.update()

    index_zero_vecotor=np.linalg.norm(inbody.G,ord=2,axis=0) < 10**(-10)
    if any(index_zero_vecotor):
        for i in alpha[index_zero_vecotor]:
            model.addConstr(i==1)
        model.update()

    pp.zonotope_subset(model, pp.zonotope(x=inbody.x,G=np.dot(inbody.G,np.diag(alpha))) ,circumbody,solver=solver)               # Putting subset constraints
    model.setObjective( sum(alpha) , GRB.MAXIMIZE )             # Setting the objective function to sum(alpha)
    model.update()
    # Result
    model.setParam("OutputFlag",False)
    model.optimize()
    alpha_max= [ alpha[i].X for i in range(len(alpha)) ]

    return np.array(alpha_max)
         
