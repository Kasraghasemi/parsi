"""
@author: kasra
Defining classes for linear systems both LTV (linear time variant) and LTI (linear time invariant)
"""
import numpy as np
try:
    import pypolycontain as pp
except:
    print('Error: pypolycontain package is not installed correctly') 

def concate_dictionaries(list_dic):
    """
    this function gets list of dictionaries and 
    returns a dictionary with the same keys and each key contains a list which is concatenatin of given dixtionaries

    Inputs:
        list_dic; a list of dictionaries which must have the same set of keys
    Output:
        dic
    """

    list_keys = list_dic[0].keys()
    n = len(list_dic)

    dic = {}
    for key in list_keys:
        dic[key]=[]
        for i in range(n):
            dic[key].append(list_dic[i][key])

    return dic


def break_subsystems(list_subsystems, subsystem_index=None):
    """
    The function looks at the coupling effects coming from other subsystems as disturbance, so it breaks the connection among subsystems using contracts.
    Inputs:
        list_subsystems; (1) it is a list of LTI systems. Each system MUST have the following values:
                            sys.A_ij[j] , sys.B_ij[j] , sys.param_set_X , sys.alpha_x , sys.param_set_U , sys.alpha_u , where j means other subsytems. 
                            Each susbystem must have its own sys.param_set_X , sys.alpha_x , sys.param_set_U , sys.alpha_u where other subsystems have access to them
                         (2) sys.X is not a list for the subsystem with index 0, so the code concludes that the parametric sets are time variable with the horizon = len(sys.X)-1
                            In that case, the subsystems can be LTI or LTV and the returing disturbance would a sequence of zonotopes for each subsystem.
        subsystem_index; None -> if you want to return a list containing the new disturbance for all subsystems
                          int -> subsystem inded, it will return only the new disturbance set for list_subsystems[subsystem_index]
    Outputs:
        disturb; which is a list of augmented disturbance sets in the form of zonotopes for each subsystems, if subsystem_index==None,
                otherwise, only for a single subsystem
                
    """

    sys_number = len(list_subsystems)

    def aug_disturbance_single_subsystem_infinite_time( i ):

        disturb = list_subsystems[i].W 
        
        for j in range(sys_number):
            if j in list_subsystems[i].A_ij:
                w=pp.zonotope(G= np.dot(np.dot( list_subsystems[i].A_ij[j], list_subsystems[j].param_set_X.G ), np.diag(list_subsystems[j].alpha_x) ) , x= np.dot( list_subsystems[i].A_ij[j], list_subsystems[j].param_set_X.x))
                disturb = disturb+ w
            if j in list_subsystems[i].B_ij:
                w=pp.zonotope(G= np.dot(np.dot( list_subsystems[i].B_ij[j], list_subsystems[j].param_set_U.G), np.diag(list_subsystems[j].alpha_u) ) , x= np.dot( list_subsystems[i].B_ij[j], list_subsystems[j].param_set_U.x))
                disturb = disturb+ w 
        return disturb


    def aug_disturbance_single_subsystem_time_sequential_param_set( i , t ):

        disturb = list_subsystems[i].W[t]
        
        for j in range(sys_number):

            if j in list_subsystems[i].A_ij[t]:
                A_ij = list_subsystems[i].A_ij[t] if list_subsystems[i].sys=='LTV' else list_subsystems[i].A_ij
                w=pp.zonotope(G= np.dot(np.dot( A_ij[j], list_subsystems[j].param_set_X[t].G ), np.diag(list_subsystems[j].alpha_x[t]) ) , x= np.dot( A_ij[j], list_subsystems[j].param_set_X[t].x))
                disturb = disturb+ w

            if j in list_subsystems[i].B_ij[t]:
                B_ij = list_subsystems[i].B_ij[t] if list_subsystems[i].sys=='LTV' else list_subsystems[i].B_ij
                w=pp.zonotope(G= np.dot(np.dot( B_ij[j], list_subsystems[j].param_set_U[t].G), np.diag(list_subsystems[j].alpha_u[t]) ) , x= np.dot( B_ij[j], list_subsystems[j].param_set_U[t].x))
                disturb = disturb+ w 

        return disturb


    if not isinstance(list_subsystems[0].X, list):
        if subsystem_index is not None:
            disturbance = aug_disturbance_single_subsystem_infinite_time( subsystem_index )
            return disturbance
        else:
            disturbance = [ aug_disturbance_single_subsystem_infinite_time( sys ) for sys in range(sys_number) ]
            return disturbance

    else:
        horizon = len(list_subsystems[0].X) - 1
        if subsystem_index is not None:
            disturbance = [ aug_disturbance_single_subsystem_time_sequential_param_set( subsystem_index , t) for t in range(horizon)]
            return disturbance
        else:
            disturbance = [ [ aug_disturbance_single_subsystem_time_sequential_param_set( sys , t) for t in range(horizon)] for sys in range(sys_number)]
            return disturbance
    
