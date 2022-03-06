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
        list_subsystems; it is a list of LTI systems. Each system MUST have the following values:
            sys.A_ij[j] , sys.B_ij[j] , sys.param_set_X , sys.alpha_x , sys.param_set_U , sys.alpha_u , where j means other subsytems. 
            Each susbystem must have its own sys.param_set_X , sys.alpha_x , sys.param_set_U , sys.alpha_u where other subsystems have access to them
        subsystem_index; None -> if you want to return a list containing the new disturbance for all subsystems
                          int -> subsystem inded, it will return only the new disturbance set for list_subsystems[subsystem_index]
    Outputs:
        disturb; which is a list of augmented disturbance sets in the form of zonotopes for each subsystems, if subsystem_index==None,
                otherwise, only for a single subsystem
    """

    sys_number = len(list_subsystems)

    def aug_disturbance_single_subsystem( i ):

        disturb = list_subsystems[i].W 
        
        for j in range(sys_number):
            if j in list_subsystems[i].A_ij:
                w=pp.zonotope(G= np.dot(np.dot( list_subsystems[i].A_ij[j], list_subsystems[j].param_set_X.G ), np.diag(list_subsystems[j].alpha_x) ) , x= np.dot( list_subsystems[i].A_ij[j], list_subsystems[j].param_set_X.x))
                disturb = disturb+ w
            if j in list_subsystems[i].B_ij:
                w=pp.zonotope(G= np.dot(np.dot( list_subsystems[i].B_ij[j], list_subsystems[j].param_set_U.G), np.diag(list_subsystems[j].alpha_u) ) , x= np.dot( list_subsystems[i].B_ij[j], list_subsystems[j].param_set_U.x))
                disturb = disturb+ w 
        return disturb

    if subsystem_index is not None:
        return aug_disturbance_single_subsystem( subsystem_index )

    else:
        disturbance = [ aug_disturbance_single_subsystem( sys ) for sys in range(sys_number) ]

        return disturbance
