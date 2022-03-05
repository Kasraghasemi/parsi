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


def break_subsystems(list_subsystems):
    """
    The function looks at the coupling effects coming from other subsystems as disturbance, so it breaks the connection among subsystems using contracts.
    Inputs:
        list_subsystems; it is a list of LTI systems. Each system MUST have the following values:
            sys.A_ij[j] , sys.B_ij[j] , sys.param_set_X , sys.alpha_x , sys.param_set_U , sys.alpha_u , where j means other subsytems. 
            Each susbystem must have its own sys.param_set_X , sys.alpha_x , sys.param_set_U , sys.alpha_u where other subsystems have access to them
    Outputs:
        disturb; which is a least of augmented disturbance sets in the form of zonotopes for each subsystems
    """

    sys_number = len(list_subsystems)
    disturb=[]

    for i in range(sys_number):
        disturb.append(list_subsystems[i].W)
    for i in range(sys_number):
        for j in range(sys_number):
            if j in list_subsystems[i].A_ij:
                w=pp.zonotope(G= np.dot(np.dot( list_subsystems[i].A_ij[j], list_subsystems[j].param_set_X.G ), np.diag(list_subsystems[j].alpha_x) ) , x= np.dot( list_subsystems[i].A_ij[j], list_subsystems[j].param_set_X.x))
                disturb[i] = disturb[i]+ w
            if j in list_subsystems[i].B_ij:
                w=pp.zonotope(G= np.dot(np.dot( list_subsystems[i].B_ij[j], list_subsystems[j].param_set_U.G), np.diag(list_subsystems[j].alpha_u) ) , x= np.dot( list_subsystems[i].B_ij[j], list_subsystems[j].param_set_U.x))
                disturb[i] = disturb[i]+ w
    
    return disturb