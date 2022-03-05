"""
@author: kasra
"""
import numpy as np
try:
    import pypolycontain as pp
except:
    raise ModuleNotFoundError("pypolycontain package is not installed correctly")
try:
    import parsi
except:
    raise ModuleNotFoundError("parsi package is not installed properly")
from gurobipy import *

zon_1_G = [
    [3,0],
    [0,3]
]
zon_2_G = [
    [1,0],
    [0,1]
]

model = Model()
z1 = pp.zonotope(x=[0,0], G=zon_1_G)
z2 = pp.zonotope(x=[1,1], G=zon_2_G)
z3 = pp.zonotope(x=[1,1], G=np.hstack(( zon_2_G , np.eye(2))))
z4 = pp.zonotope(x=[3,3], G=zon_2_G)

hd_z1_z3,_ = parsi.hausdorff_distance_condition(model,z1,z3)
hd_z1_z2,_ = parsi.hausdorff_distance_condition(model,z1,z2)
hd_z1_z1,_ = parsi.hausdorff_distance_condition(model,z1,z1)
hd_z2_z1,_ = parsi.hausdorff_distance_condition(model,z2,z1)
hd_z2_z3,_ = parsi.hausdorff_distance_condition(model,z2,z3)
hd_z2_z2,_ = parsi.hausdorff_distance_condition(model,z2,z2)
hd_z3_z1,_ = parsi.hausdorff_distance_condition(model,z3,z1)
hd_z3_z3,_ = parsi.hausdorff_distance_condition(model,z3,z3)
hd_z3_z2,_ = parsi.hausdorff_distance_condition(model,z3,z2)
hd_z1_z4,_ = parsi.hausdorff_distance_condition(model,z1,z4)
hd_z4_z1,_ = parsi.hausdorff_distance_condition(model,z4,z1)

model.setObjective( hd_z1_z1+hd_z1_z2+hd_z1_z3+hd_z2_z1+hd_z2_z2+hd_z2_z3+hd_z3_z1+hd_z3_z2+hd_z3_z3+hd_z1_z4+hd_z4_z1 , GRB.MINIMIZE ) 
model.update()
model.setParam("OutputFlag",False)
model.optimize() 

assert hd_z1_z3.X == 2
assert hd_z1_z2.X == 3
assert hd_z1_z1.X == 0
assert hd_z2_z1.X == 0
assert hd_z2_z3.X == 0
assert hd_z2_z2.X == 0
assert hd_z3_z1.X == 0
assert hd_z3_z2.X == 1
assert hd_z3_z3.X == 0
assert hd_z1_z4.X == 5
assert hd_z4_z1.X == 1





