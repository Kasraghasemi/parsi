import numpy as np
try:
    import pypolycontain as pp
except:
    print('Error: pypolycontain package is not installed correctly') 
try:
    import parsi
except:
    raise ModuleNotFoundError("parsi package is not installed properly")


number_of_subsystems = 3
horizon = 4

n=2*number_of_subsystems
m=1*number_of_subsystems

A=np.ones((n,n))
B=np.ones((n,m))

for i in range(number_of_subsystems): 
    A[2*i:2*(i+1),2*i:2*(i+1)]= np.array([[1,1],[0,1]]) 
    B[2*i:2*(i+1),i]= np.array([0,1])
A = [A]* horizon
B = [B]* horizon

X_i=pp.zonotope(G=np.eye(2),x=np.zeros(2),color='red')
U_i=pp.zonotope(G=np.eye(1),x=np.zeros(1))
W_i=pp.zonotope(G=np.eye(2)*0.1,x=np.zeros(2))

W=W_i
for _ in range(number_of_subsystems-1):
    W=W**W_i

W = [ W ] * horizon
X = [ pp.zonotope(G=np.eye(n),x=np.zeros(n),color='red') ] * horizon
U = [ pp.zonotope(G=np.eye(m),x=np.zeros(m)) ] * horizon

system=parsi.Linear_system(A,B,W=W,X=X,U=U)

sub_sys=parsi.sub_systems_LTV(
    system,
    partition_A=[2]*number_of_subsystems,
    partition_B=[1]*number_of_subsystems,
    disturbance=[ [W_i for j in range(number_of_subsystems)] for t in range(horizon)], 
    admissible_x=[ [X_i for j in range(number_of_subsystems)] for t in range(horizon)] , 
    admissible_u=[ [U_i for j in range(number_of_subsystems)] for t in range(horizon)]
)
for sys in range(number_of_subsystems):
    assert (sub_sys[sys].A == [[1,1],[0,1]]).all()
    assert (sub_sys[sys].B == [[0],[1]]).all()
    for t in range(horizon):
        assert (sub_sys[sys].W[t].G == np.eye(2)*0.1).all()
        assert (sub_sys[sys].X[t].G == np.eye(2)).all()
        assert (sub_sys[sys].U[t].G == np.eye(1)).all()
        for sys_neighbour in range(number_of_subsystems):
            if sys == sys_neighbour:
                continue
            else:
                assert (sub_sys[sys].A_ij[t][sys_neighbour] == np.ones((2,2))).all()
                assert (sub_sys[sys].B_ij[t][sys_neighbour] == np.ones((2,1))).all()