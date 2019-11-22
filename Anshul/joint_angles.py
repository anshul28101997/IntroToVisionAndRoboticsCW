import numpy as np
import math
pi = math.pi 

def H_table (tetha): # The Hartenberg convention #tetha, alpha, r, d
    return np.array([
        [(pi/2)+tetha[0], (pi/2), 0, 2],
        [(pi/2)+tetha[1], (pi/2), 0, 0],
        [tetha[2]    , (-1*pi/2), 3, 0],
        [tetha[3]    , 0        , 2, 0]
        ])

def T (param):
    
    tetha = param[0]
    alpha = param[1]
    r = param[2]
    d = param[3] 
    
    c_t = math.cos(tetha)
    s_t = math.sin(tetha)
    c_a = math.cos(alpha)
    s_a = math.sin(alpha)
    
    T_tetha = np.array([
        [c_t, -1*s_t, 0, 0],
        [s_t, c_t, 0, 0],
        [0 , 0, 1, 0],
        [0 , 0, 0, 1]
        ])
    T_d = np.array([ #
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0 ,0, 1, d],
        [0 ,0, 0, 1]
        ])
    T_r = np.array([
        [1, 0, 0, r],
        [0, 1, 0, 0],
        [0 ,0, 1, 0],
        [0 ,0, 0, 1]
        ])
    T_alpha = np.array([ #
        [1 , 0, 0, 0],
        [0 ,c_t, -1*s_t, 0],
        [0 , s_t, c_t, 0],
        [0 , 0, 0, 1]
        ])
    temp = np.eye(4)
    temp = np.matmul (temp, T_tetha)
    temp = np.matmul (temp, T_d)
    temp = np.matmul (temp, T_r)
    temp = np.matmul (temp, T_alpha)
    return temp

def H_transformation (param):
    tetha = param[0]
    alpha = param[1]
    r = param[2]
    d = param[3]
    c_t = math.cos(tetha)
    s_t = math.sin(tetha)
    c_a = math.cos(alpha)
    s_a = math.sin(alpha)
    return np.array([
        [c_t, (-1*s_t*c_a), (s_t*s_a), (r*c_t)],
        [s_t, (c_t*c_a), (-1*c_t*s_a), (r*s_t)],
        [0, s_a, c_a, d], 
        [0,0,0,1]
        ])

def H_matrix(start, end):
    matrix = np.eye(4)
    for i in range(start, end):
        matrix = np.matmul (matrix, H_transformation(H_tab[i]))
    return matrix

def R(start, end):
    matrix = np.eye(4)
    for i in range (start+1, end+1):
        matrix = np.matmul(matrix, (H_transformation(H_tab[i-1])))
    return matrix [0:3,0:3]

def d(start, end):
    matrix = np.eye(4)
    for i in range (start+1, end+1):
        matrix = np.matmul(matrix, (H_transformation(H_tab[i-1])))
    return (matrix [0:3,3]).reshape((3,1))

def Jacobian():
    n = 4
    jacobian = []
    for i in range(0,4):
        my_vec = np.matmul(R(0,i), np.array([[0],[0],[1]])).transpose()[0]
        temp = np.cross(my_vec , (d(0,n)-d(0,i)).transpose()[0])
        jacobian.append(temp)
    jacobian = np.asarray(jacobian)
    jacobian = jacobian.transpose()
    return jacobian

target = np.array([0.19858789, 4.35117671, 3.25728612])
endpos = np.array([0,0,7])
print("target", target)
tetha = np.array([float(0),float(0),float(0),float(0)])
H_tab = H_table(tetha)
for i in range (50):
    dist = target-endpos
    jac = Jacobian()
    jac_inv = np.linalg.pinv(jac)
    delta_tetha = np.matmul(jac_inv, dist)
    index = np.argmax(np.abs(delta_tetha))
    tetha[index] = (float(tetha[index])+float(delta_tetha[index]))
#     tetha = tetha+delta_tetha
    H_tab = H_table(tetha)
    T01 = T(H_tab[0])
    T12 = T(H_tab[1])
    T23 = T(H_tab[2])
    T34 = T(H_tab[3])
    T04 = np.matmul(np.matmul(np.matmul (T01, T12), T23), T34)
    endpos = (T04[0:3,3])
print("final end effector", endpos)
print("tetha", tetha)
print(np.sqrt(np.sum(np.square(np.array(target-endpos)))))
