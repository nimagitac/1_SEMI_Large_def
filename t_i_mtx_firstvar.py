import numpy as np
import math as math

def skew(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])
    

def r_mtx_i(omega_vector_i):
    '''
    In this function the R_I tensor or large rotation tensor
    is calculated.
    See:
    Eq.(18) of
    "A robust non-linear mixed hybrid quadrilateral shell element, 2005
    W. Wagner, and F. Gruttmann"
    '''
    omega_norm = np.linalg.norm(omega_vector_i)
    skew_omega = skew(omega_vector_i)
    skew_omega_p2 = skew_omega @ skew_omega
    unit_mtx = np.identity(3)
    if omega_norm == 0:
        rc_1 = 1
        rc_2 = 0.5
    else:
        rc_1 = (np.sin(omega_norm) / omega_norm)
        rc_2 = ((1-np.cos(omega_norm)) / omega_norm**2)
    r_mtx = unit_mtx + rc_1*skew_omega + rc_2*skew_omega_p2
    return r_mtx
    
    
def t_3_mtx( a_t_1, a_t_2):
    '''
     This function make t_3 matrix. This matrix is used to transform 
     beta vector which is the rotation vector in nodal coordinate system
     to omega in global coordinate system.
    '''
    t_3 = np.array([[a_t_1[0], a_t_2[0]], [a_t_1[1], a_t_2[1]], [a_t_1[2], a_t_2[2]]])
    return t_3

def t_mtx_i (omega_vector_i, nodes_coorsys_displ_i, intersection = "false"):
    """
    In this function, T_I in first variation of the director vector of shell at 
    the nodal point "i" is calculated.
    initial_nodal_coorsys_i contains a_1_i a_2_i and d_i vectors at time t=0

    See:
     Eq.(28) of
    "A robust non-linear mixed hybrid quadrilateral shell element, 2005
    W. Wagner1, and F. Gruttmann"
    -OUTPUT:
    a 3x3 matrix T_I

    """
    unit_mtx = np.identity(3)
    skew_omega = skew(omega_vector_i)
    skew_omega_p2 = skew_omega @ skew_omega
    if intersection == "false":
        a_0_1 = nodes_coorsys_displ_i[0]
        a_0_2 = nodes_coorsys_displ_i[1]
        a_0_3 = nodes_coorsys_displ_i[2]
        vdir_0 = a_0_3
        
        omega_norm = np.linalg.norm(omega_vector_i)
        r_mtx = r_mtx_i(omega_vector_i)
        
        a_t_1 = r_mtx @ a_0_1
        a_t_2 = r_mtx @ a_0_2
        a_t_3 = r_mtx @ a_0_3
        
        w_capt = skew(a_t_3)
        print(w_capt)
        t_3 = t_3_mtx( a_t_1, a_t_2)
        print(t_3)
        if omega_norm == 0:
            tc_1 = 1/2
            tc_2 = 1/6
        else:
            tc_1 = (1-np.cos(omega_norm)) / omega_norm**2
            tc_2 = (omega_norm - np.sin(omega_norm)) / omega_norm**3
            
        h_capt = unit_mtx + tc_1 * skew_omega + tc_2 * skew_omega_p2
        t_mtx = np.transpose(w_capt) @ h_capt @ t_3
        
    else:
        None# To be added
        
    return t_mtx








if __name__ == "__main__":
    
    # r_mtx = r_mtx_i(np.array([0.01, 0.02, 0.03]))
    # print(r_mtx)
    # a = np.array([0.1, 0.2, -0.3])
    # print(r_mtx @ a)
    c_sys = np.array([[1, 3, 5], [-2, 3, 2], [1, 1, 2]])
    print(r_mtx_i([0.1, 0.2, 0.1]))
    print(t_mtx_i([0.1, 0.2, 0.1], c_sys))