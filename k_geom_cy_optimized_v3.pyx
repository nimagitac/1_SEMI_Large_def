import numpy as np
cimport numpy as np


cpdef int ij_to_icapt(int number_lobatto_point, int row_num, int col_num):
    '''
    In this function, a simple map is defined. The map transform
    the number of row and column of a node to the general number 
    of node in the element. For example,(0,0)->0, (1, 1)->number_lobatto_point+1.
    
    '''
    cdef int icapt
    icapt = row_num * number_lobatto_point + col_num
    return icapt




def m_i_mtx (np.ndarray[np.float64_t, ndim=1] h_vect, \
            np.ndarray[np.float64_t, ndim=1] dir_t_intp, \
            np.ndarray[np.float64_t, ndim=1] omega_intp, float omega_limit=0.1):
    '''
    In this function the M_I matrix used in the calculation of the inner product of
    an arbitrary vector and the second variation of the director vector, is claculated.
    It is based on Eq . (34) from 
    "A robust non-linear mixed hybrid quadrilateral shell element", Wagner, Gruttman, 2005"
    
    h_vect : is an arbitrary vector. The formulae in the reference based on the calculation of
    second order variation (delta variation) of the director pre-dot product by h_vect, it means
    h_vect_I @ (\\Delta\\delta)d_I = (\\delta)w_I @ M_I(h_vect) @ (\\Delta)w_I and
    (\\Delta)w_I = H_I @ (\\Delta)\\omega_I
    
    dir_t_intp : is the director at the integration point at time t, a_t_3
    
    omega_intp : the omega vector at point I at time t
    
    -Output:
    Is a 3x3 matrix
    '''
    cdef double omega_norm, c3, c11, c_bar10, c10, 
    cdef np.ndarray[np.float64_t, ndim=1] b_i, tt_i
    cdef np.ndarray[np.float64_t, ndim=2] p1, p2, p3, m_i   
    
    omega_norm = np.linalg.norm(omega_intp)
    b_i = np.cross(dir_t_intp, h_vect)
    if omega_norm < omega_limit:
        c3 = 1/6 * (1 + 1/60 * omega_norm**2)
        c11 = -1/360 * (1 + 1/21 * omega_norm**2)
        c_bar10 = 1/6 * (1 + 1/30 * omega_norm**2)
    else:
        c3 = (omega_norm * np.sin(omega_norm) + 2 * (np.cos(omega_norm) -1)) / \
            (omega_norm ** 2 * (np.cos(omega_norm) - 1)) 
        c11 = (4 * (np.cos(omega_norm) - 1) + omega_norm ** 2 + omega_norm * np.sin(omega_norm))/\
              (2 * omega_norm ** 4 * (np.cos(omega_norm) - 1))
        c_bar10 = (np.sin(omega_norm) - omega_norm) / (2 * omega_norm * (np.cos(omega_norm) - 1))
    c10 = c_bar10 * (b_i @ omega_intp) - (dir_t_intp @ h_vect)
    tt_i = -c3 * b_i + c11 * (b_i @ omega_intp) * omega_intp
    
    p1 = 1/2 * (np.outer(dir_t_intp, h_vect) + np.outer(h_vect, dir_t_intp))
    p2 = 1/2 * (np.outer(tt_i, omega_intp) + np.outer(omega_intp, tt_i))
    p3 = c10 * np.eye(3)
    m_i = p1 + p2 + p3
    
    return m_i


def accumulate_updates(int dim, updates):
   
    cdef int icapt, kcapt, pente_icapt, pente_kcapt
    cdef np.ndarray[np.float64_t, ndim = 2] k_geom, lc1_4, lc2_5, lc3_5,\
                    lc4_3
    
    k_geom = np.zeros((5 * dim**2, 5 * dim**2))
    for (icapt, kcapt, lc1_4, lc2_5, lc3_5, lc4_3) in updates:
        pente_icapt = 5 * icapt
        pente_kcapt = 5 * kcapt 
        k_geom[pente_icapt:(pente_icapt + 3), pente_kcapt:(pente_kcapt + 3)] = lc1_4
        k_geom[(pente_icapt + 3):(pente_icapt + 5), pente_kcapt:(pente_kcapt + 3)] = lc2_5
        k_geom[pente_icapt:(pente_icapt + 3), (pente_kcapt + 3):(pente_kcapt + 5)] = lc3_5
        if pente_icapt == pente_kcapt:
            k_geom [(pente_icapt + 3):(pente_icapt + 5), (pente_kcapt + 3):(pente_kcapt + 5)] = lc4_3
    # print("Is k_geom contiguous?", k_geom.flags['C_CONTIGUOUS'])
    return k_geom


###########################################################

# def mult_sc_arr(arr, scalar):



# from cython cimport boundscheck, wraparound

# # Note: Adjust dimensions as needed
# cdef int N = 3
# cdef int M = 2

# @boundscheck(False)
# @wraparound(False)
# def multiply_static_array_by_scalar(double arr[N][M], double scalar):
#     cdef int i, j
#     cdef double result[N][M]

#     # Multiply each element by the scalar
#     for i in range(N):
#         for j in range(M):
#             result[i][j] = arr[i][j] * scalar

#     # Convert the result C array to a Python list for returning
#     py_result = []
#     for i in range(N):
#         row = []
#         for j in range(M):
#             row.append(result[i][j])
#         py_result.append(row)

#     return py_result

##########################################################

# def elem_m_i_mtx_all() unlike elem_t_i_mtx, it seems better that m_i is constructed inside the stiffness_geom_mtx
# @lprf.profile
# @profile
def  geom_stiffness_mtx(int dim, np.ndarray[np.float64_t, ndim=1] lag_xi1,\
                        np.ndarray[np.float64_t, ndim=1] lag_xi2, 
                        np.ndarray[np.float64_t, ndim=2] der_lag2d_dt, \
                        np.ndarray[np.float64_t, ndim=4] elem_h_capt_mtx_all,\
                        np.ndarray[np.float64_t, ndim=4] elem_t_3_mtx_all, \
                        np.ndarray[np.float64_t, ndim=4] elem_t_i_mtx_all,\
                        np.ndarray[np.float64_t, ndim=4] elem_displ_all, \
                        np.ndarray[np.float64_t, ndim=3] elem_updated_dir_all,\
                        np.ndarray[np.float64_t, ndim=2] der_x_t_dt,\
                        np.ndarray[np.float64_t, ndim=1] stress_vect):
    '''
    Some input parameters:
    der_lag2d_dt : is the 2x(number_lobatto_point^2) matrix that contains
                   the total number of the shape functions

    elem_h_capt_mtx_all : Is the matrix that contains all H_I matrices for all 
                        nodes of the element. In fact for each node T_I = W^Tr @ H_I @ T_3I
                        Calculated from elem_hcapt_t3_ti_mtx_all function
    elem_t_3_mtx_all: Is the matrix that contains the T_3I mtx in all the nodes
                       of the element. Calculated from elem_hcapt_t3_ti_mtx_all function
    elem_t_i_mtx_all : Is the matrix that contains the T_I matrix for all 
                      element nodes. T_I matrix in δd_I=T_I δβ_I, β is the rotation
                      vector in nodal coordinate system. Calculated from elem_hcapt_t3_ti_mtx_all function
    
    This function is calculated at each integration point. The coordinates of the integration point
    is implicitly is taken into account considering lag_xi1, lag_xi2, der_lag2d_dt, der_x_dt etc. In fact
    der_lag2d_dt, der_x_dt etc. are calculated at each integration point in the main loop of creating the 
    stiffness matrix of an element.
    
    -Output: 
    Geometrical stiffness claculated at the specified integration point.
    
    '''
    cdef int i, j, r, s, icapt, kcapt
    
    cdef double n11, n22, n12, m11, m22, m12, q1, q2, n_i, d_n_i_dt1, d_n_i_dt2, n_k, d_n_k_dt1, d_n_k_dt2, \
        lc1_1, lc1_2, lc1_3, lc2_1, lc2_2, lc2_3, lc2_4, lc3_1, lc3_2, lc3_3, lc3_4

    cdef np.ndarray[np.float64_t, ndim=1] d_n_dt1, d_n_dt2, d_xt_dt1, d_xt_dt2, h_vect, omega_vect, dirc_t
   
    cdef np.ndarray[np.float64_t, ndim=2] m_i, eye3, t_k, transp_t_i, hcapt_i, \
                 lc4_1, lc4_2, lc4_3, t_i, t_3_i, lc1_4, lc2_5, lc3_5
    
   
    
    # cdef double d_n_dt1[3], d_n_dt2[3], d_xt_dt1[3], d_xt_dt2[3], h_vect[3], omega_vect[3], dirc_t[3]
   
    
    
    # cdef double t_k[3][2]
    # cdef double t_i[3][2]
    # cdef double transp_t_i[2][3]
    # cdef double t_3_i[3][2]
    # cdef double hcapt_i[3][3]
    # cdef double transp_hcapt_i[3][3]
    # cdef double lc1_4[3][3]
    # cdef double lc2_5[2][3]
    # cdef double lc3_5[3][2]
    
    
    
    # k_geom = np.zeros((5 * dim**2, 5 * dim**2))
    eye3 = np.eye(3)
    d_n_dt1 = der_lag2d_dt[0] # Shape function N. Referring to Gruttman 2005
    d_n_dt2 = der_lag2d_dt[1]
    d_xt_dt1 = der_x_t_dt[0]
    d_xt_dt2 = der_x_t_dt[1]
    # n11, n22, n12, m11, m22, m12, q1, q2 = stress_vect
    n11 = stress_vect[0]
    n22 = stress_vect[1]
    n12 = stress_vect[2]
    m11 = stress_vect[3]
    m22 = stress_vect[4]
    m12 = stress_vect[5]
    q1 = stress_vect[6]
    q2 = stress_vect[7]
    updates = []
    for i in range(dim):
        for j in range(dim):
            icapt = ij_to_icapt(dim, i, j)
            hcapt_i = elem_h_capt_mtx_all[i, j] # np.eye(3)
            transp_hcapt_i = np.transpose(hcapt_i)
            t_3_i = elem_t_3_mtx_all[i, j]
            transp_t_3_i = np.transpose(t_3_i)
            t_i = elem_t_i_mtx_all[i, j] # Refering to the calculation of the first variation of the director
            transp_t_i = np.transpose(t_i)            
            n_i = lag_xi2[i] * lag_xi1[j] # N_I, Ith shape function in the formulation  
            d_n_i_dt1 = d_n_dt1[icapt]  
            d_n_i_dt2 = d_n_dt2[icapt]    
            h_vect = m11 * d_xt_dt1 * d_n_i_dt1 + \
                     m22 * d_xt_dt2 * d_n_i_dt2 + \
                     m12 * (d_xt_dt2 * d_n_i_dt1 + d_xt_dt1 * d_n_i_dt2) + \
                     q1 * d_xt_dt1 * n_i + q2 * d_xt_dt2 * n_i            
            omega_vect = elem_displ_all[i, j, 1]  
            dirc_t = elem_updated_dir_all[i, j]  # Director at time t       
            m_i = m_i_mtx(h_vect, dirc_t, omega_vect) # 
            for r in range(dim):
                for s in range(dim):
                    kcapt = ij_to_icapt(dim, r, s)
                    # kronecker_delta = 1 if kcapt == icapt else 0
                    t_k = elem_t_i_mtx_all[r, s] #np.array([[1, 3], [3, 4], [9, 5]])
                    n_k = lag_xi2[r] * lag_xi1[s] # N_K is the kth shape finction
                    d_n_k_dt1 = d_n_dt1[kcapt]
                    d_n_k_dt2 = d_n_dt2[kcapt]
                    
                    # t1_1 = time.time()
                    lc1_1 = n11 * d_n_i_dt1 * d_n_k_dt1 
                    lc1_2 = n22 * d_n_i_dt2 * d_n_k_dt2
                    lc1_3 = n12 * (d_n_i_dt1 * d_n_k_dt2 + d_n_i_dt2 * d_n_k_dt1)
                    lc1_4 = (lc1_1 + lc1_2 +lc1_3) * eye3
                    # k_geom[icapt:(icapt + 3), kcapt:(kcapt + 3)] = lc1_4
                    # t1_2 = time.time()
                    
                    # t2_1 = time.time()                   
                    lc2_1 = m11 * d_n_i_dt1 * d_n_k_dt1
                    lc2_2 = m22 * d_n_i_dt2 * d_n_k_dt2
                    lc2_3 = m12 *(d_n_i_dt1 * d_n_k_dt2 + d_n_i_dt2 * d_n_k_dt1)
                    lc2_4 = q1 * n_i * d_n_k_dt1 + q2 * n_i * d_n_k_dt2
                    lc2_5 =  (lc2_1 + lc2_2 + lc2_3 + lc2_4) * transp_t_i 
                    # k_geom[(icapt + 3):(icapt + 5), kcapt:(kcapt + 3)] = lc2_5
                    # t2_2 = time.time()                      
                    
                    # k_geom[(icapt + 3):(icapt + 5), kcapt:(kcapt + 3)] = \
                    #                            transp_t_i * \
                    #                          (m11 * d_n_i_dt1 * d_n_k_dt1 + \
                    #                           m22 * d_n_i_dt2 * d_n_k_dt2 + \
                    #             m12 *(d_n_i_dt1 * d_n_k_dt2 + d_n_i_dt2 * d_n_k_dt1) + \
                    #                         q1 * n_i * d_n_k_dt1 + q2 * n_i * d_n_k_dt2)
                                               
                    # t3_1 = time.time()                       
                    lc3_1 = lc2_1 # m11 * d_n_i_dt1 * d_n_k_dt1
                    lc3_2 = lc2_2 # m22 * d_n_i_dt2 * d_n_k_dt2
                    lc3_3 = lc2_3 # m12 *(d_n_i_dt1 * d_n_k_dt2 + d_n_i_dt2 * d_n_k_dt1)
                    lc3_4 = q1 * d_n_i_dt1 * n_k + q2 * d_n_i_dt2 * n_k
                    lc3_5 = (lc3_1 + lc3_2 + lc3_3 + lc3_4) * t_k
                    # k_geom[icapt:(icapt + 3), (kcapt + 3):(kcapt + 5)] = lc3_5 
                    # t3_2 = time.time() 
                                              
                    # k_geom[icapt:(icapt + 3), (kcapt + 3):(kcapt + 5)] = \
                    #                         (m11 * d_n_i_dt1 * d_n_k_dt1 + \
                    #                         m22 * d_n_i_dt2 * d_n_k_dt2 + \
                    #             m12 *(d_n_i_dt1 * d_n_k_dt2 + d_n_i_dt2 * d_n_k_dt1) + \
                    #                      q1 * d_n_i_dt1 * n_k + q2 * d_n_i_dt2 * n_k) * t_k 
                    
                    
                    # t4_1 = time.time()
                    if icapt == kcapt:
                        lc4_1 = transp_t_3_i @ transp_hcapt_i
                        lc4_2 = hcapt_i @ t_3_i
                        lc4_3 = lc4_1 @ m_i @ lc4_2
                        # k_geom [(icapt + 3):(icapt + 5), (kcapt + 3):(kcapt + 5)] = lc4_3
                    # t4_2 = time.time()
                    # k_geom[icapt:(icapt + 3), kcapt:(kcapt + 3)] = lc1_4
                    # k_geom[(icapt + 3):(icapt + 5), kcapt:(kcapt + 3)] = lc2_5
                    # k_geom[icapt:(icapt + 3), (kcapt + 3):(kcapt + 5)] = lc3_5
                    updates.append((icapt, kcapt, lc1_4, lc2_5, lc3_5, lc4_3))
                    
                    # k_geom [(icapt + 3):(icapt + 5), (kcapt + 3):(kcapt + 5)] = \
                    #                         kronecker_delta *  transp_t_3_i @ transp_hcapt_i @ m_i @ hcapt_i @ t_3_i
    # k_geom = accumulate_updates(k_geom, updates) #For avooiding thread locking, it is tried to limit the access to large k_geom by creating it outside of the nested loops
    k_geom = accumulate_updates(dim, updates)

    return  k_geom
