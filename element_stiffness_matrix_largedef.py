import numpy as np
import lagrange_der as lagd
import t_i_mtx_firstvar as tmf
import cProfile
import line_profiler as lprf
import os
import subprocess
from geomdl import exchange
import time as time
import multiprocessing as mp
from multiprocessing import shared_memory
import multiprocess_distributor as mpdistr
# from functools import lru_cache

import k_geom_cy_optimized_v6




def profile(func):
    '''This function is used for profiling the file.
    It will be used as a decorator.
    output:
    A powershell profling and process.profile file'''
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        value = func(*args, **kwargs)
        pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())
        pr.dump_stats('process.profile')
        return value
    return inner

# @lru_cache(maxsize=400)    
def ij_to_icapt(number_lobatto_point, row_num, col_num):
    '''
    In this function, a simple map is defined. The map transform
    the number of row and column of a node to the general number 
    of node in the element. For example,(0,0)->0, (1, 1)->number_lobatto_point+1.
    
    '''
    icapt = row_num * number_lobatto_point + col_num
    return icapt



def der_lag2d_dxi_node_i(number_lobatto_point, lag_xi1, lag_xi2, der_lag_dxi1,\
                  der_lag_dxi2):
    '''
    Basic function. (Will be used repeatedly and therefore should be
    calculated outside of funtions and get imported avoiding multiple calculation
    of the same thing)
    
    This function calculate the derivatives of 2D lagrange shape functions.
    The first row is d(lag2d)/dxi1 and the second is d(lag2d)/dxi2
    -Output:
    Is a 2 x number_lobatto_points*2 matrix
    '''
    der_lag2d_dxi = np.zeros((2, number_lobatto_point**2))
    for i in range(number_lobatto_point):
        for j in range(number_lobatto_point):
            icap = ij_to_icapt(number_lobatto_point, i, j)
            der_lag2d_dxi[0, icap] = lag_xi2[i] * der_lag_dxi1[j]
            der_lag2d_dxi[1, icap] = der_lag_dxi2[i] * lag_xi1[j]
    return der_lag2d_dxi


def der_lag2d_dt_node_i(number_lobatto_point, lag_xi1, lag_xi2, der_lag_dxi1,\
                  der_lag_dxi2, jacobian_at_node):
    '''
    Basic function. 
    
     This function calculate the derivatives of 2D lagrange shape functions with
     repsect to the coordinates of local nodal coordinate systemat each node regarding
     to the Jacobian matrix and calculation of the der_lagd2d_dxi.
    The first row is d(lag2d)/dti1 and the second is d(lag2d)/dti2
    -Output:
    Is a 2 x number_lobatto_points^2 matrix
    '''
    der_lag2d_dxi_node = der_lag2d_dxi_node_i(number_lobatto_point, \
                         lag_xi1, lag_xi2, der_lag_dxi1, der_lag_dxi2)
    inv_jac = np.linalg.inv(jacobian_at_node)
    der_lag_2d_dt = inv_jac @ der_lag2d_dxi_node
    
    return der_lag_2d_dt
    


# @profile
def der_x_t_dt(number_lobatto_point, der_lag2d_dt,\
                    elem_x_0_all, elem_displ_all):
    '''
    Basic function.
    
    In this function the derivatives of x (position vector at time 't')
    are calculated at the specified integration point. The specific integration
    point is introduced through der_lag2d_dti which is calculated according to the
    input coordinate.
    dim is the number of the Lobatto points.
    elem_x_0: A dim*dim*3 matrix which contains the initial coordinates of the nodes
    elem_displ_all: A dim*dim*2*3 matrix which contain at [i,j,0] 'u'(displacement vector)
    and at [i, j, 1] the 'beta' (rotation vector)
    -Output:
    is a 2 x 3 matrix. The first element is dx/dt1 and the second is dx/dt2 at the 
    specific integration points.
    '''
    dim = number_lobatto_point
    
    elem_x_t = np.zeros((dim, dim, 3))# x = X + u
    der_x_t_dt1 = np.zeros(3)
    der_x_t_dt2 = np.zeros(3)
    # for i in range(dim):# x = X + u
    #     for j in range(dim):
    #         elem_x_t[i, j] = elem_x_0[i, j] + elem_displ[i, j, 0]
    elem_x_t = elem_x_0_all[: , : ] + elem_displ_all[: , : , 0] # x = X + u
              
    for i in range(dim):
        for j in range(dim):
            icapt = ij_to_icapt(dim, i, j)
            der_x_t_dt1 = der_x_t_dt1 + der_lag2d_dt[0, icapt] * elem_x_t[i, j]
            der_x_t_dt2 = der_x_t_dt2 + der_lag2d_dt[1, icapt] * elem_x_t[i, j]
    return np.array([der_x_t_dt1, der_x_t_dt2])

            
def elem_update_dir_all(number_lobatto_point, elem_nodal_coorsys_all, elem_displ_all):
    '''
    Basic function.
    
    In this function, the updated director at time 't' for all of the nodes
    of the element is calculated and stored.
    -Output:
    A num_lobatto x num_lobbato x 3 matrix
    '''
    dim = number_lobatto_point
    elem_updated_dir_all = np.zeros((dim, dim, 3))
    for i in range(dim):
        for j in range(dim):
            omega_vect = elem_displ_all[i, j, 1]
            a_0_3 = elem_nodal_coorsys_all[i, j, 2]
            rot_mtx_i = tmf.r_mtx_node_i(omega_vect)
            a_t_3 = rot_mtx_i @ a_0_3
            elem_updated_dir_all[i, j] = a_t_3
    return elem_updated_dir_all

def der_dir_t_dt(number_lobatto_point, der_lag2d_dt, elem_updated_dir_all):
    '''
    Basic function.
    
    In this function the derivatives of director vector
    are calculated at the specified integration point. The specific integration
    point is introduced through der_lag2d_dti which is calculated according to the
    input coordinate.
    -Output:
    is a 2 x 3 matrix. The first element is da_t_3/dt1 and the second is da_t_3/dt2 at a
    specific integration point.
    '''
    dim = number_lobatto_point
    der_dir_dt1 = np.zeros(3)
    der_dir_dt2 = np.zeros(3)
    for i in range(dim):
        for j in range(dim):
            icapt = ij_to_icapt(dim, i, j)
            a_t_3 = elem_updated_dir_all[i, j]
            der_dir_dt1 = der_dir_dt1 + der_lag2d_dt[0, icapt] * a_t_3
            der_dir_dt2 = der_dir_dt2 + der_lag2d_dt[1, icapt] * a_t_3
    return (der_dir_dt1, der_dir_dt2)
            

def elem_hcapt_t3_ti_mtx_all(number_lobatto_point, elem_nodal_coorsys_all, elem_displ_all):
    '''
    Basic function.
    
    In this function, t_i matrix, which connects the variation of nodal local rotation to
    the variation of director, is calculated and stored at all the integration points of
    the element. According to Eq. (28) in 
    "A robust non-linear mixed hybrid quadrilateral shell element", Wagner, Gruttman, 2005
    -Output:
    Three (dim x dim x 3 x 3),(dim x dim x 3 x 2) and 
    (dim x dim x 3 x 2 matrix) matrices. The first one is 
    H matrxi for all nodes, the second is t_3 matrix for all
    nodes and the last one is t_i matrix for all nodes of an
    element. The definition of these matrices can be found in 
    the mentioned refrence in Eqs. (26), (27) and (34)
    '''
    dim = number_lobatto_point
    elem_h_capt_mtx_all = np.zeros((dim, dim, 3, 3))
    elem_t_3_mtx_all = np.zeros((dim, dim, 3, 2))
    elem_t_i_mtx_all = np.zeros((dim, dim, 3, 2))
    for i in range(dim):
        for j in range(dim):
            omega_vect = elem_displ_all[i, j, 1]
            a_0_1 = elem_nodal_coorsys_all[i, j, 0]
            a_0_2 = elem_nodal_coorsys_all[i, j, 1]
            a_0_3 = elem_nodal_coorsys_all[i, j, 2]
            hcapt_t3_ti = tmf.t_mtx_i (omega_vect, \
                  a_0_1, a_0_2, a_0_3, intersection = "false")
            elem_h_capt_mtx_all[i, j] = hcapt_t3_ti[0]
            elem_t_3_mtx_all[i, j] = hcapt_t3_ti[1]
            elem_t_i_mtx_all[i, j] = hcapt_t3_ti[2]
    return (elem_h_capt_mtx_all, elem_t_3_mtx_all, elem_t_i_mtx_all)


#The xi1 and xi2 are calculated and the for loops for calculting them are inside element_stiffness_function
# @profile       
def b_disp_mtx(lobatto_pw, lag_xi1, lag_xi2, der_lag2d_dt,\
                director_t_intp, der_x_t_dt_intp, der_dir_t_dt_intp,\
                elem_t_i_mtx_all):   
    '''
    This function calculates the b_linear at each integration point. The integration point is
    specified by an external nested loop. each integration point has (i_intp, j_intp)
   
    director_t_intp: The director at the integration point. It is elem_update_dir_all[i_intp, j_intp]
    
    der_x_t_dt_intp : The derivatives of current location vector'x' w.r.t (t1, t2) which are
                      nodal local cartesian coordinatesat the (i_intp, j_intp).
                      It is calculated by using der_x_t_dt function at each (i_intp, j_intp)
                      and imported to this function.
    
    der_dir_dt_intp : The derivatives of current director vector'x' w.r.t (t1, t2) which are 
                      nodal local cartesian coordinates at the (i_intp, j_intp).
                      It is calculated by using der_x_t_dt function at each (i_intp, j_intp)
                      and imported to this function.
                      
    elem_t_i_mtx_all : Is the matrix  with num_node x num_node x 3 x 2 matrix. It contains all
                       T_I matrices for all the nodes of the element.    
                      
    -Output:
    Is 8 x 5x(p+1)^2 matrix b_linear (something like Eq. (29), in
    "A robust non-linear mixed hybrid quadrilateral shell element, 2005
    W. Wagner, and F. Gruttmann") 
                      
    ''' 
    dim = np.shape(lobatto_pw)[0]
    b_linear_intp = np.zeros((8, 5*(dim**2)))
    der_n_dt1 = der_lag2d_dt[0] #ncapt means N, or the shape function. Referring to Gruttman 2005
    der_n_dt2 = der_lag2d_dt[1]
    der_x_dt1 = der_x_t_dt_intp[0]
    der_x_dt2 = der_x_t_dt_intp[1]
    der_dir_dt1 = der_dir_t_dt_intp[0]
    der_dir_dt2 = der_dir_t_dt_intp[1]
    index = 0
    for i in range(dim):
        for j in range(dim):
            icapt = ij_to_icapt(dim, i, j)
            ncapt_icapt = lag_xi2[i] * lag_xi1[j] # ncapt_icapt = N_I in the formulation or the shape function
            t_i_mtx = elem_t_i_mtx_all[i, j]
            b_linear_intp[0, index:index + 3] = der_n_dt1[icapt] * der_x_dt1
            b_linear_intp[1, index:index + 3] = der_n_dt2[icapt] * der_x_dt2
            b_linear_intp[2, index:index + 3] = der_n_dt1[icapt] * der_x_dt2 + \
                                                der_n_dt2[icapt] * der_x_dt1
                                                
            b_linear_intp[3, index:index + 3] = der_n_dt1[icapt] * der_dir_dt1
            b_linear_intp[3, index + 3:index + 5] = der_n_dt1[icapt] *\
                (der_x_dt1 @ t_i_mtx)
            
            b_linear_intp[4, index:index + 3] = der_n_dt2[icapt] * der_dir_dt2
            b_linear_intp[4, index + 3:index + 5] = der_n_dt2[icapt] *\
                (der_x_dt2 @ t_i_mtx)
            
            b_linear_intp[5, index:index + 3] = der_n_dt1[icapt] * der_dir_dt2 +\
                                                der_n_dt2[icapt] * der_dir_dt1
            b_linear_intp[5, index + 3:index + 5] =\
                                der_n_dt1[icapt] * (der_x_dt2 @ t_i_mtx) + \
                                der_n_dt2[icapt] * (der_x_dt1 @ t_i_mtx) 
            
            b_linear_intp[6, index:index + 3] = der_n_dt1[icapt] * director_t_intp
            b_linear_intp[6 , index + 3:index + 5]= \
                            ncapt_icapt * (der_x_dt1 @ t_i_mtx)
            
            b_linear_intp[7, index:index + 3] = der_n_dt2[icapt] * director_t_intp
            b_linear_intp[7, index + 3:index + 5] = \
                            ncapt_icapt * (der_x_dt2 @ t_i_mtx)
            
            index = index + 5
    return b_linear_intp

            

############################### k_geom #####################################


def der_dir_0_dt(number_lobatto_point, der_lag2d_dti, elem_nodal_coorsys_all):
    '''
    Basic function.
    
    In this function the derivatives of director vector in undeformed configuration
    are calculated (time = 0) at the specified integration point. The specific integration
    point is introduced through der_lag2d_dti which is calculated according to the
    input coordinate.
    -Output:
    is a 2 x 3 matrix. The first element is da_0_3/dt1 and the second is da_0_3/dt2 at a
    specific integration point.
    '''
    dim = number_lobatto_point
    der_dir_0_dt1 = np.zeros(3)
    der_dir_0_dt2 = np.zeros(3)
    for i in range(dim):
        for j in range(dim):
            icapt = ij_to_icapt(dim, i, j)
            a_0_3 = elem_nodal_coorsys_all[i, j, 2]
            der_dir_0_dt1 = der_dir_0_dt1 + der_lag2d_dti[0, icapt] * a_0_3
            der_dir_0_dt2 = der_dir_0_dt2 + der_lag2d_dti[1, icapt] * a_0_3
    return (der_dir_0_dt1, der_dir_0_dt2)


def der_x_0_dt(number_lobatto_point, der_lag2d_dti, elem_x_0_all):
    '''
    Basic function.
    
    In this function the derivatives of X (shown by x_0, position vector 
    in undeformed configuration i.e. at time 't = 0') are calculated
    at the specified integration point. The specific integration point is
    introduced through der_lag2d_dti which is calculated according to the
    input coordinate.
    dim is the number of the Lobatto points.
    -Output:
    is a 2 x 3 matrix. The first element is dX/dt1 and the second is dX/dt2 at the 
    specific integration points.
    '''
    dim = number_lobatto_point
   
    der_x_0_dt1 = np.zeros(3)
    der_x_0_dt2 = np.zeros(3)       
    for i in range(dim):
        for j in range(dim):
            icapt = ij_to_icapt(dim, i, j)
            der_x_0_dt1 = der_x_0_dt1 + der_lag2d_dti[0, icapt] * elem_x_0_all[i, j]
            der_x_0_dt2 = der_x_0_dt2 + der_lag2d_dti[1, icapt] * elem_x_0_all[i, j]
    return (der_x_0_dt1, der_x_0_dt2)
 
 
def strain_vector (der_x_0_dt, der_x_t_dt, \
                   dir_0_intp, dir_t_intp, \
                   der_dir_0_dt, der_dir_t_dt):
    '''
    der_x_0_dt : is the derivatives of the initial coordinate of a material point
                 with respect to nodal local coordinate system, t1 and t2,
                 at an integration point.(in our work nodal local system conincides
                 with lamina system)
    der_x_t_dt : is the dervatives of the coordinate of a material point at the 
                 deformed configuration with respect to the initial nodal local
                 coordinate system, t1 and t2, at an integration point (in our work, nodal local
                 system coincides with lamina system)
    dir_0_intp : is the director at the integration point at time 0, a_0_3
    dir_t_intp : is the director at the integration point at time  t, a_t_3
    
    der_dir_0_dt : the derivatives of the inital director vector 
                    at undeformed configuration with respect to t1 and  t2
    
    der_dir_t_dt : the derivatives of the director at deformed configuration
                   with respect to t1 and  t2
                    
    -Output:
    an strain vector 8 elements  to be used to calculate the stresses
    
    '''
    str_vec = np.empty(8)
    
    d_x0_dt1 = der_x_0_dt[0]
    d_x0_dt2 = der_x_0_dt[1]
    d_xt_dt1 = der_x_t_dt[0]
    d_xt_dt2 = der_x_t_dt[1]
    
    dir_0 = dir_0_intp
    dir_t = dir_t_intp
    
    d_d0_dt1 = der_dir_0_dt[0]
    d_d0_dt2 = der_dir_0_dt[1]   
    d_dt_dt1 = der_dir_t_dt[0]
    d_dt_dt2 = der_dir_t_dt[1]
    
    
    str_vec[0] = 0.5 * (d_xt_dt1 @ d_xt_dt1 - d_x0_dt1 @ d_x0_dt1)
    str_vec[1] = 0.5 * (d_xt_dt2 @ d_xt_dt2 - d_x0_dt2 @ d_x0_dt2)
    str_vec[2] = d_xt_dt1 @ d_xt_dt2 - d_x0_dt1 @ d_x0_dt2
    
    str_vec[3] = d_xt_dt1 @ d_dt_dt1 - d_x0_dt1 @ d_d0_dt1
    str_vec[4] = d_xt_dt2 @ d_dt_dt2 - d_x0_dt2 @ d_d0_dt2
    str_vec[5] = d_xt_dt1 @ d_dt_dt2 + d_xt_dt2 @ d_dt_dt1 - \
                 (d_x0_dt1 @ d_d0_dt2 + d_x0_dt2 @ d_d0_dt1)
                 
    str_vec[6] = d_xt_dt1 @ dir_t - d_x0_dt1 @ dir_0
    str_vec[7] = d_xt_dt2 @ dir_t - d_x0_dt2 @ dir_0
    
    return str_vec
    

def elastic_matrix(elastic_modulus, nu, thk):
    '''
    In this function elastic matrix for shell is generated.
    -Output:
    A 8x8 matrix
    '''
    ks = 5/6 # Timoschenko shear correction factor
    elastic_mtx = np.zeros((8, 8))
    cp = elastic_modulus / (1 - nu**2) * np.array([[1, nu, 0],[nu, 1, 0],[0, 0, (1-nu)/2]])
    cs = ks * elastic_modulus / (2 * (1 + nu)) * np.array([[1, 0], [0, 1]])
    elastic_mtx[0:3, 0:3] = thk * cp
    elastic_mtx[3:6, 3:6] = thk**3 / 12 * cp
    elastic_mtx[6:8, 6:8] = thk * cs
    return elastic_mtx
    


def stress_vector(strain_vect, elastic_mtx):
    '''
    In this function, the stress vector calculated. It is:
    [n11, n22, n12, m11, m22, m12, q1, q2]    
    '''
    stress_vect = elastic_mtx @ strain_vect
    return stress_vect

def m_i_mtx (h_vect, dir_t_intp, omega_intp, omega_limit=0.1):
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

def m_i_mtx_2 (h_vect, dir_t_intp, omega_intp, omega_limit=0.1):
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
    omega_norm = np.linalg.norm(omega_intp)
    sin_omega = np.sin(omega_norm)
    cos_omega = np.cos(omega_norm)
    b_i = np.cross(dir_t_intp, h_vect)
    if omega_norm < omega_limit:
        c3 = 1/6 * (1 + 1/60 * omega_norm**2)
        c11 = -1/360 * (1 + 1/21 * omega_norm**2)
        c_bar10 = 1/6 * (1 + 1/30 * omega_norm**2)
    else:
        c3 = (omega_norm * sin_omega + 2 * (np.cos_omega -1)) / \
            (omega_norm ** 2 * (cos_omega - 1)) 
        c11 = (4 * (cos_omega - 1) + omega_norm ** 2 + omega_norm * sin_omega)/\
              (2 * omega_norm ** 4 * (cos_omega - 1))
        c_bar10 = (sin_omega - omega_norm) / (2 * omega_norm * (sin_omega - 1))
    c10 = c_bar10 * (b_i @ omega_intp) - (dir_t_intp @ h_vect)
    tt_i = -c3 * b_i + c11 * (b_i @ omega_intp) * omega_intp
    
    p1 = 1/2 * (np.outer(dir_t_intp, h_vect) + np.outer(h_vect, dir_t_intp))
    p2 = 1/2 * (np.outer(tt_i, omega_intp) + np.outer(omega_intp, tt_i))
    p3 = c10 * np.eye(3)
    m_i = p1 + p2 + p3
    
    return m_i


# def kronecker_delta(i, j):
#     """
#     Compute the Kronecker delta δ_ij.
#     """
#     return 1 if i == j else 0
def accumulate_updates(dim, updates):
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

# def elem_m_i_mtx_all() unlike elem_t_i_mtx, it seems better that m_i is constructed inside the stiffness_geom_mtx
# @lprf.profile
# @profile
def geom_stiffness_mtx(number_lobatto_point, lag_xi1, lag_xi2, der_lag2d_dt, \
                         elem_h_capt_mtx_all, elem_t_3_mtx_all, elem_t_i_mtx_all,\
                       elem_displ_all, elem_updated_dir_all, der_x_t_dt, stress_vect):
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
    dim = number_lobatto_point
    # k_geom = np.zeros((5 * dim**2, 5 * dim**2))
    eye3 = np.eye(3)
    d_n_dt1 = der_lag2d_dt[0] # Shape function N. Referring to Gruttman 2005
    d_n_dt2 = der_lag2d_dt[1]
    d_xt_dt1 = der_x_t_dt[0]
    d_xt_dt2 = der_x_t_dt[1]
    n11, n22, n12, m11, m22, m12, q1, q2 = stress_vect
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
                    
                    # pente_icapt = 5 * icapt
                    # pente_kcapt = 5 * kcapt
                    
                    # t1_1 = time.time()
                    lc1_1 = n11 * d_n_i_dt1 * d_n_k_dt1 
                    lc1_2 = n22 * d_n_i_dt2 * d_n_k_dt2
                    lc1_3 = n12 * (d_n_i_dt1 * d_n_k_dt2 + d_n_i_dt2 * d_n_k_dt1)
                    lc1_4 = (lc1_1 + lc1_2 +lc1_3) * eye3
                    # k_geom[pente_icapt:(pente_icapt + 3), pente_kcapt:(pente_kcapt + 3)] = lc1_4
                    # t1_2 = time.time()
                    
                    # t2_1 = time.time()                   
                    lc2_1 = m11 * d_n_i_dt1 * d_n_k_dt1
                    lc2_2 = m22 * d_n_i_dt2 * d_n_k_dt2
                    lc2_3 = m12 *(d_n_i_dt1 * d_n_k_dt2 + d_n_i_dt2 * d_n_k_dt1)
                    lc2_4 = q1 * n_i * d_n_k_dt1 + q2 * n_i * d_n_k_dt2
                    lc2_5 = transp_t_i * (lc2_1 + lc2_2 + lc2_3 + lc2_4)
                    # k_geom[(pente_icapt + 3):(pente_icapt + 5), pente_kcapt:(pente_kcapt + 3)] = lc2_5
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
                    # k_geom[pente_icapt:(pente_icapt + 3), (pente_kcapt + 3):(pente_kcapt + 5)] = lc3_5 
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
                        #  k_geom [(pente_icapt + 3):(pente_icapt + 5), (pente_kcapt + 3):(pente_kcapt + 5)] = lc4_3
                    updates.append((icapt, kcapt, lc1_4, lc2_5, lc3_5, lc4_3))
                    
                    # k_geom [(icapt + 3):(icapt + 5), (kcapt + 3):(kcapt + 5)] = \
                    #                         kronecker_delta *  transp_t_3_i @ transp_hcapt_i @ m_i @ hcapt_i @ t_3_i
    
    k_geom = accumulate_updates(dim, updates)

    return  k_geom

      
                                                
##############################################################################################################
# @lprf.profile
# @profile
def element_stiffness_mtx(lobatto_pw, elem_x_0_coor_all, \
                          elem_nodal_coorsys_all, elem_jacobian_all,\
                          elem_displ_all, elastic_modulus, nu, thk ):
    '''
    Some input parameters:
    elem_x_0_coor_all(dim, dim, 3): The coordinates of all Lobatto points of the element at t = 0
    elem_nodal_coorsys_all (dim, dim, 3, 3): It contains coordinate system (a_0_1, a_0_2, d=a_0_3) at all
                            Lobatto points
    elem_jacobian_all (dim, dim, 2, 2) : It contains all  2 x 2 Jacobian matrix at each Lobatto point 
    elem_displ_all(dim, dim, 2, 3) : It contains [[u1, u2, u3], [β1, β2]] of the element at all nodes
    
    In this function, the stiffness matrix of the element is built by adding material stiffness matrix
    b_tr_b and geometrical stiffness matrix, k_geom.
    
    -Output: 
    Is the element stiffness matrix
    
    ''' 
    dim = lobatto_pw.shape[0]
    stiff_mtx = np.zeros((5 * dim**2, 5 * dim**2)) # 5 DOF at each node              
    elastic_mtx = elastic_matrix(elastic_modulus, nu, thk)
    elem_updated_dir_all = elem_update_dir_all(dim, elem_nodal_coorsys_all, elem_displ_all) 
    elem_ht3ti_mtx = elem_hcapt_t3_ti_mtx_all(dim, elem_nodal_coorsys_all, elem_displ_all)
    elem_hcapt_mtx_all = elem_ht3ti_mtx[0]
    elem_t_3_mtx_all = elem_ht3ti_mtx[1]
    elem_t_i_mtx_all = elem_ht3ti_mtx[2]

    for i in range(dim):
        xi2 = lobatto_pw[i, 0]
        w2 = lobatto_pw[i, 1]
        lag_xi2 = lagd.lagfunc(lobatto_pw, xi2)
        der_lag_dxi2 = lagd.der_lagfunc_dxi(lobatto_pw, xi2)
        for j in range(dim):
            xi1 = lobatto_pw[j, 0]
            w1 = lobatto_pw[j, 1]
            lag_xi1 = lagd.lagfunc(lobatto_pw, xi1)
            der_lag_dxi1 = lagd.der_lagfunc_dxi(lobatto_pw, xi1)           
            jac = elem_jacobian_all[i, j]
            der_lag2d_dt = der_lag2d_dt_node_i(dim, lag_xi1, lag_xi2, \
                           der_lag_dxi1, der_lag_dxi2, jac)  
            dirct_t = elem_updated_dir_all[i, j]
            der_xt_dt = der_x_t_dt(dim, der_lag2d_dt, elem_x_0_coor_all, \
                                   elem_displ_all)                                       
            der_dirt_dt = der_dir_t_dt(dim, der_lag2d_dt, elem_updated_dir_all) 

            b_displ = b_disp_mtx(lobatto_pw, lag_xi1, lag_xi2, der_lag2d_dt,\
                dirct_t, der_xt_dt, der_dirt_dt, elem_t_i_mtx_all) 
            
            btr_d_b = np.transpose(b_displ) @ elastic_mtx @ b_displ 
                              
            der_dir0_dt = der_dir_0_dt(dim, der_lag2d_dt, elem_nodal_coorsys_all)
            
            der_x0_dt = der_x_0_dt(dim, der_lag2d_dt, elem_x_0_coor_all) 
                                             
            strain_vect = strain_vector (der_x0_dt, der_xt_dt, \
                   elem_nodal_coorsys_all[i, j, 2], elem_updated_dir_all[i, j], \
                   der_dir0_dt, der_dirt_dt)
            stress_vect = stress_vector(strain_vect, elastic_mtx)
              
            k_geom = k_geom_cy_optimized_v6.geom_stiffness_mtx(dim, lag_xi1, lag_xi2, der_lag2d_dt, \
                         elem_hcapt_mtx_all, elem_t_3_mtx_all, elem_t_i_mtx_all,\
                       elem_displ_all, elem_updated_dir_all, der_xt_dt, stress_vect)
            # k_geom = geom_stiffness_mtx(dim, lag_xi1, lag_xi2, der_lag2d_dt, \
            #              elem_hcapt_mtx_all, elem_t_3_mtx_all, elem_t_i_mtx_all,\
            #            elem_displ_all, elem_updated_dir_all, der_xt_dt, stress_vect)
            stiff_mtx = stiff_mtx + np.linalg.det(jac) * \
                            (btr_d_b + k_geom) * w1 * w2
    return stiff_mtx  



def element_stiffness_mtx_blcok(lobatto_pw, lower_b, upper_b, elem_x_0_coor_all, \
                          elem_nodal_coorsys_all, elem_jacobian_all,\
                          elem_displ_all, elastic_modulus, nu, thk ):
    '''
    Some input parameters:
    elem_x_0_coor_all(dim, dim, 3): The coordinates of all Lobatto points of the element at t = 0
    elem_nodal_coorsys_all (dim, dim, 3, 3): It contains coordinate system (a_0_1, a_0_2, d=a_0_3) at all
                            Lobatto points
    elem_jacobian_all (dim, dim, 2, 2) : It contains all  2 x 2 Jacobian matrix at each Lobatto point 
    elem_displ_all(dim, dim, 2, 3) : It contains [[u1, u2, u3], [β1, β2]] of the element at all nodes
    
    In this function, the stiffness matrix of the element is built by adding material stiffness matrix
    b_tr_b and geometrical stiffness matrix, k_geom.
    
    -Output: 
    Is the element stiffness matrix
    
    ''' 
    dim = lobatto_pw.shape[0]
    stiff_mtx = np.zeros((5 * dim**2, 5 * dim**2)) # 5 DOF at each node               
    elastic_mtx = elastic_matrix(elastic_modulus, nu, thk)
    elem_updated_dir_all = elem_update_dir_all(dim, elem_nodal_coorsys_all, elem_displ_all) 
    elem_ht3ti_mtx = elem_hcapt_t3_ti_mtx_all(dim, elem_nodal_coorsys_all, elem_displ_all)
    elem_hcapt_mtx_all = elem_ht3ti_mtx[0]
    elem_t_3_mtx_all = elem_ht3ti_mtx[1]
    elem_t_i_mtx_all = elem_ht3ti_mtx[2]   

    for i in range(lower_b, upper_b):
        xi2 = lobatto_pw[i, 0]
        w2 = lobatto_pw[i, 1]
        lag_xi2 = lagd.lagfunc(lobatto_pw, xi2)
        der_lag_dxi2 = lagd.der_lagfunc_dxi(lobatto_pw, xi2)
        for j in range(dim):
            xi1 = lobatto_pw[j, 0]
            w1 = lobatto_pw[j, 1]
            lag_xi1 = lagd.lagfunc(lobatto_pw, xi1)
            der_lag_dxi1 = lagd.der_lagfunc_dxi(lobatto_pw, xi1)           
            jac = elem_jacobian_all[i, j]
            der_lag2d_dt = der_lag2d_dt_node_i(dim, lag_xi1, lag_xi2, \
                           der_lag_dxi1, der_lag_dxi2, jac)  
            dirct_t = elem_updated_dir_all[i, j]
            der_xt_dt = der_x_t_dt(dim, der_lag2d_dt, elem_x_0_coor_all, \
                                   elem_displ_all)                                       
            der_dirt_dt = der_dir_t_dt(dim, der_lag2d_dt, elem_updated_dir_all) 
                                            
            b_displ = b_disp_mtx(lobatto_pw, lag_xi1, lag_xi2, der_lag2d_dt,\
                dirct_t, der_xt_dt, der_dirt_dt, elem_t_i_mtx_all) 
            
            btr_d_b = np.transpose(b_displ) @ elastic_mtx @ b_displ
                              
            der_dir0_dt = der_dir_0_dt(dim, der_lag2d_dt, elem_nodal_coorsys_all)
            
            der_x0_dt = der_x_0_dt(dim, der_lag2d_dt, elem_x_0_coor_all) 
                                             
            strain_vect = strain_vector (der_x0_dt, der_xt_dt, \
                   elem_nodal_coorsys_all[i, j, 2], elem_updated_dir_all[i, j], \
                   der_dir0_dt, der_dirt_dt)
            stress_vect = stress_vector(strain_vect, elastic_mtx)
  
            k_geom = k_geom_cy_optimized_v6.geom_stiffness_mtx(dim, lag_xi1, lag_xi2, der_lag2d_dt, \
                         elem_hcapt_mtx_all, elem_t_3_mtx_all, elem_t_i_mtx_all,\
                       elem_displ_all, elem_updated_dir_all, der_xt_dt, stress_vect)
            # k_geom = geom_stiffness_mtx(dim, lag_xi1, lag_xi2, der_lag2d_dt, \
            #              elem_hcapt_mtx_all, elem_t_3_mtx_all, elem_t_i_mtx_all,\
            #            elem_displ_all, elem_updated_dir_all, der_xt_dt, stress_vect)
            stiff_mtx = stiff_mtx + np.linalg.det(jac) * \
                            (btr_d_b + k_geom) * w1 * w2
    return stiff_mtx



def element_stiffness_mtx_mp(lobatto_pw, elem_x_0_coor_all, \
                          elem_nodal_coorsys_all, elem_jacobian_all,\
                          elem_displ_all, elastic_modulus, nu, thk ):
    '''
    Some input parameters:
    elem_x_0_coor_all(dim, dim, 3): The coordinates of all Lobatto points of the element at t = 0
    elem_nodal_coorsys_all (dim, dim, 3, 3): It contains coordinate system (a_0_1, a_0_2, d=a_0_3) at all
                            Lobatto points
    elem_jacobian_all (dim, dim, 2, 2) : It contains all  2 x 2 Jacobian matrix at each Lobatto point 
    elem_displ_all(dim, dim, 2, 3) : It contains [[u1, u2, u3], [β1, β2]] of the element at all nodes
    
    In this function, the stiffness matrix of the element is built by adding material stiffness matrix
    b_tr_b and geometrical stiffness matrix, k_geom.
    
    -Output: 
    Is the element stiffness matrix
    
    ''' 
    dim = lobatto_pw.shape[0]
    stiff_mtx = np.zeros((5 * dim**2, 5 * dim**2)) # 5 DOF at each node 
    if dim <= 9:
        stiff_mtx = element_stiffness_mtx(lobatto_pw, elem_x_0_coor_all, \
                          elem_nodal_coorsys_all, elem_jacobian_all,\
                          elem_displ_all, elastic_modulus, nu, thk )
    else:       
        num_process = 4
        num_process, _, _, lower_upper_list = mpdistr.process_distributor(dim)
        
        args = [(lobatto_pw, lower_upper_list[pr, 0], lower_upper_list[pr, 1], elem_x_0_coor_all, \
                          elem_nodal_coorsys_all, elem_jacobian_all,\
                          elem_displ_all, elastic_modulus, nu, thk) for pr in range(num_process)]
 
        with mp.Pool(processes=4) as pool:
            stiff_mtx_list = pool.starmap(element_stiffness_mtx_blcok, args)        
    
        for i in range(4):
            stiff_mtx = stiff_mtx + stiff_mtx_list[i]
        
    return stiff_mtx
                                                              

      
####################################### TEST functions ##########################################
def generate_unit_vector():
    """
    Generate a random vector with unit length.
    """
    vec = np.random.normal(size=3)
    return vec / np.linalg.norm(vec)

def generate_matrix(n):
    """
    Generate a n x n x 3 x 3 matrix where each element is a random vector with unit length.
    Parameters:
    n (int): Size of the matrix.
    Returns:
    np.ndarray: n x n x 3 x 3 matrix of random unit vectors.
    """
    matrix = np.zeros((n, n, 3, 3))
    for i in range(n):
        for j in range(n):
            for k in range(3):
                matrix[i, j, k] = generate_unit_vector()
    return matrix                              
                                             
                                         
    
    
# if __name__ =='__main__':
            
    # subprocess.call("C:\\Nima\\N-Research\\DFG\\Python_programming\\Large_def\\1_SEMI_Large_def\\.P3-12-2\\Scripts\\snakeviz.exe process.profile ", \
    #               shell=False)  