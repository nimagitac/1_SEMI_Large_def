import numpy as np
import surface_geom_SEM as sgs
import surface_nurbs_isoprm as snip
from geomdl import exchange
import os
import time
import sys
import cProfile
import pstats
import io
from pstats import SortKey
import subprocess



np.set_printoptions(threshold=sys.maxsize)

# Global variables
###################################################################
ex = np.array((1, 0, 0), dtype=np.int32)
ey = np.array((0, 1, 0), dtype=np.int32)
ez = np.array((0, 0, 1), dtype=np.int32)
gauss_pw = [[[-0.577350269189626, 1],\
             [0.577350269189626, 1]],\
                    [[-0.774596669241483, 0.555555555555555],\
                      [0, 0.888888888888888], \
                      [0.774596669241483, 0.555555555555555]],\
                        [[-0.861136311594053, 0.347854845137454],\
                         [-0.339981043584856, 0.652145154862546],\
                         [0.339981043584856, 0.652145154862546],\
                         [0.861136311594053, 0.347854845137454]],
                    [[-0.906179845938664, 0.236926885056189],\
                        [-0.538469310105683, 0.478628670499366],\
                        [0, 0.568888888888889],\
                        [0.538469310105683, 0.478628670499366],\
                        [0.906179845938664, 0.236926885056189]]] #to be completed

######################################################################


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

    
def lbto_pw(data_file_address):
    '''This function gets the data from
    Mathematica provided .dat file. The data is 
    lobatto (lbto) points and weights of numeriucal integration
    -output:
    a 2D np.array in that first column is the 
    spectral node coordinates and the second column 
    is the ascociated weights
    '''
    node_data = np.genfromtxt(data_file_address)
    return node_data
 
 
def xi_to_uv(xi1, xi2, node_1_u, node_1_v, node_3_u, node_3_v):
    '''This function is the mapping form xi1-xi2 which
    are parametric space of SEM to u-v space which are
    parametric space of the IGA
    -output:
    u and v according to the input xi1 and xi2 and two oppsite
    corners of the rectangle.
    '''
    if xi1<-1 and xi1>1 and xi2<-1 and xi2>1:
        raise ValueError('-1<=xi1<=+1, -1<=xi2<=+1 must be followed')
    else:
        node_2_u = node_3_u 
        u = 1/2*((1-xi1)*node_1_u + (1+xi1)*node_2_u)
        v = 1/2*((1-xi2)*node_1_v + (1+xi2)*node_3_v)
    return u, v
     

def coorsys_director(surface_nurbs, u, v):
    '''According to P430, finite element
    procedures, K. J. Bathe
    The output coordinate systems are orhtonormal
    ones used to model the rotation of
    the director. It is used to define rotations.
    -output:
    2D (3X3) np.array in that:
    the first row is the v1 normalized and
    the last row is the director which is already
    normalized
    '''
    vdir = surface_nurbs.director(u,v)
    if abs(abs(vdir[1])-1) > 0.00001 :
        v1 = (np.cross(ey, vdir))
        v1_unit= v1 / np.linalg.norm(v1)
        v2_unit = np.cross(vdir, v1_unit)
    else:
        if vdir[1] > 0:
            v1_unit = np.array((1, 0, 0))
            v2_unit = np.array((0, 0, -1))
            vdir = np.array((0, 1, 0))
        else:
            v1_unit = np.array((1, 0, 0))
            v2_unit = np.array((0, 0, 1))
            vdir = np.array((0, 1, 0))
    
    # vdir = surface.director(u,v) # Lamina coordinate system
    # surface_der = surface.derivatives(u, v, 1)
    # g1 = np.array(surface_der[1][0])
    # g2 = np.array(surface_der[0][1])
    # g1_unit = g1 / np.linalg.norm(g1)
    # g2_unit = g2 / np.linalg.norm(g2)
    # A_xi1 = (0.5*(g1_unit + g2_unit))/np.linalg.norm(0.5*(g1_unit + g2_unit))
    # v_var_1 = np.cross(vdir, A_xi1)
    # A_xi2 = (v_var_1) / np.linalg.norm(v_var_1)
    # v1_unit = (2**0.5)/2 * (A_xi1 - A_xi2)
    # v2_unit = (2**0.5)/2 * (A_xi1 + A_xi2)
            
    return np.array([v1_unit, v2_unit, vdir])
         
    
def lagfunc(lobatto_pw, xi):
    '''Function for generating lagrangian
    shape functions
    output:
    1D np.array in that i-th element
    is the value of the i-th lagrangian shape function
    at xi. The number of elements in the array obviousluy is 
    equal to the number of nodes in 1 direction
    '''
    if xi<-1 and xi>1:
        raise ValueError('-1<=xi1<=+1, -1<=xi2<=+1 must be followed')
    else: 
        len = lobatto_pw.shape[0]
        # lag_func = np.zeros(len)
        # if xi in lobatto_pw[:,0]:
        #     xi_index = np.where(lobatto_pw[:,0] == xi)
        #     lag_func[xi_index] = 1
        # else:   No impresive increase in speed
        lag_func = np.ones(len)
        for i in range(len):
            for j in range(len): 
                if j != i:
                    lag_func[i] = lag_func[i] * (xi-lobatto_pw[j, 0]) /\
                        (lobatto_pw[i, 0]-lobatto_pw[j, 0])
        return lag_func
        

def der_lagfunc_dxi(lobatto_pw, xi):
    '''By taking 'ln' functiom from both sides of the definition
    of lagrangina shape functions, the derivatives
    can be calculated. For example:
    https://math.stackexchange.com/questions/809927/first-derivative-of-lagrange-polynomial
    however the formulation result in infinity when xi = xj therefore,
    it is modified to:
    dlag-i-dxi = (sigma (j, j!=i,(PI (k, k!=(i and j), (x-xk))))/PI(j, j!=i, (xi-xj))
    -output:
    1D np.array in that i-th element
    is the value of the derivatives of i-th lagrangian shape function
    w. r. t. xi at xi . The number of element obviousluy is 
    equal to the number of nodes
    '''    
    len = lobatto_pw.shape[0]
    der_lag_dxi = np.zeros(len)
    for i in range(len):
        numerator_sigma = 0
        denominator = 1
        for j in range(len):  
            if j != i:
                denominator = denominator * (lobatto_pw[i, 0]-lobatto_pw[j, 0])
                numerator_pi = 1
                for k in range(len):
                    if k!=i and k!=j:
                        numerator_pi = numerator_pi * (xi - lobatto_pw[k, 0])
                numerator_sigma = numerator_sigma + numerator_pi
        der_lag_dxi [i] = numerator_sigma / denominator
    return der_lag_dxi

def der2_lagfunc_dxi(lobatto_pw, xi):
    '''
    One way to calculate the second order derivative is:
    https://math.stackexchange.com/questions/1572576/lagrange-polynomial-second-order-derivative
    The other ways is:    
    The only part of the derivative of a Lagrange shape function which is the
    function of xi, is:
    (PI (k, k!=(i and j), (xi-xk))
    If we take the ln from this part and then taking the derivative of it,
    the second order derivative of the Lagrange function can be calculated as follow: 
    d2lag-i-dxi2 = (sigma j!=i  sigma k!=(i, j)  (PI p!=(i, j, k) (x-xp))))/PI(j, j!=i, (xi-xj))
    -output:
    1D np.array in that i-th element
    is the value of the second order derivatives of i-th lagrangian shape function
    w. r. t. xi at xi  bat point input(xi). The number of element obviousluy is 
    equal to the number of nodes
    '''    
    len = lobatto_pw.shape[0]
    der2_lag_dxi2 = np.zeros(len)
    for i in range(len):
        numerator_sigma_1 = 0
        denominator = 1
        for j in range(len):  
            if j!=i:
                denominator = denominator * (lobatto_pw[i, 0]-lobatto_pw[j, 0])
                numerator_sigma_2 = 0
                for k in range(len):
                    if k!=i and k!=j:
                        numerator_pi = 1
                        for p in range(len):
                            if p!=i and p!=j and p!=k:
                                numerator_pi = numerator_pi * (xi - lobatto_pw[p, 0])
                        numerator_sigma_2 = numerator_sigma_2 + numerator_pi
                numerator_sigma_1 = numerator_sigma_1 + numerator_sigma_2
        der2_lag_dxi2 [i] = numerator_sigma_1 / denominator
    return der2_lag_dxi2


def der_disp_dxi(surface_isoprm, lobatto_pw, v1_v2_array, lag_xi1, lag_xi2,\
    der_lag_dxi1, der_lag_dxi2, xi3=0):
    '''In this fuctions the derivatives of displacement(ux, uy, uz)
    with respect to natural SEM coordinates are calculated. Each node has
    5 DOFs: ux_k, uy_k, uz_k, alhpha_k and beta_k
    output:
    A tuple (der_ux_dxi, der_uy_dxi, der_uz_dxi) in which i-th element 
    is a (3x3(number of SEM nodes^2)) np.array.
    The below equations show how the tuple elements can be used.
    [dux_dxi1, dux_dxi2, dux_dt] = der_ux_dxi .[...ux_k, alpha_k,beta_K...]
    [duy_dxi1, duy_dxi2, duy_dt] = der_uy_dxi .[...uy_k, alpha_k,beta_K...]
    [duz_dxi1, duz_dxi2, duz_dt] = der_uz_dxi .[...uz_k, alpha_k,beta_K...]
    '''
    len = lobatto_pw.shape[0]
    der_ux_dxi = np.zeros((3, 3*len**2))
    der_uy_dxi = np.zeros((3, 3*len**2))
    der_uz_dxi = np.zeros((3, 3*len**2))
    num = 0 
    for i in range(len):
        for j in range(len):
            v1_unit = v1_v2_array[i, j , 0]
            v2_unit = v1_v2_array[i, j , 1]
            # regarding coorsys_director, v1 is always normal to y axis,
            # then the following Eqs. can be simplified, but it is left in 
            # the general form as a different director coordinate system
            # may be used in future.
            
            der_ux_dxi [:,num:num+3] = [[der_lag_dxi1[j] * lag_xi2[i], \
                der_lag_dxi1[j] * lag_xi2[i] * xi3/2 * surface_isoprm.thk * (-v2_unit[0]),\
                der_lag_dxi1[j] * lag_xi2[i] * xi3/2 * surface_isoprm.thk * (v1_unit[0])],\
                [lag_xi1[j] * der_lag_dxi2[i],\
                lag_xi1[j] * der_lag_dxi2[i] * xi3/2 * surface_isoprm.thk * (-v2_unit[0]),\
                lag_xi1[j] * der_lag_dxi2[i] * xi3/2 * surface_isoprm.thk * (v1_unit[0])],\
                [0, lag_xi1[j] * lag_xi2[i] * surface_isoprm.thk/2 * (-v2_unit[0]),\
                    lag_xi1[j] * lag_xi2[i] * surface_isoprm.thk/2 * (v1_unit[0])]]
            
            der_uy_dxi [:,num:num+3] = [[der_lag_dxi1[j] * lag_xi2[i], \
                der_lag_dxi1[j] * lag_xi2[i] * xi3/2 * surface_isoprm.thk * (-v2_unit[1]),\
                der_lag_dxi1[j] * lag_xi2[i] * xi3/2 * surface_isoprm.thk * (v1_unit[1])],\
                [lag_xi1[j] * der_lag_dxi2[i], \
                lag_xi1[j] * der_lag_dxi2[i] * xi3/2 * surface_isoprm.thk * (-v2_unit[1]),\
                lag_xi1[j] * der_lag_dxi2[i] * xi3/2 * surface_isoprm.thk * (v1_unit[1])],\
                [0, lag_xi1[j] * lag_xi2[i] * surface_isoprm.thk/2 * (-v2_unit[1]),\
                    lag_xi1[j] * lag_xi2[i] * surface_isoprm.thk/2 * (v1_unit[1])]]

            der_uz_dxi [:,num:num+3] = [[der_lag_dxi1[j] * lag_xi2[i], \
                der_lag_dxi1[j] * lag_xi2[i] * xi3/2 * surface_isoprm.thk * (-v2_unit[2]),\
                der_lag_dxi1[j] * lag_xi2[i] * xi3/2 * surface_isoprm.thk * (v1_unit[2])],\
                [lag_xi1[j] * der_lag_dxi2[i], \
                lag_xi1[j] * der_lag_dxi2[i]* xi3/2*surface_isoprm.thk*(-v2_unit[2]),\
                lag_xi1[j] * der_lag_dxi2[i] * xi3/2 * surface_isoprm.thk * (v1_unit[2])],\
                [0, lag_xi1[j] * lag_xi2[i]  *surface_isoprm.thk/2 * (-v2_unit[2]),\
                    lag_xi1[j] * lag_xi2[i] * surface_isoprm.thk/2 * (v1_unit[2])]]
            num = num + 3
    return (der_ux_dxi, der_uy_dxi, der_uz_dxi)

# def der_disp(surface, lobatto_pw, v1_v2_array, lag_xi1, lag_xi2,\
#             der_lag_dxi1, der_lag_dxi2, node_1_u, node_1_v,\
#             node_3_u, node_3_v, jac2, t=0):

def der_disp(surface_isoprm, lobatto_pw, v1_v2_array, lag_xi1, lag_xi2,\
            der_lag_dxi1, der_lag_dxi2, jac, xi3=0):
    '''In this fuctions the derivatives of displacement in x direction
    ux, in y direction uy and in z direction uz with respect to
    physical coordinates with the help of mapping (jacobian) matrix
    are calculated (according to P439, finite element
    procedures, K. J. Bathe)
    output:
    A tuple (der_ux, der_uy, der_uz)in which i-th element
    is a 3x3(number of 2D-SEM nodes).
    [dux_dx, dux_dy, dux_dz] = der_ux .[...ux_k, alpha_k,beta_K...]
    [duy_dx, duy_dy, duy_dz] = der_uy .[...uy_k, alpha_k,beta_K...]
    [duz_dx, duz_dy, duz_dz] = der_uz .[...uz_k, alpha_k,beta_K...]'''
    
    len = lobatto_pw.shape[0]
    # jac = np.zeros((3,3))
    # inv_jac_total = np.zeros((3,3))
    der_ux = np.zeros((3, 3*len**2))
    der_uy = np.zeros((3, 3*len**2))
    der_uz = np.zeros((3, 3*len**2))
    # jac1 = np.array([[(node_3_u - node_1_u)/2, 0 , 0],\
    #                 [0, (node_3_v - node_1_v)/2, 0],[0, 0, 1]]) #From (xi1, xi2) space to (u,v) space
    # jac2 = surface.ders_uvt(u, v, t)
    inv_jac_total = np.linalg.inv(jac)
    der_disp_dxi_var = der_disp_dxi(surface_isoprm, lobatto_pw, v1_v2_array, lag_xi1, lag_xi2,\
    der_lag_dxi1, der_lag_dxi2, xi3)
    der_ux = np.matmul(inv_jac_total, der_disp_dxi_var[0])
    der_uy = np.matmul(inv_jac_total, der_disp_dxi_var[1])
    der_uz = np.matmul(inv_jac_total, der_disp_dxi_var[2])
    return (der_ux, der_uy, der_uz)


def b_matrix(surface_isoprm, lobatto_pw, v1_v2_array, lag_xi1, lag_xi2,\
            der_lag_dxi1, der_lag_dxi2, jac, xi3):
    '''according to P440, finite element
    procedures, K. J. Bathe, B matrix is calculated and
    strain = B.d
    output:
    A 2D numpy array. Its dimension is (6x5(number of nodes))
    as each node has 5 DOFs. Strain vector is assumed as
    [exx, eyy, ezz, exy, eyz, exz]
    '''
    len = lobatto_pw.shape[0]
    
    der_disp_var = der_disp(surface_isoprm, lobatto_pw, v1_v2_array, lag_xi1, lag_xi2,\
                           der_lag_dxi1, der_lag_dxi2, jac, xi3)
    der_ux_var = der_disp_var[0]
    der_uy_var = der_disp_var[1]
    der_uz_var = der_disp_var[2]
    b_matrix_var = np.array([[der_ux_var[0,0], 0, 0,
                              der_ux_var[0,1],der_ux_var[0,2]],
                             [0, der_uy_var[1,0], 0,
                              der_uy_var[1,1], der_uy_var[1,2]],
                             [0, 0, der_uz_var[2,0],
                              der_uz_var[2,1], der_uz_var[2,2]],
                             [der_ux_var[1,0], der_uy_var[0,0], 0,
                              der_ux_var[1,1] + der_uy_var[0,1],
                              der_ux_var[1,2] + der_uy_var[0,2]],
                             [0, der_uy_var[2,0], der_uz_var[1,0],
                              der_uy_var[2,1] + der_uz_var[1,1],
                              der_uy_var[2,2] + der_uz_var[1,2]],
                             [der_ux_var[2,0], 0, der_uz_var[0,0], 
                              der_ux_var[2,1] + der_uz_var[0,1],
                              der_ux_var[2,2]+ der_uz_var[0,2]]])
    i = 1
    j = 3
    len_2 = len**2
    while i <= (len_2)-1:
        b_local = np.array([[der_ux_var[0,j], 0, 0,
                              der_ux_var[0,j+1], der_ux_var[0,j+2]],
                             [0, der_uy_var[1,j], 0,
                              der_uy_var[1,j+1], der_uy_var[1,j+2]],
                             [0, 0, der_uz_var[2,j],
                              der_uz_var[2,j+1], der_uz_var[2,j+2]],
                             [der_ux_var[1,j], der_uy_var[0,j], 0,
                              der_ux_var[1,j+1] + der_uy_var[0,j+1],
                              der_ux_var[1,j+2] + der_uy_var[0,j+2]],
                             [0, der_uy_var[2,j], der_uz_var[1,j],
                              der_uy_var[2,j+1] + der_uz_var[1,j+1],
                              der_uy_var[2,j+2] + der_uz_var[1,j+2]],
                             [der_ux_var[2,j], 0, der_uz_var[0,j], 
                              der_ux_var[2,j+1] + der_uz_var[0,j+1],
                              der_ux_var[2,j+2]+ der_uz_var[0,j+2]]])
        b_matrix_var = np.append(b_matrix_var, b_local, axis=1)
        j += 3
        i += 1
    return b_matrix_var





def physical_crd_xi(surface, xi1, xi2, node_1_u, node_1_v,\
                    node_3_u, node_3_v, t=0):
    u = xi_to_uv(xi1, xi2, node_1_u, node_1_v, node_3_u, node_3_v)[0]
    v =  xi_to_uv(xi1, xi2, node_1_u, node_1_v, node_3_u, node_3_v)[1]
    physical_coor_var = surface.physical_crd(u, v, t)
    x = physical_coor_var[0]
    y = physical_coor_var[1]
    z = physical_coor_var[2]
    return x, y, z
     

 
def coorsys_material(coorsys_tanvec_mtx, row_num, col_num):
    '''According to P438, finite element
    procedures, K. J. Bathe Fig. 5.33, an orthonormal coordinate
    system, in which the plane stress assumption is valid,
    is generated in this function.row_num and col_num show the
    location of node, equivalent to xi1 and xi2.
    -output:
    2D (3X3) np.array in that:
    the first row is r_hat_unit normalized and
    the last row is director which is already
    normalized.
    '''
    g2 = coorsys_tanvec_mtx[row_num, col_num, 4]
    vdir = coorsys_tanvec_mtx[row_num, col_num, 2]
    var_1 = np.cross(g2, vdir)
    r_hat_unit = var_1 / np.linalg.norm(var_1)
    var_2 = np.cross(vdir, r_hat_unit)
    s_hat_unit = var_2 / np.linalg.norm(var_2)
    return np.array([r_hat_unit, s_hat_unit, vdir])
       

    
            
def mat_transform_matrix(coorsys_tanvec_mtx, row_num, col_num):
    '''According to P441, finite element
    procedures, K. J. Bathe, a transformation matrix
    named in the book as Qsh is generated in this function.
    row_num and col_num show the location of node,
    equivalent to xi1 and xi2.
    -output:
    2D (6X6) np.array
    '''
    material_coorsys_vectors = \
        coorsys_material(coorsys_tanvec_mtx, row_num, col_num)   
    r_hat_unit = material_coorsys_vectors[0]
    s_hat_unit = material_coorsys_vectors[1]
    vdir = material_coorsys_vectors[2]
    k_1 = np.inner(ex, r_hat_unit) # as both are unit vectors, this is similarity cosine
    k_2 = np.inner(ex, s_hat_unit)
    k_3 = np.inner(ex, vdir)
    m_1 = np.inner(ey, r_hat_unit)
    m_2 = np.inner(ey, s_hat_unit)
    m_3 = np.inner(ey, vdir)
    n_1 = np.inner(ez, r_hat_unit)
    n_2 = np.inner(ez, s_hat_unit)    
    n_3 = np.inner(ez, vdir)
    trans_matrix = np.array(((k_1**2, m_1**2, n_1**2, k_1*m_1, m_1*n_1, k_1*n_1),\
                    (k_2**2, m_2**2, n_2**2, k_2*m_2, m_2*n_2, k_2*n_2),\
                    (k_3**2, m_3**2, n_3**2, k_3*m_3, m_3*n_3, k_3*n_3),
                    (2*k_1*k_2, 2*m_1*m_2, 2*n_1*n_2, k_1*m_2 + k_2*m_1,\
                        m_1*n_2 + m_2*n_1, k_1*n_2 + k_2*n_1),\
                    (2*k_2*k_3, 2*m_2*m_3, 2*n_2*n_3, k_2*m_3 + k_3*m_2,\
                        m_2*n_3 + m_3*n_2, k_2*n_3 + k_3*n_2),\
                    (2*k_1*k_3, 2*m_1*m_3, 2*n_1*n_3, k_1*m_3 + k_3*m_1,\
                        m_1*n_3 + m_3*n_1, k_1*n_3 + k_3*n_1)))
    return trans_matrix


# @profile
def element_stiffness_matrix(surface_isoprm, lobatto_pw, coorsys_tanvec_mtx,\
                                elastic_modulus=3*(10**6), \
                                nu=0.3, number_gauss_point=2):
    '''stiffness matrix and numerical integration
    -output:
    Element stiffness matrix
    '''
    len = lobatto_pw.shape[0]
    # jac1 =np.array(np.array([[(node_3_u - node_1_u)/2, 0 , 0],\
    #                 [0, (node_3_v - node_1_v)/2, 0],[0, 0, 1]]))
    gauss_pw_nparray = np.array(gauss_pw[number_gauss_point-2])
    len_gauss = gauss_pw_nparray.shape[0]
    elastic_matrix_planestress = elastic_modulus/(1-nu**2)*\
            np.array([[1, nu, 0, 0, 0, 0],[nu, 1, 0, 0, 0, 0],\
                [0, 0, 0, 0, 0, 0], [0, 0, 0, (1-nu)/2, 0, 0],
               [0, 0, 0, 0, 5/6*(1-nu)/2, 0],[0, 0, 0, 0, 0, 5/6*(1-nu)/2]])
    v1_v2_at_lobatto_points = coorsys_tanvec_mtx[:, :, 0:2, :]    
    # for i in range(len):
    #     for j in range(len):
    #         uv_var = xi_to_uv(lobatto_pw[j,0], lobatto_pw[i,0], node_1_u,\
    #                         node_1_v, node_3_u, node_3_v)
    #         v1_v2_at_lobatto_points[i, j , 0] = \
    #             coorsys_director(surface, uv_var[0], uv_var[1])[0]
    #         v1_v2_at_lobatto_points[i, j , 1] = \
    #             coorsys_director(surface, uv_var[0], uv_var[1])[1]
    stiff_mtx = np.zeros((5*len**2, 5*len**2)) # 5 DOF at each node
    # with open("b_isoprm.dat", 'w') as b_file:
    #     pass
    # with open ('qsh_isoprm.dat', 'w') as qsh_file:
    #     pass
    # with open("kwojac_isoprm.dat", 'w') as k_file:
    #     pass
    # with open("director_unit_isoprm.dat", 'w') as du:
    #     pass
    # with open("k_isoprm.dat", 'w') as k_file:
    #     pass
    # with open("jac_isoprm.dat", 'w') as jac_file:
    #     pass
    for i in range(len):
        # print(i, '\n')
        xi2 = lobatto_pw[i, 0]
        w2 = lobatto_pw[i, 1]
        # xi1 = lobatto_pw[i, 0]
        # w1 = lobatto_pw[i, 1]
        lag_xi2 = lagfunc(lobatto_pw, xi2)
        der_lag_dxi2 = der_lagfunc_dxi(lobatto_pw, xi2)
        for j in range(len):
            xi1 = lobatto_pw[j, 0]
            w1 = lobatto_pw[j, 1]
            # xi2 = lobatto_pw[j, 0]
            # w2 = lobatto_pw[j, 1]
            # uv_var = xi_to_uv(lobatto_pw[j,0], lobatto_pw[i,0], node_1_u,\
            #                 node_1_v, node_3_u, node_3_v)
            qsh = mat_transform_matrix(coorsys_tanvec_mtx, i, j)
            # with open ('qsh_isoprm.dat', 'a') as qsh_file:
            #         np.savetxt(qsh_file, qsh)
            elastic_matrix = np.transpose(qsh).\
                    dot(elastic_matrix_planestress).dot(qsh)
            lag_xi1 = lagfunc(lobatto_pw, xi1)
            der_lag_dxi1 = der_lagfunc_dxi(lobatto_pw, xi1)
            crv_content = surface_isoprm.curvature_mtx(coorsys_tanvec_mtx, i, j,\
                          lag_xi1, lag_xi2, der_lag_dxi1, der_lag_dxi2)
            gauss_crv = crv_content
            with open ('gauss_crv_isoprm.dat', 'a') as g_file:
                    np.savetxt(g_file,  np.array([[xi1, xi2, gauss_crv]]))
           
            # second_surf_der = surface.derivatives(uv_var[0], uv_var[1], order=2)    
            for k in range(len_gauss):
                xi3 = gauss_pw_nparray[k, 0]
                w3 =gauss_pw_nparray[k, 1]
                # jac = surface.ders_uvt(t, second_surf_der) 
                jac = surface_isoprm.jacobian_mtx(coorsys_tanvec_mtx, i, j, xi3,\
                     lag_xi1, lag_xi2, der_lag_dxi1, der_lag_dxi2) #np.identity(3)
                with open ('jac_isoprm.dat', 'a') as jac_file:
                    # np.savetxt(jac_file, jac) 
                    np.savetxt(jac_file, [np.linalg.det(jac)]) 
                    jac_file.write('\n')
                b = b_matrix(surface_isoprm, lobatto_pw, v1_v2_at_lobatto_points,\
                            lag_xi1, lag_xi2,der_lag_dxi1,\
                            der_lag_dxi2, jac, xi3)                   
                btr_d_b = np.transpose(b) @ elastic_matrix @ b
                # with open ('b_isoprm.dat', 'a') as b_file:
                #     np.savetxt(b_file, btr_d_b)
                stiff_mtx = stiff_mtx + np.linalg.det(jac) * \
                            btr_d_b * w1 * w2 * w3 
    # with open ('kwojac_isoprm.dat', 'a') as k_file:
    #                 np.savetxt(k_file, stiff_mtx)
    # with open ('k_isoprm.dat', 'a') as k_file:
    #                 np.savetxt(k_file, stiff_mtx)
    return stiff_mtx


def check_symmetric(a, rtol=1e-05, atol=1e-05):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def boundary_condition(lobatto_pw):
    # bc_h_bott = [0, 0, 1, 1, 1] #cantilever Plate.zero means clamped DOF
    # bc_h_top = [0, 0, 0, 0, 0]
    # bc_v_left = [0, 0, 1, 1, 1]
    # bc_v_right = [0, 0, 1, 1, 1]
    # bc_h_bott = [1, 1, 1, 1, 1] #cantilever quarter cylinder.zero means clamped DOF
    # bc_h_top = [0, 0, 0, 0, 0]
    # bc_v_left = [1, 1, 1, 1, 1]
    # bc_v_right = [1, 1, 1, 1, 1]
    bc_h_bott = [1, 0, 1, 0, 1] #pinched shell. zero means clamped DOF
    bc_h_top = [0, 1, 0, 1, 0]
    bc_v_left = [1, 1, 0, 1, 0] #xi1 = 1, director direction is inwards
    bc_v_right = [0, 1, 1, 1, 0]
    # bc_h_bott = [0, 0, 0, 0, 0] #Total clamped. zero means clamped DOF
    # bc_h_top = [0, 0, 0, 0, 0]
    # bc_v_left = [0, 0, 0, 0, 0]
    # bc_v_right = [0, 0, 0, 0, 0]
    # bc_h_bott = [0, 0, 0, 0, 0] # Finding stiffness zero means clamped DOF
    # bc_h_top = [0, 0, 0, 0, 0]
    # bc_v_left = [0, 0, 1, 0, 0]
    # bc_v_right = [0, 0, 0, 0, 0]
    len = lobatto_pw.shape[0]
    bc_line_h_bott = np.zeros(5*len) #bottom horizontal line
    bc_line_h_top = np.zeros(5*len) #top horizontal line
    bc_line_v_left = np.zeros(5*(len - 2))
    bc_line_v_right = np.zeros(5*(len - 2))
    i = 1
    c_node = len * (len-1)
    while i <= len:
        bc_line_h_bott[5*i - 5] = (1 - bc_h_bott[0]) * (5*i-4)
        bc_line_h_bott[5*i - 4] = (1 - bc_h_bott[1]) * (5*i-3)
        bc_line_h_bott[5*i - 3] = (1 - bc_h_bott[2]) * (5*i-2)
        bc_line_h_bott[5*i - 2] = (1 - bc_h_bott[3]) * (5*i-1)
        bc_line_h_bott[5*i - 1] = (1 - bc_h_bott[4]) * (5*i)
      
        bc_line_h_top[5*i - 5] = (1 - bc_h_top[0]) * (5*(c_node + i) - 4)
        bc_line_h_top[5*i - 4] = (1 - bc_h_top[1]) * (5*(c_node + i) - 3)
        bc_line_h_top[5*i - 3] = (1 - bc_h_top[2]) * (5*(c_node + i) - 2)
        bc_line_h_top[5*i - 2] = (1 - bc_h_top[3]) * (5*(c_node + i) - 1)
        bc_line_h_top[5*i - 1] = (1 - bc_h_top[4]) * (5*(c_node + i))
        i += 1
    i = 1
    while i <= len-2:
        bc_line_v_left[5*i - 5] = (1 - bc_v_left[0]) * (5*(i*len + 1) - 4)
        bc_line_v_left[5*i - 4] = (1 - bc_v_left[1]) * (5*(i*len + 1) - 3)
        bc_line_v_left[5*i - 3] = (1 - bc_v_left[2]) * (5*(i*len + 1) - 2)
        bc_line_v_left[5*i - 2] = (1 - bc_v_left[3]) * (5*(i*len + 1) - 1)
        bc_line_v_left[5*i - 1] = (1 - bc_v_left[4]) * (5*(i*len + 1))

        bc_line_v_right[5*i - 5] = (1 - bc_v_right[0]) * (5*(i*len + len) - 4)
        bc_line_v_right[5*i - 4] = (1 - bc_v_right[1]) * (5*(i*len + len) - 3)
        bc_line_v_right[5*i - 3] = (1 - bc_v_right[2]) * (5*(i*len + len) - 2)
        bc_line_v_right[5*i - 2] = (1 - bc_v_right[3]) * (5*(i*len + len) - 1)
        bc_line_v_right[5*i - 1] = (1 - bc_v_right[4]) * (5*(i*len + len))
        i += 1
    bc_1 = np.concatenate((bc_line_h_bott, bc_line_h_top, bc_line_v_left,\
                        bc_line_v_right))
#####################################################
    '''Only for pinched shell'''
    node_3_number = (len-1)*len + 1 #xi1 = -1, xi2 = 1
    node_4_number = node_3_number + len - 1  #xi1 = 1, xi2 = 1
    bc_node_1 = np.array([0, 2, 3, 4, 5])
    bc_node_2 = np.array([5*len-4, 5*len-3, 0, 5*len-1, 5*len])
    bc_node_3 = np.array([5*node_3_number-4, 0, 5*node_3_number-2, 0, 5*node_3_number])
    bc_node_4 = np.array([5*node_4_number-4, 0, 5*node_4_number-2, 0, 5*node_4_number])
    for i in range(5):
        bc_1[i] = bc_node_1[i]
        bc_1[5*len - 5 + i] = bc_node_2[i]
        bc_1[5*(len + 1) - 5 + i] = bc_node_3[i]
        bc_1[5*(len + 1 + len - 1) - 5  + i] = bc_node_4[i]
#####################################################
    bc_2 = np.sort(np.delete(bc_1, np.where(bc_1 == 0))) - 1 # in python arrays start from zero
    return (bc_2.astype(int))


def stiffness_matrix_bc_applied(stiff_mtx, bc):
    stiff = np.delete(np.delete(stiff_mtx, bc, 0), bc, 1)
    return stiff
    




if __name__ == '__main__':
        ############## Visualization
    # surf_cont = multi.SurfaceContainer(data)
    # surf_cont.sample_size = 30
    # surf_cont.vis = vis.VisSurface(ctrlpts=False, trims=False)
    # surf_cont.render()

    ########################## TESTs
    # Test for area and mapping matrix

    data = exchange.import_json("pinched_shell.json") # curved_beam_lineload_2_kninsertion curved_beam_lineload_2 pinched_shell_kninsertion_changedeg.json pinched_shell.json rectangle_cantilever square square_kninsertion generic_shell_kninsertion foursided_curved_kninsertion foursided_curved_kninsertion2  rectangle_kninsertion
    # visualization(data)
    surfs = sgs.SurfaceGeo(data, 0, 3)
    lobatto_pw_all =lbto_pw("node_weight_all.dat")
    i_main = 5
    if i_main == 1:
        lobatto_pw = lobatto_pw_all[1:3,:]
    else:  
        index = np.argwhere(lobatto_pw_all==i_main)
        lobatto_pw = lobatto_pw_all[index[0, 0] + 1:\
                            index[0, 0] + (i_main+1) +1, :]
    node_1_ub = 0
    node_1_vb = 0
    node_3_ub = 1
    node_3_vb = 1
    surf_isoprm = snip.SurfaceGeneratedSEM(surfs, lobatto_pw, node_1_ub,\
                  node_1_vb, node_3_ub, node_3_vb)

    coorsys_tanvec_mtx = surf_isoprm.coorsys_director_tanvec_allnodes()
    print(der2_lagfunc_dxi(lobatto_pw, -0.765055))
    # stiff = element_stiffness_matrix(surf_isoprm, lobatto_pw, coorsys_tanvec_mtx)
    # bc = boundary_condition(lobatto_pw)
    # print(bc)
    # stiff_bc = stiffness_matrix_bc_applied(stiff, bc)

    # len = lobatto_pw.shape[0]
    # print('number of nodes:',len,'\n')
    # load = np.zeros(np.shape(stiff)[0])
    # p = 0.25
    # # # load[5*int((len-1)/2) + 3 - 1] = p #for cantilever
    # # load[8:(5*int((len))-1 - 1):5] = p #for cantilever subjected to moment at free end
    # # load[3] = p/2                      #for cantilever subjected to moment at free end
    # # load[(5*int((len))-1 - 1)] =p/2    #for cantilever subjected to moment at free end
    # # # load[5*int((len-1)/2*len+(len-1)/2) + 3 - 1] = 1
    # load[0] = p #for pinched
    # load = np.delete(load, bc, 0)
    # d = np.linalg.solve(stiff_bc, load)
    # # print(d)
    # # print("displacement:", d[int(3*(len-1)/2)+1-1], '\n')    #for cantilever subjected to moment at free end
    # time2 = time.time()
    # print( "time is:",time2-time1, '\n')
    # print("displacement:", d[np.where(load==p)[0][0]]/(1.83*10**-5), '\n')
    # print(f'condition number of the stiffness matrix :  {np.linalg.cond(stiff_bc)}\n')
# print("determinant of stiff matrix is", np.linalg.det(stiff_bc))

    print("End")

    # subprocess.call("C:\\Users\\azizi\Desktop\\DFG_Python\\.P394\\Scripts\\snakeviz.exe process.profile ", \
    #               shell=False)

    # "editor.hover.enabled": false,
    # "editor.hover.sticky": false,
    # "editor.hover.above": false,
    # "editor.parameterHints.enabled": false,
    # "editor.semanticHighlighting.enabled": true,

