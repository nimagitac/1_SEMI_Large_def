import numpy as np
import surface_geom_SEM as sgs
import lagrange_der as lagd
import t_i_mtx_firstvar as tmf
import cProfile
import subprocess

from geomdl import exchange



# def jacobian_element_ncoorsys_mtx (lobatto_pw, coorsys_tanvec_mtx):
#     '''
#     In this function the jacobian matrix for all the nodes of one element
#     is calculated. It is based on Eq. (14) of 
#     "A robust non-linear mixed hybrid quadrilateral shell element, 2005
#     W. Wagner1, and F. Gruttmann"
#     -Output:
#     Is a 2x2 matrix.
#     '''
#     dim = lobatto_pw.shape[0]
#     for i in range(dim):
#         for j in range(dim):
#             a_0_1 = coorsys_tanvec_mtx[i, j, 0]
#             a_0_2 = coorsys_tanvec_mtx[i, j, 1]
#             g1 = coorsys_tanvec_mtx[i, j, 3]
#             g2 = coorsys_tanvec_mtx[i, j, 4]
    
#     jac_elem_mtx = np.array([[g1 @ a_0_1, g1 @ a_0_2], [g2 @ a_0_1, g2 @ a_0_2]])
#     return jac_elem_mtx

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

def der_lag2d_dti_node_i(jacobian_at_node, der_lag2d_dxi_node_i):
    '''
     This function calculate the derivatives of 2D lagrange shape functions with
     repsect to the coordinates of local nodal coordinate systemat each node regarding
     to the Jacobian matrix and calculation of the der_lagd2d_dxi.
    The first row is d(lag2d)/dti1 and the second is d(lag2d)/dti2
    -Output:
    Is a 2 x number_lobatto_points^2 matrix
    '''
    inv_jac = np.linalg.inv(jacobian_at_node)
    der_lag_2d_dti = inv_jac @ der_lag2d_dxi_node_i
    
    return der_lag_2d_dti
    
    
# @profile
# def der_x_t_dt(number_lobatto_point, der_lag2d_dti\
#                     , elem_x_0, elem_displ):
#     '''
#     In this f
#     '''
#     dim = number_lobatto_point
    
#     elem_x_t = elem_x_0[:dim, :dim] + elem_displ[:dim, :dim] # x = X + u

#     x_t_1_vect = np.zeros((dim**2))
#     x_t_2_vect = np.zeros((dim**2))
#     x_t_3_vect = np.zeros((dim**2))
#     for i in range(dim):
#         for j in range(dim):
#             icapt = ij_to_icapt(dim, i, j)
#             x_t_1_vect[icapt] = elem_x_t[i, j, 0]
#             x_t_2_vect[icapt] = elem_x_t[i, j, 1]
#             x_t_3_vect[icapt] = elem_x_t[i, j, 2]    
#     der_x_t_dt1 = np.array([der_lag2d_dti[0] @ x_t_1_vect, der_lag2d_dti[0] @ x_t_2_vect,\
#                             der_lag2d_dti[0] @ x_t_3_vect])
#     der_x_t_dt2 = np.array([der_lag2d_dti[1] @ x_t_1_vect, der_lag2d_dti[1] @ x_t_2_vect,\
#                             der_lag2d_dti[1] @ x_t_3_vect])
#     return (der_x_t_dt1, der_x_t_dt2)

# @profile
def der_x_t_dt(number_lobatto_point, der_lag2d_dti,\
                    elem_x_0, elem_displ):
    '''
    In this function the derivatives of x (position vector at time 't')
    are calculated at the specified integration point. The specific integration
    point is introduced through der_lag2d_dti which is calculated according to the
    input coordinate.
    dim is the number of the Lobatto points.
    elem_x_0: A dim*dim*3 matrix which contains the initial coordinates of the nodes
    elem_displ: A dim*dim*2*3 matrix which contain at [i,j,0] 'u'(displacement vector)
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
    elem_x_t = elem_x_0[: , : ] + elem_displ[: , : , 0] # x = X + u
              
    for i in range(dim):
        for j in range(dim):
            icapt = ij_to_icapt(dim, i, j)
            der_x_t_dt1 = der_x_t_dt1 + der_lag2d_dti[0, icapt] * elem_x_t[i, j]
            der_x_t_dt2 = der_x_t_dt2 + der_lag2d_dti[1, icapt] * elem_x_t[i, j]
    return (der_x_t_dt1, der_x_t_dt2)
            
def elem_update_dir_all(number_lobatto_point, elem_nodal_coorsys, elem_displ):
    '''
    In this function, the updated director at time 't' for all of the nodes
    of the element is calculated and stored.
    -Output:
    A num_lobatto x numlobbato x 3 matrix
    '''
    dim = number_lobatto_point
    elem_updated_dir_all = np.zeros((dim, dim, 3))
    for i in range(dim):
        for j in range(dim):
            omega_vect = elem_displ[i, j, 1]
            a_0_3 = elem_nodal_coorsys[i, j, 2]
            rot_mtx_i = tmf.r_mtx_node_i(omega_vect)
            a_t_3 = rot_mtx_i @ a_0_3
            elem_updated_dir_all[i, j] = a_t_3
    return elem_updated_dir_all

def der_dir_t_dt(number_lobatto_point, der_lag2d_dti, elem_updated_dir_all):
    '''
    In this function the derivatives of director vector
    are calculated at the specified integration point. The specific integration
    point is introduced through der_lag2d_dti which is calculated according to the
    input coordinate.
    -Output:
    is a 2 x 3 matrix. The first element is da_t_3/dt1 and the second is da_t_3/dt2 at the 
    specific integration points.
    '''
    dim = number_lobatto_point
    der_dir_dt1 = np.zeros(3)
    der_dir_dt2 = np.zeros(3)
    for i in range(dim):
        for j in range(dim):
            icapt = ij_to_icapt(dim, i, j)
            der_dir_dt1 = der_dir_dt1 + der_lag2d_dti[0, icapt] * elem_updated_dir_all[i, j]
            der_dir_dt2 = der_dir_dt2 + der_lag2d_dti[1, icapt] * elem_updated_dir_all[i, j]
    return (der_dir_dt1, der_dir_dt2)
            


        
# def der_dir_t_dt(number_lobatto_point, der_lag2d_dti\
#                     , elem_x_0, elem_displ)       
    
    
 
    
    
    
    
if __name__ =='__main__':
    data = exchange.import_json("scordelis_corrected.json") #  pinched_shell_kninsertion_changedeg.json pinched_shell.json rectangle_cantilever square square_kninsertion generic_shell_kninsertion foursided_curved_kninsertion foursided_curved_kninsertion2  rectangle_kninsertion
    # sgs.visualization(data)
    surfs = sgs.SurfaceGeo(data, 0, 0.25)
    p_1 = surfs.physical_crd(0., 0.)
    p_2 = surfs.physical_crd(1., 0.)
    p_3 = surfs.physical_crd(1., 1.)
    print("p_1:", p_1, "  p_2:", p_2, "  p_3:", p_3)
    print("\n\n")
    lobatto_pw_all = lagd.lbto_pw("node_weight_all.dat")
    number_element_u = 2
    number_element_v = 2
    i_main = 2
    if i_main == 1:
        lobatto_pw = lobatto_pw_all[1:3,:]
    else:  
        index = np.argwhere(lobatto_pw_all==i_main)
        lobatto_pw = lobatto_pw_all[index[0, 0] + 1:\
                            index[0, 0] + (i_main+1) +1, :]
    dim = lobatto_pw.shape[0]
    jacobian_at_node = np.array([[1, 2],[3, 4]])
    elem_displ = np.random.randint(0, 4, size=(dim, dim, 2, 3))
    elem_x_0 = np.random.randint(0, 10, size=(dim, dim, 3))
    elem_dir_all = np. random.randint(0, 5, size=(dim, dim, 3))
    print('\n', elem_displ, '\n', elem_x_0)
    for i in range( dim):
        xi2 = lobatto_pw[i, 0]
        lag_xi2 = lagd.lagfunc(lobatto_pw, xi2)
        der_lag_dxi2 = lagd.der_lagfunc_dxi(lobatto_pw, xi2)
        for j in range( dim):
            xi1 = lobatto_pw[j, 0]
            lag_xi1 = lagd.lagfunc(lobatto_pw, xi1)
            der_lag_dxi1 = lagd.der_lagfunc_dxi(lobatto_pw, xi1)
           
            der_lag2d_dxi = der_lag2d_dxi_node_i( dim, lag_xi1, lag_xi2, der_lag_dxi1,\
                  der_lag_dxi2)
            der_lag2d_dti = der_lag2d_dti_node_i(jacobian_at_node, der_lag2d_dxi)
            der_x_dt = der_x_t_dt(dim, der_lag2d_dti, elem_x_0, elem_displ)
            der_x_dt_test = der_x_t_dt(dim, der_lag2d_dti, elem_x_0, elem_displ)
            der_dir = der_dir_t_dt(dim, der_lag2d_dti, elem_dir_all)
            print(der_x_dt,'\n', der_x_dt_test,'\n\n\n')
            
    # subprocess.call("C:\\Nima\\N-Research\\DFG\\Python_programming\\Large_def\\1_SEMI_Large_def\\.P3-12-2\\Scripts\\snakeviz.exe process.profile ", \
    #               shell=False)  