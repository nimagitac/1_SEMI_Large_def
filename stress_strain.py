import numpy as np
import lagrange_der as lagd
import element_stiffness_matrix_largedef as esmlrg





def stress_mtx_all(lobatto_pw, number_element_u, number_element_v, \
    x_0_coor_all, nodal_coorsys_all, jacobian_all, node_displ_all, \
        elastic_modulus, nu, thk):
    '''
    In this function, stresses and strains are calculated and stored for
    each element at all integration points.
    -Output:
    A numelemu x numelemv x dim x dim x 2 x 8 matrix. The first
    2 x 8 part contains strains and stresses.

    '''
    dim = lobatto_pw.shape[0]
    elastic_mtx = esmlrg.elastic_matrix(elastic_modulus, nu, thk)
    strs_mtx = np.zeros((number_element_v, number_element_u,\
                                        dim, dim, 2, 8))
    for i_main in range(number_element_v):
        for j_main in range(number_element_u):
            
            elem_x_0_coor_all = x_0_coor_all[i_main, j_main] 
            elem_nodal_coorsys_all = nodal_coorsys_all[i_main, j_main]  
            elem_jacobian_all = jacobian_all[i_main, j_main]  
            elem_displ_all = node_displ_all[i_main, j_main] 
            elem_updated_dir_all = esmlrg.elem_update_dir_all(dim, \
                            elem_nodal_coorsys_all, elem_displ_all)  
            
           
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
                    der_lag2d_dt = esmlrg.der_lag2d_dt_node_i(dim, lag_xi1, lag_xi2, \
                                    der_lag_dxi1, der_lag_dxi2, jac)
                    der_x0_dt = esmlrg.der_x_0_dt(dim, der_lag2d_dt, elem_x_0_coor_all)
                    der_xt_dt = esmlrg.der_x_t_dt(dim, der_lag2d_dt, elem_x_0_coor_all, \
                                   elem_displ_all)
                    der_dir0_dt = esmlrg.der_dir_0_dt(dim, der_lag2d_dt,\
                                    elem_nodal_coorsys_all)
                    der_dirt_dt = esmlrg.der_dir_t_dt(dim, der_lag2d_dt,\
                                    elem_updated_dir_all) 
                    
                    strain_vect = esmlrg.strain_vector (der_x0_dt, der_xt_dt, \
                                    elem_nodal_coorsys_all[i, j, 2], elem_updated_dir_all[i, j], \
                                    der_dir0_dt, der_dirt_dt)
                    
                    stress_vect = esmlrg.stress_vector(strain_vect, elastic_mtx)
                    strs_mtx[i_main, j_main, i, j, 0] = strain_vect
                    strs_mtx[i_main, j_main, i, j, 1] = stress_vect
    return strs_mtx