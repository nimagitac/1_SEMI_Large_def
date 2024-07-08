import numpy as np
import lagrange_der as lagd
import element_stiffness_matrix_largedef as esmlrg


def element_intern_force(lobatto_pw, elem_x_0_coor_all, \
                          elem_nodal_coorsys_all, elem_jacobian_all,\
                          elem_displ_all, elastic_mtx):
    '''
    
    '''
    dim = lobatto_pw.shape[0]
    elem_intern_force = np.zeros(5 * dim**2)
    elem_updated_dir_all = esmlrg.elem_update_dir_all(dim, elem_nodal_coorsys_all, elem_displ_all) 
    elem_ht3ti_mtx = esmlrg.elem_hcapt_t3_ti_mtx_all(dim, elem_nodal_coorsys_all, elem_displ_all)
    # elem_hcatp_mtx_all = elem_ht3ti_mtx[0]
    # elem_t_3_mtx_all = elem_ht3ti_mtx[1]
    elem_t_i_mtx_all = elem_ht3ti_mtx[2]
    ##################################################
    elem_ht3ti_mtx = esmlrg.elem_hcapt_t3_ti_mtx_all(dim, elem_nodal_coorsys_all,np.zeros((dim, dim, 2, 3)))
    # elem_hcatp_mtx_all = elem_ht3ti_mtx[0]
    # elem_t_3_mtx_all = elem_ht3ti_mtx[1]
    elem_t_i_0_mtx_all = elem_ht3ti_mtx[2]
    #################################################
    for i in range(dim):
        # print(i, "\n")
        xi2 = lobatto_pw[i, 0]
        w2 = lobatto_pw[i, 1]
        #with open ('qsh_isoprm.dat', 'w') as qsh_file:
        #     pass
        lag_xi2 = lagd.lagfunc(lobatto_pw, xi2)
        der_lag_dxi2 = lagd.der_lagfunc_dxi(lobatto_pw, xi2)
        for j in range(dim):
            xi1 = lobatto_pw[j, 0]
            w1 = lobatto_pw[j, 1]
            # with open ('qsh_isoprm.dat', 'a') as qsh_file:
            #         np.savetxt(qsh_file, qsh)
            lag_xi1 = lagd.lagfunc(lobatto_pw, xi1)
            der_lag_dxi1 = lagd.der_lagfunc_dxi(lobatto_pw, xi1)           
            jac = elem_jacobian_all[i, j]
            der_lag2d_dt = esmlrg.der_lag2d_dt_node_i(dim, lag_xi1, lag_xi2, \
                           der_lag_dxi1, der_lag_dxi2, jac)
            dirct_t = elem_updated_dir_all[i, j]
            der_dir0_dt = esmlrg.der_dir_0_dt(dim, der_lag2d_dt, elem_nodal_coorsys_all)
            der_xt_dt = esmlrg.der_x_t_dt(dim, der_lag2d_dt, elem_x_0_coor_all, \
                                   elem_displ_all)
            der_dirt_dt = esmlrg.der_dir_t_dt(dim, der_lag2d_dt, elem_updated_dir_all)
            der_x0_dt = esmlrg.der_x_0_dt(dim, der_lag2d_dt, elem_x_0_coor_all)
            # b_displ = esmlrg.b_disp_mtx(lobatto_pw, lag_xi1, lag_xi2, der_lag2d_dt,\
            #     dirct_t, der_xt_dt, der_dirt_dt, elem_t_i_mtx_all)
            ####################################
             
            b_displ = esmlrg.b_disp_mtx_0(lobatto_pw, lag_xi1, lag_xi2, der_lag2d_dt,\
                elem_nodal_coorsys_all[i, j, 2], der_x0_dt,  der_dir0_dt,\
                elem_t_i_0_mtx_all)
            #################################
            strain_vect = esmlrg.strain_vector (der_x0_dt, der_xt_dt, \
                   elem_nodal_coorsys_all[i, j, 2], elem_updated_dir_all[i, j], \
                   der_dir0_dt, der_dirt_dt)
            stress_vect = esmlrg.stress_vector(strain_vect, elastic_mtx) 
            
            elem_intern_force = elem_intern_force + \
                        np.transpose(b_displ) @ stress_vect * \
                            np.linalg.det(jac) * w1 * w2
    return elem_intern_force
    