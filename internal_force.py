import numpy as np
import lagrange_der as lagd
import element_stiffness_matrix_largedef as esmlrg


def element_intern_force(lobatto_pw, elem_x_0_coor_all, \
                          elem_nodal_coorsys_all, elem_jacobian_all,\
                          elem_displ_all, elastic_mtx):
    '''
    This function calculate the internal force of the elements or Tanspose(B)σ
    
    -Output:
    A vector with the dimension of 5*num_node**2 
    '''
    dim = lobatto_pw.shape[0]
    elem_intern_force = np.zeros(5 * dim**2)
    elem_updated_dir_all = esmlrg.elem_update_dir_all(dim, elem_nodal_coorsys_all, elem_displ_all) 
    elem_ht3ti_mtx = esmlrg.elem_hcapt_t3_ti_mtx_all(dim, elem_nodal_coorsys_all, elem_displ_all)
    # elem_hcatp_mtx_all = elem_ht3ti_mtx[0]
    # elem_t_3_mtx_all = elem_ht3ti_mtx[1]
    elem_t_i_mtx_all = elem_ht3ti_mtx[2]
    ##################################################
    # elem_ht3ti_mtx = esmlrg.elem_hcapt_t3_ti_mtx_all(dim, elem_nodal_coorsys_all,np.zeros((dim, dim, 2, 3)))
    # elem_hcatp_mtx_all = elem_ht3ti_mtx[0]
    # elem_t_3_mtx_all = elem_ht3ti_mtx[1]
    # elem_t_i_0_mtx_all = elem_ht3ti_mtx[2]
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
            b_displ = esmlrg.b_disp_mtx(lobatto_pw, lag_xi1, lag_xi2, der_lag2d_dt,\
                dirct_t, der_xt_dt, der_dirt_dt, elem_t_i_mtx_all)
            strain_vect = esmlrg.strain_vector (der_x0_dt, der_xt_dt, \
                   elem_nodal_coorsys_all[i, j, 2], elem_updated_dir_all[i, j], \
                   der_dir0_dt, der_dirt_dt)
            stress_vect = esmlrg.stress_vector(strain_vect, elastic_mtx) 
            
            elem_intern_force = elem_intern_force + \
                        np.transpose(b_displ) @ stress_vect * \
                            np.linalg.det(jac) * w1 * w2
    return elem_intern_force

def global_intern_force(lobatto_pw, element_boundaries_u, element_boundaries_v,\
                        x_0_coor_all, nodal_coorsys_all, jacobian_all, node_displ_all,\
                        elastic_mtx):
    '''
    This function assembles the element internal forces and result in the 
    global internal force vector
    
    -Output:
    A vector with the dimension of the all DOFs
    
    '''
    number_element_u = len(element_boundaries_u) - 1
    number_element_v = len(element_boundaries_v) - 1
    number_lobatto_node = lobatto_pw.shape[0]
    number_node_one_row = number_element_u*(number_lobatto_node - 1) + 1
    node_global_3 = number_element_v * (number_lobatto_node-1) * number_node_one_row +\
                    number_node_one_row #This is the node equal to u = v = 1 in IGA parametric space
    global_intern_frc = np.zeros((5*node_global_3))
    for i_main in range(number_element_v):
        for j_main in range(number_element_u):
            # print('row {} out of {}'.format(i_main, number_element_v-1))
            # node_1_u = element_boundaries_u[j_main]
            # node_1_v = element_boundaries_v[i_main]
            # node_3_u = element_boundaries_u[j_main+1]
            # node_3_v = element_boundaries_v[i_main+1]
            
            elem_x_0_coor_all = x_0_coor_all[i_main, j_main] #### To be changed in meshing
            elem_nodal_coorsys_all = nodal_coorsys_all[i_main, j_main]  #### To be changed in meshing
            elem_jacobian_all = jacobian_all[i_main, j_main]  #### To be changed in meshing
            elem_displ_all = node_displ_all[i_main, j_main]  #### To be changed in 
            elem_intern_frc = element_intern_force(lobatto_pw, elem_x_0_coor_all, \
                          elem_nodal_coorsys_all, elem_jacobian_all,\
                          elem_displ_all, elastic_mtx)
            # print('element load vector is : ', elem_load_v,'\n')
            node_1_number = i_main * (number_lobatto_node - 1) * number_node_one_row +\
                            j_main * (number_lobatto_node - 1) + 1
            # node_2_number = i_main * (number_lobatto_node-1) * number_node_one_row +\
            #                 (j_main+1) * (number_lobatto_node-1) + 1
            # node_3_number = node_1_number + (number_lobatto_node-1) * number_node_one_row
            # node_4_number = node_2_number + (number_lobatto_node-1) * number_node_one_row 
            number_dof_element = 5 * number_lobatto_node**2
            connectivity = np.zeros(number_dof_element)
            p = 0
            for i in range(number_lobatto_node):
                for j in range(number_lobatto_node):
                    h = 5 * i * number_lobatto_node + 5 * j
                    connectivity[h] = (5*(node_1_number + p + j) - 5)
                    connectivity[h+1] = (5*(node_1_number + p + j) - 4)
                    connectivity[h+2] = (5*(node_1_number + p + j) - 3)
                    connectivity[h+3] = (5*(node_1_number + p + j) - 2)
                    connectivity[h+4] = (5*(node_1_number + p + j) - 1)
                p = p + number_node_one_row
            connectivity = connectivity.astype(int)
            for i in range(number_dof_element):
                global_intern_frc[connectivity[i]] = \
                global_intern_frc[connectivity[i]] + elem_intern_frc[i]
    return global_intern_frc
    
  