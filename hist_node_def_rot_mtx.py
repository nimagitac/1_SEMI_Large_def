import numpy as np
import surface_nurbs_isoprm as snip
import t_i_mtx_firstvar as tmf


def beta_to_omega(a_0_1, a_0_2, omega_prv_step, beta_vector):
    '''
    This function take the initial nodal basis system and 
    the rotation angle vector (omega) which belongs to the 
    previous step and calculate the delta omega vector to be 
    added to the previous omega vector to make it current.
    -Output:
    delta_omega which is a vector with 3 components
    '''
    r_mtx = tmf.r_mtx_i(omega_prv_step)
    a_t_1 = r_mtx @ a_0_1
    a_t_2 = r_mtx @ a_0_2
    t_3 = tmf.t_3_mtx(a_t_1, a_t_2)
    delta_omega= t_3 @ beta_vector
    return delta_omega


def initiate_nodal_coorsys(surface, lobatto_pw, element_boundaries_u,\
                       element_boundaries_v, hist_mtx):
    '''
    This fucntion calculates the nodal coordinate system
    based on the isoparametric assumption for each element
    and add these three vector: A_1 (a_0_1), A_2 (a_0_2) and A_3 (a_0_3) 
    to the first three members out of 5 members of
    hist_mtx (which is numel*numel*numnode*numnode*5*3)
    
    '''
    dim = lobatto_pw.shape[0]
    number_element_u = len(element_boundaries_u) - 1
    number_element_v = len(element_boundaries_v) - 1
    for i in range(number_element_v):
        for j in range(number_element_u):
            node_1_u = element_boundaries_u[j]
            node_1_v = element_boundaries_v[i]
            node_3_u = element_boundaries_u[j + 1]
            node_3_v = element_boundaries_v[i + 1]
            surface_isoprm = snip.SurfaceGeneratedSEM(surface, lobatto_pw, node_1_u,\
                  node_1_v, node_3_u, node_3_v)
            coorsys_tanvec_mtx = surface_isoprm.coorsys_director_tanvec_allnodes()
            for r in range(dim):
                for s in range(dim):
                    hist_mtx[i, j, r, s, 0] = coorsys_tanvec_mtx[r, s, 0]
                    hist_mtx[i, j, r, s, 1] = coorsys_tanvec_mtx[r, s, 1]
                    hist_mtx[i, j, r, s, 2] = coorsys_tanvec_mtx[r, s, 2]
    return hist_mtx



def make_dipl_hist(lobatto_pw, element_boundaries_u, element_boundaries_v, \
                   displ, hist_mtx):
    '''
    This functin takes the number of elements through element boundaries,
    complete displacement vector (displ_mtx) and complete history 
    matrix of deformation matrix (hist_mtx) and update the
    hist_mtx according to new displ_mtx calculated at each 
    step.
    '''
    dim = lobatto_pw.shape[0] #number of Lobatto points
    number_element_u = len(element_boundaries_u) - 1
    number_element_v = len(element_boundaries_v) - 1
    number_node_one_row = number_element_u*(dim - 1) + 1
    number_node_one_column = number_element_v*(dim - 1) + 1
    for i in range(number_element_v):
        for j in range(number_element_u):
            node_1_number = i * (dim-1) * number_node_one_row +\
                            j * (dim-1) + 1
            connectivity = np.zeros(5*dim**2)
            p = 0
            for r in range(dim):
                for s in range(dim):
                    h = 5 * r * dim + 5 * s #The first LOCAL element dof assigned to the node (r,s) starting from 0 to be used in Python
                    connectivity[h] = (5 * (node_1_number + p + s) - 5)
                    connectivity[h+1] = (5 * (node_1_number + p + s) - 4)
                    connectivity[h+2] = (5 * (node_1_number + p + s) - 3)
                    connectivity[h+3] = (5 * (node_1_number + p + s) - 2)
                    connectivity[h+4] = (5 * (node_1_number + p + s) - 1)
                p = p + number_node_one_row
            connectivity = connectivity.astype(int)
            
            number_element_dof = 5*dim**2
            connec_index = 0
            for m in range(number_element_dof):
                for n in range(number_element_dof):                    
                    hist_mtx[i, j, m, n, 3, 0] = \
                        hist_mtx[i, j, m, n, 3, 0] + displ[connectivity[connec_index]]
                    hist_mtx[i, j, m, n, 3, 1] = \
                        hist_mtx[i, j, m, n, 3, 1] + displ[connectivity[connec_index + 1]]
                    hist_mtx[i, j, m, n, 3, 2] = \
                        hist_mtx[i, j, m, n, 3, 2] + displ[connectivity[connec_index + 2]]
                    hist_mtx[i, j, m, n, 4, 0] = \
                        hist_mtx[i, j, m, n, 4, 0] + displ[connectivity[connec_index + 3]]
                    hist_mtx[i, j, m, n, 4, 1] = \
                        hist_mtx[i, j, m, n, 4, 1] + displ[connectivity[connec_index + 4]]
                    hist_mtx[i, j, m, n, 4, 2] = \
                        hist_mtx[i, j, m, n, 4, 2] + displ[connectivity[connec_index + 5]]
                        
                    connec_index = connec_index + 5
            
                    
    