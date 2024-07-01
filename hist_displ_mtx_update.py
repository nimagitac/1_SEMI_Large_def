import numpy as np
import surface_nurbs_isoprm as snip
import t_i_mtx_firstvar as tmf
import lagrange_der as lagd
from geomdl import exchange
import surface_geom_SEM as sgs


def deltabeta_to_deltaomega(a_0_1, a_0_2, omega_prv_step_vector, deltabeta_vector):
    '''
    This function take the initial nodal basis system and 
    the rotation angle vector (omega_prv_step) which belongs to the 
    previous step and calculate the delta omega vector to be 
    added to the previous omega vector to make it current.
    -Output:
    delta_omega which is a vector with 3 components
    
    '''
    r_mtx = tmf.r_mtx_node_i(omega_prv_step_vector)
    a_t_1 = r_mtx @ a_0_1
    a_t_2 = r_mtx @ a_0_2
    t_3 = tmf.t_3_mtx(a_t_1, a_t_2)
    delta_omega= t_3 @ deltabeta_vector
    return delta_omega



def initiate_x_0_ncoorsys_jacmtx_all(surface, lobatto_pw, element_boundaries_u,\
                       element_boundaries_v, x_0_coor_all, \
                       nodal_coorsys_all, jacobian_ncoorsys_all):
    '''
    -This function calculate the initial coordinate of all the nodes of all the elements
    -This fucntion calculates the nodal coordinate system
    based on the isoparametric assumption for each element
    and add these three vector: A_1 (a_0_1), A_2 (a_0_2) and A_3 (a_0_3) 
    to the first three members out of 5 members of
    hist_mtx (which is (number_element*number_element)*(number_node*number_node)*(5*3))
    -This function also calculate the Jacobina matrix from convected coordiate to
    initial nodal coordinate system according to "A robust non-linear mixed
    hybrid quadrilateral shell element" Wagenr, Gruttmann, 2005 Eq. (14)
    
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
            x_0_coor_all[i, j] = surface_isoprm.nodes_physical_coordinate()
            for r in range(dim):
                for s in range(dim):
                    a_0_1 = coorsys_tanvec_mtx[r, s, 0]
                    a_0_2 = coorsys_tanvec_mtx[r, s, 1]
                    a_0_3 = coorsys_tanvec_mtx[r, s, 2]
                    g1 = coorsys_tanvec_mtx[r, s, 3]
                    g2 = coorsys_tanvec_mtx[r, s, 4]
                    nodal_coorsys_all[i, j, r, s, 0] = a_0_1
                    nodal_coorsys_all[i, j, r, s, 1] = a_0_2
                    nodal_coorsys_all[i, j, r, s, 2] = a_0_3
                    jacobian_ncoorsys_all[i, j, r, s]=\
                    np.array([[g1 @ a_0_1, g1 @ a_0_2], [g2 @ a_0_1, g2 @ a_0_2]])
                   
    return (x_0_coor_all, nodal_coorsys_all, jacobian_ncoorsys_all)



def update_displ_hist(lobatto_pw, number_element_u, number_element_v, \
                   displ_compl_vect, node_displ_all, nodal_coorsys_all):
    '''
    This functin takes the number of elements,
    complete displacement vector (displ_compl) and complete history 
    matrix of deformation matrix (hist_displ_mtx) and update the
    hist_mtx according to new displ_mtx calculated at each 
    step. In addition to the displacement increment which can be directly added 
    to the previous displacement vector, it extracts the beta_1 and beta_2 increment
    from the displ_compl and transforms them to global coordinate system by using t_3. 
    The result is the omega increment which is added to the omega vector from the previous
    step.
    -Output:
    updated hist_mtx which is a (number_element*number_element)*(number_node*number_node)*(5*3)
    matrix
    '''
    dim = lobatto_pw.shape[0] #number of Lobatto points
    number_element_dof = 5 * dim**2 #in local coordinate system
    # number_element_u = len(element_boundaries_u) - 1
    # number_element_v = len(element_boundaries_v) - 1
    number_node_one_row = number_element_u*(dim - 1) + 1
    # number_node_one_column = number_element_v*(dim - 1) + 1
    for i_main in range(number_element_v):
        for j_main in range(number_element_u):
            node_1_number = i_main * (dim-1) * number_node_one_row +\
                            j_main * (dim-1) + 1
            connectivity = np.zeros(number_element_dof)
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
            print('connectivity is:\n', connectivity)
            connec_index = 0
            for m in range(dim):
                for n in range(dim): 
                    node_displ_all[i_main, j_main, m, n, 0, 0] = \
                        node_displ_all[i_main, j_main, m, n, 0, 0] + \
                            displ_compl_vect[connectivity[connec_index]]
                    node_displ_all[i_main, j_main, m, n, 0, 1] = \
                        node_displ_all[i_main, j_main, m, n, 0, 1] + \
                            displ_compl_vect[connectivity[connec_index + 1]]
                    node_displ_all[i_main, j_main, m, n, 0, 2] = \
                        node_displ_all[i_main, j_main, m, n, 0, 2] + \
                            displ_compl_vect[connectivity[connec_index + 2]]
                                            
                    beta_1 = displ_compl_vect[connectivity[connec_index + 3]]
                    beta_2 = displ_compl_vect[connectivity[connec_index + 4]]
                    delta_beta_vector = np.array([beta_1, beta_2])
                    omega_1_prv = node_displ_all[i_main, j_main, m, n, 1, 0]
                    omega_2_prv = node_displ_all[i_main, j_main, m, n, 1, 1]
                    omega_3_prv = node_displ_all[i_main, j_main, m, n, 1, 2]
                    omega_prv_step_vector =\
                        np.array([omega_1_prv, omega_2_prv, omega_3_prv])
                    a_0_1 = nodal_coorsys_all[i_main, j_main, m, n, 0]
                    a_0_2 = nodal_coorsys_all[i_main, j_main, m, n, 1]
                    delta_omega_vector = deltabeta_to_deltaomega(a_0_1, a_0_2,\
                        omega_prv_step_vector, delta_beta_vector)             
                    
                    node_displ_all[i_main, j_main, m, n, 1, 0] = \
                        node_displ_all[i_main, j_main, m, n, 1, 0] + delta_omega_vector[0]
                    node_displ_all[i_main, j_main, m, n, 1, 1] = \
                        node_displ_all[i_main, j_main, m, n, 1, 1] + delta_omega_vector[1]
                    node_displ_all[i_main, j_main, m, n, 1, 2] = \
                        node_displ_all[i_main, j_main, m, n, 1, 2] + delta_omega_vector[2]
                        
                    connec_index = connec_index + 5
    return node_displ_all
            
#    for m in range(dim):
#                 for n in range(dim): 
#                     node_displ_all[i_main, j_main, m, n, 3, 0] = \
#                         node_displ_all[i_main, j_main, m, n, 3, 0] + \
#                             displ_compl[connectivity[connec_index]]
#                     node_displ_all[i_main, j_main, m, n, 3, 1] = \
#                         node_displ_all[i_main, j_main, m, n, 3, 1] + \
#                             displ_compl[connectivity[connec_index + 1]]
#                     node_displ_all[i_main, j_main, m, n, 3, 2] = \
#                         node_displ_all[i_main, j_main, m, n, 3, 2] + \
#                             displ_compl[connectivity[connec_index + 2]]
                                            
#                     betha_1 = displ_compl[connectivity[connec_index + 3]]
#                     betha_2 = displ_compl[connectivity[connec_index + 4]]
#                     beta_vector = np.array([betha_1, betha_2])
#                     omega_1_prv = node_displ_all[i_main, j_main, m, n, 4, 0]
#                     omega_2_prv = node_displ_all[i_main, j_main, m, n, 4, 1]
#                     omega_3_prv = node_displ_all[i_main, j_main, m, n, 4, 2]
#                     omega_prv_step_vector =\
#                         np.array([omega_1_prv, omega_2_prv, omega_3_prv])
#                     a_0_1 = node_displ_all[i_main, j_main, m, n, 0]
#                     a_0_2 = node_displ_all[i_main, j_main, m, n, 1]
#                     delta_omega_vector = beta_to_deltaomega(a_0_1, a_0_2, omega_prv_step_vector, beta_vector)             
                    
#                     node_displ_all[i_main, j_main, m, n, 4, 0] = \
#                         node_displ_all[i_main, j_main, m, n, 4, 0] + delta_omega_vector[0]
#                     node_displ_all[i_main, j_main, m, n, 4, 1] = \
#                         node_displ_all[i_main, j_main, m, n, 4, 1] + delta_omega_vector[1]
#                     node_displ_all[i_main, j_main, m, n, 4, 2] = \
#                         node_displ_all[i_main, j_main, m, n, 4, 2] + delta_omega_vector[2]
                        
#                     connec_index = connec_index + 5                 
    
    
    
if __name__ == "__main__":
    data = exchange.import_json("scordelis_corrected.json") #  pinched_shell_kninsertion_changedeg.json pinched_shell.json rectangle_cantilever square square_kninsertion generic_shell_kninsertion foursided_curved_kninsertion foursided_curved_kninsertion2  rectangle_kninsertion
    # visualization(data)
    surfs = sgs.SurfaceGeo(data, 0, 0.25)
    p_1 = surfs.physical_crd(0., 0.)
    p_2 = surfs.physical_crd(1., 0.)
    p_3 = surfs.physical_crd(1., 1.)
    print("p_1:", p_1, "  p_2:", p_2, "  p_3:", p_3)
    print("\n\n")
    lobatto_pw_all = lagd.lbto_pw("node_weight_all.dat")
    number_element_u = 1
    number_element_v = 1
    i_main = 2
    if i_main == 1:
        lobatto_pw = lobatto_pw_all[1:3,:]
    else:  
        index = np.argwhere(lobatto_pw_all==i_main)
        lobatto_pw = lobatto_pw_all[index[0, 0] + 1:\
                            index[0, 0] + (i_main+1) +1, :]
    dim = lobatto_pw.shape[0]
    
    
    #####################################################################################
    print("dim is:", dim)
    number_lobatto_node = lobatto_pw.shape[0]
    number_node_one_row = number_element_u*(number_lobatto_node - 1) + 1
    number_node_one_column = number_element_v*(number_lobatto_node - 1) + 1
    node_global_a = 1 #  u = v = 0 . Four nodes at the tips of the square in u-v parametric space
    node_global_b = number_node_one_row
    node_global_c = node_global_a + number_element_v*(number_lobatto_node-1)\
                        *number_node_one_row
    node_global_d = node_global_c + number_node_one_row - 1
    total_dof = node_global_d * 5
    hist_mtx = np.random.uniform(0.5, 1, size=(number_element_u, number_element_v, dim, dim, 2, 3))
    nodal_coorsys = np.random.uniform(-1, 1, size=(number_element_u, number_element_v, dim, dim, 3, 3))
    # for i in range(number_element_v):
    #     for j in range(number_element_u):
    #         for r in range(dim):
    #             for s in range(dim):
    #                 hist_mtx[i, j, r, s] = np.random.uniform(-0.01, 0.01, size=(2,3))
    
    # displ_compl = np.random.uniform(0, 0.01, size=(total_dof))
    displ_compl = np.array([3.81051317e-03, 3.70935431e-03, 7.67199128e-03, 9.08921653e-03,
        7.65445923e-04, 1.48747421e-03, 9.26348091e-03, 1.59701444e-03,
        8.42850182e-03, 2.94031266e-03, 8.99073939e-03, 8.83575662e-03,
        5.84884547e-03, 9.51557365e-05, 5.56895777e-03, 8.42714007e-03,
        8.09599920e-03, 3.12179729e-04, 8.06873430e-03, 3.37561938e-04,
        6.09422453e-03, 3.18572677e-03, 8.11884232e-03, 8.78619014e-04,
        1.96866917e-03, 2.82855180e-03, 9.02744573e-04, 5.88541804e-03,
        4.10795423e-03, 5.41088869e-03, 8.70190805e-03, 7.43859003e-03,
        5.25129517e-03, 5.75928858e-03, 1.93310337e-03, 7.82838350e-03,
        4.27779177e-03, 8.01410451e-03, 6.25062423e-03, 5.78838154e-03,
        6.61302996e-03, 8.36463755e-03, 9.14542004e-03, 2.26432310e-03,
        5.41657083e-03])
    print("history matrix is:", hist_mtx)
    print("displacement:", displ_compl) 
    update_hist_mtx = update_displ_hist(lobatto_pw, number_element_u, number_element_v, \
                   displ_compl, hist_mtx, nodal_coorsys)
    # print(displ_compl.shape[0])
    
    print('\n\n updated history:', update_hist_mtx[0, 0, 0, 1])
    
    print('\n\n updated history:', update_hist_mtx[0, 0, 0, 0])
###########################################################################################################
    nodes_coorsys_displ_all = np.zeros((number_element_v, number_element_u,\
                                        i_main + 1, i_main + 1, 5, 3)) #To record the history of deformation. Dimensions are: number of elment in u and v, number of nodes in xi1 and xi2, (5 for A_1, A_2, A_3, u, omega) each has 3 components.
    jacobian_ncoorsys_all = np.zeros((number_element_v, number_element_u,\
                                        i_main + 1, i_main + 1, 2, 2)) 
    xcapt_coor_all = np.zeros((number_element_v, number_element_u, i_main + 1, i_main + 1, 3))
    element_boundaries_u = [0, 1] 
    element_boundaries_v = [0, 1]
    ncoorsys_jac_local= initiate_x_0_ncoorsys_jacmtx_all(surfs, lobatto_pw, element_boundaries_u,\
                        element_boundaries_v, xcapt_coor_all, nodes_coorsys_displ_all, jacobian_ncoorsys_all)
    xcapt_all = ncoorsys_jac_local[0]
    nodes_coorsys_displ_all = ncoorsys_jac_local[1]
    jacobian_ncoorsys_all = ncoorsys_jac_local[2]

