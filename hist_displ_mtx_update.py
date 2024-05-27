import numpy as np
import surface_nurbs_isoprm as snip
import t_i_mtx_firstvar as tmf
import lagrange_der as lagd
from geomdl import exchange
import surface_geom_SEM as sgs


def beta_to_deltaomega(a_0_1, a_0_2, omega_prv_step_vector, beta_vector):
    '''
    This function take the initial nodal basis system and 
    the rotation angle vector (omega_prv_step) which belongs to the 
    previous step and calculate the delta omega vector to be 
    added to the previous omega vector to make it current.
    -Output:
    delta_omega which is a vector with 3 components
    
    '''
    r_mtx = tmf.r_mtx_i(omega_prv_step_vector)
    a_t_1 = r_mtx @ a_0_1
    a_t_2 = r_mtx @ a_0_2
    t_3 = tmf.t_3_mtx(a_t_1, a_t_2)
    delta_omega= t_3 @ beta_vector
    return delta_omega


def initiate_ncoorsys_plus_jacmtx(surface, lobatto_pw, element_boundaries_u,\
                       element_boundaries_v, nodes_coorsys_displ, jacobian_ncoorsys_all):
    '''
    This fucntion calculates the nodal coordinate system
    based on the isoparametric assumption for each element
    and add these three vector: A_1 (a_0_1), A_2 (a_0_2) and A_3 (a_0_3) 
    to the first three members out of 5 members of
    hist_mtx (which is (number_element*number_element)*(number_node*number_node)*(5*3))
    
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
                    a_0_1 = coorsys_tanvec_mtx[r, s, 0]
                    a_0_2 = coorsys_tanvec_mtx[r, s, 1]
                    a_0_3 = coorsys_tanvec_mtx[r, s, 2]
                    g1 = coorsys_tanvec_mtx[r, s, 3]
                    g2 = coorsys_tanvec_mtx[r, s, 4]
                    nodes_coorsys_displ[i, j, r, s, 0] = a_0_1
                    nodes_coorsys_displ[i, j, r, s, 1] = a_0_2
                    nodes_coorsys_displ[i, j, r, s, 2] = a_0_3
                    pp= g1 @ a_0_1
                    pp1= np.dot(g1, a_0_1)
                    jacobian_ncoorsys_all[i, j, r, s]= np.array([[g1 @ a_0_1, g1 @ a_0_2], [g2 @ a_0_1, g2 @ a_0_2]])
    return (nodes_coorsys_displ, jacobian_ncoorsys_all)



def update_displ_hist(lobatto_pw, number_element_u, number_element_v, \
                   displ_compl, nodes_coorsys_displ):
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
                    nodes_coorsys_displ[i_main, j_main, m, n, 3, 0] = \
                        nodes_coorsys_displ[i_main, j_main, m, n, 3, 0] + \
                            displ_compl[connectivity[connec_index]]
                    nodes_coorsys_displ[i_main, j_main, m, n, 3, 1] = \
                        nodes_coorsys_displ[i_main, j_main, m, n, 3, 1] + \
                            displ_compl[connectivity[connec_index + 1]]
                    nodes_coorsys_displ[i_main, j_main, m, n, 3, 2] = \
                        nodes_coorsys_displ[i_main, j_main, m, n, 3, 2] + \
                            displ_compl[connectivity[connec_index + 2]]
                                            
                    betha_1 = displ_compl[connectivity[connec_index + 3]]
                    betha_2 = displ_compl[connectivity[connec_index + 4]]
                    beta_vector = np.array([betha_1, betha_2])
                    omega_1_prv = nodes_coorsys_displ[i_main, j_main, m, n, 4, 0]
                    omega_2_prv = nodes_coorsys_displ[i_main, j_main, m, n, 4, 1]
                    omega_3_prv = nodes_coorsys_displ[i_main, j_main, m, n, 4, 2]
                    omega_prv_step_vector =\
                        np.array([omega_1_prv, omega_2_prv, omega_3_prv])
                    a_0_1 = nodes_coorsys_displ[i_main, j_main, m, n, 0]
                    a_0_2 = nodes_coorsys_displ[i_main, j_main, m, n, 1]
                    delta_omega_vector = beta_to_deltaomega(a_0_1, a_0_2, omega_prv_step_vector, beta_vector)             
                    
                    nodes_coorsys_displ[i_main, j_main, m, n, 4, 0] = \
                        nodes_coorsys_displ[i_main, j_main, m, n, 4, 0] + delta_omega_vector[0]
                    nodes_coorsys_displ[i_main, j_main, m, n, 4, 1] = \
                        nodes_coorsys_displ[i_main, j_main, m, n, 4, 1] + delta_omega_vector[1]
                    nodes_coorsys_displ[i_main, j_main, m, n, 4, 2] = \
                        nodes_coorsys_displ[i_main, j_main, m, n, 4, 2] + delta_omega_vector[2]
                        
                    connec_index = connec_index + 5
    return nodes_coorsys_displ
            
                    
    
    
    
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
    
    
    #####################################################################################
    # print("dim is:", dim)
    # number_lobatto_node = lobatto_pw.shape[0]
    # number_node_one_row = number_element_u*(number_lobatto_node - 1) + 1
    # number_node_one_column = number_element_v*(number_lobatto_node - 1) + 1
    # node_global_a = 1 #  u = v = 0 . Four nodes at the tips of the square in u-v parametric space
    # node_global_b = number_node_one_row
    # node_global_c = node_global_a + number_element_v*(number_lobatto_node-1)\
    #                     *number_node_one_row
    # node_global_d = node_global_c + number_node_one_row - 1
    # total_dof = node_global_d * 5
    # hist_mtx = np.random.uniform(0.5, 1, size=(number_element_u, number_element_v, dim, dim, 5, 3))
    # for i in range(number_element_v):
    #     for j in range(number_element_u):
    #         for r in range(dim):
    #             for s in range(dim):
    #                 hist_mtx[i, j, r, s, 3:5, :] = np.random.uniform(-0.01, 0.01, size=(2,3))
    # hist_mtx =np.array([[[[[[ 5.10392199e-01,  6.20121686e-01,  9.62887746e-01],
    #                [ 8.98183183e-01,  8.07365241e-01,  9.81782390e-01],
    #                [ 9.57496503e-01,  6.22985134e-01,  6.30505495e-01],
    #                [-9.32965878e-03, -8.10178420e-03,  5.22167537e-03],
    #                [-9.96982671e-03, -6.23812780e-03, -6.52004910e-03]],
    #               [[ 9.67808067e-01,  7.66511589e-01,  8.66098753e-01],
    #                [ 7.60003879e-01,  9.77013174e-01,  5.88317424e-01],
    #                [ 8.24084107e-01,  9.83668747e-01,  6.04654452e-01],
    #                [ 5.57993335e-03,  3.34648826e-03,  3.46193036e-03],
    #                [ 1.92574264e-03, -9.11287412e-03, -7.30148856e-03]]],
    #              [[[ 8.49124697e-01,  7.56808044e-01,  9.61265197e-01],
    #                [ 9.70594789e-01,  5.23040880e-01,  5.07768899e-01],
    #                [ 8.16088497e-01,  5.47092738e-01,  5.33872056e-01],
    #                [-4.17156409e-03, -3.27360876e-03, -7.56617528e-03],
    #                [ 9.95248245e-03, -7.49564164e-03,  5.44680680e-03]],
    #               [[ 5.38799188e-01,  5.62158639e-01,  9.88266100e-01],
    #                [ 7.58655417e-01,  7.03329240e-01,  5.38504846e-01],
    #                [ 9.55126576e-01,  9.40891346e-01,  6.11265709e-01],
    #                [-9.01191836e-03,  7.65624291e-03, -1.36948010e-03],
    #                [-4.69373572e-03,  2.64593970e-03, -5.13116178e-03]]]],
    #             [[[[ 9.65430493e-01,  6.78479122e-01,  9.49200266e-01],
    #                [ 5.39033902e-01,  5.03477360e-01,  6.03302006e-01],
    #                [ 6.48390982e-01,  9.02824461e-01,  8.23083879e-01],
    #                [-8.22574030e-03, -9.14144196e-04, -9.41918576e-03],
    #                [ 6.30631900e-03,  7.36331972e-03, -6.12084743e-03]],
    #               [[ 8.66362690e-01,  9.15625812e-01,  7.77679806e-01],
    #                [ 8.76626804e-01,  6.84172196e-01,  8.66576222e-01],
    #                [ 6.45826530e-01,  9.34146950e-01,  7.75243546e-01],
    #                [ 9.68663171e-03, -9.88030911e-03,  9.31125565e-03],
    #                [ 8.87988625e-03,  7.18426784e-03,  8.85738912e-03]]],
    #              [[[ 5.85666127e-01,  6.95353180e-01,  6.70484091e-01],
    #                [ 6.93631556e-01,  8.44368243e-01,  6.95262853e-01],
    #                [ 5.71536788e-01,  8.33196813e-01,  7.88785004e-01],
    #                [-2.99999445e-03,  3.19434951e-03, -4.97265438e-03],
    #                [-3.72727907e-03, -1.16150094e-03, -3.54912164e-03]],
    #               [[ 5.41325805e-01,  9.30440871e-01,  5.28663539e-01],
    #                [ 9.70994054e-01,  5.35939936e-01,  8.74365918e-01],
    #                [ 8.60406929e-01,  9.58753494e-01,  5.34711032e-01],
    #                [-6.98724437e-03,  3.68034347e-03, -5.48391738e-03],
    #                [-1.56898496e-03,  3.97930480e-04, -6.87881820e-03]]]]],
    #            [[[[[ 9.60053963e-01,  6.37336333e-01,  5.98982095e-01],
    #                [ 8.26698985e-01,  6.67805344e-01,  8.31876360e-01],
    #                [ 6.55180522e-01,  7.09738044e-01,  8.94176991e-01],
    #                [ 3.04089148e-03, -9.76839652e-03,  3.70862880e-03],
    #                [ 3.82449289e-03, -2.47816893e-03,  1.41661138e-03]],
    #               [[ 5.03455971e-01,  7.20857876e-01,  8.08034403e-01],
    #                [ 7.96857834e-01,  8.22222384e-01,  5.97515306e-01],
    #                [ 5.18347735e-01,  7.19520573e-01,  6.58344139e-01],
    #                [ 2.92321051e-03,  4.66143102e-03, -5.03630183e-03],
    #                [-7.58112113e-03, -7.40881435e-03,  7.69251271e-03]]],
    #              [[[ 9.85662314e-01,  7.83748559e-01,  5.09137307e-01],
    #                [ 9.83640030e-01,  6.88563358e-01,  5.55694963e-01],
    #                [ 9.30063074e-01,  5.35063515e-01,  7.30759495e-01],
    #                [-7.47740890e-03, -4.58815987e-03, -8.44042603e-03],
    #                [-3.45342046e-03, -4.98299581e-04,  6.76304374e-03]],
    #               [[ 8.38675056e-01,  7.56642183e-01,  7.99059353e-01],
    #                [ 9.33532201e-01,  8.18303087e-01,  7.81433488e-01],
    #                [ 7.27900733e-01,  6.37673275e-01,  9.19615820e-01],
    #                [-3.36110561e-03,  5.86700330e-03, -8.39568246e-03],
    #                [ 4.20378549e-03,  4.82864106e-03,  1.13981591e-03]]]],
    #             [[[[ 6.12316435e-01,  9.51273153e-01,  6.63375880e-01],
    #                [ 6.46174698e-01,  5.31869440e-01,  9.45978651e-01],
    #                [ 9.29971720e-01,  9.33673474e-01,  9.77391278e-01],
    #                [ 4.93781258e-03, -9.79682792e-04,  8.08745951e-03],
    #                [-5.89717734e-04,  3.41299207e-03, -1.81021152e-03]],
    #               [[ 7.76439848e-01,  8.56369059e-01,  5.63326614e-01],
    #                [ 5.38013044e-01,  7.85086437e-01,  9.82382396e-01],
    #                [ 8.85152883e-01,  7.99575482e-01,  7.10284803e-01],
    #                [ 2.20195976e-03, -8.75605826e-03, -4.05914252e-03],
    #                [-1.59854062e-03, -1.97411115e-03,  7.17284402e-03]]],
    #              [[[ 9.23154922e-01,  9.63583193e-01,  6.75526362e-01],
    #                [ 5.94193584e-01,  5.15235926e-01,  9.61825507e-01],
    #                [ 5.89756003e-01,  5.13717125e-01,  8.24998908e-01],
    #                [ 3.41827583e-03,  5.26683409e-03,  2.53154804e-03],
    #                [ 1.90026157e-03, -2.36184408e-03,  4.45958913e-03]],
    #               [[ 7.55743890e-01,  9.11481149e-01,  5.25291054e-01],
    #                [ 9.80614958e-01,  7.19283534e-01,  7.31831751e-01],
    #                [ 5.90299521e-01,  5.27008839e-01,  5.97452896e-01],
    #                [ 1.86230025e-03, -2.18029858e-03,  7.78889965e-03],
    #                [ 2.71757516e-03, -7.89188198e-03, -3.38029838e-03]]]]]])

    # # displ_compl = np.random.uniform(0, 0.01, size=(total_dof))
    # displ_compl = np.array([3.81051317e-03, 3.70935431e-03, 7.67199128e-03, 9.08921653e-03,
    #     7.65445923e-04, 1.48747421e-03, 9.26348091e-03, 1.59701444e-03,
    #     8.42850182e-03, 2.94031266e-03, 8.99073939e-03, 8.83575662e-03,
    #     5.84884547e-03, 9.51557365e-05, 5.56895777e-03, 8.42714007e-03,
    #     8.09599920e-03, 3.12179729e-04, 8.06873430e-03, 3.37561938e-04,
    #     6.09422453e-03, 3.18572677e-03, 8.11884232e-03, 8.78619014e-04,
    #     1.96866917e-03, 2.82855180e-03, 9.02744573e-04, 5.88541804e-03,
    #     4.10795423e-03, 5.41088869e-03, 8.70190805e-03, 7.43859003e-03,
    #     5.25129517e-03, 5.75928858e-03, 1.93310337e-03, 7.82838350e-03,
    #     4.27779177e-03, 8.01410451e-03, 6.25062423e-03, 5.78838154e-03,
    #     6.61302996e-03, 8.36463755e-03, 9.14542004e-03, 2.26432310e-03,
    #     5.41657083e-03])
    # print("history matrix is:", hist_mtx)
    # print("displacement:", displ_compl) 
    # update_hist_mtx = update_displ_hist(lobatto_pw, number_element_u, number_element_v, \
    #                displ_compl, hist_mtx)
    # # print(displ_compl.shape[0])
    
    # print('\n\n updated history:', update_hist_mtx[0, 0, 0, 1, 3:5])
    
    # print('\n\n updated history:', update_hist_mtx[0, 1, 0, 0, 3:5])
###########################################################################################################
nodes_coorsys_displ_all = np.zeros((number_element_v, number_element_u,\
                                    i_main + 1, i_main + 1, 5, 3)) #To record the history of deformation. Dimensions are: number of elment in u and v, number of nodes in xi1 and xi2, (5 for A_1, A_2, A_3, u, omega) each has 3 components.
jacobian_ncoorsys_all = np.zeros((number_element_v, number_element_u,\
                                    i_main + 1, i_main + 1, 2, 2)) 
element_boundaries_u = [0, 1] 
element_boundaries_v = [0, 1]
ncoorsys_jac_local= initiate_ncoorsys_plus_jacmtx(surfs, lobatto_pw, element_boundaries_u,\
                       element_boundaries_v, nodes_coorsys_displ_all, jacobian_ncoorsys_all)
nodes_coorsys_displ_all = ncoorsys_jac_local[0]
jacobian_ncoorsys_all = ncoorsys_jac_local[1]

