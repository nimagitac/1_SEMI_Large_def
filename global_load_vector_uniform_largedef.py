import numpy as np
import time
from geomdl import exchange
import surface_geom_SEM as sgs
import element_stiff_matrix_small as esmsml
import global_stiff_matrix_small as gsmsml

# For test
# uniform_load_x = 0
# uniform_load_y = 0
# uniform_load_z = 1

def shape_function_2d(lobatto_pw, xi1, xi2):
    """
    In this function the matrix N_2D in 
    (u1, u2, u3) = N_2D (u1_node, u2_node, u3_node, β1, β2) at each
    integration point xi1 and xi2 is claculated. 
    The conincidence of integration and nodal points is included to make
    the generation of N_2D fast.
    -Output
    a 3x3, 5 * len**
    """
    len = lobatto_pw.shape[0]
    shp_func = np.zeros((3, 5 * len**2))
    xi1_index = np.where(lobatto_pw == xi1)[0]
    xi2_index = np.where(lobatto_pw == xi2)[0]
    n_index = 5 * (xi2_index[0]*len + xi1_index[0])
    shp_func[0, n_index] = 1
    shp_func[1, n_index + 1] = 1
    shp_func[2, n_index + 2] = 1
    return shp_func

def shape_function_2d_general(lobatto_pw, lag_xi1, lag_xi2): #in a case that xi1 and xi2 are not coincided with SEM nodes
    len = lobatto_pw.shape[0]
    shp_func = np.zeros((3, 5 * len**2)) 
    len = lobatto_pw.shape[0] 
    num = 0
    for i in range(len):
        for j in range(len):
            shp_func[:, num:num+5] = np.array([[lag_xi1[j] * lag_xi2[i], 0, 0, 0, 0],
                        [0, lag_xi1[j] * lag_xi2[i], 0, 0, 0],\
                        [0, 0, lag_xi1[j] * lag_xi2[i], 0, 0]])
            num += 5
    return shp_func        
        
    
def element_load_vector(lobatto_pw, elem_jacobian_all,\
                uniform_load_x, uniform_load_y, uniform_load_z):
    '''
    -Output:
    Element load vector which is a numpy 1D array.
    '''
    len = lobatto_pw.shape[0]
   
    elem_load_v = np.zeros(5 * len**2) # 5 DOF at each node
    for i in range(len):
        # print(i, '\n')
        xi2 = lobatto_pw[i, 0]
        w2 = lobatto_pw[i, 1]
        for j in range(len):
            xi1 = lobatto_pw[j, 0]
            w1 = lobatto_pw[j, 1]
            
            jac_mtx = elem_jacobian_all[i, j]
            # print("jac2 and w1 and w2 is", jac2,'  ',w1,'  ',w2, '\n')
            elem_load_v_var =\
                np.transpose(shape_function_2d(lobatto_pw, xi1, xi2))\
                @ np.transpose((uniform_load_x, uniform_load_y, uniform_load_z))
            elem_load_v = elem_load_v + elem_load_v_var *\
                                np.linalg.det(jac_mtx) * w1 * w2
    return elem_load_v
            

def global_load_vector(lobatto_pw, jacobian_all, element_boundaries_u, element_boundaries_v,\
                        uniform_load_x, uniform_load_y, uniform_load_z):
    number_element_u = len(element_boundaries_u) - 1
    number_element_v = len(element_boundaries_v) - 1
    number_lobatto_node = lobatto_pw.shape[0]
    number_node_one_row = number_element_u*(number_lobatto_node - 1) + 1
    node_global_3 = number_element_v * (number_lobatto_node-1) * number_node_one_row +\
                    number_node_one_row #This is the node equal to u = v = 1 in IGA parametric space
    global_load_v = np.zeros((5*node_global_3))
    for i_main in range(number_element_v):
        for j_main in range(number_element_u):
            # print('row {} out of {}'.format(i_main, number_element_v-1))
            # node_1_u = element_boundaries_u[j_main]
            # node_1_v = element_boundaries_v[i_main]
            # node_3_u = element_boundaries_u[j_main+1]
            # node_3_v = element_boundaries_v[i_main+1]
            elem_jacobian_all = jacobian_all[i_main, j_main]
            elem_load_v = element_load_vector( lobatto_pw, elem_jacobian_all,\
                            uniform_load_x, uniform_load_y, uniform_load_z)
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
                    h = 5*i*number_lobatto_node + 5*j
                    connectivity[h] = (5*(node_1_number + p + j) - 5)
                    connectivity[h+1] = (5*(node_1_number + p + j) - 4)
                    connectivity[h+2] = (5*(node_1_number + p + j) - 3)
                    connectivity[h+3] = (5*(node_1_number + p + j) - 2)
                    connectivity[h+4] = (5*(node_1_number + p + j) - 1)
                p = p + number_node_one_row
            connectivity = connectivity.astype(int)
            for i in range(number_dof_element):
                global_load_v[connectivity[i]] = \
                global_load_v[connectivity[i]] + elem_load_v[i]
    return global_load_v


def load_vector_bc_applied(load_vector, bc):
    load_v = np.delete(load_vector, bc, 0)
    return load_v




# def element_load_vector(surface, lobatto_pw, uniform_load_x, uniform_load_y,\
#                         uniform_load_z, node_1_u, node_1_v, node_3_u,\
#                             node_3_v):
#     '''
#     -Output:
#     Element load vector which is a numpy 1D array.
#     '''
#     len = lobatto_pw.shape[0]
#     jac1_mtx = np.array(np.array([[(node_3_u - node_1_u)/2, 0],\
#                     [0, (node_3_v - node_1_v)/2]]))
#     # xi1 = 0.5
#     # xi2 = 0.0
#     # second_surf_der = surface.derivatives(xi1, xi2, order=2)    
#     # ders = surface.ders_uvt(0, second_surf_der)
#     # vector_1 = ders[0,:]
#     # vector_2 = ders[1,:]
#     # jac2 = np.linalg.norm((np.cross(vector_1,vector_2)))
#     # print("jac2 is", jac2, '\n')
#     elem_load_v = np.zeros(5 * len**2) # 5 DOF at each node
#     for i in range(len):
#         # print(i, '\n')
#         xi2 = lobatto_pw[i, 0]
#         w2 = lobatto_pw[i, 1]
#         for j in range(len):
#             xi1 = lobatto_pw[j, 0]
#             w1 = lobatto_pw[j, 1]
#             uv_var = esmsml.xi_to_uv(xi1, xi2, node_1_u,\
#                             node_1_v, node_3_u, node_3_v)
#             # print(surfs.physical_crd(uv_var[0], uv_var[1]))
#             second_surf_der = surface.derivatives(uv_var[0], uv_var[1], order=2)    
#             ders = surface.ders_uvt(0, second_surf_der)
#             vector_1 = ders[0,:]
#             vector_2 = ders[1,:]
#             jac2 = np.linalg.norm((np.cross(vector_1,vector_2)))
#             # print("jac2 and w1 and w2 is", jac2,'  ',w1,'  ',w2, '\n')
#             elem_load_v_var =\
#                 np.transpose(shape_function_2d(lobatto_pw, xi1, xi2))\
#                 @ np.transpose((uniform_load_x, uniform_load_y, uniform_load_z))
#             elem_load_v = elem_load_v + elem_load_v_var *\
#                                 np.linalg.det(jac1_mtx) * jac2 * \
#                                 w1 * w2
#     return elem_load_v
            

# def global_load_vector(surface, lobatto_pw, element_boundaries_u, element_boundaries_v,\
#                         uniform_load_x, uniform_load_y, uniform_load_z):
#     number_element_u = len(element_boundaries_u) - 1
#     number_element_v = len(element_boundaries_v) - 1
#     number_lobatto_node = lobatto_pw.shape[0]
#     number_node_one_row = number_element_u*(number_lobatto_node - 1) + 1
#     node_global_3 = number_element_v * (number_lobatto_node-1) * number_node_one_row +\
#                     number_node_one_row #This is the node equal to u = v = 1 in IGA parametric space
#     global_load_v = np.zeros((5*node_global_3))
#     for i_main in range(number_element_v):
#         for j_main in range(number_element_u):
#             # print('row {} out of {}'.format(i_main, number_element_v-1))
#             node_1_u = element_boundaries_u[j_main]
#             node_1_v = element_boundaries_v[i_main]
#             node_3_u = element_boundaries_u[j_main+1]
#             node_3_v = element_boundaries_v[i_main+1]
#             elem_load_v = element_load_vector(surface, lobatto_pw, \
#                             uniform_load_x, uniform_load_y,\
#                             uniform_load_z, node_1_u, node_1_v,\
#                             node_3_u, node_3_v)
#             # print('element load vector is : ', elem_load_v,'\n')
#             node_1_number = i_main * (number_lobatto_node-1) * number_node_one_row +\
#                             j_main * (number_lobatto_node-1) + 1
#             # node_2_number = i_main * (number_lobatto_node-1) * number_node_one_row +\
#             #                 (j_main+1) * (number_lobatto_node-1) + 1
#             # node_3_number = node_1_number + (number_lobatto_node-1) * number_node_one_row
#             # node_4_number = node_2_number + (number_lobatto_node-1) * number_node_one_row 
#             number_dof_element = 5*number_lobatto_node**2
#             connectivity = np.zeros(number_dof_element)
#             p = 0
#             for i in range(number_lobatto_node):
#                 for j in range(number_lobatto_node):
#                     h = 5*i*number_lobatto_node + 5*j
#                     connectivity[h] = (5*(node_1_number + p + j) - 5)
#                     connectivity[h+1] = (5*(node_1_number + p + j) - 4)
#                     connectivity[h+2] = (5*(node_1_number + p + j) - 3)
#                     connectivity[h+3] = (5*(node_1_number + p + j) - 2)
#                     connectivity[h+4] = (5*(node_1_number + p + j) - 1)
#                 p = p + number_node_one_row
#             connectivity = connectivity.astype(int)
#             for i in range(number_dof_element):
#                 global_load_v[connectivity[i]] = \
#                 global_load_v[connectivity[i]] + elem_load_v[i]
#     return global_load_v



if __name__ == '__main__':
    lobatto_pw = esmsml.lbto_pw("node_weight.dat")
    # lag_xi1 = esm.lagfunc(lobatto_pw, lobatto_pw[1, 0])
    # lag_xi2 = esm.lagfunc(lobatto_pw, lobatto_pw[2, 0])
    # sh1 = shape_function_2d(lobatto_pw, lobatto_pw[1, 0], lobatto_pw[2, 0])[0]
    # sh2 = shape_function_2d(lobatto_pw, lobatto_pw[1, 0], lobatto_pw[2, 0])[0]
    
    data = exchange.import_json("pinched_shell.json") #  pinched_shell_kninsertion_changedeg.json pinched_shell.json pinched_shell_half.json rectangle_cantilever square square_kninsertion generic_shell_kninsertion foursided_curved_kninsertion foursided_curved_kninsertion2  rectangle_kninsertion
    # gsm.visualization(data)
    surfs = sgs.SurfaceGeo(data, 0, 3)
    # load = element_load_vector(surfs, lobatto_pw, 0, 0,\
    #                     10, 0, 0, 1,\
    #                         1)
  
    bc_h_bott = [1, 0, 1, 0, 1] #pinched shell. zero means clamped DOF
    bc_h_top = [0, 1, 0, 1, 0]
    bc_v_left = [1, 1, 0, 1, 0]
    bc_v_right = [0, 1, 1, 1, 0]
    
    time1 = time.time()
   
    u_manual = [0, 0.5, 1]
    v_manual = [0, 1]
    mesh = gsmsml.mesh_func(surfs, u_manual, v_manual)
    # mesh = gsm.mesh_func(surfs)
    element_boundaries_u = mesh[0]
    element_boundaries_v = mesh[1]           
    bc1 = gsmsml.global_boundary_condition(lobatto_pw, bc_h_bott, bc_h_top,\
                                bc_v_left, bc_v_right, element_boundaries_u,\
                                element_boundaries_v)
    global_load = global_load_vector(surfs, lobatto_pw, element_boundaries_u, element_boundaries_v)
    print(global_load)
    print("End")
    

    
    
