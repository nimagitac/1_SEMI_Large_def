import numpy as np
import lagrange_der as lagder
from geomdl import exchange
import surface_geom_SEM as sgs
import global_stiff_matrix_small as gsmsml
import hist_displ_mtx_update as hidmtu



#INPUT *************************************************************
u_analytic = 0.3020247
elastic_modulus = 4.32*10**8
nu = 0
uniform_load_x = 0
uniform_load_y = 0
uniform_load_z = 90
bc_h_bott = [0, 1, 0, 1, 0] #Scordelis shell. zero means clamped DOF
bc_h_top = [1, 0, 1, 0, 1]
bc_v_left = [1, 1, 1, 1, 1]
bc_v_right = [0, 1, 1, 1, 0]

#*************************************************************




print("\nImporting Lobatto points and weights from data-base ...")
# time.sleep(1)
lobatto_pw_all = lagder.lbto_pw("node_weight_all.dat")
print("\nImporting geometry from json file ...")
# time.sleep(1)
data = exchange.import_json("scordelis_corrected.json") #  pinched_shell_kninsertion_changedeg.json pinched_shell.json rectangle_cantilever square square_kninsertion generic_shell_kninsertion foursided_curved_kninsertion foursided_curved_kninsertion2  rectangle_kninsertion
    # visualization(data)
surfs = sgs.SurfaceGeo(data, 0, 0.25)

min_order_elem = int(input("\nEnter the minimum order of elements (minimum order = 1):\n"))
max_order_elem = int(input("Enter the maximum order of elements (maximum order = 30):\n"))
min_number_elem = int(input("\nEnter the minimum number of elements in u and v direction:\n"))
max_number_elem = int(input("Enter the maximum number of elements in u and v direction:\n"))
print("\nEnter the order of continuity at knots to be used for auto detection of elements boundaries in u direction")
print("The default value is '1'")
c_order_u =int(input())
print("\nEnter the order of continuity at knots to be used for auto detection of elements boundaries in v direction")
print("The default value is '1'")
c_order_v =int(input())

i_main = min_order_elem
while i_main <= max_order_elem:
    if i_main==1:
        lobatto_pw = lobatto_pw_all[1:3,:]
    else:
        index = np.argwhere(lobatto_pw_all==i_main)
        lobatto_pw = lobatto_pw_all[index[0, 0] + 1:\
                            index[0, 0] + (i_main+1) + 1, :]
    j_main = min_number_elem
    # elemnum_displm_array = np.zeros((max_number_elem - min_number_elem + 1, 2))
    # time_assembling = np.zeros((max_number_elem - min_number_elem + 1, 2))
    # time_solver = np.zeros((max_number_elem - min_number_elem + 1, 2))
    # dof_displm_array = np.zeros((max_number_elem - min_number_elem + 1, 2)) 
    # dof_time_assembling = np.zeros((max_number_elem - min_number_elem + 1, 2)) 
    # dof_time_solver = np.zeros((max_number_elem - min_number_elem + 1, 2)) 
    # cond_elem =np.zeros((max_number_elem - min_number_elem + 1, 2)) 
    elemnum_counter = 0
    while j_main <= max_number_elem:
        print("\n\n\nNumber of elements manually given in u and v: {}    Order of elements: {} ".\
            format(str(j_main)+'x'+str(j_main), i_main))
        print("\nProgram starts to generate mesh according to continuity at knots and manual input of number of elements ...") 
        u_manual = np.linspace(0, 1, j_main + 1) #np.linspace(a, b, c) divide line ab to c-1 parts or add c points to it.
        v_manual = np.linspace(0, 1, j_main + 1)
        mesh = gsmsml.mesh_func(surfs, u_manual, v_manual, c_order_u, c_order_v)
        element_boundaries_u = mesh[0]
        element_boundaries_v = mesh[1]
        
        bc = gsmsml.global_boundary_condition(lobatto_pw, bc_h_bott, bc_h_top,\
                                        bc_v_left, bc_v_right, element_boundaries_u,\
                                        element_boundaries_v)
        number_element_u = len(element_boundaries_u) - 1
        number_element_v = len(element_boundaries_v) - 1
        number_lobatto_node = lobatto_pw.shape[0]
        number_node_one_row = number_element_u*(number_lobatto_node - 1) + 1
        number_node_one_column = number_element_v*(number_lobatto_node - 1) + 1
        node_global_a = 1 #  u = v = 0 . Four nodes at the tips of the square in u-v parametric space
        node_global_b = number_node_one_row
        node_global_c = node_global_a + number_element_v*(number_lobatto_node-1)\
                            *number_node_one_row
        node_global_d = node_global_c + number_node_one_row - 1
        total_dof = node_global_d * 5
        displm_complete = np.zeros(total_dof)
        
        
        nodes_coorsys_displ_all = np.zeros((number_element_v, number_element_u,\
                                    i_main + 1, i_main + 1, 5, 3)) #To record the history of deformation. Dimensions are: number of elment in u and v, number of nodes in xi1 and xi2, (5 for A_1, A_2, A_3, u, omega) each has 3 components.
        jacobian_ncoorsys_all = np.zeros((number_element_v, number_element_u,\
                                    i_main + 1, i_main + 1, 2, 2))
        coorsys_jac_temp = hidmtu.initiate_ncoorsys_plus_jacmtx(surfs,\
                                      lobatto_pw, element_boundaries_u,\
                       element_boundaries_v, nodes_coorsys_displ_all, \
                                                  jacobian_ncoorsys_all)
        nodes_coorsys_displ_all = coorsys_jac_temp[0]
        jacobian_ncoorsys_all = coorsys_jac_temp[1] #To avoide repitition calculation of Jacobian matrix, the Jacobian matrix is calculated for all elements at all GLL points
        
        pass
                            
        
        
        
        
        
        
    #     print("\nAssembling global stiffness matrix ...")
    #     t_1_assembling = time.perf_counter()
    #     k_global = gsmsml.global_stiffness_matrix(surfs, lobatto_pw, element_boundaries_u, \
    #                                         element_boundaries_v, elastic_modulus, nu)
    #     t_2_assembling = time.perf_counter()
    #     k_global_bc = esm.stiffness_matrix_bc_applied(k_global, bc) 
    #     print("\nAssembling global load vector ...")
    #     global_load = glv.global_load_vector(surfs, lobatto_pw, element_boundaries_u,\
    #                             element_boundaries_v, uniform_load_x,\
    #                             uniform_load_y, uniform_load_z)
    #     global_load_bc = np.delete(global_load, bc, 0)
    #     print("\nLinear solver in action! ...")
    #     t_1_solver = time.perf_counter()
    #     d = np.linalg.solve(k_global_bc, global_load_bc)
    #     t_2_solver = time.perf_counter()
    #     n_dimension = k_global.shape[0]
    #     displm_compelete = np.zeros(n_dimension)
    #     i = 0
    #     j = 0
    #     while i < n_dimension:
    #         if i in bc:
    #             i += 1 
    #         else:
    #             displm_compelete[i] = d[j]
    #             i += 1
    #             j += 1
    #     number_lobatto_node = lobatto_pw.shape[0]
    #     number_element_u = len(element_boundaries_u) - 1
    #     number_element_v = len(element_boundaries_v) - 1
    #     number_node_one_row = number_element_u*(number_lobatto_node - 1) + 1
    #     number_node_one_column = number_element_v*(number_lobatto_node - 1) + 1
    #     node_global_a = 1 #  u = v = 0 . Four nodes at the tips of the square in u-v parametric space
    #     node_global_c = node_global_a + number_element_v*(number_lobatto_node-1)\
    #                         *number_node_one_row
    #     print('\nDisplacement ratio: {}'.\
    #         format(displm_compelete[5*(node_global_c)-3]/u_analytic))
    #     # elemnum_displm_array[elemnum_counter] = [i_main,\
    #     #     displm_compelete[5*(node_global_c)-3]/u_analytic]
    #     # time_assembling [elemnum_counter] = [i_main, t_2_assembling - t_1_assembling]
    #     # time_solver [elemnum_counter] = [i_main, t_2_solver - t_1_solver]
    #     # n_dimension_bc = global_load_bc.shape[0]
    #     # dof_displm_array[elemnum_counter] = [n_dimension_bc, \
    #     #                 displm_compelete[5*(node_global_c)-3]/u_analytic]
    #     # dof_time_assembling [elemnum_counter] = [n_dimension_bc, t_2_assembling - t_1_assembling] 
    #     # dof_time_solver [elemnum_counter] = [n_dimension_bc, t_2_solver - t_1_solver]
    #     # if j_main == max_order_elem:###############
    #     if i_main in [6, 7, 8]:
    #         cond_elem[elemnum_counter] = [j_main, np.linalg.cond(k_global_bc)]
    #     elemnum_counter +=1
        j_main += 1
    # # np.savetxt(f'scordelis_h_ref_displm_p_{j_main}.dat', elemnum_displm_array)
    # # np.savetxt(f'scordelis_h_ref_asmtime_p_{j_main}.dat', time_assembling)
    # # np.savetxt(f'scordelis_h_ref_solvertime_p_{j_main}.dat', time_solver)
    # # np.savetxt(f'scordelis_h_ref_displm_dof_p_{j_main}.dat', dof_displm_array)
    # # np.savetxt(f'scordelis_h_ref_asmtime_dof_p_{j_main}.dat', dof_time_assembling)
    # # np.savetxt(f'scordelis_h_ref_solvertime_dof_p_{j_main}.dat', dof_time_solver)
    # # if j_main == max_order_elem:
    # if i_main in [6, 7, 8]:
    #         np.savetxt(f'scordelis_h_ref_cond_elem_p_{i_main}.dat', cond_elem)
    i_main += 1
    # # f_global = glv.global_load_vector(surfs, lobatto_pw, element_boundaries_u, element_boundaries_v)
    # # f_global_bc = glv.load_vector_bc_applied(f_global, bc)
    # # d = np.linalg.