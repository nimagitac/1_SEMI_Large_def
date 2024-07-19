import numpy as np
import lagrange_der as lagder
from geomdl import exchange
import surface_geom_SEM as sgs
import global_stiff_matrix_small as gsmsml
import hist_displ_mtx_update as hdu
import element_stiff_matrix_small as esmsml
import global_load_vector_uniform_SEMN_small as glvsml
import global_load_vector_uniform_largedef as glvlgr
import element_stiffness_matrix_largedef as esmlrg
import internal_force as intf
import cProfile
import line_profiler as lprf
import os
import subprocess
import time as time



#INPUT *************************************************************
u_analytic = 5.86799
elastic_modulus = 6.825*(10**7)
nu = 0.3
thk = 0.04
point_load = 1
bc_h_bott = [1, 1, 1, 1, 1] 
bc_h_top = [1, 1, 1, 1, 1]
bc_v_left = [1, 0, 1, 0, 1]
bc_v_right = [0, 1, 1, 1, 0]

#*************************************************************




print("\nImporting Lobatto points and weights from data-base ...")
# time.sleep(1)
lobatto_pw_all = lagder.lbto_pw("node_weight_all.dat")
print("\nImporting geometry from json file ...")
# time.sleep(1)
data = exchange.import_json("Hemispherical-shell_pantheon.json") #  pinched_shell_kninsertion_changedeg.json pinched_shell.json rectangle_cantilever square square_kninsertion generic_shell_kninsertion foursided_curved_kninsertion foursided_curved_kninsertion2  rectangle_kninsertion
    # visualization(data)
surfs = sgs.SurfaceGeo(data, 1, thk)

p_1 = surfs.physical_crd(0., 0.)
p_2 = surfs.physical_crd(1., 0.)
p_3 = surfs.physical_crd(1., 1.)
print("p_1:", p_1, "  p_2:", p_2, "  p_3:", p_3)

min_order_elem = int(input("\nEnter the minimum order of elements (minimum order = 1):\n"))
max_order_elem = int(input("Enter the maximum order of elements (maximum order = 30):\n"))
min_number_elem = 1 # int(input("\nEnter the minimum number of elements in u and v direction:\n"))
max_number_elem = 1 # int(input("Enter the maximum number of elements in u and v direction:\n"))
print("\nEnter the order of continuity at knots to be used for auto detection of elements boundaries in u direction")
print("The default value is '1'")
c_order_u = 1 # int(input())
print("\nEnter the order of continuity at knots to be used for auto detection of elements boundaries in v direction")
print("The default value is '1'")
c_order_v = 1 # int(input())

number_load_step = int(input("Enter the total number of load steps  "))
error_force = 10**-5 # input("Enter the error in the length of residucal force")
newton_rep = 100 # input("Enter the maximum steps in the Newton-Raphson")

i_main = min_order_elem
while i_main <= max_order_elem:
    with open(f'scordelis_p_ref_displm_p_{i_main}_.dat', 'w') as result:
        pass
    if i_main==1:
        lobatto_pw = lobatto_pw_all[1:3,:]
    else:
        index = np.argwhere(lobatto_pw_all==i_main)
        lobatto_pw = lobatto_pw_all[index[0, 0] + 1:\
                            index[0, 0] + (i_main+1) + 1, :]
    j_main = min_number_elem
    dim = lobatto_pw.shape[0]
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
        
        
        node_displ_all = np.zeros((number_element_v, number_element_u,\
                                    i_main + 1, i_main + 1, 2, 3)) #To record the history of deformation. Dimensions are: number of elment in u and v, number of nodes in xi1 and xi2, 2x3 for u, omega, each has 3 components.
        nodal_coorsys_all = np.zeros((number_element_v, number_element_u,\
                                    i_main + 1, i_main + 1, 3, 3)) #TDimensions are: number of elment in u and v, number of nodes in xi1 and xi2, 3x3 for A_1, A_2, A_3
        jacobian_all = np.zeros((number_element_v, number_element_u,\
                                    i_main + 1, i_main + 1, 2, 2))
        x_0_coor_all = np.zeros((number_element_v, number_element_u, i_main + 1, i_main + 1, 3)) # The initial coordinate of each element node for each element
        inital_coor_coorsys_jac = hdu.initiate_x_0_ncoorsys_jacmtx_all(surfs,\
                                    lobatto_pw, element_boundaries_u,\
                                element_boundaries_v, x_0_coor_all,\
                                nodal_coorsys_all, jacobian_all)
        x_0_coor_all = inital_coor_coorsys_jac[0]
        nodal_coorsys_all = inital_coor_coorsys_jac[1]
        jacobian_all = inital_coor_coorsys_jac[2] #To avoide repitition calculation of Jacobian matrix, the Jacobian matrix is calculated for all elements at all GLL points
        elem_x_0_coor_all = x_0_coor_all[0, 0] #### To be changed in meshing
        elem_nodal_coorsys_all = nodal_coorsys_all[0, 0]  #### To be changed in meshing
        elem_jacobian_all = jacobian_all[0, 0]  #### To be changed in meshing
        
        t1 = time.time()      
        bc = gsmsml.global_boundary_condition(lobatto_pw, bc_h_bott, bc_h_top,\
                                bc_v_left, bc_v_right, element_boundaries_u,\
                                element_boundaries_v)
        # bc = np.append(bc, [2, 5 * node_global_b - 3] )
        bc = np.append(bc, 2)
        bc = np.sort(bc)
        
       
        # global_total_load = glvlgr.global_load_vector(lobatto_pw, jacobian_all, \
        #                        element_boundaries_u, element_boundaries_v,\
        #                       uniform_load_x, uniform_load_y, uniform_load_z)
        # global_load_incr =  global_total_load / number_load_step
        # global_load_incr_bc = np.delete(global_load_incr, bc, 0)
        # global_load_incr_bc_norm = np.linalg.norm(global_load_incr_bc)
        
        global_total_load = np.zeros(total_dof)
        global_total_load[0] = point_load
        global_total_load[5 * node_global_b - 4] = - point_load
        global_load_step = global_total_load / number_load_step
        global_load_step_bc = np.delete(global_load_step, bc, 0)
        global_load_step_bc_norm = np.linalg.norm(global_load_step_bc)
        
        elastic_mtx = esmlrg.elastic_matrix(elastic_modulus, nu, thk)
        elem_displ_all = node_displ_all[0, 0]  #### To be changed in meshing
        
        # error = 1
        for p_main in range(number_load_step):
            
            print("load step number:  ", p_main,"\n")
            # print("error ratio:", error / global_load_incr_bc_norm,'\n\n')
            global_load_incr = (p_main + 1) * global_load_step
            global_load_incr_bc = np.delete(global_load_incr, bc, 0)
            # print(global_stepload)            
            k_elem = esmlrg.element_stiffness_mtx(lobatto_pw, elem_x_0_coor_all, \
                        elem_nodal_coorsys_all, elem_jacobian_all,\
                        elem_displ_all, elastic_modulus, nu, thk) #### To be changed in meshing
            t2 = time.time()
            
            k_global = k_elem
            k_global_bc = esmsml.stiffness_matrix_bc_applied(k_elem, bc)
           
            
            error = 10000
            newton_step_counter = 0 
            
            while (error /  global_load_step_bc_norm) >= error_force \
                        and newton_step_counter <= newton_rep:
                            
                print("Newton iteration number:", newton_step_counter)
                global_internal_force = intf.element_intern_force(lobatto_pw, elem_x_0_coor_all, \
                          elem_nodal_coorsys_all, elem_jacobian_all,\
                          elem_displ_all, elastic_mtx)
                # print(global_stepload,'\n')
                global_internal_force_bc = np.delete(global_internal_force, bc, 0)
                global_res_load = global_load_incr -  global_internal_force
                # print(global_res_load,'\n')
                
                global_res_load_bc = np.delete(global_res_load, bc, 0)
                
                d = np.linalg.solve(k_global_bc, global_res_load_bc)
                # n_dimension = k_global.shape[0]
                # displm_compelete = np.zeros(n_dimension)
                i = 0
                j = 0
                while i < total_dof:
                    if i in bc:
                        i += 1 
                    else:
                        displm_complete[i] = d[j]
                        i += 1
                        j += 1
                node_displ_all = hdu.update_displ_hist(lobatto_pw, number_element_u, number_element_v, \
                   displm_complete, node_displ_all, nodal_coorsys_all)
                elem_displ_all = node_displ_all[0, 0]  #### To be changed in meshing
                # print('\nDisplacement ratio: {}'\
                #     .format(displm_complete[5*(node_global_c)-3]/u_analytic))  
                error = np.linalg.norm(global_res_load_bc)
                print("error ratio:", error /  global_load_step_bc_norm,'\n')
                # print(node_displ_all[0, 0, 0, 0, 0, 0])
                newton_step_counter += 1
                if newton_step_counter > newton_rep:
                    print("Not converged in defined Newton steps")
            step_deformation = np.array([[p_main, node_displ_all[0, 0, 0, 0, 0, 0]]])
            print(node_displ_all[0, 0, 0, 0, 0, 0]) 
            # print(node_displ_all[0, 0, 0, node_global_b - 1, 0, 1])        
            with open(f'hemisphere_p_ref_displm_p_{i_main}_.dat', 'a') as result:
                # result.write(f"Step {p_main}:\n")
                np.savetxt(result, step_deformation )
        j_main += 1
    i_main += 1
    pass
                
                
# print(t2 - t1)
# subprocess.call("C:\\Nima\\N-Research\\DFG\\Python_programming\\Large_def\\1_SEMI_Large_def\\.P3-12-2\\Scripts\\snakeviz.exe process.profile ", \
#                 shell=False)
                            
        
        
        
        
        
        
   