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
import global_stiff_matrix_largedef as gsmlrg
import internal_force as intf
import cProfile
import line_profiler as lprf
import os
import subprocess
import time as time
import sys

if __name__ == "__main__":
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    CYAN = '\033[36m'
    #INPUT *************************************************************
    u_analytic = -0.25356483
    elastic_modulus = 4.32*10**8
    thk = 0.25
    nu = 0
    uniform_load_x = 0
    uniform_load_y = 0
    uniform_load_z = -90
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

    p_1 = surfs.physical_crd(0., 0.)
    p_2 = surfs.physical_crd(1., 0.)
    p_3 = surfs.physical_crd(1., 1.)
    print("p_1:", p_1, "  p_2:", p_2, "  p_3:", p_3)

    min_order_elem = int(input("\nEnter the minimum order of elements (minimum order = 1):\n"))
    max_order_elem = int(input("Enter the maximum order of elements (maximum order = 30):\n"))
    min_number_elem = int(input("\nEnter the minimum number of elements in u and v direction:\n"))
    max_number_elem = int(input("Enter the maximum number of elements in u and v direction:\n"))
    print("\nEnter the order of continuity at knots to be used for auto detection of elements boundaries in u direction")
    print("The default value is '1'")
    c_order_u = 1 # int(input())
    print("\nEnter the order of continuity at knots to be used for auto detection of elements boundaries in v direction")
    print("The default value is '1'")
    c_order_v = 1 # int(input())

    number_load_step = int(input("Enter the total number of load steps  "))
    error_force = 10**-7 # input("Enter the error in the length of residucal force")
    newton_iter_max = 100 # input("Enter the maximum steps in the Newton-Raphson")

    i_main = min_order_elem
    while i_main <= max_order_elem:
        with open(f'scordelis_displm_p_{i_main}.dat', 'w') as result:
            pass
        if i_main==1:
            lobatto_pw = lobatto_pw_all[1:3,:]
        else:
            index = np.argwhere(lobatto_pw_all==i_main)
            lobatto_pw = lobatto_pw_all[index[0, 0] + 1:\
                                index[0, 0] + (i_main+1) + 1, :]
        j_main = min_number_elem
        dim = lobatto_pw.shape[0]
        elemnum_counter = 0
        while j_main <= max_number_elem:
            # with open(f'scordelis_displm_p_{i_main}.dat', 'w') as result:
            #     pass
            print("\n\n*************************************************************************")
            print("*************************************************************************\n")
            print("Number of elements manually given in u and v: {}    Order of elements: {} ".\
            format(str(j_main)+'x'+ str(j_main), i_main))
            print("Program starts to generate mesh according to continuity at knots and manual input of number of elements ...") 
            print("\n*************************************************************************")
            print("*************************************************************************\n\n")
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
            
            # t1 = time.time()      
            bc = gsmsml.global_boundary_condition(lobatto_pw, bc_h_bott, bc_h_top,\
                                    bc_v_left, bc_v_right, element_boundaries_u,\
                                    element_boundaries_v)
            global_total_load = glvlgr.global_load_vector(lobatto_pw, jacobian_all, \
                                element_boundaries_u, element_boundaries_v,\
                                uniform_load_x, uniform_load_y, uniform_load_z)
            global_load_incr =  global_total_load / number_load_step
            global_load_incr_bc = np.delete(global_load_incr, bc, 0)
            global_load_incr_bc_norm = np.linalg.norm(global_load_incr_bc)
            elastic_mtx = esmlrg.elastic_matrix(elastic_modulus, nu, thk)
            newton_iter_counter_total = 0
            for p_main in range(number_load_step):
                print("\n******************************************\n")
                # print("\nload step number:  ", p_main,"\n")
                # print(f"Mesh of {j_main}x{j_main} element with order{i_main}")
                # print("error ratio:", error / global_load_incr_bc_norm,'\n\n')
                global_stepload = (p_main + 1) * global_load_incr
                global_stepload_bc = np.delete(global_stepload, bc, 0)               
                error = 10000
                newton_iter_counter = 0 
                while (error / global_load_incr_bc_norm) >= error_force:
                    print(f"{GREEN}{j_main}x{j_main}-p{i_main}")
                    print(f"{YELLOW}load step number:  ", p_main)           
                    print(f"{CYAN}Newton iteration number:", newton_iter_counter,f"{RESET}")
                    k_global =gsmlrg.global_stiffness_matrix(lobatto_pw,\
                            element_boundaries_u, element_boundaries_v,
                            x_0_coor_all, nodal_coorsys_all, jacobian_all,
                            node_displ_all, elastic_modulus, nu, thk)
                    k_global_bc = esmsml.stiffness_matrix_bc_applied(k_global, bc)
                    
                    
                    global_internal_force = intf.global_intern_force(lobatto_pw, \
                                    element_boundaries_u, element_boundaries_v,\
                                    x_0_coor_all, nodal_coorsys_all, jacobian_all,\
                                    node_displ_all, elastic_mtx)
                    # print(global_stepload,'\n')
                    # global_internal_force_bc = np.delete(global_internal_force, bc, 0)
                    global_res_load = global_stepload -  global_internal_force
                    # print(global_res_load,'\n')
                    
                    global_res_load_bc = np.delete( global_res_load, bc, 0)
                    
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
                    nodal_coorsys_all, displm_complete, node_displ_all)
                    elem_displ_all = node_displ_all[0, 0]  #### To be changed in meshing
                    # print('\nDisplacement ratio: {}'\
                    #     .format(displm_complete[5*(node_global_c)-3]/u_analytic))  
                    error = np.linalg.norm(global_res_load_bc)
                    print("Error ratio:", error / global_load_incr_bc_norm)
                    # print(displm_complete[5*(node_global_c)-3],"\n")
                    print('Displacement:', node_displ_all[j_main-1, 0, dim-1, 0, 0, 2],"\n\n")
                    newton_iter_counter += 1
                    if newton_iter_counter > newton_iter_max:
                        print("Not converged in the defined number of Newton steps")
                        sys.exit()
                newton_iter_counter_total += newton_iter_counter
                # step_deformation = np.array([[p_main, node_displ_all[j_main-1, 0, dim-1, 0, 0, 2]]])        
                # with open(f'scordelis_p_ref_displm_p_{i_main}_newton.dat', 'a') as result:
                #     # result.write(f"Step {p_main}:\n")
                #     np.savetxt(result, step_deformation )
            output_array = np.array([[i_main, j_main, j_main**2, \
                            global_res_load_bc.shape[0],\
                            node_displ_all[j_main-1, 0, dim-1, 0, 0, 2],\
                            node_displ_all[j_main-1, 0, dim-1, 0, 0, 2]/u_analytic,\
                            newton_iter_counter_total]])  
            with open(f'scordelis_displm_p_{i_main}.dat', 'a') as result:
                #     # result.write(f"Step {p_main}:\n")
                np.savetxt(result, output_array, fmt='%d %d %d %d %.10f %.10f %d')  
            j_main += 1
        i_main += 1
        pass
                
                
# print(t2 - t1)
# subprocess.call("C:\\Nima\\N-Research\\DFG\\Python_programming\\Large_def\\1_SEMI_Large_def\\.P3-12-2\\Scripts\\snakeviz.exe process.profile ", \
#                 shell=False)
                            
        
        
        
        
        
        
    