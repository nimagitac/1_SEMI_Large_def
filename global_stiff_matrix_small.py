# from setuptools import sic
import surface_geom_SEM as sgs
import surface_nurbs_isoprm as snip
import element_stiff_matrix_small as esm
import numpy as np
# from surface_geom_SEM import *
from geomdl import exchange
from geomdl import multi
from geomdl.visualization import VisMPL as vis
import time
import cProfile
import pstats
import io
from pstats import SortKey
import subprocess
import collections


def profile(func):
    '''This function is used for profiling the file.
    It will be used as a decorator.
    output:
    A powershell profling and process.profile file'''
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        value = func(*args, **kwargs)
        pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())
        pr.dump_stats('process.profile')
        return value
    return inner


def mesh_func(surface, u_manual=[0 ,1], v_manual=[0, 1],\
         c_order_u=1, c_order_v=1, round_error=16):
    '''This function provides the locations of boundaries of elements 
    in row and columns in the mesh. The borders can be imported manually
    symbolized as u_ and v_manual and also according to degree of repetition
    of knots.
    c_order_u and c_order_v show the degrees of the NURBS in which the element
    borders is set in u and v direction. The targetrepetition of knot
    at the border will be (degree of NURBS - c_order_u or v).
    -Output:
    two series that epecify mesh in u and v direcetion. 
    '''
    element_boundaries_u =np.array(u_manual)
    element_boundaries_v = np.array(v_manual)
    target_repnum_u = surface.degree_u - c_order_u
    target_repnum_v = surface.degree_v - c_order_v
    dic_u = collections.Counter(np.round(surface.knotvector_u, decimals=round_error))
    dic_v = collections.Counter(np.round(surface.knotvector_v, decimals=round_error))
    for key in dic_u:
        if dic_u[key] >= target_repnum_u:
            element_boundaries_u = np.append(element_boundaries_u, key)
    for key in dic_v:
        if dic_v[key] >= target_repnum_v:
            element_boundaries_v = np.append(element_boundaries_v, key)
    return np.sort(np.unique(element_boundaries_u)),\
           np.sort(np.unique(element_boundaries_v))



def global_boundary_condition(lobatto_pw, bc_h_bott, bc_h_top,\
                              bc_v_left, bc_v_right, element_boundaries_u,\
                              element_boundaries_v):
    number_element_u = len(element_boundaries_u) - 1
    number_element_v = len(element_boundaries_v) - 1
    number_lobatto_node = lobatto_pw.shape[0]
    number_node_one_row = number_element_u*(number_lobatto_node - 1) + 1
    number_node_one_column = number_element_v*(number_lobatto_node - 1) + 1
    node_global_a = 1 #  u = v = 0 . Four nodes at the tips of the squar in u-v parametric space
    node_global_b = number_node_one_row
    node_global_c = node_global_a + number_element_v*(number_lobatto_node-1)\
                    *number_node_one_row
    node_global_d = node_global_b +  number_element_v*(number_lobatto_node-1)\
                    *number_node_one_row
    bc_line_h_bott = np.zeros(5*number_node_one_row) #bottom horizontal line
    bc_line_h_top = np.zeros(5*number_node_one_row) #top horizontal line
    bc_line_v_left = np.zeros(5*(number_node_one_column))
    bc_line_v_right = np.zeros(5*(number_node_one_column))
    i = 1
    j = 1
    while i <= node_global_b:
        bc_line_h_bott[5*j - 5] = (1 - bc_h_bott[0]) * (5*i-4)
        bc_line_h_bott[5*j - 4] = (1 - bc_h_bott[1]) * (5*i-3)
        bc_line_h_bott[5*j - 3] = (1 - bc_h_bott[2]) * (5*i-2)
        bc_line_h_bott[5*j - 2] = (1 - bc_h_bott[3]) * (5*i-1)
        bc_line_h_bott[5*j - 1] = (1 - bc_h_bott[4]) * (5*i)
        i += 1
        j += 1     
    i = node_global_c
    j = 1
    while i <= node_global_d:  
        bc_line_h_top[5*j - 5] = (1 - bc_h_top[0]) * (5*i - 4)
        bc_line_h_top[5*j - 4] = (1 - bc_h_top[1]) * (5*i - 3)
        bc_line_h_top[5*j - 3] = (1 - bc_h_top[2]) * (5*i - 2)
        bc_line_h_top[5*j - 2] = (1 - bc_h_top[3]) * (5*i - 1)
        bc_line_h_top[5*j - 1] = (1 - bc_h_top[4]) * (5*i)
        i += 1
        j += 1  
    i = 1
    j = 1
    while i <= node_global_c:
        bc_line_v_left[5*j - 5] = (1 - bc_v_left[0]) * (5*i- 4)
        bc_line_v_left[5*j - 4] = (1 - bc_v_left[1]) * (5*i - 3)
        bc_line_v_left[5*j - 3] = (1 - bc_v_left[2]) * (5*i - 2)
        bc_line_v_left[5*j - 2] = (1 - bc_v_left[3]) * (5*i - 1)
        bc_line_v_left[5*j - 1] = (1 - bc_v_left[4]) * (5*i)
        i += number_node_one_row
        j += 1  
    i = node_global_b 
    j = 1
    while i <= node_global_d:
        bc_line_v_right[5*j - 5] = (1 - bc_v_right[0]) * (5*i- 4)
        bc_line_v_right[5*j - 4] = (1 - bc_v_right[1]) * (5*i - 3)
        bc_line_v_right[5*j - 3] = (1 - bc_v_right[2]) * (5*i - 2)
        bc_line_v_right[5*j - 2] = (1 - bc_v_right[3]) * (5*i - 1)
        bc_line_v_right[5*j - 1] = (1 - bc_v_right[4]) * (5*i)
        i += number_node_one_row
        j += 1
    bc_1 = np.concatenate((bc_line_h_bott, bc_line_h_top, bc_line_v_left,\
                        bc_line_v_right))

    bc_2 = np.sort(np.delete(bc_1, np.where(bc_1 == 0))) - 1 # in python arrays start from zero
    # print('new bc:  ', bc_`2)
    bc_3 = np.unique(bc_2)
    return (bc_3.astype(int))



# @profile
def global_stiffness_matrix(surface, lobatto_pw, element_boundaries_u, \
                           element_boundaries_v, elastic_modulus, nu):
    number_element_u = len(element_boundaries_u) - 1
    number_element_v = len(element_boundaries_v) - 1
    number_lobatto_node = lobatto_pw.shape[0]
    number_node_one_row = number_element_u*(number_lobatto_node - 1) + 1
    node_global_3 = number_element_v * (number_lobatto_node-1) * number_node_one_row +\
                    number_node_one_row #This is the node equal to u = v = 1 in IGA space
    k_global = np.zeros((5*node_global_3, 5*node_global_3))

    for i_main in range(number_element_v):
        for j_main in range(number_element_u):
            # print('row {} out of {}'.format(i_main, number_element_v-1))
            node_1_u = element_boundaries_u[j_main]
            node_1_v = element_boundaries_v[i_main]
            node_3_u = element_boundaries_u[j_main + 1]
            node_3_v = element_boundaries_v[i_main + 1]
            surface_isoprm = snip.SurfaceGeneratedSEM(surface, lobatto_pw, node_1_u,\
                  node_1_v, node_3_u, node_3_v)
            coorsys_tanvec_mtx = surface_isoprm.coorsys_director_tanvec_allnodes()
            
            k_element = esm.element_stiffness_matrix(surface_isoprm, lobatto_pw,\
                                                coorsys_tanvec_mtx,\
                                                elastic_modulus, nu, \
                                                number_gauss_point=2)
                    
            node_1_number = i_main * (number_lobatto_node-1) * number_node_one_row +\
                            j_main * (number_lobatto_node-1) + 1
            # node_2_number = i_main * (number_lobatto_node-1) * number_node_one_row +\
            #                 (j_main+1) * (number_lobatto_node-1) + 1
            # node_3_number = node_1_number + (number_lobatto_node-1) * number_node_one_row
            # node_4_number = node_2_number + (number_lobatto_node-1) * number_node_one_row 
            connectivity = np.zeros(5*number_lobatto_node**2)
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
            number_dof_element = 5*number_lobatto_node**2
            
            for i in range(number_dof_element):
                for j in range(number_dof_element):
                    k_global[connectivity[i], connectivity[j]] = \
                        k_global[connectivity[i], connectivity[j]] + k_element[i , j]
    return k_global


def visualization(data):
    surf_cont = multi.SurfaceContainer(data)
    surf_cont.sample_size = 30
    surf_cont.vis = vis.VisSurface(ctrlpts=False, trims=False)
    surf_cont.render()
    return
    



if __name__ == '__main__':
    
    elastic_modulus = 3*(10**6)
    nu = 0.3
    
    ###### Test for bc for one element
    # element_boundaries_u = [0,  1]
    # element_boundaries_v = [0,  1]
    # lobatto_pw = lbto_pw("node_weight.dat")
    # bc = global_boundary_condition(lobatto_pw, element_boundaries_u, element_boundaries_v)
    # print(bc)

    # os.system('cls')
    # bc_h_bott = [0, 0, 1, 1, 1] #cantilever Plate.zero means clamped DOF
    # bc_h_top = [0, 0, 0, 0, 0]
    # bc_v_left = [0, 0, 1, 1, 1]
    # bc_v_right = [0, 0, 1, 1, 1]
    # bc_h_bott = [1, 1, 1, 1, 1] #cantilever quarter cylinder.zero means clamped DOF
    # bc_h_top = [0, 0, 0, 0, 0]
    # bc_v_left = [1, 1, 1, 1, 1]
    # bc_v_right = [1, 1, 1, 1, 1]
    bc_h_bott = [1, 0, 1, 0, 1] #pinched shell. zero means clamped DOF
    bc_h_top = [0, 1, 0, 1, 0]
    bc_v_left = [1, 1, 0, 1, 0]
    bc_v_right = [0, 1, 1, 1, 0]
    # bc_h_bott = [0, 1, 0, 1, 0] #Scordelis shell. zero means clamped DOF
    # bc_h_top = [1, 0, 1, 0, 1]
    # bc_v_left = [1, 1, 1, 1, 1]
    # bc_v_right = [0, 1, 1, 1, 0]
    # bc_h_bott = [0, 0, 0, 0, 0] #Total clamped. zero means clamped DOF
    # bc_h_top = [0, 0, 0, 0, 0]
    # bc_v_left = [0, 0, 0, 0, 0]
    # bc_v_right = [0, 0, 0, 0, 0]
    # bc_h_bott = [0, 0, 0, 0, 0] # Finding stiffness zero means clamped DOF
    # bc_h_top = [0, 0, 0, 0, 0]
    # bc_v_left = [0, 0, 1, 0, 0]
    # bc_v_right = [0, 0, 0, 0, 0]
    time1 = time.time()
    data = exchange.import_json("pinched_shell.json") #  pinched_shell_kninsertion_changedeg.json pinched_shell.json rectangle_cantilever square square_kninsertion generic_shell_kninsertion foursided_curved_kninsertion foursided_curved_kninsertion2  rectangle_kninsertion
    # visualization(data)
    surfs = sgs.SurfaceGeo(data, 0, 3)
    lobatto_pw_all =esm.lbto_pw("node_weight_all.dat")
    i_main = 4
    if i_main == 1:
        lobatto_pw = lobatto_pw_all[1:3,:]
    else:  
        index = np.argwhere(lobatto_pw_all==i_main)
        lobatto_pw = lobatto_pw_all[index[0, 0] + 1:\
                            index[0, 0] + (i_main+1) +1, :]
    node_1_ub = 0
    u_manual = [0, 0.25, 0.5, 0.75, 1]
    v_manual = [0, 0.25, 0.5, 0.75, 1]
    # u_manual = [0, 1]
    # v_manual = [0, 1]
    # element_boundaries_u = [0, 0.5, 0.75, 1]
    # element_boundaries_v = [0, 0.2, 0.3, 1]  
    mesh = mesh_func(surfs, u_manual, v_manual)
    # mesh = mesh_func(surfs)
    element_boundaries_u = mesh[0]
    element_boundaries_v = mesh[1]
    # print(mesh[0], mesh[1])            
    bc1 = global_boundary_condition(lobatto_pw, bc_h_bott, bc_h_top,\
                                bc_v_left, bc_v_right, element_boundaries_u,\
                                element_boundaries_v)
    # print('bc1 is :' , bc1)

    k_global = global_stiffness_matrix(surfs, lobatto_pw, element_boundaries_u, \
                                          element_boundaries_v, elastic_modulus, nu)
    k_global_bc = esm.stiffness_matrix_bc_applied(k_global, bc1) 
    load = np.zeros(np.shape(k_global)[0])
    p =0.25
    load[0] = p #for pinched                    
    load = np.delete(load, bc1, 0)
    d = np.linalg.solve(k_global_bc, load)  
    print("\n1D number of nodes:", lobatto_pw.shape[0],'\n')
    print('number of elements:', (element_boundaries_u.shape[0] - 1)\
                            * (element_boundaries_v.shape[0] - 1),'\n')
    print( "time is:",time.time()-time1, '\n')                
    print("displacement:", d[np.where(load==p)[0][0]]/(1.83*10**-5), '\n')
    # print(f'condition number of the stiffness matrix :  {np.linalg.cond(k_global_bc)}\n')
    # print("End")
    # subprocess.call("C:\\Users\\azizi\Desktop\\DFG_Python\\.P394\\Scripts\\snakeviz.exe process.profile ", \
    #                 shell=False)
