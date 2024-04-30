import numpy as np
import surface_nurbs_isoprm as snip
import lagrange_der as lagd



def initiate_jacobian_mtx_all(surface, lobatto_pw, element_boundaries_u,\
                       element_boundaries_v,  jacobian_all_node):
    '''
    This function claculate the Jacobian matrix in all elements at all nodes.
    -Output:
    jacobian_all_node(which is (number_element*number_element)*(number_node*number_node)*(3*3))
    
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
                xi2 = lobatto_pw[i, 0]
                # w2 = lobatto_pw[i, 1]
                lag_xi2 =  lagd.lagfunc(lobatto_pw, xi2)
                der_lag_dxi2 = lagd.der_lagfunc_dxi(lobatto_pw, xi2)
                for s in range(dim):
                    xi1 = lobatto_pw[j, 0]
                    # w1 = lobatto_pw[j, 1]
                    lag_xi1 = lagd.lagfunc(lobatto_pw, xi1)
                    der_lag_dxi1 = lagd.der_lagfunc_dxi(lobatto_pw, xi1)
                    jacobian_all_node[i, j, r, s, : , :] = \
                        surface_isoprm.jacobian_mtx(coorsys_tanvec_mtx, r, s,\
                         0, lag_xi1, lag_xi2, der_lag_dxi1, der_lag_dxi2)
                    
    return jacobian_all_node

