import numpy as np


# def jacobian_element_ncoorsys_mtx (lobatto_pw, coorsys_tanvec_mtx):
#     '''
#     In this function the jacobian matrix for all the nodes of one element
#     is calculated. It is based on Eq. (14) of 
#     "A robust non-linear mixed hybrid quadrilateral shell element, 2005
#     W. Wagner1, and F. Gruttmann"
#     -Output:
#     Is a 2x2 matrix.
#     '''
#     dim = lobatto_pw.shape[0]
#     for i in range(dim):
#         for j in range(dim):
#             a_0_1 = coorsys_tanvec_mtx[i, j, 0]
#             a_0_2 = coorsys_tanvec_mtx[i, j, 1]
#             g1 = coorsys_tanvec_mtx[i, j, 3]
#             g2 = coorsys_tanvec_mtx[i, j, 4]
    
#     jac_elem_mtx = np.array([[g1 @ a_0_1, g1 @ a_0_2], [g2 @ a_0_1, g2 @ a_0_2]])
#     return jac_elem_mtx
    


def der_displ_elem(elem_row_num, elem_col_num, result_all_mtx, jacobian_elem_ncoorsys):
    '''
    In this f
    '''
    
    