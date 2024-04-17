import numpy as np


def lagfunc(lobatto_pw, xi):
    '''
    Function for generating lagrangian
    shape functions
    output:
    1D np.array, the value of the i-th element
    represents the value of the i-th Lagrangian shape function 
    at xi. The number of elements in the array obviousluy is 
    equal to the number of nodes in 1 direction
    '''
    if xi<-1 and xi>1:
        raise ValueError('-1<=xi1<=+1, -1<=xi2<=+1 must be followed')
    else: 
        len = lobatto_pw.shape[0]
        # lag_func = np.zeros(len)
        # if xi in lobatto_pw[:,0]:
        #     xi_index = np.where(lobatto_pw[:,0] == xi)
        #     lag_func[xi_index] = 1
        # else:   No impresive increase in speed
        lag_func = np.ones(len)
        for i in range(len):
            for j in range(len): 
                if j != i:
                    lag_func[i] = lag_func[i] * (xi-lobatto_pw[j, 0]) /\
                        (lobatto_pw[i, 0]-lobatto_pw[j, 0])
        return lag_func
        

def der_lagfunc_dxi(lobatto_pw, xi):
    '''By taking 'ln' functiom from both sides of the definition
    of lagrangina shape functions, the derivatives
    can be calculated. For example:
    https://math.stackexchange.com/questions/809927/first-derivative-of-lagrange-polynomial
    however the formulation result in infinity when xi = xj therefore,
    it is modified to:
    dlag-i-dxi = (sigma (j, j!=i,(PI (k, k!=(i and j), (x-xk))))/PI(j, j!=i, (xi-xj))
    -output:
    1D np.array, the value of the i-th element
    represents the derivative of the i-th Lagrangian shape function at xi
    w. r. t. xi at xi . The number of element obviousluy is 
    equal to the number of nodes
    '''    
    len = lobatto_pw.shape[0]
    der_lag_dxi = np.zeros(len)
    for i in range(len):
        numerator_sigma = 0
        denominator = 1
        for j in range(len):  
            if j != i:
                denominator = denominator * (lobatto_pw[i, 0]-lobatto_pw[j, 0])
                numerator_pi = 1
                for k in range(len):
                    if k!=i and k!=j:
                        numerator_pi = numerator_pi * (xi - lobatto_pw[k, 0])
                numerator_sigma = numerator_sigma + numerator_pi
        der_lag_dxi [i] = numerator_sigma / denominator
    return der_lag_dxi

def der2_lagfunc_dxi(lobatto_pw, xi):
    '''
    One way to calculate the second order derivative is:
    https://math.stackexchange.com/questions/1572576/lagrange-polynomial-second-order-derivative
    The other ways is:    
    The only part of the derivative of a Lagrange shape function which is the
    function of xi, is:
    (PI (k, k!=(i and j), (xi-xk))
    If we take the ln from this part and then taking the derivative of it,
    the second order derivative of the Lagrange function can be calculated as follow: 
    d2lag-i-dxi2 = (sigma j!=i  sigma k!=(i, j)  (PI p!=(i, j, k) (x-xp))))/PI(j, j!=i, (xi-xj))
    -output:
    1D np.array in that i-th element
    is the value of the second order derivatives of i-th lagrangian shape function
    w. r. t. xi at xi  bat point input(xi). The number of element obviousluy is 
    equal to the number of nodes
    '''    
    len = lobatto_pw.shape[0]
    der2_lag_dxi2 = np.zeros(len)
    for i in range(len):
        numerator_sigma_1 = 0
        denominator = 1
        for j in range(len):  
            if j!=i:
                denominator = denominator * (lobatto_pw[i, 0]-lobatto_pw[j, 0])
                numerator_sigma_2 = 0
                for k in range(len):
                    if k!=i and k!=j:
                        numerator_pi = 1
                        for p in range(len):
                            if p!=i and p!=j and p!=k:
                                numerator_pi = numerator_pi * (xi - lobatto_pw[p, 0])
                        numerator_sigma_2 = numerator_sigma_2 + numerator_pi
                numerator_sigma_1 = numerator_sigma_1 + numerator_sigma_2
        der2_lag_dxi2 [i] = numerator_sigma_1 / denominator
    return der2_lag_dxi2