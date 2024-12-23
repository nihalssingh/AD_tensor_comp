import cupy as cp
import numpy as np
import autograd as ag

def fun(x):
    fun = x
    return fun

def j_derv(A):
    j_prod = ag.jacobian(fun)
    # print(j_prod(A))
    return j_prod(A)

#Function to initialise the value der.a = dA/dA
def derv(A):
    #Creating a zero vector of size equal to the number of elements in dA/dA
    s = cp.shape(A)
    l_s = len(s)
    p = 1
    for i in range(0, l_s):
        p = s[i]*p
    A_der = cp.zeros(p*p, dtype = int)
    
    #Setting all the symmetrically located elements to 1 (eg. x(i,j,k,i,j,k) for a 3-order tensor)
    count = 0
    while count < p*p:
        A_der[count] = 1
        count = count + p + 1
    #reshaping to the shape of dA/dA
    s_der = cp.append(s, s)
    A_der = cp.reshape(A_der, s_der.tolist())
    return A_der

