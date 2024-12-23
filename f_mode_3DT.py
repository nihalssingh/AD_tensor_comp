import numpy as np
from autograd import jacobian    
from derv_cpu import derv
from derv_cpu import j_derv
import timeit
import autograd
import torch as pt
import matplotlib.pyplot as plt
import pickle
# from opt_einsum import contract
our_times = []
g_ot = []
autograd_times = []
g_at = []
pytorch_times = []
g_pt = []


# EINSUM 

# Class Definition for forwaard node: stores the function value and its derivative
class f_node:
    def __init__(self, val, der):
       self.val = val
       self.der = der
    
    #Addition Nodes-- Operator Overloading
    def __add__(self, a):
        v = f_node(self.val + a.val, self.der + a.der)
        return v

    
    #Product Nodes
    # C' = B*(s2,s1s4,s3s4) A' + A*(s1,s2s4,s3s4) B'
    # C = A #(s1,s2,s3) B
    def einsum(s1, s2, s3, s4, a, b):
        # Calculating s4
        # templates for the first argument of einsum
        s_template = '{}, {}->{}'
        if s3 == '':
            s_template = '{}, {}'
        s_prod = s_template.format(s1, s2, s3) #einsum ip/op description for c.val
        s_dera = s_template.format(s2, s1+s4, s3+s4)
        s_derb = s_template.format(s1, s2+s4, s3+s4)
       
        #Calculating the product and the derivative: (prod, t1+t2)
        prod = np.einsum(s_prod, a.val, b.val, optimize=True)
        # prod = contract(s_prod, a.val, b.val)
        # b.der = 0 tensor
        if np.any(b.der) != True:
            der = np.einsum(s_dera, b.val, a.der, optimize=True)
            # der = contract(s_dera, b.val, a.der)
        elif np.any(a.der) != True:
            der = np.einsum(s_derb, a.val, b.der, optimize=True)
            # der = contract(s_derb, a.val, b.der)
        else:
            der = np.einsum(s_dera, b.val, a.der, optimize=True) + np.einsum(s_derb, a.val, b.der, optimize=True)
            # der = contract(s_dera, b.val, a.der) + contract(s_derb, a.val, b.der)
        # t1 = np.einsum(s_dera, b.val, a.der)
        # t2 = np.einsum(s_derb, a.val, b.der)
      
        # der = t1 + t2
        # v = f_node(prod, der)
        return f_node(prod, der)
    
    #Element-wise unary
    # sin
    def sin(s1, s2, a):
        s_template = '{}, {}->{}'
        s_der = s_template.format(s1, s1+s2, s1+s2)
        val = np.sin(a.val)
        der = np.einsum(s_der, np.cos(a.val), a.der)
        # der = contract(s_der, np.cos(a.val), a.der)
        v = f_node(val, der)
        return v
    
    #cos
    def cos(s1, s2, a):
        s_template = '{}, {}->{}'
        s_der = s_template.format(s1, s1+s2, s1+s2)
        val = np.cos(a.val)
        der = np.einsum(s_der, -np.sin(a.val), a.der)
        v = f_node(val, der)
        return v
        
def prod(x,y,z):                 # Define a function
    if len(x.shape) == 3:
        prod1 = autograd.numpy.einsum('ijk,lmn->ijklmn', x, z)
        prod2 = autograd.numpy.einsum('ijk,lmn->ijklmn', y, z)
    if len(x.shape) == 4:
        prod1 = autograd.numpy.einsum('ijkl,mnop->ijklmnop', x, z)
        prod2 = autograd.numpy.einsum('ijkl,mnop->ijklmnop', y, z)
    return prod1+prod2

def fun(x,y,z):
    if len(x.shape) == 3:
            f = pt.einsum('ijk,lmn->ijklmn', x, z) + pt.einsum('ijk,lmn->ijklmn', y, z)
    if len(x.shape) == 4:
        f = pt.einsum('ijkl,mnop->ijklmnop', x, z) + pt.einsum('ijkl,mnop->ijklmnop', y, z) 
    return f

# Initialising Inputs 
for iters in range (1,10):
    n_inp = 3
    
    A = np.random.rand(3+iters*1, 3+iters*1, 2)
    B = np.random.rand(3+iters*1, 3+iters*1, 2)
    C = np.random.rand(3, 3+iters*1, 3+iters*1)
    
    # AA = pt.rand(1+iters*1, 1+iters*1, 2)
    # BB = pt.rand(1+iters*1, 1+iters*1, 2)
    # CC = pt.rand(3, 1+iters*1, 1+iters*1)
    
    # A = np.random.rand(2, 4, 3, 3)
    # B = np.random.rand(2, 4, 3, 3)
    # C = np.random.rand(2, 4, 3, 3)
    
    # AA = pt.rand(2, 4, 3, 3)
    # BB = pt.rand(2, 4, 3, 3)
    # CC = pt.rand(2, 4, 3, 3)
    
    I = [A, B, C]
    for t in range(1, 10):
        
        # f_der[i] = dF/dXi
        f_der = []
        
        # Calculating dXi/dXi for all inputs to be used later
        d_I = []
        
        # to1d = timeit.default_timer()
        to1 = timeit.default_timer()
    
        for i in range(0, n_inp):
            d_I.append(derv(I[i]))
        # to2d = timeit.default_timer()
        # print("Total Time Ours Derv: ", (to2d-to1d)*1000)
        # our_times_d.append((to2d-to1d)*1000)
        
        # Looping over all inputs variables Xj to calculate dF/dXj
        for i in range(0, n_inp):
            x = []
            shape = np.zeros([])
            s4 = ''
            
            #Initialising the input nodes with derivative values: dXj/dXi o=for all i and j
            for j in range(0, n_inp):
                # s4 and dXi/dXi calculation 
                if i==j:
                    x.append(f_node(I[i], d_I[i]))
                    
                    #Calculating s4
                    for l in range(0, len(np.shape(I[i]))):
                        s4 = s4 + chr(ord('z')-l)
                # dXj/dXi
                else:
                    shape = np.shape(I[j])
                    shape = np.append(shape, np.shape(I[i]))
                    
                    x.append(f_node(I[j], np.zeros(shape)))
          
            #Function Definitions go here:
        
            # f = f_node.einsum('ij','jk','ik', s4, f_node.einsum('ij','jk','ik', s4, x[0], x[1]), x[2])
            # f= f_node.einsum('ijk','lmn','ijklmn', s4, x[0], x[1])
            if len(A.shape) == 4:
                
                f = f_node.einsum('ijkl','mnop','ijklmnop', sinp, x[0], x[2]) 
                + f_node.einsum('ijkl','mnop','ijklmnop', sinp, x[1], x[2])
                
            if len(A.shape) == 3:
                f = f_node.einsum('ijk','lmn','ijklmn', s4, x[0], x[2]) + f_node.einsum('ijk','lmn','ijklmn', s4, x[1], x[2])
            # f = f_node.sin('ijk', s4, x[0]) + f_node.cos('ijk', s4, x[1])
            
            f_der.append(f.der)
            # print(np.shape(f_der[i]))
        
        f_val = f.val
        to2 = timeit.default_timer()
        # print("Total Time Ours: ", (to2-to1)*1000)
        our_times.append((to2-to1)*1000)
    
    print("Total Time Ours: ", sum(our_times)/len(our_times))  
    g_ot.append(sum(our_times)/len(our_times))
    
    # for t in range(1, 10):
    #     ta1 = timeit.default_timer()
    #     jacobian_prod = jacobian(prod)       # Obtain its gradient function
    #     aa = jacobian_prod(A,B,C)
    #     ta2 = timeit.default_timer()
    #     autograd_times.append((ta2-ta1)*1000)
    #     # print("Total Time Autograd: ", (ta2-ta1)*1000)
    
    # print("Total Time Autograd: ", sum(autograd_times)/len(autograd_times))  
    # g_at.append(sum(autograd_times)/len(autograd_times))
    
    # II = (AA,BB,CC)
    # for t in range(1, 10):
    #     tt1 = timeit.default_timer()
    #     ja1cob_f = pt.autograd.functional.jacobian(fun, II)
    #     tt2 = timeit.default_timer()
    #     pytorch_times.append((tt2-tt1)*1000)
    #     # print("Total Time Autograd: ", (ta2-ta1)*1000)
    
    # print("Total Time Torch: ", sum(pytorch_times)/len(pytorch_times))  
    # g_pt.append(sum(pytorch_times)/len(pytorch_times))
    
    plt.title('AutoDiff for 3D Tensors ijk,lmn->ijklmn')
    plt.xlabel('No. of Elements of o/p')
    plt.ylabel('Computation Time (ms)')
    plt.grid()
    plt.scatter(np.prod(f_val.shape), g_ot[iters-1], color='green')
    # plt.scatter(np.prod(f_val.shape), g_at[iters-1], color='blue')
    # plt.scatter(np.prod(f_val.shape), g_pt[iters-1], color='red')
    
np.save("C:\\Coursework\\g_ot_cpu.npy",g_ot)
# print(g_ot)
# print(g_at)
# print(g_pt)

plt.show()
