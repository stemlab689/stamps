# -*- coding: utf-8 -*-
# @Author: Chieh-Han Lee
# @Date:   2017-02-24 11:21:31
# @Last Modified by:   Chieh-Han Lee
# @Last Modified time: 2017-02-26 22:14:14

import numpy

from ..general import tensor
from scipy.linalg import svd

def HOSVD(A):

    '''
    High Order Singular Value Decomposition

    Input:


    '''

    U = []
    Uh = []
    nModeSingularValues = []

    for mode in range(A.ndim):
        
        u, s, vh = svd( tensor.unfolding(A, mode+1) )
        U.append(u)
        Uh.append(vh)
        nModeSingularValues.append(s)
    
    S = tensor.ttm(A, U, True)

    return U, S, nModeSingularValues

if __name__ == '__main__':

    A = numpy.zeros((3,3,3))

    A[:,:,0] = numpy.asmatrix([[0.9073, 0.8924, 2.1488],
                            [0.7158, -0.4898, 0.3054],
                            [-0.3698, 2.4288, 2.3753]]).transpose()

    A[:,:,1] = numpy.asmatrix([[1.7842, 1.7753, 4.2495],
                            [1.6970, -1.5077, 0.3207],
                            [0.0151, 4.0337, 4.7146]]).transpose()

    A[:,:,2] = numpy.asmatrix([[2.1236, -0.6631, 1.8260],
                            [-0.0740, 1.9103, 2.1335],
                            [1.4429, -1.7495,-0.2716]]).transpose()

    U, S, D = HOSVD(A)