# -*- coding: utf-8 -*-
# @Author: Chieh-Han Lee
# @Date:   2015-08-05 19:40:44
# @Last Modified by:   Chieh-Han Lee
# @Last Modified time: 2016-10-31 23:26:00
# -*- coding: utf-8 -*-
'''
Created on 2012/4/11

@author: KSJ
'''

import numpy as np

from scipy.spatial import cKDTree as KDTree
from scipy.spatial.distance import cdist as scipy_cdist

def idw_est( x, y, z, x_est, y_est ,power = 2):
    x, y, z, x_est, y_est =\
    map( lambda x : np.array( x, ndmin = 2 ),
         ( x, y, z, x_est, y_est ) )
    #dist_matrix = np.linalg.norm(
    #   np.hstack((x.T - x_est, y.T - y_est)) , axis=0 ) + 10**-10
    dist_matrix =\
        np.sqrt( ( x.T - x_est ) **2 + ( y.T - y_est ) **2 ) + 10**-10
    weight_matrix = np.reciprocal( dist_matrix ** power )
    up_matrix = weight_matrix * z.T
    up_matrix = up_matrix.sum( axis = 0 ) #sum column
    down_matrix = weight_matrix.sum( axis = 0 ) #sum column
    z_est = up_matrix / down_matrix
    return z_est

def idw_est_coord_value(coord, value, coord_est, power = 2):
    '''
    coord: a 2d array, r x d, row is data count, column is dimension
    value: a 2d array, r x 1, row is data count, column is value
    coord_est: dito coord
    '''
    coord_matrix = scipy_cdist(coord_est, coord) #coord_est by coord
    weight_matrix = np.reciprocal(coord_matrix**power)
    # remove dupliacted localtion (Set 0 wieght)
    weight_matrix[np.isinf(weight_matrix)] = 0.
    up_matrix = weight_matrix * value.T
    up_matrix = up_matrix.sum(axis=1, keepdims=True) #sum column
    down_matrix = weight_matrix.sum(axis=1, keepdims=True) #sum column
    value_est = up_matrix / down_matrix
    return value_est
 
def idw_kdtree( grid_s, grid_v, grid_s_est, nnear=10, eps=0, power=2, weights=None, leafsize=16 ):
    '''
    Inverse distance weighting (IDW) method using KDtree

    Syntax
        interp = idw_kdtree( grid_s, grid_v, grid_s_est, nnear=10, eps=0, power=2, weights=None, leafsize=10 ):
    
    Input
        grid_s:
            [r1 x d]. Coordinates in grid format.
        grid_v:
            [r1 x 1]. 
        grid_s_est:
            [r2 x d].
        nnear:
            integer. The list of k-th nearest neighbors to return. f k is an integer it is 
            treated as a list of [1, ... k] (range(1, k+1)). Note that the counting starts 
            from 1.
        eps:
            nonnegative float. Return approximate nearest neighbors;the k-th returned 
            value is guaranteed to be no further than (1+eps) times the distance to 
            the real k-th nearest neighbor.
        power:
            integer. Power parameter. Greater values of p assign greater influence to values 
            closest to the interpolated point, with the result turning into a mosaic of tiles 
            (a Voronoi diagram) with nearly constant interpolated value for large values of p
        weights:
            []. Weighted matrix.
        leafsize:
            positive integer. The number of points at which the algorithm switches over to brute-force.

    Output
        interp:
        [r2 x 1].Interpolation result of IDW.

    '''


    tree = KDTree(grid_s, leafsize=leafsize)

    distances, indices = tree.query(grid_s_est, k=nnear, eps=eps)
    interp = np.zeros( (len(grid_s_est),) + np.shape(grid_v[0]) )
    iternum = 0
    for dist, idx in zip(distances, indices):
        z0 = grid_v[idx[0]]
        if nnear == 1:
            weighted_v = grid_v[idx]
        elif dist[0] < 1e-10 and ~np.isnan(z0):
            weighted_v = grid_v[idx[0]]
        else:
            ix = np.where(dist==0)[0]
            if ix.size:
                dist = np.delete(dist, ix)
                idx = np.delete(idx, ix)
            ix = np.where(np.isnan(grid_v[idx]))[0]
            dist = np.delete(dist, ix)
            idx = np.delete(idx, ix)

            weight_matrix = np.reciprocal( dist ** power )
            if weights is not None:
                weight_matrix *= weights[idx]

            weight_matrix /= np.sum(weight_matrix)
            weighted_v = np.dot(weight_matrix, grid_v[idx])

        interp[iternum] = weighted_v
        iternum += 1

    return interp


if __name__ == "__main__":
    x = np.random.random(5)
    y = np.random.random(5)
    z = np.random.random(5)
    
    x_est = np.random.random(7)
    y_est = np.random.random(7)
    
    print idw_est( x, y, z, x_est, y_est)

    grid_s = np.random.random((100,2))
    grid_v = np.random.random((100,1))
    grid_s_est = np.random.random((7000,2))
    print idw_kdtree( grid_s, grid_v, grid_s_est )

    