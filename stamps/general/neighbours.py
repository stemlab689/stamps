# -*- coding:utf-8 -*-
import numpy
from scipy.spatial import cKDTree
try:
  from scipy.spatial.distance import cdist as coord2dist
except:
  from .coord2dist import coord2dist  


def neighbours_index_kd(ck, ctree, nmax, dmax):
    '''let ck group by ctree
        ctree can be a numpy array with shape (n, nd) or
        scipy kd-tree

        nmax is first nearest nmax neighbors
        dmax here is only max distance

        return a dict.
            it's key is a tuple, ctree index
            it's value is a list, ck_index
    '''
    if not isinstance(ctree, cKDTree):
        try:
            ctree = cKDTree(ctree)
        except Exception as e:
            import ipdb
            ipdb.set_trace()
    dd, ii = ctree.query(ck, k=range(1, nmax+1), distance_upper_bound=dmax)
    marr = numpy.ma.masked_array(ii, numpy.isinf(dd))
    marr = numpy.sort(marr)

    res = {}
    for k_idx, marr_i in enumerate(marr):
        try:
            res[tuple(marr_i.compressed())].append(k_idx)
        except KeyError:
            res[tuple(marr_i.compressed())] = [k_idx]
    return res
        
def neighbours_kd(c_one, c, z, nmax, dmax, tree=None):
  '''
  input
  c_one: 1 by nd
        nd can be:
            1 for space or time,
            2 for space,
            3 for space-time
  c: n by nd
  z: n by ??
  nmax: int
  dmax: 1 by rd float 
        rd can be:
            1 for space or time
            3 for space-time
  return
  c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr
  '''  
  empty_result = [ numpy.array([]).reshape( ( 0, c_one.shape[1] ) ),
                   numpy.array([]).reshape( ( 0, 1 ) ),
                   numpy.array([]).reshape( ( 0, 1 ) ),
                   0,
                   numpy.array([]).reshape( ( 0, 1 ) ) ]

  isST = 1 if dmax.size == 3 else 0

  if c.size == 0:
    print('no data')
    return empty_result

  if nmax == 0:
    print('nmax is 0')
    return empty_result  
    
  if isST==0: # pure spatial or temporal cases
    if tree is None:
      try:
        tree=cKDTree(c,leafsize=15)
      except:
        import sys
        sys.setrecursionlimit(10000)
        tree=cKDTree(c,leafsize=30)
    d_nebr,idx_nebr=tree.query(
        c_one, k=range(1, nmax+1), distance_upper_bound=dmax[0][0])
    idx_nebr=idx_nebr[0]    
    c_nebr=c[idx_nebr,:]
    z_nebr=z[idx_nebr,:]    
    n_nebr=idx_nebr.size
    return c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr
  
  elif isST == 1:#space time case  
    if tree is None:
      #get distance of time
      d_t = numpy.abs( c[:,2:3] - c_one[:,2:3] )
      index_t = numpy.where( d_t <= dmax[0][1] )
      if len(index_t[0]) == 0:
        print("noneighbor")
        return empty_result
  
      #get distance of space which already match time
      d_xy = coord2dist( c[index_t[0],0:2], c_one[:,0:2] )
      index_s = numpy.where( d_xy <= dmax[0][0] )
      if len(index_s[0]) == 0:
        print("noneighbor")
        return empty_result  
    
      #calculate all distance which matched perfectly
      c_one_n=c_one
      c_n=c[index_t[0][index_s[0]],:]
      c_one_n[:,2]=c_one[:,2]*dmax[0][2]
      c_n[:,2]=c_n[:,2]*dmax[0][2]
      tree=cKDTree(c_n,leafsize=15)

    d_nebr,idx_nebr=tree.query(c_one_n, k=range(1, nmax+1))  
    idx_nebr=index_t[0][index_s[0]][idx_nebr[0]]
    d_nebr=d_nebr.T
    c_nebr=c[idx_nebr,:]
    z_nebr=z[idx_nebr]
    n_nebr=idx_nebr.size   
    return c_nebr, z_nebr, d_nebr, n_nebr,\
        numpy.sort(idx_nebr.reshape((-1,1)), axis=0)
    
    
def neighbours( c_one, c, z, nmax, dmax ):
    '''
    input
    c_one: 1 by nd
        nd can be:
            1 for space or time,
            2 for space,
            3 for space-time
    c: n by nd
    z: n by ??(any), means abservations
    nmax: int
    dmax: 1 by rd float 
        rd can be:
            1 for space or time
            3 for space-time
    return
    c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr
    '''

    empty_result = [ numpy.array([]).reshape( ( 0, c_one.shape[1] ) ),
                     numpy.array([]).reshape( ( 0, 1 ) ),
                     numpy.array([]).reshape( ( 0, 1 ) ),
                     0,
                     numpy.array([]).reshape( ( 0, 1 ) ) ]

    isST = 1 if dmax.size == 3 else 0

    if c.size == 0:
    #    print('no data')
        return empty_result

    if nmax == 0:
    #    print('nmax is 0')
        return empty_result

    if isST == 0: #
        #get distance of space (only)
        d_xy = coord2dist( c, c_one )
        index_s = numpy.where( d_xy <= dmax[0][0] )
        if len(index_s[0]) == 0:
            print("noneighbor")
            return empty_result
        elif len( index_s[0] ) <= nmax:
            c_nebr = c[index_s[0],:]
            z_nebr = z[index_s[0],:]
            d_nebr = d_xy[index_s[0],:]
            n_nebr = len( index_s[0] )
            idx_nebr = index_s[0].reshape( ( -1, 1 ) )
            return c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr
        elif len( index_s[0] ) > nmax:
            d_nebr = d_xy[index_s[0],:]
            index_s = ( numpy.sort( d_nebr[:,0].argsort()[:nmax] ), 0 ) #dummy 0 for consistence
            c_nebr = c[index_s[0],:]
            z_nebr = z[index_s[0],:]
            d_nebr = d_xy[index_s[0],:]
            n_nebr = len( index_s[0] )
            idx_nebr = index_s[0].reshape( ( -1, 1 ) )
            return c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr

    elif isST == 1:#space time case

        #get distance of time
        d_t = numpy.float64(numpy.abs( c[:,2:3] - c_one[:,2:3] ))
        index_t = numpy.where( d_t <= dmax[0][1] )
        if len(index_t[0]) == 0:
            # print("noneighbor")
            return empty_result

        #get distance of space which already match time
        d_xy = coord2dist( c[index_t[0],0:2], c_one[:,0:2] )
        index_s = numpy.where( d_xy <= dmax[0][0] )
        if len(index_s[0]) == 0:
            # print("noneighbor")
            return empty_result
        
        #calculate all distance which matched perfectly
        d_r = d_xy[index_s[0],0:1] + dmax[0][2] * d_t[ index_t[0] [ index_s[0] ],0:1]
        index_r = numpy.where( d_r <= dmax[0][0] + dmax[0][2] * dmax[0][1] )
        
        if len( index_r[0] ) == 0:
            n_nebr = 0
            return empty_result
        elif len( index_r[0] ) <= nmax:
            c_nebr = c[index_t[0],:][index_s[0],:][index_r[0],:]
            z_nebr = z[index_t[0],:][index_s[0],:][index_r[0],:]
            d_nebr = d_r[index_r[0],:]
            n_nebr = len( index_r[0] )
            idx_nebr = index_t[0].reshape( ( -1, 1 ) )[index_s[0][index_r[0]],:]
            return c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr
        elif len( index_r[0] ) > nmax:
            d_nebr = d_r[index_r[0],:]
            index_r = ( numpy.sort( d_nebr[:,0].argsort()[:nmax] ), 0 ) #dummy 0 for consistence
            c_nebr = c[index_t[0],:][index_s[0],:][index_r[0],:]
            z_nebr = z[index_t[0],:][index_s[0],:][index_r[0],:]
            d_nebr = d_r[index_r[0],:]
            n_nebr = len( index_r[0] )
            idx_nebr = index_t[0].reshape( ( -1, 1 ) )[index_s[0][index_r[0]],:]
            return c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr
   
