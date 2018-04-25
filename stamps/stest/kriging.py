# -*- coding: utf-8 -*-
from six.moves import range
import numpy as np

from ..general.neighbours import neighbours
from ..general.coord2K import coord2K
from ..stest.designmatrix import designmatrix

def kriging(ck,ch,zh,model,param,nhmax,dmax,order=None,options=None):
  ''' 
% kriging                   - prediction using kriging methods 
%
% Standard linear kriging algorithm for processing hard data
% that can reasonably be assumed as Gaussian distributed. The
% function is intendend to be as general as possible, covering
% various situations, like non-stationarity of the mean,
% multivariate cases, nested models, space-time estimations,
% etc. Depending on the case, specific format are needed for
% the input variables. This function is a special case of the more
% general BMEprobaMoments.mfunction, which processes both hard and 
% soft data.
%
% SYNTAX :
%
% [zk,vk]=kriging(ck,ch,zh,model,param,nhmax,dmax,order,options);
%
% INPUT :
%
% ck        nk by d   matrix of coordinates for the estimation locations.
%                     A line corresponds to the vector of coordinates at
%                     an estimation location, so the number of columns
%                     corresponds to the dimension of the space. There is
%                     no restriction on the dimension of the space.
% ch        nh by d   matrix of coordinates for the hard data locations,
%                     with the same convention as for ck.
% zh        nh by 1   vector of values for the hard data at the coordinates
%                     specified in ch.
  covmodel  list      list of m nested covariance models in each of 
                      which the spatial and temporal components are put 
                      in a list as [covmodelS,covmodelT] 
  covparam0 list      list of intial values covariance parameters for 
                      m covmodels in each component the parameters are 
                      listed as [sill,[covparamS1,covparamS2,..],
                      [covparamT1,covparamT2,..]]
% nhmax     scalar    maximum number of hard data values that are considered
%                     for the estimation at the locations specified in ck.
% dmax      scalar    maximum distance between an estimation location and
%                     existing hard data locations. All hard data locations
%                     separated by a distance smaller than dmax from an
%                     estimation location will be included in the estimation
%                     process for that location, whereas other hard data
%                     locations are neglected.
% order     scalar    order of the polynomial mean along the spatial axes at
%                     the estimation locations. For the zero-mean case, NaN
%                     (Not-a-Number) is used. Note that order=NaN can only be
%                     used with covariance models and not with variogram models.
% options   scalar    optional parameter that can be used if the default value
%                     is not satisfactory (otherwise it can simply be omitted
%                     from the input list of variables). options(1) is taking
%                     the value 1 or 0 if the user wants or does not want to
%                     display the order number of the location which is
%                     currently processed, respectively.
%
% OUTPUT :
%
% zk        nk by 1   vector of estimated values at the estimation locations. A
%                     value coded as NaN means that no estimation has been performed
%                     at that location due to the lack of available data. 
% vk        nk by 1   vector of estimation (kriging) variances at the estimation
%                     locations. As for zk, a value coded as NaN means that no
%                     estimation has been performed at the corresponding location.
%       
    
    
    input
    ck: nk by d float
    ch: nh by d float
    zh: nh by 1 float
    model: mn by 1 string
    param: mn by 3 float
    nhmax: int
    dmax: 1 by 3 float
    order: None or 0
    options: not no use here
    
    return
    zk: nk by 1 float
    vk: nk by 1 float
  '''
    
    #initial return variable
  nk = ck.shape[0]
  zk = np.zeros( ( nk, 1 ) )
  vk = np.zeros( ( nk, 1 ) )
  zk[:] = np.nan
  vk[:] = np.nan
  
  if order is None:
    order = np.nan
    
  for i in range( ck.shape[0] ):
    ci = ck[i:i+1]
    ci_nebr,zi_nebr,di_nebr,ni_nebr,idxi_nebr=neighbours(ci,ch,zh,nhmax,dmax)
    if ni_nebr > 0:
      K, dummyKK = coord2K( ci_nebr, ci_nebr, model, param )
      k, dummykk = coord2K( ci_nebr, ci, model, param ) #kk is a list contain ki
      k0, dummykk0 = coord2K(ci, ci, model, param)
      X=designmatrix(ci_nebr,order)[0]  
      x=designmatrix(ci,order)[0]
#            unit = np.ones(k.shape) # n by 1, X in matlab
#            unit_t_add = np.append( unit.T, [[0]], axis = 1 ) # 1 by n+1, x in matlab
                 
            #change shape for kriging
#            Kadd = np.hstack( ( K, unit ) )
#            Kadd = np.vstack( ( Kadd, unit_t_add ) )
      Kadd = K
      Xm,Xn = X.shape
      Kadd = np.vstack([np.hstack([K,X]),
                           np.hstack([X.T,np.zeros((Xn,Xn))])])
#            kadd = np.append( k, [[0]], axis = 0 )
      kadd = np.vstack([k,x.T])
      weight = np.linalg.solve(Kadd,kadd)[0:Xm]
      weight_t = weight.T
            #compute zk, vk
      zk[i] = weight_t.dot( zi_nebr )
      vk[i] = (k0-2*weight_t.dot(k)+weight_t.dot(K).dot(weight))[0]
            
    else:
      pass #already give NaN
  
  return zk, vk

def krigconstr(c0,c,order=None):
  '''
  order is None or 0 for now
  '''
  

  return X,x
    
def krigingFac( ck, ch, zh, model, param, nhmax, dmax, order = 0, options = None):

    ''' 
    input
    ck: nk by d float
    ch: nh by d float
    zh: nh by 1 float
    model: mn by 1 string ( ex. "exponential/gaussian" )
    param: mn by 3 float (ex. [ c, s, t ] )
    nhmax: int
    dmax: 1 by 3 float
    order: always 0, equals NaN in matlab
    options: not no use here, 0 or 1 in matlab for display echo 
    
    return
    zk: mn+1 by 1 float
    vk: mn+1 by 1 float
    '''
    
    #initial return variable
    nk = ck.shape[0]
    mn = len(model) #model.shape[0]
    zk = np.zeros( ( nk, mn+1 ) )
    vk = np.zeros( ( nk, mn+1 ) )
    zk[:] = np.nan
    vk[:] = np.nan
    
    for i in range( ck.shape[0] ):
        ci = ck[i:i+1]
        ci_nebr, zi_nebr, di_nebr, ni_nebr, idxi_nebr = neighbours(ci, ch, zh, nhmax, dmax )
        if ni_nebr > 0:
            K, dummyKK = coord2K( ci_nebr, ci_nebr, model, param )
            dummyk, kk = coord2K( ci_nebr, ci, model, param ) #kk is a list contain ki
            dummyk0, kk0 = coord2K(ci, ci, model, param)
            unit = np.ones(kk[0].shape) # n by 1
            unit_t_add = np.append( unit.T, [[0]], axis = 1 ) # 1 by n+1
            for idx_k, (ki,k0i) in enumerate( zip( kk, kk0 ) ):          
                #change shape for kriging
                Kadd = np.hstack( ( K, unit ) )
                Kadd = np.vstack( ( Kadd, unit_t_add ) )
                kkadd = np.append( ki, [[0]], axis = 0 )
                weight = np.dot( np.linalg.inv( Kadd ), kkadd )[:-1,:]
                weight_t = weight.T
                #compute zk, vk
                zk[i:i+1,idx_k:idx_k+1] = weight_t.dot( zi_nebr )
                vk[i:i+1,idx_k:idx_k+1] = ( k0i - 2 * weight_t.dot( ki ) + weight_t.dot( K ).dot( weight ) )[0]
            #compute local mean trend
            kkadd = np.append( np.zeros( kk[0].shape ), [[1]], axis = 0 )
            weight = np.dot( np.linalg.inv( Kadd ), kkadd )[:-1,:]
            weight_t = weight.T
            zk[i:i+1,-1:] = weight_t.dot( zi_nebr )
            vk[i:i+1,-1:] = weight_t.dot( K ).dot( weight )[0]
        else:
            pass #already give NaN
    return zk, vk

