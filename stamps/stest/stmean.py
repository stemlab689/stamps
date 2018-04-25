# -*- coding: utf-8 -*-
from six.moves import range
import numpy as np
from scipy.interpolate import griddata

from ..general.coord2dist import coord2dist
from ..stats.eof import eof
from ..stats.stl import stl
from ..stest import idw


#try:
#    from pylab import griddata as pylab_griddata #qgis
#except ImportError:
#    from matplotlib.pylab import griddata as pylab_griddata #OSGEO4W qgis

def _tricube(x):
    w=np.zeros(x.size)
    xx=np.abs(x)
    idx=np.where(np.bitwise_and(xx>=0,xx<1))
    w[idx]=(1-xx[idx]**3)**3
    return w

def _bicube(x):
    w=np.zeros(x.size)
    idx=np.where(np.abs(x)<1)
    w[idx]=(1-np.abs(x[idx])**2)**2
    return w 
def _exp(x):
    w=np.zeros(x.size)
    w=np(-x)
    return w   

def stmean( grid_s, grid_t, grid_z, pars=None,kernel='tricube', DataObj = None ):
    '''
% stmean                  - Estimates space/time mean values from measurements
%
% Assuming a separable additive space/time mean trend model, this 
% function calculates the spatial mean component ms and temporal mean 
% component mt of space/time random field Z, using measurements at 
% fixed measuring  sites cMS and fixed measuring events tME. 
% The spatial mean component ms is obtained by averaging the measurements
% at each measuring sites. Then a smoothed spatial mean component mss
% is obtained by applying a specified spatial filter to ms.
% Similarly mt is obtained by averaging the measurement for each
% measuring event, and a smoothed temporal mean component mts is obtained
% by applying the same function of temporal filter to mt.
% Then the space/time mean trend is simply given by 
% mst(s,t)=mss(s)+mts(t)-mean(mts)
%
% SYNTAX :
%
% [ms,mss,mt,mts,stmean]=stmean(cMS,tME,Z,pars,DataObj=None);
%
% INPUTS :  
%  cMS    nMS by 2   matrix of spatial x-y coordinates for the nMS monitoring
%                    sites
%  idMS   nMS by 1   vector with a unique id number for each monitoring site
%  tME    1 by nME   vector with the time of the measuring events
%  Z      nMS by nME matrix of measurements for Z at the nMS monitoring
%                    sites and nME measuring events. Z may have NaN values.
%  pars   1 by 4     parameters to smooth the spatial and temporal average
%                    p[0]=dNeib  distance (radius) of spatial neighborhood
%                    p[1]=tNeib  time (radius) of temporal neighborhood
                     p[2]=ar  spatial range of kernel smoothing function 
%                    p[3]=at  temporal range of kernel smoothing function
     method string     specified kernel function, including exp, tricube, bicube 
                                         functions. default is tricube function. 

% OUTPUT :
%
%  ms     nMS by 1   vector of spatial average
%  mss    nMS by 1   vector of smoothed spatial average
%  mt     1 by nME   vector of temporal average
%  mts    nMS by 1   vector of smoothed temporal average   
    '''

#    if not DataObj:
#        from nousedataobj import NoUseDataObj
#        DataObj = NoUseDataObj()
    if DataObj:    
        title = DataObj.getProgressText()    
        DataObj.setProgressRange(0,len(grid_s))
        DataObj.setCurrentProgress(0, title + "\n- By STMean...")

    grid_t=np.reshape(grid_t,(1,grid_t.size))  
    grid_t=grid_t.astype(np.float64)

    mask_grid_z = np.ma.masked_array(grid_z,np.isnan(grid_z))
    mean_s = np.array( mask_grid_z.mean( axis = 1 ) , ndmin = 2).T
    mean_t = np.array( mask_grid_z.mean( axis = 0 ) , ndmin = 2)
#  mean_st = mask_grid_z.mean()


    mss=np.zeros(mean_s.shape)
    mts=np.zeros(mean_t.shape)

    s_dist=coord2dist(grid_s,grid_s)
    t_dist=coord2dist(grid_t.T,grid_t.T,1) 
    if pars is None:
        dNeib=np.max(s_dist)*0.4
        tNeib=np.max(t_dist)*0.4
        ar=dNeib
        at=tNeib    
    else:
        dNeib=pars[0]
        tNeib=pars[1]
        ar=pars[2]
        at=pars[3]
    
    if kernel is 'exp':
        fun='_exp'
    elif kernel is 'tricube':
        fun='_tricube'
    elif kernel is 'bicube':
        fun='_bicube'
        
    for i in range(grid_s.shape[0]):
        ids=np.where(s_dist[i]<=dNeib)
        w=eval(fun+'(s_dist[i][ids]/ar)')#_tricube(s_dist[i]/dNeib)
        w=w/np.sum(w)  # normalize
        mss[i]=(w*mean_s.flat[ids].flat[:]).sum()
        
    for j in range(grid_t.size):
        idt=np.where(t_dist[j]<=tNeib)
        w=eval(fun+'(t_dist[j][idt]/at)')#w=_tricube(t_dist[j]/tNeib)
        w=w/np.sum(w)  # normalize
        mts[0,j]=(w*mean_t.flat[idt].flat[:]).sum()

#  grid_trend = mss + mts - mean_st    
#  grid_trend[np.where(np.isnan(grid_z))] = np.nan 
            
    return mean_s,mss,mean_t,mts
        
#    print grid_z
#    print mean_s
#    print mean_t
#    print grid_trend
#    print mean_s + mean_t - mean_st

def stmean_stl( grid_s, grid_t, grid_z, pars=None, kernel='tricube', DataObj = None ):
    '''
% stmean_stl                  - Estimates space/time mean values from measurements
%
% Assuming a separable additive space/time mean trend model, this 
% function calculates the spatial mean component ms and temporal mean 
% component mt of space/time random field Z, using measurements at 
% fixed measuring  sites cMS and fixed measuring events tME. 
% The spatial mean component ms is obtained by averaging the measurements
% at each measuring sites. Then a smoothed spatial mean component mss
% is obtained by applying an exponential spatial filter to ms.
% In this function, a smoothed temporal mean component mts is obtained
% by applying the stl function to extract the periodic trend of mt. 
% Then the space/time mean trend is simply given by 
% mst(s,t)=mss(s)+mts(t)-mean(mts)
%
% SYNTAX :
%
% [ms,mss,mt,mts,stmean]=stmean(cMS,tME,Z,pars,DataObj=None);
%
% INPUTS :  
%  cMS    nMS by 2   matrix of spatial x-y coordinates for the nMS monitoring
%                    sites
%  idMS   nMS by 1   vector with a unique id number for each monitoring site
%  tME    1 by nME   vector with the time of the measuring events
%  Z      nMS by nME matrix of measurements for Z at the nMS monitoring
%                    sites and nME measuring events. Z may have NaN values.
%  pars   1 by 2     parameters to smooth the spatial and temporal average
%                    p[0]=dNeib  distance (radius) of spatial neighborhood
%                    p[1]=np     the time span for seasonal periodic variation 
                                                                 (default=12)
                                         p[2]=ar  spatial range of kernel smoothing function             
%
% OUTPUT :
%
%  ms     nMS by 1   vector of spatial average
%  mss    nMS by 1   vector of smoothed spatial average
%  mt     1 by nME   vector of temporal average
%  mts    nMS by 1   vector of smoothed temporal average   

    Note: See stats.stl 
    '''

#    if not DataObj:
#        from nousedataobj import NoUseDataObj
#        DataObj = NoUseDataObj()
    if DataObj:    
        title = DataObj.getProgressText()    
        DataObj.setProgressRange(0,len(grid_s))
        DataObj.setCurrentProgress(0, title + "\n- By STMean...")

    grid_t=np.reshape(grid_t,(1,grid_t.size))  
    grid_t=grid_t.astype(np.float64)

    mask_grid_z = np.ma.masked_array(grid_z,np.isnan(grid_z))
    mean_s = np.array( mask_grid_z.mean( axis = 1 ) , ndmin = 2).T
    mean_t = np.array( mask_grid_z.mean( axis = 0 ) , ndmin = 2)
#  mean_st = mask_grid_z.mean()

    if kernel is 'exp':
        fun='_exp'
    elif kernel is 'tricube':
        fun='_tricube'
    elif kernel is 'bicube':
        fun='_bicube'
        
    mss=np.zeros(mean_s.shape)
    mts=np.zeros(mean_t.shape)

    s_dist=coord2dist(grid_s,grid_s)
    if pars is None:
        dNeib=np.max(s_dist)*0.4
        np=12
    else:
        dNeib=pars[0]
        np=pars[1]
        ar=pars[2]
        
    for i in range(grid_s.shape[0]):
        ids=np.where(s_dist[i]<=dNeib)
        w=eval(fun+'(s_dist[i][ids]/ar)')#_tricube(s_dist[i]/dNeib)
        w=w/np.sum(w)  # normalize
        mss[i]=(w*mean_s.flat[ids].flat[:]).sum()
        
    mts=stl(mean_t.ravel(),ns='per',np=np)
            
    return mean_s,mss,mean_t,mts  

def stmean_eof(grid_s, grid_t, grid_z, n=5, norm=0, norms_direc=0, method='svd'):

    '''
    Remove mean trend with Empirical Orthogonal Function
    '''

    L, lambdaU2, PC, EOFs, ECs, error, norms = eof(grid_z, n, norm, norms_direc, method)  
    meantrend = np.dot(ECs,EOFs.T)

    return meantrend

def stmeaninterp(grid_s, grid_t, grid_z, est_grid_s, est_grid_t,
                             method='idw', pars=None, smooth=True, DataObj = None):
    '''
    Calculates space/time interpolation of mean trend values using 
    an additive separable s/t mean trend model.
    Using the spatial smoothed mean component ms  and the temporal
    smoothed mean component mt obtained by the function mst from
    from measurements at monitoring sites coordinates cMS and measuring
    events times tME, the s/t mean trend at cooridnate cI and time cI
    is given by
                 mst(s,t)= ms(cI) + mt(tI) - mean(mt),
    where ms(cI) is interpolated from ms at the monitoirng sites, and 
    mt(tI) is interpolated from the mt at the measuring events tME.  

    SYNTAX:
    mstI = stmean_est(cMS, tME, data, xyi, ti, 
                                        method='idw', pars=None, smooth=True, DataObj = None):
    Input:
    
    cMS     m by 2      2D np array of spatial coordinate of the monitors                              
    tME     1 by n      2D np array of times of observed events
    data    m by n      2D np array of observations
    xyi     si by 2     2D np array for spatial coordinates of si interpolated 
                                            locations
    ti      1 by ti     2D np array for times to be interpolated
    method  string      method for spatial trend interpolation. 
                                            Default is idw and griddata is another options including 
                                            linear, cubic, and nearest 
    pars    1 by 2      parameters to smooth the spatial and temporal average
                                            p[0]=dNeib  distance (radius) of spatial neighborhood
                                            p[1]=tNeib  time (radius) of temporal neighborhood 
                                            Default is the 2/5 of the spatial and temporal ranges are 
                                            used as neighborhoods, respectively.
    smooth  bool        optioal. If the smoothed spatial and temporal means are used.
                                            Default is True.                   
    
    Output:    
    mstI    si by ti    2D np array for space-time interpolation
    '''
#    if not DataObj:
#        from nousedataobj import NoUseDataObj
#        DataObj = NoUseDataObj()
    if DataObj:        
        title = DataObj.getProgressText()
        DataObj.setProgressRange(0,len(grid_s))
        DataObj.setCurrentProgress(0, title + "\n- By STMean...")    
        
    # grid_t can be np.datetime or other datetime formats
    # Create the ordinal values for grid_t(tME) for the following operations 
    # by H-L Yu   
        
    # TO-DO add try except for error capture from the wrong input format  
    
    grid_t=np.reshape(grid_t,(1,grid_t.size))  
    grid_t=grid_t.astype(np.float64)
    est_grid_t=np.reshape(est_grid_t,(1,est_grid_t.size))
    est_grid_t=est_grid_t.astype(np.float64) 
    
    ms,mss,mt,mts = stmean(grid_s,grid_t,grid_z,pars)
    if smooth:
        mean_s=mss
        mean_t=mts
    else:
        mean_s=ms
        mean_t=mt
    
    mask_grid_z = np.ma.masked_array(grid_z,np.isnan(grid_z))
#  mean_s = np.array( mask_grid_z.mean( axis = 1 ) , ndmin = 2).T
#  mean_t = np.array( mask_grid_z.mean( axis = 0 ) , ndmin = 2)
    mean_st = mask_grid_z.mean()
        
    if method=='linear':
        mean_s_est = griddata(grid_s, mean_s,est_grid_s)
        mean_s_est[np.isnan(mean_s_est)]=np.mean(mean_s) 
        #Set the nan values to average of mean_s
    elif method=='nearest':
        mean_s_est = griddata(grid_s, mean_s,est_grid_s,method='nearest')
    elif method=='cubic':
        mean_s_est = griddata(grid_s, mean_s,est_grid_s,method='cubic')
    else:
        temp_x,temp_y = map(np.array,zip(*grid_s))
        temp_x_est, temp_y_est = map(np.array,zip(*est_grid_s))
        mean_s_est = idw.idw_est( temp_x, temp_y, mean_s.T[0], temp_x_est, temp_y_est, power = 2 )
    
    mean_s_est=mean_s_est.ravel()  
#    mean_s_est = pylab_griddata(temp_x,temp_y,mean_s.T[0],
#                                temp_x_est,temp_y_est,interp = 'nn')
        #temp_x_est and temp_y_est is not mono increasing
        #i don't know whether there is a bug in it
#    
#    mean_s_est = mean_s_est.diagonal()
    mean_s_est = np.array( mean_s_est,ndmin=2 ).T
    # enforce the temporally linear interpolation
    est_tmin=np.min(np.hstack([est_grid_t[0],grid_t[0]]))
    est_tmax=np.max(np.hstack([est_grid_t[0],grid_t[0]]))
    data_tgrid=np.hstack([est_tmin,grid_t[0],est_tmax])
    data_tmean=np.hstack([mean_t[0][0],mean_t[0],mean_t[0][-1]])
    mean_t_est = np.array(np.interp(est_grid_t[0],data_tgrid,data_tmean),ndmin=2)
        
        #est_z_2d = pylab_griddata(pylab_x, pylab_y, point_value,est_x_2d, est_y_2d,interp = 'nn')
                
    grid_trend_est = mean_s_est + mean_t_est - mean_st
        
    return grid_trend_est
        
        
        

if __name__ == "__main__":
        grid_s=np.array([[1,3.],[1,8],[4,1],[3,2]])
        grid_t=np.array([[1,3,5,7,9.]])
        grid_z = np.array([[1,np.nan,3.,4,5],
                                                    [5,6,1,7,8],
                                                    [1,np.nan,4,2,5],
                                                    [5,2,6,3,1.]])
        
        grid_s_est=grid_s#np.array([[1,6.],[1,6]])[::-1]
        grid_t_est=grid_t#np.array([[1]])

        ms,mss,mt,mts = stmean(grid_s,grid_t,grid_z)
        print ms
        print mss
        print mt
        print mts
        grid_trend = stmeaninterp(grid_s,grid_t,grid_z,grid_s_est,grid_t_est)
        print grid_trend