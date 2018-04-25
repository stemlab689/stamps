# -*- coding: utf-8 -*-
# import math
from six.moves import range
import numpy as np
# import scipy.ndimage.filters.gaussian_filter as gaussian_filter
#import logging
#
#log_manager = logging.getLogger('debug_log')
#log_manager.setLevel(logging.DEBUG)
#fh = logging.FileHandler('debug.txt')
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#fh.setFormatter(formatter)
#log_manager.addHandler(fh)
            
def kernelsmoothing(grid_s,grid_t,grid_z,bs,bt,ktype = "gaussian", DataObj = None):
    '''
    grid_s: row by 2 np array
    grid_t: 1 by col np array
    grid_z: row by col np array
    bs: float
    bt: flaot
    ktype: string
    DataObj: for GUI use
    
    return grid_trend, raw by col np array, or return False if fail
    '''
    if not DataObj:
        from ..bme.nousedataobj import NoUseDataObj
        DataObj = NoUseDataObj()
        
    title = DataObj.getProgressText()
        
    #use old code ( grid_t was 1d np array )
    grid_t = grid_t[0]
    
    #determ func
    func = TypeToFunc(ktype)
    
    #create grid_trend
    grid_trend = grid_z.copy()
    
    DataObj.setProgressRange(0,len(grid_s))
    DataObj.setCurrentProgress(0, title + "\n- By Kernel Smoothing...")
    for index_s in range(grid_s.shape[0]):
        if not DataObj.wasProgressCanceled():  
            for index_t in range(grid_t.shape[0]):
                if np.isnan(grid_z[index_s][index_t]):
                    pass
                else:
                    ds = np.sqrt(((grid_s - grid_s[index_s])**2).sum(axis=1))
                    dt = np.abs(grid_t - grid_t[index_t])
                    
#                    index_ss = np.where(ds<=bs)
#                    index_tt = np.where(dt<=bt)
#                    selected_ds = ds[index_ss]
#                    selected_dt = dt[index_tt]
#                    
#                    selected_grid_z = grid_z[np.ix_(index_ss[0],index_tt[0])]
                    
                    ds_nomal = (np.array([ds]).T / bs)**2
                    dt_nomal = (dt / bt)**2
#                    ds_nomal = (np.array([selected_ds]).T / bs)**2
#                    dt_nomal = (selected_dt / bt)**2
                    
                    dr_matrix = ds_nomal + dt_nomal
                    kernel_w = func(dr_matrix)
                    
                    up = kernel_w * grid_z
                    kernel_w[np.isnan(grid_z)] = np.nan
                    down = kernel_w
                    grid_trend[index_s][index_t] = up[~np.isnan(up)].sum() / down[~np.isnan(down)].sum()
            
#                    up = kernel_w * selected_grid_z
#                    kernel_w[np.isnan(selected_grid_z)] = np.nan
#                    down = kernel_w
#                    grid_trend[index_s][index_t] = up[~np.isnan(up)].sum() / down[~np.isnan(down)].sum()
                    
            DataObj.setCurrentProgress(index_s + 1)
        else:
            return False
        
    DataObj.setCurrentProgress(text = title)
    return grid_trend
 
def kernelsmoothing_est(grid_s, grid_t, grid_z, 
                        est_grid_s, est_grid_t,
                        bs, bt, ktype = "gau", DataObj = None):



    if not DataObj:
        from ..bme.nousedataobj import NoUseDataObj
        DataObj = NoUseDataObj()

    title = DataObj.getProgressText()
    
    #determ func
    func = TypeToFunc(ktype)
    
    grid_t = grid_t[0]
    est_grid_t = est_grid_t[0]
    
    #create grid_trend
    est_grid_trend = np.empty((est_grid_s.shape[0],est_grid_t.shape[0]))
    est_grid_trend[:] = np.nan
    
    DataObj.setProgressRange(0,len(est_grid_s))
    DataObj.setCurrentProgress(0, title + "\n- By Kernel Smoothing...")
    for index_s in range(len(est_grid_s)):
        if not DataObj.wasProgressCanceled():  
            for index_t in range(len(est_grid_t)):
                ds = np.sqrt(((grid_s - est_grid_s[index_s])**2).sum(axis=1))
                dt = np.abs(grid_t - est_grid_t[index_t])
                
    #            index_ss = np.where(ds<=bs)
    #            index_tt = np.where(dt<=bt)
    #            selected_ds = ds[index_ss]
    #            selected_dt = dt[index_tt]
    #            selected_grid_z = grid_z[np.ix_(index_ss[0],index_tt[0])]
                
                ds_nomal = (np.array([ds]).T / bs)**2
                dt_nomal = (dt / bt)**2
    #            ds_nomal = (np.array([selected_ds]).T / bs)**2
    #            dt_nomal = (selected_dt / bt)**2
                
                dr_matrix = ds_nomal + dt_nomal
                kernel_w = func(dr_matrix)
                
                up = kernel_w * grid_z
                kernel_w[np.isnan(grid_z)] = np.nan
                down = kernel_w
                est_grid_trend[index_s][index_t] = up[~np.isnan(up)].sum() / down[~np.isnan(down)].sum()
            DataObj.setCurrentProgress(index_s + 1)
        else:
            return False
#            up = kernel_w * selected_grid_z
#            kernel_w[np.isnan(selected_grid_z)] = np.nan
#            down = kernel_w
#            est_grid_trend[index_s][index_t] = up[~np.isnan(up)].sum() / down[~np.isnan(down)].sum()
            
#    print 'grid_s=', grid_s
#    print 'grid_t=', grid_t
#    print 'grid_z=', grid_z
#    print 'grid_s_est=', est_grid_s
#    print 'grid_t_est=', est_grid_t
#    print 'grid_tr_est=', est_grid_trend
    return est_grid_trend

#def kernelsmoothing_cv(grid_s,grid_t,grid_z,bs,bt,ktype = "gaussian",sample = None):
#    #determ func
#    func = TypeToFunc(ktype)
#    
#    #create grid_trend
#    grid_trend = grid_z.copy()
#    
##    #sample list 
##    try:
##        sample_index_list = random.sample([[i,j] for i in range(len(grid_s)) for j in range(len(grid_t)) if ~np.isnan(grid_z[i][j])],sample)
##    except ValueError:
##        sample_index_list = [[i,j] for i in range(len(grid_s)) for j in range(len(grid_t)) if ~np.isnan(grid_z[i][j])]
#
#    
##    #get true sample number
##    samplenumber = len(sample_index_list)
##    
##    for (index_s,index_t) in sample_index_list:
#    for index_s in range(len(grid_s)):
#        for index_t in range(len(grid_t)):
#            if np.isnan(grid_z[index_s][index_t]):
#                pass
#            else:
#                ds = np.sqrt(((grid_s - grid_s[index_s])**2).sum(axis=1))
#                dt = np.abs(grid_t - grid_t[index_t])
#                
#                index_ss = np.where(ds<=bs)
#                index_tt = np.where(dt<=bt)
#                selected_ds = ds[index_ss]
#                selected_dt = dt[index_tt]
#                selected_grid_z = grid_z[np.ix_(index_ss[0],index_tt[0])]
#                          
#                ds_nomal = (np.array([selected_ds]).T / bs)**2
#                dt_nomal = (selected_dt / bt)**2
#                
#                dr_matrix = ds_nomal + dt_nomal
#                kernel_w = func(dr_matrix)
#                
#                up = kernel_w * selected_grid_z
#                kernel_w[np.isnan(selected_grid_z)] = np.nan
#                down = kernel_w
#                
#                #abstract self
#                
#                grid_trend[index_s][index_t] = (up[~np.isnan(up)].sum() - grid_z[index_s][index_t]) / (down[~np.isnan(down)].sum() - 1.)
#    error = (grid_z - grid_trend )**2
#    error = error[~np.isnan(error)]
#    samplenumber = error.size
#    square_error = error.sum()
#    mean_square_error = square_error/samplenumber
#    return mean_square_error
                 
def TypeToFunc(kerneltype):
    dictionary={"gau":gaussian,
                "gaussian":gaussian,
                "qua":quadratic,
                "quadratic":quadratic}
    return dictionary[kerneltype]

def gaussian(dr_matrix):
    answer = np.exp(-3 * dr_matrix)
    #answer[(dr_matrix > 1)] = 0.0
    return answer

def quadratic(dr_matrix):
    #log_manager.info(str(np.sqrt(dr_matrix)))
    answer =  1 - dr_matrix
    answer[dr_matrix > 1] = 0.0
    #log_manager.info(answer)
    #answer[dr_matrix < 0] = 0.0
    return answer

if __name__ == "__main__":
    
    import time
    func=gaussian

    grid_z = np.array([[1,np.nan,3.,4,5],
                          [5,6,1,7,8],
                          [1,np.nan,4,2,5],
                          [5,2,6,3,1.]])
    #grid_trend = grid_z.copy()
    grid_s=np.array([[1,3.],[1,8],[3,2],[4,1]])
    grid_t=np.array([[1,3,5,7,9.]])
    bs=8
    bt=2
    
    print kernelsmoothing(grid_s,grid_t,grid_z,bs,bt,ktype = "gaussian")
    
    print kernelsmoothing_est(grid_s, grid_t, grid_z, 
                        est_grid_s = grid_s, est_grid_t = grid_t,
                        bs = bs, bt = bt, ktype = "gau")