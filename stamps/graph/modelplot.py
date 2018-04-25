# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 22:16:03 2015

@author: hdragon689
"""
from six.moves import range
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..stats.stcovfit import covmodelest,anisocovmodelest
from ..general.coord2K import coord2dist
from ..stats.stcovfit import cal_cov_mod

def modelplot(C,r,covmodel=None,covparam=None):
  '''
  Plot the empirical and modeled covariances for both spatial and S/T cases
  
  Syntax: modelplot(C,rLag,tLag,covmodel,covparam)

  Input:
  C     ns by nt    2D array of empirical covariance with ns spatial lags 
                    and nt temporal lags
  r     list        list of two 1D arrays of spatial and temporal lags 
                    [rLags, tLags]
  covmodel  list    default is None that only plot the empirical covariance                             
  covparam  list        
  
  Remark: detais of covmodel and covparam can refer to stamps.general.coord2K
  '''
  if len(r)==2 and type(r) is list: # space-time case
    rLag=r[0]
    tLag=r[1] 
    tLagM,rLagM=np.meshgrid(tLag,rLag)  
    if (covmodel is not None) and (covparam is not None):    
      rLagI=np.linspace(rLag[0],rLag[-1],50)
      tLagI=np.linspace(tLag[0],tLag[-1],50)   
      tLagMI,rLagMI=np.meshgrid(tLagI,rLagI)      
      modelcov,covi=covmodelest(rLagMI,tLagMI,covmodel,covparam)
    else:
      modelcov=np.zeros(1)*np.nan
      # Create cov plot and cov fit functions
    plt.figure()
    ax1=plt.subplot2grid((2,2),(0,0))
    ax2=plt.subplot2grid((2,2),(1,0))
    ax3=plt.subplot2grid((2,2),(0,1),rowspan=2,projection='3d')
    ax1.plot(rLag,C[:,0],'bo')
    ax2.plot(tLag,C[0,:],'bo') 
    ax3.scatter(rLagM,tLagM,C, color='b')
    ax1.set_xlabel('Spatial distance')
    ax1.set_ylabel('Covariance')
    ax2.set_xlabel('Temporl lag')
    ax2.set_ylabel('Covariance')
    ax3.set_xlabel('r')
    ax3.set_ylabel(r'$\tau$')
    ax3.set_zlabel('C')
    ax1.set_title('Spatial Covariance')
    ax2.set_title('Temporal Covariance')
    ax3.set_title('S/T Covariance')
    ax3.set_xlim([rLag[0],rLag[-1]])
    ax3.set_ylim([tLag[0],tLag[-1]])
    Cint=(C[~np.isnan(C)].max()-C[~np.isnan(C)].min())*0.02
    ax3.set_zlim([C[~np.isnan(C)].min()-Cint,C[~np.isnan(C)].max()+Cint])
    if not np.isnan(modelcov).all():
      ax1.plot(rLagI,modelcov[:,0],'r-')
      ax2.plot(tLagI,modelcov[0,:],'r-')
      ax3.plot_wireframe(rLagMI, tLagMI, modelcov,color='r', rstride=3, cstride=3)
      maxz=np.max([modelcov.max(),C[~np.isnan(C)].max()])
      minz=np.max([modelcov.min(),C[~np.isnan(C)].min()])
      ax3.set_zlim([minz,maxz])
      ax3.view_init(15,45); plt.draw()      
  else: 
    if len(r)==1 and type(r) is list:
      rLag=r[0].flat[:]
    elif type(r) is np.ndarray:
      rLag=r
    if (covmodel is not None) and (covparam is not None):    
      rLagI=np.linspace(rLag[0],rLag[-1],50)
      tLagI=np.array([0])
      tLagMI,rLagMI=np.meshgrid(tLagI,rLagI)  
      modelcov,covi=covmodelest(rLagMI,tLagMI,covmodel,covparam)
    plt.figure()
    plt.hold(True)
    plt.plot(rLag,C.flat[:],'o',label='Empirical covariance')
    if (covmodel is not None) and (covparam is not None):
      plt.plot(rLagI,modelcov.flat[:],'r-',label='Covariance model')
    plt.xlabel('Distance')
    plt.ylabel('Covariance')
    plt.xlim([rLag[0],rLag[-1]])
    Cint=(C[~np.isnan(C)].max()-C[~np.isnan(C)].min())*0.02
    plt.ylim([C[~np.isnan(C)].min()-Cint,C[~np.isnan(C)].max()+Cint])
    if (covmodel is not None) and (covparam is not None):
      plt.legend(loc='upper right')
  plt.show()    


def anisomodelplot(C,r,covmodel=None,covparam=None,theta=None,ratio=None):  
  '''
  Plot the empirical and modeled covariances
  
  Syntax: anisomodelplot(C,rLag,tLag,covmodel,covparam)

  Input:
  C     ns by nt    2D array of empirical covariance with ns spatial lags 
                    and nt temporal lags
  r     list        list of three 1D arrays of spatial and temporal lags 
                    [rLags, tLags, angles]. In pure spatial case, [rLags, angles] 
                    is used
  covmodel  list    default is None that only plot the empirical covariance                             
  covparam  list    
  theta     scalar  the theta for the covariance model (in radian)
  ratio     scalar  the ratio between secondary and principle axis of the 
                    anisotropic ellipse. The value is in the range between 0 and 1
  
  Remark: 
  detais of covmodel and covparam can refer to stamps.general.coord2K
  details of anisotropic parameters can refer to stamps.stats.stcovfit.anisocovmodelest
  
  '''  
  nang=len(C)
  if len(r)==3:
    rLag=r[0]
    tLag=r[1]
    aLag=r[2]  
  elif len(r)==2:
    rLag=r[0]
    aLag=r[1]
    tLag=None    
    
  modelcov=[None]*nang

  if covmodel is not None:
    # some error check may be required
    for i in range(nang):
      ang=np.array([aLag[i]])
      if tLag is None:
        tLag=np.array([0])
      tLagI,rLagI=np.meshgrid(tLag,rLag)
      covang,_= anisocovmodelest(covmodel,covparam,theta,ratio,ang,rLagI,tLagI)
      modelcov[i]=covang[0]

  if tLag is None or tLag.size == 1: # pure spatial case
    ax=[None]*nang
    cols=2
    rows=np.int(np.ceil(nang/cols))
    plt.figure()
    for i in range(nang):
      if i<rows:
        ax[i]=plt.subplot2grid((rows,cols),(i,0))
      else:
        ax[i]=plt.subplot2grid((rows,cols),(i-rows,1))
      ax[i].plot(rLag,C[i][:,0],'bo')
      ax[i].set_xlabel('Spatial distance')
      ax[i].set_ylabel('Covariance')
      ax[i].set_xlim([rLag[0],rLag[-1]])
      ax[i].text(0.95, 0.95, '$%5.2f^o$' % (aLag[i]/np.pi*180.),
            verticalalignment='top', horizontalalignment='right',
            transform=ax[i].transAxes,color='black', fontsize=10)      
      if modelcov[0] is not None:
        ax[i].plot(rLag,modelcov[i][:,0],'r-')
      if i==0 or i-rows==0:
        ax[i].set_title('Spatial Covariance')     
  else:
    ax=[None]*nang*2
    cols=2
    rows=nang
    plt.figure()
    for i in range(nang):
      ax[i]=plt.subplot2grid((rows,cols),(i,0))
      ax[i].plot(rLag,C[i][:,0],'bo')
      ax[i+rows]=plt.subplot2grid((rows,cols),(i,1))
      ax[i+rows].plot(tLag,C[i][0,:],'bo')
      ax[i].text(0.95, 0.95, '$%5.2f^o$' % (aLag[i]/np.pi*180.),
        verticalalignment='top', horizontalalignment='right',
        transform=ax[i].transAxes,color='black', fontsize=10)
      ax[i+rows].text(0.95, 0.95, '$%5.2f^o$' % (aLag[i]/np.pi*180.),
        verticalalignment='top', horizontalalignment='right',
        transform=ax[i+rows].transAxes,color='black', fontsize=10)  
      if i==0:
        ax[i].set_title('Spatial Covariance' % (aLag[i]/np.pi*180.))  
        ax[i+rows].set_title('Temporal Covariance' % (aLag[i]/np.pi*180.))               
      if i==nang-1:
        ax[i].set_xlabel('Spatial distance')
        ax[i].set_ylabel('Covariance')
        ax[i].set_xlim([rLag[0],rLag[-1]])
        ax[i+rows].set_xlabel('Temporal lag')
        ax[i+rows].set_ylabel('Covariance')
        ax[i+rows].set_xlim([tLag[0],tLag[-1]])
      else:
        ax[i].axes.get_xaxis().set_ticks([])
        ax[i+rows].axes.get_xaxis().set_ticks([])

      if modelcov[0] is not None:
        ax[i].plot(rLag,modelcov[i][:,0],'r-')    
        ax[i+rows].plot(tLag,modelcov[i][0,:],'r-') 


def mlemodelplot(ch,zh,covmodel=None,covparam=None):
  
  zh=zh.reshape(zh.size,1)
 
  cov=zh.dot(zh.T)
  covu=cov[np.triu_indices(zh.size)]
  dist_s=coord2dist(ch[:,:2],ch[:,:2])
  distu_s=dist_s[np.triu_indices(zh.size)]
  if ch.shape[1]>2:
    if type(ch[0,-1])==np.datetime64:
      origin=ch[0,-1]
      ch[:,-1]=np.double(np.asarray(ch[:,-1],dtype='datetime64')-origin)
      ch=ch.astype(np.double)
    dist_t=coord2dist(ch[:,2],ch[:,2])
    distu_t=dist_t[np.triu_indices(zh.size)]
    
  if (covmodel is not None) and (covparam is not None):
    rLagI=np.linspace(0,dist_s.max()*2./3,50)
    if ch.shape[1]>2:
      tLagI=np.linspace(0,dist_t.max()*2./3,50)   
    else:
      tLagI=np.array([0])
    tLagMI,rLagMI=np.meshgrid(tLagI,rLagI)      
    modelcov,covi=covmodelest(rLagMI,tLagMI,covmodel,covparam)  
    
  if ch.shape[1]>2:
    print ''
    # to be written
  else:
    plt.figure()
    plt.hold(True)
    plt.scatter(distu_s.flat[:],covu.flat[:])
    plt.plot(rLagI,modelcov,'r-')
    plt.xlim(0,rLagI.max())
    plt.ylim(covu.min(),covu.max())
    plt.xlabel('Distance')
    plt.ylabel('Covariance')


def covariance_model_plot(cov_model, s_range=None, t_range=None, show=True):
    s_range = max([i[2] for i in cov_model]) if not s_range else s_range
    t_range = max([i[4] for i in cov_model]) if not t_range else t_range

    cov_s_lags = np.linspace(0, s_range, 100)
    cov_t_lags = np.linspace(0, t_range, 100)
    cov_t_lags_grid, cov_s_lags_grid = np.meshgrid(cov_t_lags,cov_s_lags)


    cov_st_lags = np.array(
        zip(*map(
            lambda x:x.flatten(),
            np.meshgrid(cov_s_lags,cov_t_lags)
            ))
        )
    cov_z = cal_cov_mod(cov_st_lags, cov_model)
    cov_z_grid = cov_z.reshape((cov_s_lags.shape[0],cov_t_lags.shape[0]))

    if show:
        plt.figure(1)
        plt.subplot(211)
        plt.plot(cov_s_lags_grid[:,0], cov_z_grid[:,0], 'b--')
        plt.subplot(212)
        plt.plot(cov_t_lags_grid[0], cov_z_grid[0], 'r--')

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(2)
        ax3d = Axes3D(fig)
        ax3d.plot_wireframe(cov_t_lags_grid, cov_s_lags_grid, cov_z_grid)

    
        plt.show()

    return cov_t_lags_grid, cov_s_lags_grid, cov_z_grid
