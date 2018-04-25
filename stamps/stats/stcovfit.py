# -*- coding: utf-8 -*-
from six.moves import range
import numpy as np
from copy import deepcopy

from ..general import bobyqa as bobyqa
from ..models.covmodel import get_model
from ..general.isspacetime import isspacetime

def covmodelwls(covmodel,covparam,lag_cov_s,lag_cov_t,
  lag_cov_v,lag_cov_n,lag_th=None,theta=None,ratio=None):
  '''

  This function calculate the value of objective function of 
  weighted least square for covaraince function fitting 
  proposed by Cressie in 1985

  Input:    
    
  covmodel    m             list of m nested covariance models in each of 
                            which the spatial and temporal components are put 
                            in a list as [covmodelS,covmodelT] 
  covparam    m             list of covariance parameters for m covmodels in 
                            each component the parameters are listed as 
                            [sill,[covparamS1,covparamS2,..],
                            [covparamT1,covparamT2,..]]
  lag_cov_s   nls by nlt    2D np array of the meshgrid of spatial lags. nls 
                            and nlt denote number of spatial and temporal lags
                            respectively                                        
  lag_cov_t   nls by nlt    2D np array of the meshgrid of temporal lags
  
  lag_cov_v   nls by nlt    2D np array of the empirical covariance values 
                            in the S/T lag meshgrid, ref. stcov function
  lag_cov_n   nls by nlt    2D np array of the number of data pair counts 
                            in the S/T lag meshgrid
  lag_th      na by 1       Optional. For anisotropic case, the directions to 
                            be evaluted   
  theta       1 by nd-1     vector of angle values that characterize the anisotropy. 
                            In a two dimensional space, angle is the trigonometric angle
                            between the horizontal axis and the principal axis of the
                            ellipse. In a three dimensional space, spherical coordinates
                            are used, such that angle(1) is the horizontal trigonometric
                            angle and angle(2) is the vertical trigonometric angle for the
                            principal axis of the ellipsoid. All the angles are measured
                            counterclockwise in degrees and are between -pi/2 and pi/2.                           
  ratio       1 by nd-1     1D array of vector that characterize the ratio for the length of the axes
                            for the ellipse (in 2D) or ellipsoid (in 3D). In a two dimensional
                            space, ratio is the length of the principal axis of the ellipse
                            divided by the length of the secondary axis, so that ratio<1. 
                            i.e., a_max=a, a_min=ratio*a. 
                            In a three dimensional space, ratio(1) is the length of the principal
                            axis of the ellipsoid divided by the length of the second axis, 
                            whereas ratio(2) is length of the principal axis of the ellipsoid
                            divided by the length of the third axis, so that ratio(1)<1 and
                            ratio(2)>1                                 

  Output: 
        
  objfv       scalar        the objective function from Cressie's 1985 paper

  '''
  if lag_th is None:
    lag_cov_est, dummy = covmodelest(lag_cov_s,lag_cov_t,covmodel,covparam)              
    ersqr = (lag_cov_v - lag_cov_est) ** 2
    wt = lag_cov_n / (lag_cov_est[0][0] ** 2 + lag_cov_est ** 2)
    wt /= wt.sum()
    objfv = wt * ersqr
    objfv = objfv[np.where(~np.isnan(objfv))].sum()
  else:
    lag_cov_est,_=anisocovmodelest(covmodel,covparam,theta,ratio,
                       lag_th,lag_cov_s,lag_cov_t)
    objfv=0.
    for i in range(lag_th.size):
      ersqr = (lag_cov_v[i] - lag_cov_est[i]) ** 2
      wt = lag_cov_n[i] / (lag_cov_est[i][0][0] ** 2 + lag_cov_est[i] ** 2)
      wt /= wt.sum()
      objfvi = wt * ersqr
      objfvii = objfvi[np.where(~np.isnan(objfvi))].sum()                 
      objfv=objfv+objfvii 
    
  return objfv

def covmodelfit(lag_cov_s,lag_cov_t,lag_cov_v,lag_cov_n,covmodel,covparam0,
  theta0=None,ratio0=None,lag_th=None,lower_bnd=None, upper_bnd=None): 
  '''

  This function automatically fits the S/T covariance models to S/T empiricl 
  covariances 

  Input:    
    
  lag_cov_s   nls by nlt    2D np array of the meshgrid of spatial lags. nls 
                            and nlt denote number of spatial and temporal lags
                            respectively                                        
  lag_cov_t   nls by nlt    2D np array of the meshgrid of temporal lags
  
  lag_cov_v   nls by nlt    2D np array of the empirical covariance values 
                            in the S/T lag meshgrid, ref. stcov function. For 
                            the anisotropic case, lag_cov_v should be a list 
                            containing na elements with nls by nlt covariance 
                            estimates at different theta speicified in lag_th. 
  lag_cov_n   nls by nlt    2D np array of the number of data pair counts 
                            in the S/T lag meshgrid
  covmodel    m             list of m nested covariance models in each of 
                            which the spatial and temporal components are put 
                            in a list as [covmodelS,covmodelT] 
  covparam0   m             list of intial values covariance parameters for 
                            m covmodels in each component the parameters are 
                            listed as [sill,[covparamS1,covparamS2,..],
                            [covparamT1,covparamT2,..]]
  theta0      scalar        Direction of principle axis of geometric anisotropy 
                            (in radian)
  ratio0      scalar        The ratio between the maximum and minimum ranges of 
                            the anisotropic ellipse. ratio=maximum/minimum and
                            therefore it has range of [0,1]
  lag_th      1 by na       the spatial or S/T covariance model at angle theta 
                            to be evaluated                       
  lower_bnd   m             Optional. The lower bound for covaraince parameters 
                            with the same format of covparam0. Default is None.
  lower_bnd   m             Optional. The lower bound for covaraince parameters 
                            with the same format of covparam0. Default is None.                           
  Output: 
        
  covparam    m             Optimal covariance estimation
  theta       scalar        Optimal direction of principle axis in anisotropic case.
                            Only availabe if anisotropy parameters are specified.
                            E.g., theta0. 
  ratio       scalar        Optimal ratio between principle and secondary axis in 
                            anistropic case
  opt_val     scalar        Optimal value of the objective function

  Remark: 
  1) the fitting uses the weighted least square criteria for covaraince 
  function fitting proposed by Cressie in 1985. The optimization algorithm is 
  bobyqa obtained from nlopt package at http://ab-initio.mit.edu/nlopt/
  2) details of theta and ratio can refer to stamps.stats.stcovfit.anisocovmodelest

  '''
  def objwls(param,ns,nt):
    ''' This function is used by nlopt to assess the objective function
    At this stage, the objective is based upon the covmodelwls function
    '''  
    if theta0 is None:
      covparam=par2covpar(param,ns,nt)
      return covmodelwls(covmodel,covparam,lag_cov_s,lag_cov_t,
                    lag_cov_v,lag_cov_n)  
    else:
      # I should add some check about the inputs for anisotropy 
      covparam=par2covpar(param[:-2],ns,nt)
      return covmodelwls(covmodel,covparam,lag_cov_s,lag_cov_t,
                    lag_cov_v,lag_cov_n,lag_th,param[-2],param[-1]) 
      
  
  def par2covpar(param,ns,nt):   
    '''This function transforms data formats from the form used for nlopt to 
       the form in the regular covariance model
       The nlopt input format is an 1-D np array, in which the covariance 
       parameters are listed in the form of
       [s1, r11, r12, t11, t12, s2, r2, t2, .....] 
       where 
       si is sill of nested model i, 
       rki is the kth spatial range parameters of nested spatial model i
       tki is the kth temporal range parameters of nested temporal model i
    ''' 
    covparam=[]
    k=0
    for m in range(len(covmodel)):
      if ns[m]>0 and nt[m]>0:
        covparam.append([param[k],param[k+1:k+ns[m]+1],
                       param[k+ns[m]+1:k+ns[m]+nt[m]+1]])
      elif ns[m]>0 and nt[m]==0:
        covparam.append([param[k],param[k+1:k+ns[m]+1]])
      elif ns[m]==0 and nt[m]>0:
        covparam.append([param[k],param[k+1:k+nt[m]+1]]) 
      else:
        covparam.append([param[k],param[k+1:k+ns[m]+1],
                       param[k+ns[m]+1:k+ns[m]+nt[m]+1]])   
      
      k=k+ns[m]+nt[m]+1        
    return covparam
    
  def covpar2par(covparam):    
    '''This function transforms data formats from regular covariance model to 
       the form used for nlopt. 
       
       Remark: The regular covariance model can refer to covmodeldef function
       '''                 
    fit_params=[]
    nm=len(covparam)
    npars=np.zeros(nm,dtype=np.int)
    npart=np.zeros(nm,dtype=np.int)
    for k in range(nm):
      for i,par in enumerate(covparam[k]):
        try:
          for j in range(len(par)):
            if not (par[j] is None):
              fit_params.append(par[j])
          if i==1:
            if par[0] is not None:
              npars[k]=len(par)
          else:
            if par[0] is not None:
              npart[k]=len(par)
        except:
          fit_params.append(par)  
    fit_params=np.array(fit_params)      
    return fit_params,npars,npart          
  
  pars,ns,nt=covpar2par(covparam0)
  if theta0 is not None:
    pars=np.append(pars,[theta0,ratio0])
    
  args=[ns,nt]
  nm=len(covparam0)
  idsill=[0]
  idnrg=[]
  if nm>1:
    k=0
    for i in range(nm):
      if ns[i]>1:        
        idnrg.append(np.arange(k+2,k+ns[i]+1))
      if nt[i]>1:
        idnrg.append(np.arange(k+ns[i]+2,k+ns[i]+nt[i]+1))
      if i < nm-1:  
        idsill.append(k+ns[i]+nt[i]+1)
        k=k+ns[i]+nt[i]+1
  
  idnrg=[item for sublist in idnrg for item in sublist]
  idnrg=np.array(idnrg)
      
  npars=len(pars)
  if theta0 is None:    
    idrg=np.setdiff1d(np.arange(npars),idsill)
    idrg=np.setdiff1d(idrg,idnrg)
  else:
    idrg=np.setdiff1d(np.arange(npars-2),idsill)
    idaniso=np.arange(npars-2,npars)

  if lower_bnd:
    low_bnd=covpar2par(lower_bnd)
  else:  
    low_bnd=np.empty(npars)
    low_bnd[idsill]=pars[idsill]*0.3    
    low_bnd[idrg]=pars[idrg]*0.3
    if idnrg.size>0:
      low_bnd[idnrg]=pars[idnrg]*0.8
      low_bnd[idnrg[np.where(low_bnd[idnrg]<0.5)]]=0.5
    if theta0 is not None:
      low_bnd[idaniso[0]]=pars[idaniso[0]]-np.pi/4
      low_bnd[idaniso[1]]=pars[idaniso[1]]*0.5
  if upper_bnd: 
    up_bnd=covpar2par(upper_bnd)
  else:
    up_bnd=np.empty(npars)
    up_bnd[idsill]=pars[idsill]*2.
    up_bnd[idrg]=pars[idrg]*5.
    if idnrg.size>0:
      up_bnd[idnrg]=pars[idnrg]*1.2
    if theta0 is not None:
      up_bnd[idaniso[0]]=pars[idaniso[0]]+np.pi/4
      up_bnd[idaniso[1]]=np.min([pars[idaniso[1]]*2,1.])
          
  result, opt_val=bobyqa.bobyqa( objwls, pars, args, low_bnd, up_bnd )
  
  if theta0 is None:
    covparam=par2covpar(result,ns,nt)
    return covparam, opt_val
  else:
    covparam=par2covpar(result[:-2],ns,nt)
    theta=result[-2]
    ratio=result[-1]
    return covparam, theta, ratio, opt_val
      
  
    
    
    #    
    ##===============================================================================
    # def autofitcov(CSTguess,COVparams,Smodels,Tmodels):
    #    CSTguess=CSTguess.reshape((-1,3)).tolist()
    #    fittingmodels=[]
    #    for cst,smodel,tmodel in zip(CSTguess,Smodels,Tmodels):
    #        cst.insert(1,smodel)
    #        cst.insert(3,tmodel)
    #        fittingmodels.append(cst)
    # 
    #    fittingmodels = tuple(fittingmodels)
    #    return fitcovariance(fittingmodels,COVparams)
    #===============================================================================

def covmodeldef(covmodel, covparam):
  ''' covariance model definition
  
  SYNTAX:
    fit_models=covmodeldef(covmodel,covparam)
  
  INPUTS:  
  covmodel    m             list of m nested covariance models in each of 
                            which the spatial and temporal components are put 
                            in a list as [covmodelS,covmodelT] 
  covparam    m             list of covariance parameters for m covmodels in 
                            each component the parameters are listed as 
                            [sill,[covparamS1,covparamS2,..],
                            [covparamT1,covparamT2,..]]
  OUTPUTS:                          
  fit_models  n_model by 5  list in which row for nest model number and 
                            col for cell, spatial model, spatial range,
                            temporal model, temporal range
  
  Remark: the use of fit_models is originally designed for widely-used 
  covariance models that has only two parameters for both spatial and 
  temporal models                           
  '''
  fit_models=[]
  for i,(modelS,modelT) in enumerate(covmodel):
    if modelS=='nugget' or modelS=='nug':
      rangeS=np.NaN
      rangeT=np.NaN
    else:
      rangeS=covparam[i][1][0]
      rangeT=covparam[i][2][0]
    fit_models.append([covparam[i][0],modelS,rangeS,
                                      modelT,rangeT])
  return fit_models

def covmodelest(lag_cov_s, lag_cov_t, covmodel, covparam):

  '''
  This function calculate the model predicted value at spatial-temporal coordinate that user input

  Syntax:
  COV_est =covmodelest(lag_cov_s,lag_cov_t,covmodel,covparam)
  
  Input:    
    
  lag_cov_s   nls by nlt    2D np array of the meshgrid of spatial lags. nls 
                            and nlt denote number of spatial and temporal lags
                            respectively
                                        
  lag_cov_t   nls by nlt    2D np array of the meshgrid of temporal lags
  covmodel    m             list of m nested covariance models in each of 
                            which the spatial and temporal components are put 
                            in a list as ['covmodelS/covmodelT']. The details of
                            available covariance model see below
  covparam    m             list of covariance parameters for m covmodels in 
                            each component the parameters are listed as 
                            [sill,[covparamS1,covparamS2,..],
                            [covparamT1,covparamT2,..]]                           
    
  Output:
  COV_est     nls by nlt    2D np array of the model predicted value of 2D meshgrid
  Cov_i       list          list includes m nls by nlt 2D np array for
                            all nested models      
  
  Remark: Five covariance models are available for use now, including
  gaussian(gau), exponential, spherical, holecos, and nugget
    
  
  
  '''

  isST, isSTsep, model_res = isspacetime(covmodel)
  if isST:
    if isSTsep:
      modelS, modelT = model_res
      Ki = []
      for model_s, model_t, param_i in zip(modelS, modelT, covparam):
        if len(param_i) == 1:
          if model_s=='nuggetC' or model_s=='nuggetC':
            sill=param_i
            param_s=[None]
            param_t=[None]
          else:
            print 'covparam is not consisent with covmodel'
            raise 
        else:
          sill, param_s, param_t = param_i
        model_s = get_model(model_s)
        model_t = get_model(model_t)                       
        Ki.append( sill * model_s( lag_cov_s, 1., param_s ) * model_t( lag_cov_t, 1., param_t ) )
      return sum(Ki), Ki # K, KK in matlab
    else:
      (modelS,) = model_res
      Ki=[]
      for model_s, param_i in zip(modelS, covparam):
        sill, param_s, s_t_ratio = param_i
        model_s = get_model(model_s)
        Ki.append(
          sill * model_s(lag_cov_s + s_t_ratio * lag_cov_t, 1., param_s))
      return sum(Ki), Ki # K, KK in matlab
  else:
    (modelS,) = model_res
    Ki = []
    for model_s, param_i in zip(modelS, covparam):
      if len(param_i) == 1:
        if model_s == 'nuggetC' :
          sill = param_i
          param_s = [None]
        else:
          print 'covparam is not consisent with covmodel'
          raise 
      sill, param_s = param_i
      model_s = get_model(model_s)
      Ki.append( sill * model_s(lag_cov_s, 1., param_s)) 
    return sum(Ki), Ki # K, KK in matlab

def anisocovmodelest(covmodel,covparam,theta,ratio,lag_th,lag_cov_s,lag_cov_t=None):
  '''
  Evaluate the spatially-aniostrpy covariance with given directional angle of 
  the prinipal axis with maximum range and its ratios to the other axes, and 
  isotropic model
  
  INPUT:
  covmodel    m             list of m nested isotropic covariance models in each of 
                            which the spatial and temporal components are put 
                            in a list as ['covmodelS/covmodelT']. The details of
                            available covariance model see below
  covparam    m             list of isotropic covariance parameters for m covmodels in 
                            each component the parameters are listed as 
                            [sill,[covparamS1,covparamS2,..],
                            [covparamT1,covparamT2,..]]      
  theta       1 by nd-1     vector of angle values that characterize the anisotropy. 
                            In a two dimensional space, angle is the trigonometric angle
                            between the horizontal axis and the principal axis of the
                            ellipse. In a three dimensional space, spherical coordinates
                            are used, such that angle(1) is the horizontal trigonometric
                            angle and angle(2) is the vertical trigonometric angle for the
                            principal axis of the ellipsoid. All the angles are measured
                            counterclockwise in degrees and are between -pi/2 and pi/2.                           
  ratio       1 by nd-1     1D array of vector that characterize the ratio for the length of the axes
                            for the ellipse (in 2D) or ellipsoid (in 3D). In a two dimensional
                            space, ratio is the length of the principal axis of the ellipse
                            divided by the length of the secondary axis, so that ratio<1. 
                            i.e., a_max=a, a_min=ratio*a. 
                            In a three dimensional space, ratio(1) is the length of the principal
                            axis of the ellipsoid divided by the length of the second axis, 
                            whereas ratio(2) is length of the principal axis of the ellipsoid
                            divided by the length of the third axis, so that ratio(1)<1 and
                            ratio(2)>1
  lag_th     1 by na        the spatial or S/T covariance model at angle theta 
                            to be evaluated
  lag_cov_s  nls by nlt     the spatial or S/T covariance model with spatial 
                            distances to be evaluated
  lag_cov_t  nls by nlt     the temporal lags of S/T covariance model to be evaluated

  OUTPUT:

  Csta       (ns by nt) by na  a list with na length contains S/T covariance model  
                               at every angle      
  Cstai      list          a list with na length. Each component has a list 
                           containing the details of each component of nested
                           model
  
  '''  
    
  isST, isSTsep, model_res = isspacetime(covmodel)
  Ka = [None]*lag_th.size
  Kai = [None]*lag_th.size
  for i, phi in enumerate(lag_th):
    if isST:
      if isSTsep:
        modelS, modelT = model_res
        Ki = []
        for model_s, model_t, param_i in zip(modelS, modelT, covparam):
          if len(param_i)==1:
            if model_s == 'nuggetC' or model_t == 'nuggetC':
              sill=param_i
              param_s = [None]
              param_t = [None]
            else:
              print 'covparam is not consisent with covmodel'
              raise 
          else:
            sill, param_s, param_t = param_i
            if param_s.size==0: 
              param_s=[None]
            if param_t.size==0:
              param_t=[None]
          model_s = get_model(model_s)
          model_t = get_model(model_t)
          if param_s[0] is not None:  #param_s[0] is the correlation length
            param_s = (
              param_s * ratio
              / np.sqrt(
                ratio**2 * np.cos(phi-theta)**2
                + np.sin(phi-theta)**2)
              )
          Ki.append(
            sill * model_s(lag_cov_s, 1., param_s)
            * model_t(lag_cov_t, 1., param_t)
            )
          Ka[i] = sum(Ki)
          Kai[i] = Ki  # K, KK in matlab
      else:
        (modelS,) = model_res
        Ki = []
        for model_s, param_i in zip(modelS, covparam):
          sill, param_s, s_t_ratio = param_i
          model_s = get_model(model_s)
          # Assume param_s[0] is the correlation length
          # This may not be true in non-seperable S/T covariance
          if param_s[0] is not None:  
            param_s = param_s*ratio/np.sqrt(ratio**2*np.cos(phi-theta)**2+np.sin(phi-theta)**2)          
          Ki.append( sill * model_s( lag_cov_s + s_t_ratio * lag_cov_t, 1., param_s ) )
        Ka[i] = sum(Ki)
        Kai[i] = Ki # K, KK in matlab
    else:
      (modelS,) = model_res
      Ki = []
      for model_s, param_i in zip(modelS, covparam):
        if len(param_i) == 1:
          if model_s == 'nuggetC' :
            sill = param_i
            param_s = [None]
          else:
            print 'covparam is not consisent with covmodel'
            raise 
        sill, param_s = param_i
        model_s = get_model(model_s)
        if param_s[0] is not None:
          param_s = (
            param_s * ratio
            / np.sqrt(
              ratio**2 * np.cos(phi-theta)**2
              + np.sin(phi-theta)**2)
            )
        Ki.append(sill * model_s(lag_cov_s, 1., param_s))
      Ka[i]=sum(Ki)
      Kai[i]=Ki # K, KK in matlab  
  return Ka, Kai

def covdownscale(bigcovmod, scale, method=None):
    '''
        covmodel:
            [[c,model_s, bs, model_t, bt],
              ...,
             [c,model_s, bs, model_t, bt]]

        return estimated downscale covariance model
    '''

    model_count = len(bigcovmod)
    #parse covmod to get lower and upper bound
    low_bnd = []
    up_bnd = []
    for cov in bigcovmod:
        low_bnd += [0.5*cov[0], 0.5*cov[2], 0.5*cov[4]]
        up_bnd += [1.5*cov[0], 1.5*cov[2], 1.5*cov[4]]
    low_bnd = np.array(low_bnd)
    up_bnd = np.array(up_bnd)
    big_bt = int(up_bnd[2::3].max()) # bt max, e.g. opt function range

    low_bnd[1::3] = up_bnd[1::3] #spatial not change
    up_bnd[2::3] *= scale #temparal change scale
    init_gss = low_bnd + (up_bnd - low_bnd)*0.5

    smallcovmod = deepcopy(bigcovmod)
    big_cov_mod_f = get_cov_mod_func(bigcovmod)
    
    def opt_func(x, big_cov_mod_f, smallcovmod, big_bt):
        for i, cst in enumerate(np.split(x, len(x)/3)):
            smallcovmod[i][0] = cst[0]
            smallcovmod[i][2] = cst[1]
            smallcovmod[i][4] = cst[2]
        small_cov_mod_f = get_cov_mod_func(smallcovmod)

        range_big_bt = range(big_bt)
        big_lags = np.zeros((len(range_big_bt),2))
        big_lags[:,1] = range_big_bt
        big_cov_z = big_cov_mod_f(big_lags)

        range_small_bt = range(scale)
        small_lags = np.zeros((len(range_small_bt), 2))
        small_lags[:,1] = range_small_bt
        multiplier = np.arange(
            scale,0,-1, dtype=np.float64
            ).reshape((scale,1))
        multiplier[1:,:] *=2
        opt_val = 0.0

        for i in range_big_bt:
            small_lags_copy = deepcopy(small_lags)
            small_lags_copy[:,1] += i*scale
            small_cov_z = small_cov_mod_f(small_lags_copy)
            opt_val +=\
                abs(big_cov_z[i][0]\
                    - (small_cov_z * multiplier / scale**2).sum()
                    )
        print x, opt_val
        return opt_val
            
    args = (big_cov_mod_f, smallcovmod, big_bt)

    result, opt_val = bobyqa.bobyqa(
        opt_func, init_gss, args, low_bnd, up_bnd, stop_val=10**-8)

    #scale small covariance
    for cov in smallcovmod:
        cov[4] /= scale


    return smallcovmod, opt_val
  
def get_cov_mod_func(covmodel):
    '''
        get covariance model as a python function with input lag
    '''
    def cov_func(lag):
        '''
            lag: n by 2 2D np array, column0 is for spatial, 1 for time
        '''
        s_lag = lag[:, 0]
        t_lag = lag[:, 1]
        Ki = []
        for cov in covmodel:
            c, fs, bs, ft, bt = cov
            fs = get_model(fs)
            ft = get_model(ft)
            Ki.append(c*fs(s_lag, 1., bs)*ft(t_lag, 1., bt))
        return sum(Ki).reshape((-1, 1))
    return cov_func

def cal_cov_mod(lag, cov_mod):
    ''' calculate covariance value at each lag
    lag: n by 2 np 2d array, [[s_lag, t_lag],...,[s_lag, t_lag]]
    cov_mod: [[c,model_s, bs, model_t, bt], ..., [c,model_s, bs, model_t, bt]]
    '''
    cov_f = get_cov_mod_func(cov_mod)
    cov_z = cov_f(lag)
    return cov_z

def _index_to_fit_func(index):
    dictionary={"gaussian":_Gau,
                "exponential":_Exp,
                "spherical":_Sph,
                "holecos":_HoC,
                "nugget":_Nug,
                "gau":_Gau,
                "exp":_Exp,
                "sph":_Sph,
                "hoc":_HoC,
                "nug":_Nug,}
    return dictionary[index]
  

def _Gau(bandwidth,lag):
    value = np.exp(-3*(lag/bandwidth)**2)
    return value

def _Exp(bandwidth,lag):
    value = np.exp(-3*(lag/bandwidth))
    return value

def _Sph(bandwidth,lag):
    value=lag.copy()
    boollag = lag <= bandwidth
    value[boollag] = 1.0-1.5*(lag[boollag]/bandwidth)\
                    +0.5*(lag[boollag]/bandwidth)**3;
    boollag = lag > bandwidth 
    value[boollag] = 0.0
    return value;

def _HoC(bandwidth,lag):
    value = np.cos(3.1415926*lag/bandwidth)
    return value

def _Nug(bandwidth,lag):
    value=lag.copy()
    boollag = lag == 0
    value[boollag] = 1.
    boollag = lag != 0
    value[boollag] = 0.
    return value;

if __name__ == "__main__":
    covmodel=[['nuggetC/nuggetC'],['exponentialC/exponentialC'],['exponentialC/exponentialC']]
    # covmodel=[['nugget','nugget'],['exponential','exponential'],['exponential','exponential']]
    covparam=[[0.2,[None],[None]],[0.7,[100],[10]],[0.3,[50],[5]]]
    # fit_models = [[0.7,'exponential',100,'exponential',10],
    #               [0.3,'exponential',50,'exponential',5]]
    lag_s = np.linspace(0,100,11)
    lag_t = np.linspace(0,10,6)
    lag_th = np.linspace(-np.pi/2,np.pi/2,10)
    lag_cov_t, lag_cov_s = np.meshgrid( lag_t, lag_s )
    lag_cov_est,cov_est_i = covmodelest( lag_cov_s,lag_cov_t,covmodel, covparam)
    Ka,Kai=anisocovmodelest(covmodel,covparam,np.pi/6,0.8,lag_th,lag_cov_s,lag_cov_t)
    lag_cov_n = np.ones( lag_cov_est.shape )
    lag_cov_v = lag_cov_est + 0.2*(np.random.rand( *lag_cov_est.shape ) - 0.5)
    objv = covmodelwls(covmodel,covparam,lag_cov_s,lag_cov_t,
                  lag_cov_v,lag_cov_n)
    result, opt_val=covmodelfit(lag_cov_s,lag_cov_t,lag_cov_v,lag_cov_n,covmodel,covparam)
    print objv

    try:
        from matplotlib import pyplot as plt
        lag_s_line = np.linspace(0,100,51)
        lag_t_line = np.linspace(0,10,51)
        lag_cov_t_line, lag_cov_s_line = np.meshgrid( lag_t_line, lag_s_line )
        lag_cov_v_line, dummy = covmodelest( lag_cov_s_line,lag_cov_t_line,
                                     covmodel,covparam )

        plt.figure(1)
        plt.subplot(211)
        plt.plot(lag_s, lag_cov_v[:,0], 'bo', lag_s_line, lag_cov_v_line[:,0], 'b--')
        plt.subplot(212)
        plt.plot(lag_t, lag_cov_v[0], 'ro', lag_t_line, lag_cov_v_line[0], 'r--')

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(2)
        ax3d = Axes3D(fig)
        ax3d.plot_wireframe(lag_cov_s_line, lag_cov_t_line, lag_cov_v_line)
        ax3d.scatter(lag_cov_s, lag_cov_t,lag_cov_v,c='r')
        plt.show()
    except ImportError,e:
        print 'Warning: Import matplotlib fault, cannot draw.'
        print 'Error Message:',e.message
