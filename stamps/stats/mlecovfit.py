# -*- coding: utf-8 -*-
import numpy

from six.moves import range

from ..general.isspacetime import isspacetime
from ..general import bobyqa as bobyqa

from ..general.coord2K import coord2K, coord2dist
from ..models.covmodel import get_model



def _par2op(covparam):
    pars = []
    for i in range(len(covparam)):
        for j in range(len(covparam[0])):
            if j is not 0:
                pars.append(covparam[i][j][0])
            else:
                pars.append(covparam[i][j])
    indices = [i for i, e in enumerate(pars) if e != None]
    return numpy.array(pars), indices

def _op2covpar(pars_slice, indices,covmodel):
    pars = [None] * (indices[-1]+1)
    for i, e in enumerate(indices):
        pars[e] = pars_slice[i]
    covparam = []
    nm = range(len(covmodel))
    k = 0
    for m in nm:
        covparam.append([pars[k],pars[k+1:k+2],pars[k+2:k+3]])
        k = len(covmodel) * (m+1)
    return covparam

def _covpar2par(covparam):    
    '''This function transforms data formats from regular covariance model to 
    the form used for nlopt. 
       
    Remark: The regular covariance model can refer to covmodeldef function
    '''                 
    fit_params=[]
    nm=len(covparam)
    npars=numpy.zeros(nm,dtype=numpy.int)
    npart=numpy.zeros(nm,dtype=numpy.int)
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
    fit_params=numpy.array(fit_params)      
    return fit_params,npars,npart    


def _par2covpar(param,ns,nt,covmodel):   
    '''This function transforms data formats from the form used for nlopt to 
    the form in the regular covariance model
       The nlopt input format is an 1-D numpy array, in which the covariance 
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
        if any(ns)>0 and any(nt)>0:
            covparam.append([param[k],param[k+1:k+ns[m]+1],
                     param[k+ns[m]+1:k+ns[m]+nt[m]+1]])
            k=k+ns[m]+nt[m]+1 
        elif all(nt)==0:
            covparam.append([param[k],param[k+1:k+ns[m]+1]])
            k=k+ns[m]+1
        elif all(ns)==0:
            covparam.append([param[k],param[k+1:k+nt[m]+1]])
            k=k+nt[m]+1
             
    return covparam

def _par2covSTpar(param,ns,nt):
    covparamS=[]
    covparamT=[]
    nm=len(ns)
    k=0
    for m in range(nm):
        if any(ns)>0 and any(nt)>0:
            covparamS.append([param[k],param[k+1:k+ns[m]+1]])
            covparamT.append([param[k],param[k+ns[m]+1:k+ns[m]+nt[m]+1]])
            k=k+ns[m]+nt[m]+1 
        elif all(nt)==0:
            covparamS.append([param[k],param[k+1:k+ns[m]+1]])
            k=k+ns[m]+1
        elif all(ns)==0:
            covparamT.append([param[k],param[k+1:k+nt[m]+1]])
            k=k+nt[m]+1
             
    return covparamS,covparamT           
  
def _covjac(pars, ns, nt, covmodel, ch):
    '''
    Obtain the Jacobian of the covariance with each of parameters
    Namely, the first derivatives of the covariance with respect to parameters
    '''
    
    covparam = _par2covpar(pars,ns,nt,covmodel)
    isST, isSTsep, model_res = isspacetime(covmodel)
    if isST:
        if isSTsep:
            modelS, modelT = model_res
            dist_s = coord2dist(ch[:, 0:2], ch[:, 0:2])
            dist_t = coord2dist(ch[:, 2:3], ch[:, 2:3])
            jac = []
        
            for model_s, model_t, param_i in zip(modelS, modelT, covparam):
                sill, param_s, param_t = param_i
                model_s = get_model(model_s)
                model_t = get_model(model_t)      
                jac.append(model_s(dist_s, 1., param_s) * model_t(dist_t, 1., param_t))
                jacar=model_s(dist_s, sill, param_s, jac=True, jacpar='ar')
                if jacar.size>0:
                    jac.append(jacar)  
                jacar=model_t(dist_s, sill, param_t, jac=True, jacpar='ar')     
                if jacar.size>0:
                    jac.append(jacar)  
        else:
            (modelS,) = model_res
            #To be implemented
            return
    else:        
        jac=[]
        dist_s = coord2dist(ch, ch)
        (modelS,) = model_res
        for model_s, param_i in zip(modelS, covparam):
            sill, param_s = param_i
            model_s = get_model(model_s)
            jac.append(model_s(dist_s, 1., param_s))  
            jacar=model_s(dist_s, sill, param_s, jac=True, jacpar='ar')
            if jacar.size>0:
                jac.append(jacar)  
        
    return jac     
  
def _covjac2(pars, ns, nt, covmodel, ch):
    '''
    Obtain the second derivatives of the covariance with every of parameters
    '''    
    covparam = _par2covpar(pars, ns, nt, covmodel)
    isST, isSTsep, model_res = isspacetime(covmodel)
    if isST:
        if isSTsep:
            modelS, modelT = model_res
            dist_s = coord2dist(ch[:, 0:2], ch[:, 0:2])
            dist_t = coord2dist(ch[:, 2:3], ch[:, 2:3])
            jac2 = [[None]*pars.size for i in range(pars.size)]
            k = 0            
            for model_s, model_t, param_i in zip(modelS, modelT, covparam):
                sill, param_s, param_t = param_i
                model_s = get_model(model_s)
                model_t = get_model(model_t)  
                parnum = len(param_i)
                jac2[k][k] = numpy.zeros(dist_s.shape) # dsds
                # dsdar
                jac2[k][k+1] = jac2[k+1, k]=\
                  model_s(dist_s, 1., param_s, jac=True, jacpar='ar')*model_t(dist_t, 1., param_t) 
                # dsdat
                jac2[k][k+2] = jac2[k+2,k]=\
                  model_s(dist_s, 1., param_s)*model_t(dist_s, 1., param_t, jac=True, jacpar='ar')
                jac2[k+1][k+1] = model_s(dist_s, sill, param_s, jac=True, jacpar='ar2')*model_t(dist_t, 1., param_t)
                jac2[k+2][k+2] = model_s(dist_s, 1., param_s)*model_t(dist_s, sill, param_t, jac=True, jacpar='ar2')
                k=k+parnum
        else:
            (modelS,) = model_res
            #To be implemented
            return
    else:
        (modelS,) = model_res      
        jac2 = [[None]*pars.size for i in range(pars.size)]
        dist_s = coord2dist(ch, ch)
        k = 0
        for model_s, param_i in zip(modelS, covparam):
            parnum = 1+len(param_i[1])
            sill, param_s = param_i
            model_s = get_model(model_s)
            jac2[k][k] = numpy.zeros(dist_s.shape) # dsds
            for j in range(1, 1+len(param_i[1])):
                # dsdar
                jac2[k][k+j]=jac2[k+j][k]=\
                  model_s(dist_s, 1., param_s, jac=True, jacpar='ar')   
                jac2[k+j][k+j]= model_s(dist_s, sill, param_s, jac=True, jacpar='ar2')
            for m in range(k,k+1+len(param_i[1])):  
                for i in range(k+parnum,pars.size):
                    jac2[m][i]=jac2[i][m]=numpy.zeros(dist_s.shape)
            k=k+parnum
        
    return jac2       
  
def mlecovfitv(ch,zh,covmodel,covparam):
    '''
    Maximum likelihood method for vector format data
    
    opt_param=mlecovfit_v(ch,zh,covmodel,covparam0)
    
    Input
    
    ch          n by d      array consisting of observation locations
    zh          n by 1      array containing obseved values
    covmodel    m           list of m nested covariance models in each of 
                            which the spatial and temporal components are put 
                            in a list as [covmodelS,covmodelT] 
    covparam    m           list of covariance parameters for m covmodels in 
                            each component the parameters are listed as 
                            [sill,[covparamS1,covparamS2,..],
                            [covparamT1,covparamT2,..]] 
    ''' 
    def llikjac(pars,ns,nt):

        covparam = _par2covpar(pars,ns,nt,covmodel)
        jac=_covjac(pars,ns,nt,covmodel,ch)   
        V,_=coord2K(ch,ch,covmodel,covparam)  
        Viy=numpy.linalg.solve(V,zh)
        invV=numpy.linalg.inv(V)
        jacv=numpy.zeros(pars.size)
        for k in range(len(jac)):     
            jacv[k]=0.5*numpy.trace(invV.dot(jac[k])) \
                    -0.5*Viy.T.dot(jac[k]).dot(Viy)#/(zh.T.dot(Viy))

        return jacv  
    
    def hess(pars,ns,nt):
        '''
        Hessain estimation with reference in 
        Kitanidis P., and R. Lane. 1985. Maximum likelihood parameter estimation of 
        hydrological spatial processes by the Gauss-Newton method. Journal of 
        hydrology 79. 
        
        Hessian is calculated by its simplication from (11) to (12) in the paper
        Note that this implementation assumes the mean is constant of zero
        
        '''    
    
        covparam = _par2covpar(pars,ns,nt,covmodel)
        jac=_covjac(pars,ns,nt,covmodel,ch)
        #jac2=_covjac2(pars,ns,nt,covmodel,ch)
        V,_=coord2K(ch,ch,covmodel,covparam) 
        V[numpy.where(V/V[0,0]<10e-6)]=0
        #Viy=numpy.linalg.solve(V,zh)
        invV=numpy.linalg.inv(V)
        hes=numpy.zeros((pars.size,pars.size))
        for j in range(pars.size):
            for k in range(j,pars.size):
                comp1=-0.5*numpy.trace(invV.dot(jac[j]).dot(invV).dot(jac[k]))  
                #comp2=0.5*numpy.trace(invV.dot(jac2[j][k]))
                #comp4=Viy.T.dot(jac[j]).dot(invV).dot(jac[k]).dot(Viy)
                #comp5=-0.5*Viy.T.dot(jac2[j][k]).dot(Viy)
                hes[j,k]=-comp1#comp1+comp2+comp4+comp5
                hes[k,j]=hes[j,k]
        
        return hes    
        
    #  def Fishermat(pars,ns,nt):        
    def loglik(pars,ns,nt):
        if any(pars<=0):
            llik=100000000000
            return llik
        
        covparam = _par2covpar(pars,ns,nt,covmodel)
        V, V_list = coord2K(ch, ch, covmodel, covparam)
        # covariance tapering
        V[numpy.where(V/V[0,0]<10e-6)]=0
        Lv=numpy.linalg.cholesky(V)
        logdetV=2.*numpy.sum(numpy.log(numpy.diag(Lv)))
        #detV = numpy.linalg.det(V)
        n=len(zh)
        Viy=numpy.linalg.solve(V,zh)
        ytViy=zh.T.dot(Viy)
        llik=n/2*numpy.log(2*numpy.pi)+0.5*logdetV+0.5*ytViy    
        return llik    
    
    pars,ns,nt=_covpar2par(covparam)
    zh=zh.reshape(zh.size,)
    #  V=numpy.zeros((ch.size,ch.size))
    #  Viy=numpy.zeros((ch.size,1))
    #  global V, Viy
    #pars, indices = _par2op(covparam)
    #pars_slice = numpy.take(pars,indices)
    if type(ch[0,-1])==numpy.datetime64:
        ch[:,-1]=numpy.double(numpy.asarray(ch[:,-1],dtype='datetime64'))
        ch=ch.astype(numpy.double)
  
    #  _llikjac = lambda pars,ns,nt: llikjac(pars,ns,nt)
    #  _hess = lambda pars,ns,nt: hess(pars,ns,nt)
    #result = op.minimize(loglik, pars_slice, args=(indices),
    #                       options={'maxiter': 10000},method='BFGS')
  
 
    low_bnd=numpy.ones(pars.size)*numpy.finfo(float).eps
    up_bnd=numpy.array([k for k in pars*3])
    args=[ns,nt]
    result, opt_val=bobyqa.bobyqa(loglik, pars, args, low_bnd, up_bnd )

    #  bnds=[(numpy.finfo(float).eps,k) for k in pars*3]   
    #  result=op.differential_evolution(loglik,bnds, args=(ns,nt), maxiter=5)  
    ##  result = op.minimize(loglik, result['x'], args=(ns,nt),jac=_llikjac,hess=_hess,
    ##                       options={'gtol': 1e-6, 'disp': True},method='BFGS')
    #  if not result['success']:                     
    #  result = op.minimize(loglik, pars, args=(ns,nt),jac=_llikjac,hess=_hess,
    #                       options={'xtol': 1e-4, 'disp': True},method='Newton-CG')                     
    #  if not result['success']:  
    #    bnds=[(0,None)]*pars.size
    #    result = op.minimize(loglik, result['x'], args=(ns,nt),jac=_llikjac,bounds=bnds,
    #                       options={'gtol': 1e-4, 'disp': True},method='TNC') 

    param = _par2covpar(result,ns,nt,covmodel) 
#  param = _par2covpar(result['x'],ns,nt,covmodel)   

                     
    return param#,result['success']    

def mlecovfitg(grid_s,grid_t,grid_v,covmodel,covparam0):
    '''
    Maximum likelihood method for grid format data
    
    opt_param=mlecovfit_v(ch,zh,covmodel,covparam)
    
    Input
    
    ch          n by d      array consisting of observation locations
    zh          n by 1      array containing obseved values
    covmodel    m           list of m nested covariance models in each of 
                            which the spatial and temporal components are put 
                            in a list as [covmodelS,covmodelT] 
    covparam    m           list of covariance parameters for m covmodels in 
                            each component the parameters are listed as 
                            [sill,[covparamS1,covparamS2,..],
                            [covparamT1,covparamT2,..]] 
    maxLagS     scalar      maximum spatial distance to be evaluated
    maxLagT     scalar      maximum temporal distance to be evaluated                         
    ''' 

    def loglik(pars,ch,ch_t,ns,nt,covmodelS,covmodelT):
        
        if any(pars<=0):
            llik=100000000000
            return llik
        
        covparamS,covparamT=_par2covSTpar(pars,ns,nt)

        # spatial part
        #covparamS = _par2covpar(par_S,ns,ns2,covmodelS)
        #Vs, V_list_s = coord2K(grid_s, grid_s, covmodelS, covparamS)
        Vs, V_list_s = coord2K(ch, ch, covmodelS, covparamS)
        # covariance tapering
        Vs[numpy.where(Vs/Vs[0,0]<10e-6)]=0
        Lv_s=numpy.linalg.cholesky(Vs)
        logdetV_s=2.*numpy.sum(numpy.log(numpy.diag(Lv_s)))
        #detV = numpy.linalg.det(V)
        n,m=grid_v.shape
        vs_m=grid_v.mean(1).reshape(n,1)
        vs_r=grid_v-vs_m.dot(numpy.ones((1,m)))
        S_s=vs_r.dot(vs_r.T)*1./(m-1)
        ddt_s=vs_m.dot(vs_m.T)
        term2_s=numpy.linalg.solve(Vs,S_s+ddt_s)
        llik_s=m*(logdetV_s+numpy.trace(term2_s))   

        # temporal part        
        #covparamT = _par2covpar(par_T,nt,nt2,covmodelT)
        #Vt, V_list_t = coord2K(grid_t, grid_t, covmodelT, covparamT)  
        Vt, V_list_t = coord2K(ch_t, ch_t, covmodelT, covparamT)
        Vt[numpy.where(Vt/Vt[0,0]<10e-6)]=0 # covariance tapering
        Lv_t=numpy.linalg.cholesky(Vt)
        logdetV_t=2.*numpy.sum(numpy.log(numpy.diag(Lv_t)))
        n,m=grid_v.shape
        vt_m=grid_v.mean(0).reshape(1,m)
        vt_r=grid_v-numpy.ones((n,1)).dot(vt_m)
        S_t=vt_r.T.dot(vt_r)*1./(n-1)
        ddt_t=vt_m.T.dot(vt_m)
        term2_t=numpy.linalg.solve(Vt,S_t+ddt_t)
        llik_t=m*(logdetV_t+numpy.trace(term2_t))        
        
        llik=llik_s+llik_t
        
        return llik    
         
    isST, isSTsep, modelS, modelT = isspacetime(covmodel)
    covmodelS=[]
    covmodelT=[]
    covparamS0=[]
    covparamT0=[]
    if isST:
        if isSTsep:       
            for model_s, model_t, param_i in zip( modelS,modelT,covparam0):
                if len(param_i)==1:
                    if model_s=='nuggetC' or model_s=='nuggetC':
                        sill=param_i
                        param_s=[None]
                        param_t=[None]
                    else:
                        print 'covparam is not consisent with covmodel'
                        raise 
                else:
                    sill, param_s, param_t = param_i                
                    covmodelS.append(model_s)
                    covmodelT.append(model_t)
                    covparamS0.append([sill,param_s])
                    covparamT0.append([sill,param_t])

        else:
            print 'S/T separability is currently assumed in MLE'
    else:
        print 'Pure spatial or temporal cases should use mlecovfitv'       
    
#    par_S,ns,ns2=_covpar2par(covparamS0)
#    par_T,nt,nt2=_covpar2par(covparamT0) 
#    numS=len(par_S)
#    numT=len(par_T)
#    pars = numpy.hstack([par_S,par_T])
    pars,ns,nt=_covpar2par(covparam0)
    
    ch=grid_s
    if type(grid_t[0])==numpy.datetime64:
        ch_t=numpy.double(numpy.asarray(grid_t,dtype='datetime64'))
    else:
        ch_t=grid_t
    ch_t=ch_t.reshape(len(ch_t),1)    

    low_bnd=numpy.ones(pars.size)*numpy.finfo(float).eps
    up_bnd=numpy.array([k for k in pars*3])
    args=[ch,ch_t,ns,nt,covmodelS,covmodelT]
    result, opt_val=bobyqa.bobyqa(loglik, pars, args, low_bnd, up_bnd )

    param = _par2covpar(result,ns,nt,covmodel) 

    return param
    
#    # Assure the following parameters have proper dimension or format, e.g., 
#    # 1D numpy array 
#    if grid_t is None:
#        grid_t=numpy.array([0]).reshape(1,1)
#  
#    if len(grid_s.shape)<2:
#        grid_s=numpy.reshape(grid_s,(grid_s.size,1))
#    
#    if maxLagS is None:
#        maxLagS=numpy.max(pdist(grid_s))
#    if maxLagT is None:
#        maxLagT=numpy.max(pdist(grid_t.reshape(grid_t.size,1)))
#        
#        
#    grid_t=numpy.asarray(grid_t)
#    grid_t=numpy.reshape(grid_t,(grid_t.size,1))
#    grid_v=grid_v.reshape((grid_s.shape[0],grid_t.size))
#
#    if grid_s.shape[0]<8000:
#        s_diff_i_left,s_diff_i_right,s_diff_v, _=diffarray(grid_s)
#    else:
#        s_diff_i_left,s_diff_i_right,s_diff_v, _= \
#          diffarray(grid_s,maxLagS)
#    nd=grid_s.shape[1]
#  
#    if nd==1:
#        s_diff_v=numpy.abs(s_diff_v).ravel()
#    elif nd==2:           
#        s_diff_v=numpy.sqrt(s_diff_v[:,0]**2+s_diff_v[:,1]**2)
#    elif nd==3:
#        s_diff_v=numpy.sqrt(s_diff_v[:,0]**2+s_diff_v[:,1]**2+s_diff_v[:,2]**2)
#  
#    if len(grid_t)<8000:
#        t_diff_i_left,t_diff_i_right,t_diff_v,_=diffarray(grid_t)
#    else:
#        t_diff_i_left,t_diff_i_right,t_diff_v,_= \
#            diffarray(grid_t,maxLagT)
#    t_diff_v=numpy.abs(t_diff_v).ravel()    
#    
#    idxs=numpy.where(s_diff_v<=maxLagS)
#    s_diff_i_left=s_diff_i_left[idxs].astype(numpy.int)
#    s_diff_i_right=s_diff_i_right[idxs].astype(numpy.int)
#    s_diff_v=s_diff_v[idxs]
#  
#    idxt=numpy.where(t_diff_v<=maxLagT)
#    t_diff_i_left=t_diff_i_left[idxt].astype(numpy.int)
#    t_diff_i_right=t_diff_i_right[idxt].astype(numpy.int)
#    t_diff_v=t_diff_v[idxt]      
#
#    lagS=numpy.unique(s_diff_v)
#    lagT=numpy.unique(t_diff_v)
    
    
    
    #for s in lagS:
        
    
def mlecovfit_sub(ch,zh,covmodel,covparam0):
    '''
    
    ch      n by d
    zh      n by m
    
    '''
    
    def loglik(pars,ns,nt):
        
        if any(pars <= 0):
            llik = 100000000000
            return llik
        
        covparam = _par2covpar(pars, ns, nt, covmodel)
        V, V_list = coord2K(ch, ch, covmodel, covparam)
        # covariance tapering
        V[numpy.where(V/V[0,0]<10e-6)]=0
        Lv=numpy.linalg.cholesky(V)
        logdetV=2.*numpy.sum(numpy.log(numpy.diag(Lv)))
        #detV = numpy.linalg.det(V)
        n,m=zh.shape
        zh_m=zh.mean(1).reshape(n,1)
        zh_r=zh-zh_m.dot(numpy.ones((1,m)))
        S=zh_r.dot(zh_r.T)*1./(m-1)
        ddt=zh_m.dot(zh_m.T)
        term2=numpy.linalg.solve(V,S+ddt)
        llik=m*(logdetV+numpy.trace(term2))
        return llik
        
    pars, ns, nt =_covpar2par(covparam0)
    if len(ch.shape) < 2:
        ch = ch.reshape(ch.size, 1)
    if type(ch[0,-1]) ==n umpy.datetime64:
        ch[:,-1] = numpy.double(numpy.asarray(ch[:,-1],d type='datetime64'))
        ch = ch.astype(numpy.double)

    low_bnd = numpy.ones(pars.size)*numpy.finfo(float).eps
    up_bnd = numpy.array([k for k in pars*3])
    args = [ns, nt]
    result, opt_val = bobyqa.bobyqa(loglik, pars, args, low_bnd, up_bnd )

    param = _par2covpar(result, ns, nt, covmodel) 

    return param#,result['success']     
