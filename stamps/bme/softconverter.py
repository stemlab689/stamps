# -*- coding: utf-8 -*-
import csv

import numpy as np
from six.moves import range

from .pystks_variable import get_standard_soft_pdf_type


def ud2ud(file_name, usecols, skiprows, delimiter):
    '''使用者自定型
    mean, var: n by 1 numpy array
    
    '''
    limi_idx = usecols[0] # idx of limi
    f=open(file_name,"rb")
    c=csv.reader(f,delimiter = delimiter )

    #skip rows
    for skiprow in range(skiprows):
        c.next()
    try:
        nol = []
            
        for line in c :
            d=line
            
            a=int(d[limi_idx])+1
            nol.append(a)
                
    except StopIteration:
        pass
    
    nolmax=max(nol)

    f2=open(file_name,"rb")
    readcsv=csv.reader( f2,delimiter= delimiter )
    #skip rows
    for skiprow in range(skiprows):
        readcsv.next()
    try:
        limi = []
        probdens = []
        for i in readcsv:
            d2=i
            a2=int(d2[limi_idx])
            px1=tuple(d2[limi_idx +1 : limi_idx +1 + a2+1])
            px1=px1+tuple([0]*(nolmax-a2 ))
            px=map(float,px1)
            limi.append(px)

            py1=tuple(d2[limi_idx +1 + a2+1 : limi_idx +1 + a2+1+ a2+1])
            py1=py1+tuple([0]*(nolmax-a2 ))
            py=map(float,py1)
            probdens.append(py)
                

    except StopIteration:
        pass
        
        
        
    nl=np.array(nol,ndmin=2).T
    limi=np.array(limi,ndmin=2)
    probdens=np.array(probdens,ndmin=2)
    return nl,limi,probdens

def gs2ud(mean, var):
    '''轉換高斯型資料至使用者自定型
    mean, var: n by 1 numpy array
    
    '''
    softpdftype = 2
    limi_norm = np.array([-3.719,-3.090,-2.326,
                          -1.645,-0.524,-0.253,
                           0.000,
                           0.253, 0.524, 1.645,
                           2.326, 3.090, 3.719,])
    limi_n = len(limi_norm)

    nl = np.ones((mean.shape[0],1), dtype=int) * limi_n
    limi = np.kron(np.sqrt(var),limi_norm).reshape((-1,limi_n))+mean
    probdens = limi-mean
    probdens = -(probdens**2)/2/var
    probdens = 1/np.sqrt(2*np.pi*var)*np.exp(probdens)

    nl, limi, probdens, _ = proba2probdens(softpdftype, nl, limi, probdens)

    return nl, limi, probdens

def uf2ud(low, up):

    softpdftype = 2
    limi_norm = np.linspace(-1,1,5)
    limi_n = len(limi_norm)
    nl = np.ones((low.shape[0],1), dtype=int) * limi_n
    limi = np.kron((up-low)/2.,limi_norm)+(up+low)/2.
    probdens = np.ones((low.shape[0],limi_n))*(1./(up-low))

    nl, limi, probdens, _ = proba2probdens(softpdftype, nl, limi, probdens)

    return nl, limi, probdens

def ud2zs(softpdftype, nl, limi, probdens):
    '''
    user defined to new zs data
    new zs data: a sequence of zsdata,
        zsdata is a sequence of pdftype, *pdf_args
        e.g. zsdata1 = (2, nl, limi, probdens)
        e.g. zsdata2 = (10, mean, var)
        e.g. new zs data = (zsdata1, zsdata2)
    '''
    zsdata = []
    if nl.size != 0:
        for n_i, l_i, p_i in zip(nl, limi, probdens):
            zsdata.append([softpdftype, n_i, l_i, p_i])

    return zsdata

def uf2zs(softpdftype, low, up):
    '''
    new zs data
    '''
    limi_norm = np.linspace(-1,1,5)
    limi_n = len(limi_norm)
    nl = np.ones((low.shape[0],1), dtype=int) * limi_n
    limi = np.kron((up-low)/2.,limi_norm)+(up+low)/2.
    probdens = np.ones((low.shape[0],limi_n))*(1./(up-low))
    zsdata = []
    for n_i, l_i, p_i in zip(nl, limi, probdens):
        zsdata.append([softpdftype, n_i, l_i, p_i])

    return zsdata

def zs2ud(zs):
    '''new zs data'''
    softpdftype = 2
    mean = np.array([zsi[1] for zsi in zs]).reshape((-1,1))
    var = np.array([zsi[2] for zsi in zs]).reshape((-1,1))
    nl, limi, probdens = gs2ud(mean, var)

    return softpdftype, nl, limi, probdens

def proba2probdens(softpdftype, nl, limi, probdens):
    '''
    proba2probdens            - Normalizes the probability density function (Jan 1, 2001)
  
    Calculates the norm (area under the curve) of a function template, and returns
    a normalized pdf. This function uses the syntax of probabilistic data (see 
    probasyntax) to define the pdf.
   
    SYNTAX :
   
    [probdens,norm]=proba2probdens(softpdftype,nl,limi,probdenstemplate);
   
    INPUT :
   
    softpdftype scalar      indicates the type of soft pdf representing the  
                            probabilitic soft data.  
                            softpdftype may take value 1, 2, 3 or 4, as follow:
                            1 for Histogram, 2 for Linear, 3 for Grid histogram, 
                            and 4 for Grid Linear.
                            In current status, only softpdftype 2 is avaiable for use 
    nl          ns by 1     2D array of the number of interval limits. nl(i) is the number  
                            of interval limits used to define the soft pdf for soft data 
                            point i. (see probasyntax for more explanations)
    limi        ns by l     2D array of interval limits, where l is equal to
                            either max(nl) or 3 depending of the softpdftype.
                            limi(i,:) are the limits of intervals for the i-th 
                            soft data. (see probasyntax for more explanations)
    probdenstemplate        ns by p matrix of non-normalized probability density values,  
                            where p is equal to either max(nl)-1 or max(nl), depending on 
                            the softpdftype. probdenstemplate(i,:) are the values of the  
                            non-normalized probability density corresponding to the intervals  
                            for the i-th soft data defined in limi(i,:). (see probasyntax for 
                            more explanations)
   
    OUTPUT :
    nl          ns by 1     2D array of the number of interval limits. nl(i) is the number  
                            of interval limits used to define the soft pdf for soft data 
                            point i. (see probasyntax for more explanations)
    limi        ns by l     2D array of interval limits, where l is equal to
                            either max(nl) or 3 depending of the softpdftype.
                            limi(i,:) are the limits of intervals for the i-th 
                            soft data. (see probasyntax for more explanations)
    probdens    ns by p     2D array of normalized probability density values,
                            each row of probadens is equal to the corresponding row
                            of probadenstemplate divided by its normalization constant
    norm        ns by 1     2D array of the normalization constants (area under the curve)
                            of each row of probadenstemplat, e.g. norm(is) is the
                            normalization constant for probdenstemplate(is,:)  
    ''' 
    softpdftype = 2 # always 2 for now
    
    if type(nl) is int: # only one soft data
      limi = limi.reshape(1, limi.size)
      probdens = probdens.reshape(1, probdens.size)
      nl = np.array(nl).reshape(1, 1)
    elif limi.shape is not probdens.shape:
      limi = limi.reshape((len(nl), int(limi.size/len(nl))))
      probdens = probdens.reshape((len(nl), int(limi.size/len(nl))))
      nl = nl.reshape((len(nl), 1))
    norm_probdens = np.zeros(probdens.shape)
    area = np.zeros(nl.shape)
    m = 0
    for nl_i, limi_i, probdens_i in zip(nl, limi, probdens):
        nl_i = int(nl_i[0])
        limi_i = limi_i[: nl_i]
        probdens_i_original = np.copy(probdens_i) #copy
        probdens_i = probdens_i[:nl_i]

        height = limi_i[1:] - limi_i[:-1]
        sum_up_low = probdens_i[:-1] + probdens_i[1:]
        area[m] = (sum_up_low * height / 2.).sum()
        norm_probdens_i = probdens_i / area[m]
        probdens_i_original[ :nl_i] = norm_probdens_i
        norm_probdens[m]=probdens_i_original#.append( probdens_i_original )
        m = m+1
    
    if len(nl) == 1: 
      limi = limi.reshape(limi.size,)
      norm_probdens = norm_probdens.reshape(limi.size,)    
      
    return nl, limi, norm_probdens, area

def proba2stat(softpdftype, nl, limi, probdens):
    def range_include_end(start, end, step):
        include_end_indexs = np.where( (end - start)%step == 0 )[0]
        revised_end = end.copy()
        revised_end[ include_end_indexs ] += step[ include_end_indexs ]
        result = map( lambda x:np.arange(*x), zip(start, revised_end, step) )
        return result

    softpdftype = get_standard_soft_pdf_type(softpdftype)
    if nl.shape[0] == 0:
        return np.array([]).reshape( (0,1) ), np.array([]).reshape( (0,1) )

    if softpdftype == 1 or softpdftype == 2:
        L1 = limi[:,:-1]
        L2 = limi[:,1:]
    elif softpdftype == 3 or softpdftype == 4: #grid
        nlMax = nl.max()
        limi_expand = np.zeros( (nl.shape[0], nlMax) )
        for idx, i in enumerate( range_include_end(limi[:,0], limi[:,2], limi[:,1]) ):
            limi_expand[idx][:i.size] = i
        L1 = limi_expand[:,:-1]
        L2 = limi_expand[:,1:]

    if softpdftype == 1 or softpdftype == 3:
        P1 = probdens
        XsMean_mat = (1/2.) * P1 * (L2**2 - L1**2)
        Xs2Mean_mat = (1/3.) * P1 * (L2**3 - L1**3)
    elif softpdftype == 2 or softpdftype == 4:
        P1 = probdens[:,:-1]
        P2 = probdens[:,1:]
        fsp = (P2 - P1) / (L2 - L1)
        fso = P1 - L1 * fsp

        L1p2 = L1 * L1
        L1p3 = L1p2 * L1
        L1p4 = L1p3 * L1

        L2p2 = L2 * L2
        L2p3 = L2p2 * L2
        L2p4 = L2p3 * L2
        
        XsMean_mat = ( 1 / 2. ) * ( fso * (L2p2 - L1p2) ) + ( 1 / 3. ) * (fsp * (L2p3 - L1p3) )
        Xs2Mean_mat = ( 1 / 3. ) * ( fso * (L2p3 - L1p3) ) + ( 1 / 4. ) * ( fsp * (L2p4 - L1p4) )

    XsMean = []
    Xs2Mean = []
    for nl_i, XsMean_i, Xs2Mean_i in zip( nl, XsMean_mat, Xs2Mean_mat ):
        XsMean.append( [ XsMean_i[ : nl_i[0] - 1].sum()] )
        Xs2Mean.append( [ Xs2Mean_i[ : nl_i[0] - 1].sum()] )

    XsMean, Xs2Mean = np.array( XsMean ), np.array( Xs2Mean )
    softmean = XsMean
    softvar = Xs2Mean - XsMean**2

    return softmean, softvar

def proba2quantile(softdpftype, nl, limi, probdens, quantiles = []):
    '''give a discrete pdf, return the quantiles user gave'''

    #default quantiles
    if quantiles:
        pass
    else:
        quantiles = [ .05, .25, .50, .75, .95, ]

    
    #get cdf
    probdens_quantile = []
    # probdens_cdf = []

    for nl_i, limi_i, probdens_i in zip( nl, limi, probdens ):

        #clip
        nl_i = int( nl_i[ 0 ] )
        limi_i = limi_i[ : nl_i ]
        probdens_i = probdens_i[ :nl_i ]

        #get area_i
        height = limi_i[ 1: ] - limi_i[ :-1 ]
        sum_up_low = probdens_i[ :-1 ] + probdens_i[ 1: ]
        area_i = (sum_up_low * height / 2.)
        
        #set cdf
        probdens_cdf_i = [ 0. ]
        for p in area_i:
            probdens_cdf_i.append( probdens_cdf_i[ -1 ] + p )

        #get interp
        probdens_quantile.append( np.interp( quantiles, probdens_cdf_i, limi_i ) )

    return np.array( probdens_quantile )

def pdf2cdf(zs):
    Fs = []
    for idx_k, zsi in enumerate(zs):
        softpdftype = zsi[0]
        if softpdftype == 2:
            nl = zsi[1]
            limi = zsi[2]
            probdens = zsi[3]
            height = np.diff(limi)
            sum_up_low = probdens[:-1] + probdens[1:]    
            cumsum = np.hstack([0,np.cumsum(height*sum_up_low/2)])
            probCDFs = cumsum/float(cumsum[nl-1])
            Fs.append([softpdftype,nl,limi,probdens,probCDFs])     
    return Fs

def probaUniform(zlow, zup):
    '''
    function [softpdftype,nl,limi,probdens]=probaUniform(zlow,zup)
    % probaUniform      - creates Uniform soft probabilistic data  at a set of data points
    %
    % This program generates uniform probabilistic soft pdf at a set of data points.
    % At each data point the lower and upper bound of the z values are known to be zlow and zup.
    % The soft probrabilistic distribution is given by the uniform pdf as follow
    % pdf(z)=1/(zup-low) for zlow<z<zup, 0 otherwise.
    %
    % SYNTAX :
    %
    % [softpdftype,nl,limi,probdens]=probaUniform(zlow,zup);
    %
    % INPUT :
    %
    % zlow     ns by 1       vector of the lower bound value at ns soft data point
    % zup      ns by 1       vector of the upper bound value at ns soft data point
    %
    % OUTPUT :
    %
    % softpdftype scalar=2   indicates the type of soft pdf representing the probabilitic soft data.  
    % nl          ns by 1    vector of number of limits  nl (see probasyntax)
    % limi        ns by 2    matrix representing the limits  values (see probasyntax)
    % probdens    ns by 2    matrix representing the probability densities (see probasyntax)
    %
    % SEE ALSO probaGaussian, probaStundentT and probasyntax.m 
    '''

    ns=len(zlow)
    softpdftype=2#*np.ones((ns,1))
    nl=2*np.ones((ns,1)).astype(int)
    zlow=np.array(zlow).reshape((ns,1))
    zup=np.array(zup).reshape((ns,1))
    limi=np.hstack([zlow, zup])
    probdens=np.kron(1./np.diff(limi),[1,1])
    zs=[softpdftype,nl,limi,probdens]
    #zs=[[softpdftype[m],nl[m],limi[m],probdens[m]] for m in range(ns)]

    return zs
