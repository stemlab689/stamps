# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:37:12 2015

@author: hdragon689
"""
import numpy
from ..general.coord2K import coord2K
from ..stats.anisotropy import aniso2iso

def simuchol(ch,covmodel,covparam,ns=1):
  '''
% simuchol                  - simulation by the Cholesky method
%
% Implementation of the traditional non conditional
% simulation method based on a Choleski decomposition
% of the covariance matrix. This simulation method is
% especially recommended for simulating independantly
% several sets of a limited number of hard values
% (less than few hundreds). Simulated values are zero
% mean Gaussian distributed. 
%
% SYNTAX :
%
% [Zh,L]=simuchol(ch,covmodel,covparam,ns);
%
% INPUT :
%
% ch         nh by d   matrix of coordinates for the locations
%                      where hard data have to be simulated. A
%                      line corresponds to the vector of coordinates
%                      at a simulation location, so the number of
%                      columns corresponds to the dimension of the
%                      space. There is no restriction on the dimension
%                      of the space.
% covmodel   string    string that contains the name of the covariance
%                      model which is used for the simulation (see the
%                      MODELS directory). Variogram models are not
%                      available for this function.
% covparam   1 by k    vector of values for the parameters of covmodel,
%                      according to the convention for the corresponding
%                      covariance model.
% ns         scalar    number of sets of simulated values which are required.
%                      If the optional ns variable is omitted from the input
%                      list of variables, simuchol.m produces a single set
%                      of simulated values.
% 
% OUTPUT :
%
% Zh         nh by ns  matrix of zero mean simulated Gaussian distributed hard
%                      values at the coordinates specified in ch. Each column
%                      corresponds to a different simulation, so that if ns=1
%                      or if ns is omitted, Zh is a column vector of values.
%                      Each simulated vector of values is statistically
%                      independant from the others.
% L          nh by nh  upper triangular matrix obtained from the Choleski
%                      decomposition of the global covariance matrix C for
%                      values at the ch coordinates, such that C=L'*L.

  '''  

#rand('state',sum(100*clock));
  K,_=coord2K(ch,ch,covmodel,covparam)
  L=numpy.linalg.cholesky(K)
#  Lt=L.T
#  n=Lt.shape[0]
  n=K.shape[0]
  Zh=numpy.zeros([n,ns])
  for i in xrange(ns): 
    Zh[:,i:i+1]=L.dot(numpy.random.randn(n,1))
  
  return Zh  
    
def anisosimuchol(ch,covmodel,covparam,theta,ratio,ns=1):
  newch=aniso2iso(ch,theta,ratio)
  K,_=coord2K(newch,newch,covmodel,covparam)
  L=numpy.linalg.cholesky(K)
#  Lt=L.T
#  n=Lt.shape[0]
  n=K.shape[0]
  Zh=numpy.zeros([n,ns])
  for i in xrange(ns): 
    Zh[:,i:i+1]=L.dot(numpy.random.randn(n,1))
    
  return Zh  

    
if __name__ == "__main__":  
  
  import matplotlib.pyplot as plt
  
  x=numpy.linspace(0,50,50)
  xi,yi=numpy.meshgrid(x,x)
  ch=numpy.hstack([xi.reshape(xi.size,1),yi.reshape(xi.size,1)])
#  covmodelS=['nuggetC','exponentialC','exponentialC']
#  covparamS=[[1,[None]],[0.45,[500]],[0.4,[300000]]]

  covmodelS=['exponentialC']
  covparamS=[[10,[20]]]  
  Z=simuchol(ch,covmodelS,covparamS)
  ZZ1=Z.reshape(xi.shape)
  
  plt.figure()
  plt.pcolor(xi,yi,ZZ1)
  plt.title('Large scale')
  plt.colorbar()
  plt.show()
  
  Za=anisosimuchol(ch,covmodelS,covparamS,theta=30,ratio=0.5)
  ZZ1a=Za.reshape(xi.shape)  
  
  plt.figure()
  plt.pcolor(xi,yi,ZZ1a)
  plt.title('Ansiotropic Large scale')
  plt.colorbar()
  plt.show()  
  
#  covmodelS=['exponentialC']
#  covparamS=[[10,[10]]]  
#  Z=simuchol(ch,covmodelS,covparamS)
#  ZZ2=Z.reshape(xi.shape)
#  
#  plt.figure()
#  plt.pcolor(xi,yi,ZZ2)
#  plt.title('Small scale')
#  plt.colorbar()
#  plt.show()
#  
#  plt.figure()
#  plt.pcolor(xi,yi,ZZ1+ZZ2)
#  plt.title('True')
#  plt.colorbar()
#  plt.show()  
#    