# -*- coding: utf-8 -*-
'''
Created on 2012/7/6

@author: KSJ
'''
# import os
import numpy
import matplotlib.pyplot as plt

from .stcov import stcov
from ..stest import stmean

def polarstcovmap(cMS,tME,Zst,rLag,rLagTol,tLag=None,tLagTol=None,angLag=None,angTol=None,
                  plot=True):
  '''
  Plot polar map for empirical spatial covariance of space-time data
  SYNTAX :

  polarstcovmap(cMS,tME,Zst,rLag,rLagTol,tLag=None,tLagTol=None,angLag=None
                ,angTol=None,plot=True)

  INPUT : 

  cMS       ns by d       
  tME       1 by nt       
  Zst       ns by nt      Data
  rLag      nls by 1      1D array of vector with the r lags    
  rLagTol   nls by 1      1D array of vector with the tolerance for the r lags
  angLag    na by 1       1D array of angles to be evaluated (in radian). The 
                          two ends should be [-pi/2,pi/2) 
                          The default is [-pi/2:30/180*pi:pi/2)
  angTol    na by 1       1D array of angle tolerance around the each of angles
                          the default is the array of 15/180*pi
  plot      bool          True for plot the polar covariance. Default is True                       
                         
  OUTPUT:
  CC        list          a list of na directional empirical covariances. Each 
                          has shape of nls by nlt
  CCn       list          a list of na directional empirical covariances. Each 
                          has shape of nls by nlt
  lagr      nls by nlt    meshed spatial distances
  lagt      nls by nlt    meshed temporal lags                    
  angLag    na by 1       1D array of the diretions being evaluated      

  Remark: 
  For the purposes of plotting polar empirical covariance,               
  '''
  
  if angLag is None:
    ang_inic=-numpy.pi/2
    ang_step=30/180.*numpy.pi
    angLag=numpy.arange(ang_inic,numpy.pi/2,ang_step)
    angTol=numpy.ones(angLag.shape)*ang_step/2
  
  CC=[]
  CCn=[]
  CC4plot=numpy.empty([rLag.size,0])   
  for i,ang in enumerate(angLag):
    if tLag is not None:
      C,Cn,lagr,lagt=stcov(cMS,tME,Zst,rLag,rLagTol,tLag,tLagTol,ang=ang,angtol=angTol[i])
    else:
      C,Cn,lagr,lagt=stcov(cMS,tME,Zst,rLag,rLagTol,ang=ang,angtol=angTol[i])
    CC.append(C)
    CCn.append(Cn)
    CC4plot=numpy.hstack((CC4plot,C[:,0:1]))       
  
  if plot is True:
    idx=numpy.where(numpy.isnan(CC4plot))
    if idx[0].size>0:  
      CC3=numpy.hstack((CC4plot,numpy.hstack((CC4plot,CC4plot))))
      angLag3=numpy.hstack((angLag-numpy.pi,numpy.hstack((angLag,angLag+numpy.pi)))) 
      try:
        CC1=stmean.stmeaninterp(lagr,angLag3,CC3,lagr,angLag,method='linear')
      except:  
        CC1=stmean.stmeaninterp(lagr,angLag3,CC3,lagr,angLag,method='nearest')
      for k in xrange(numpy.int(idx[0].size)):
        CC4plot[idx[0][k],idx[1][k]]=CC1[idx[0][k],idx[1][k]]
      
    angLag4plot=numpy.hstack((angLag,angLag+numpy.pi))
    CC4plot=numpy.hstack((CC4plot,CC4plot))
    angLag4plot=numpy.append(angLag4plot,angLag4plot[-1]+angTol[0]*2)
    CC4plot=numpy.hstack((CC4plot,CC4plot[:,0:1]))
    
    angLag4plot=angLag4plot-angTol[0]
    theta,radius = numpy.meshgrid(angLag4plot, rLag) #rectangular plot of polar data
    fig=plt.figure()
    ax = fig.add_subplot(111,polar=True)
    im=ax.pcolor(theta, radius, CC4plot,cmap='hot_r')  
    plt.colorbar(im)  
    plt.show()  
  
  return CC,CCn,lagr,lagt,angLag

          
def iso2aniso(ciso,angle,ratio):
  '''
% iso2aniso                 - convert isotropic to anisotropic coordinates 
%
% Do the transformation which is reciprocal to the transformation made by
% the aniso2iso.m function, so that applying successively both transformation
% has no effect on the coordinates. The function maps a set of coordinates
% in an isotropic space into a set of coordinates in an anisotropic one. 
%
% SYNTAX :
%
% [c]=iso2aniso(ciso,angle,ratio); 
%
% INPUT :
%
% ciso    n by d   An 2D array of coordinates for the locations in the isotropic
%                  space. A line corresponds to the vector of coordinates at a
%                  location, so that the number of columns corresponds to the
%                  dimension of the space. Only two dimensional or three dimensional
%                  space coordinates can be processed by this function.
% angle   1 by d-1 vector of angle values that characterize the anisotropy. 
%                  In a two dimensional space, angle is the trigonometric angle
%                  between the horizontal axis and the principal axis of the
%                  ellipse. In a three dimensional space, spherical coordinates
%                  are used, such that angle(1) is the horizontal trigonometric
%                  angle and angle(2) is the vertical trigonometric angle for the
%                  principal axis of the ellipsoid. All the angles are measured
%                  counterclockwise in degrees and are between -90 and 90.
% ratio   1 by d-1 vector that characterize the ratio for the length of the axes
%                  for the ellipse (in 2D) or ellipsoid (in 3D). In a two dimensional
%                  space, ratio is the length of the secondary axis of the ellipse
%                  divided by the length of the principal axis, so that ratio in [0,1]. In a
%                  three dimensional space, ratio(1) is the length of the second
%                  axis of the ellipsoid divided by the length of the principal axis, 
%                  whereas ratio(2) is length of the third axis of the ellipsoid
%                  divided by the length of the principal axis, so that ratio(1) and
%                  ratio(2) are both in the range of [0,1].
%
% OUTPUT :
%
% c       n by d   An 2D array of coordinates having the same size as ciso, that gives the new coordinates
%                  in the anisotropic space.
%
% NOTE :
%
% It is possible to specify an additional index vector, taking integer values from 1
% to nv. The values in index specifies which of the nv variable is known at each one
% of the corresponding coordinates. The ciso matrix of coordinates and the index vector
% are then grouped together using the MATLAB cell array notation, so that ciso={ciso,index}.
% This allows to perform the same coordinate transformation at once on a set of possibly
% different variables. The output variable c is then also a cell array that contains
% both the new matrix of coordinates and the index vector.
  '''
    
  # Determine the dimension of the space and set ratio 
  if type(ciso) is numpy.ndarray:  
    d=ciso.shape[1]
  else:
    ciso=numpy.array(ciso,ndmin=2)
    d=ciso.shape[1]
  angle=angle*2*numpy.pi/360
  angle=-angle

  # When d<2 or d>3, error

  if (d<2) or (d>3):
    print 'iso2aniso function requires coordinates in a 2D or 3D space'
    return

  # Case for d=2

  if d==2:
    R=numpy.array([[numpy.cos(angle),numpy.sin(angle)],
                   [-numpy.sin(angle),numpy.cos(angle)]])
    ciso[:,1]=ciso[:,1]*ratio
    c=ciso.dot(R.T)
    

  # Case for d=3

  if d==3:
    phi=angle[0]
    teta=angle[1]
    ratioy=ratio[0]
    ratioz=ratio[1]

    R1=numpy.array([[numpy.cos(phi),numpy.sin(phi),0],
                    [-numpy.sin(phi),numpy.cos(phi),0],
                    [0,0,1]])
    R2=numpy.array([[numpy.cos(teta),0,numpy.sin(teta)],
                    [0,1,0],
                    [-numpy.sin(teta),0,numpy.cos(teta)]])
    R=(R2.T).dot(R1.T)
    ciso[:,1]=ciso[:,1]*ratioy
    ciso[:,2]=ciso[:,2]*ratioz    
    c=ciso.dot(R)

    
  return c


def aniso2iso(c,angle,ratio):
  '''
% aniso2iso                 - convert anisotropic to isotropic coordinates (Jan 1,2001)
%
% Transform a set of two dimensional or three dimensional coordinates
% using rotations and dilatations of the axes, in order to map an
% anisotropic space into an isotropic one. The geometric anisotropy
% is characterized by the angle(s) of the principal axis of the ellipse
% (in 2D) or ellipsoid (in 3D), and by the ratio(s) of the principal axis
% length by the other axes lengths. Using this function, an ellipse
% (ellipsoid) is thus mapped into a circle (sphere) having as radius the
% length of the principal axis. The transformation consist in a rotation
% of the axes followed by a dilatation. 
% 
% SYNTAX :
%
% [ciso]=aniso2iso(c,angle,ratio); 
%
% INPUT :
%
% c       n by d   matrix of coordinates for the locations in the anisotropic
%                  space. A line corresponds to the vector of coordinates at a
%                  location, so that the number of columns corresponds to the
%                  dimension of the space. Only two dimensional or three dimensional
%                  space coordinates can be processed by this function.
% angle   1 by d-1 vector of angle values that characterize the anisotropy. 
%                  In a two dimensional space, angle is the trigonometric angle
%                  between the horizontal axis and the principal axis of the
%                  ellipse. In a three dimensional space, spherical coordinates
%                  are used, such that angle(1) is the horizontal trigonometric
%                  angle and angle(2) is the vertical trigonometric angle for the
%                  principal axis of the ellipsoid. All the angles are measured
%                  counterclockwise in degrees and are between -90∞ and 90∞.
% ratio   1 by d-1 vector that characterize the ratio for the length of the axes
%                  for the ellipse (in 2D) or ellipsoid (in 3D). In a two dimensional
%                  space, ratio is the length of the secondary axis of the ellipse
%                  divided by the length of the principal axis, so that ratio in [0,1]. In a
%                  three dimensional space, ratio(1) is the length of the second
%                  axis of the ellipsoid divided by the length of the principal axis, 
%                  whereas ratio(2) is length of the third axis of the ellipsoid
%                  divided by the length of the principal axis, so that ratio(1) and
%                  ratio(2) are both in the range of [0,1].
%
% OUTPUT :
%
% ciso    n by d   matrix having the same size as c, that gives the new coordinates
%                  in the isotropic space.
%
% NOTE :
%
% It is possible to specify an additional index vector, taking integer values from 1
% to nv. The values in index specify which one of the nv variable is known at each one
% of the corresponding coordinates. The c matrix of coordinates and the index vector
% are then grouped together using the MATLAB cell array notation, so that c={c,index}.
% This allows to perform the same coordinate transformation at once on a set of possibly
% different variables. The output variable ciso is then also a cell array that contains
% both the new matrix of coordinates and the index vector.
  '''
  
  # Determine the dimension of the space and set angle
  if type(c) is numpy.ndarray:  
    d=c.shape[1]
  else:
    c=numpy.array(c,ndmin=2)
    d=c.shape[1]
  angle=angle*2*numpy.pi/360;

  # When d<2 or d>3, error

  if (d<2) or (d>3):
    print 'aniso2iso requires coordinates in a 2D or 3D space'
    return

  # Case for d=2

  if d==2:
    R=numpy.array([[numpy.cos(angle),numpy.sin(angle)],
                  [-numpy.sin(angle),numpy.cos(angle)]])
    ciso=c.dot(R.T)
    ciso[:,1]=ciso[:,1]*1./ratio

  # Case for d=3

  if d==3:
    phi=angle[0]
    teta=angle[1]
    ratioy=ratio[0]
    ratioz=ratio[1]
  
    R1=numpy.array([[numpy.cos(phi),numpy.sin(phi),0],
                    [-numpy.sin(phi),numpy.cos(phi),0],
                    [0,0,1]]) 
    R2=numpy.array([[numpy.cos(teta),0,numpy.sin(teta)],
                    [0,1,0],
                    [-numpy.sin(teta),0,numpy.cos(teta)]])
    R=(R1.T).dot(R2.T)
    ciso=c.dot(R)
    ciso[:,1]=ciso[:,1]*1./ratioy
    ciso[:,2]=ciso[:,2]*1./ratioz
  
  return ciso   
