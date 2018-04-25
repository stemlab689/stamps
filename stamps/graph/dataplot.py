# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 09:46:39 2015

@author: hdragon689
"""
import numpy
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas
from pandas.tools.plotting import scatter_matrix


def colorplot(c,z,ax=None,cmap='hot_r',zrange=None,colorbar=True):  
  '''
  Plot colored symbols for the values of a vector at a set of two dimensional
  coordinates. The function uses a colormap such that the color displayed at
  these coordinates is a function of the corresponding values.  
  
  Syntax:
  ax=colorplot(c,z,ax=None,cmap='hot_r',zrange=None)
  
  Input:
  c         n by 2    2D numpy array of spatial coordinates
  z         n by 1    2D numpy array of observations
  ax        ax        optional. Axes object for colorplot. Default is None that 
                      creates new plot
  cmap      string    the color scheme. default is the reverse of hot scheme. 
                      Popular scheme are jet, hot, gray and their reverse xxx_r
                      the complete color scheme can refer to 
                      http://matplotlib.org/examples/color/colormaps_reference.html
  zrange    list      two scalar contains min and max of color range, [zmin, zmax]                
  colorbar  bool      default is True to display the colorbar    
  
  '''
  if ax is None:
    ax=plt.figure().add_subplot(111)
  else:
    ax=ax
    
  z=z.reshape(z.size,1)  
  if zrange is not None:
    plt.scatter(x=c[:,0],y=c[:,1],c=z,s=100, cmap=cmap, \
                  vmin=zrange[0], vmax=zrange[1])
  else:
    plt.scatter(x=c[:,0],y=c[:,1],c=z,s=100, cmap=cmap)    
  if colorbar:        
    plt.colorbar(ax=ax)
  
  xmax=numpy.max(c[:,0]); xmin=numpy.min(c[:,0])
  ymax=numpy.max(c[:,1]); ymin=numpy.min(c[:,1])
  xpadding=0.05*(xmax-xmin)
  ypadding=0.05*(ymax-ymin)
  plt.xlim(xmin-xpadding,xmax+xpadding)  
  plt.ylim(ymin-ypadding,ymax+ypadding)
  plt.show()
  plt.draw()
  
  return ax  

def markerplot(c,z,ax=None,symsize=300,zrange=None): 
  '''
  Plot the values of a vector at a set of two dimensional coordinates
  using symbols of varying sizes such that the size of the displayed
  symbols at these coordinates is a function of the corresponding values. 
  
  Syntax:
  ax=markerplot(c,z,ax=None,simsize=300,zrange=None)
  
  Input:
  c       n by 2    2D numpy array of spatial coordinates
  z       n by 1    2D numpy array of observations
  ax      ax        optional. Axes object for markerplot. Default is None that 
                    creates new plot
  symsize scalar    the size scheme. The size scale for marker display. 
                    Default is 300  
  zrange list       two scalar contains min and max of marker size range, [zmin, zmax]                
  
  '''  
  
  if ax is None:
    ax=plt.figure().add_subplot(111)
  else:
    ax=ax
    
  z=z.reshape(z.size,1)  
  zmax=numpy.max(z); zmin=numpy.min(z)
  z=z/(zmax-zmin)
  if zrange is not None:
    plt.scatter(x=c[:,0],y=c[:,1],s=z*symsize,vmin=zrange[0], vmax=zrange[1])
  else:
    plt.scatter(x=c[:,0],y=c[:,1], s=z*symsize)    

  xmax=numpy.max(c[:,0]); xmin=numpy.min(c[:,0])
  ymax=numpy.max(c[:,1]); ymin=numpy.min(c[:,1])
  xpadding=0.05*(xmax-xmin)
  ypadding=0.05*(ymax-ymin)
  plt.xlim(xmin-xpadding,xmax+xpadding)  
  plt.ylim(ymin-ypadding,ymax+ypadding)          
  plt.show()
  plt.draw()  
  
  return ax
  
  
def tsplot(t,z,ax=None,fmt='b-',displayscale=None):
  ''' 
  Plot time series data
  Syntax:
    tsplot(t,z,ax=None,fmt='b-',displayscale=None)
  
  Input:
  t             1 by n    1D or 2D array of time in datetime or 
                          numpy.datetime64 formats
  z             1 by n    1D or 2D array of observations
  ax            axes      the axes object to be plotted
  fmt           string    line format for time series plot. 
                          Details can refer to plt.plot?    
  displayscale  string    'm' and 'y' refers to the pre-specified xtick formats
                          for monthly or yearly data plotting. If None, the 
                          plot goes with the default in plt.plot_date function
                          
  Remark: Details of time xticker format can refer to 
          http://linux.die.net/man/3/strftime    
  '''
  
  if ax is None:  
    ax = plt.figure().add_subplot(111)
  else:
    ax=ax
  
  ax.plot_date(t,z,fmt=fmt)
  if displayscale=='m':
  #  xlocator=dates.YearLocator(1,month=7)
    xlocator=dates.MonthLocator([1,7],interval=1)
    months = dates.MonthLocator()  # every month
    ax.xaxis.set_major_locator(xlocator)
  #  ax.xaxis.set_minor_locator(x2locator)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b\n%Y')) 
    ax.xaxis.set_minor_locator(months)
  #  ax.xaxis.set_minor_formatter(dates.DateFormatter('%b\n%Y'))
  # Format can re
  elif displayscale=='y':  
    years = dates.YearLocator()   # every year
    months = dates.MonthLocator()  # every month
    yearsFmt = dates.DateFormatter('%Y')    
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)  
    
  ax.autoscale_view()
  
  return ax

def histscatterplot(data,columns=None):
  '''
  Input
  data      m by n    a 2D array with m observations and n variables. This input
                      can also be a pandas.DataFrame 
  columns   list      list of the titles of variables
  
  Note: this function is an interface to the scatter_matrix function in pandas
  '''  
  
  
  if type(data) is not pandas.core.frame.DataFrame:
    data=pandas.DataFrame(data,columns=columns)    
  scatter_matrix(data,diagonal='hist',hist_kwds={'normed':'True'},alpha=0.2)    
  plt.show()
  plt.draw()    
  


if __name__ == "__main__":
  c=numpy.random.rand(30,2)
  z=numpy.random.rand(30,1)
  zr=[0,0.5]
  colorplot(c,z,ax=None,cmap='jet_r',zrange=zr)
  markerplot(c,z,symsize=300,zrange=zr)
  
