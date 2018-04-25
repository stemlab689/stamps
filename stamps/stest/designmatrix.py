# -*- coding: utf-8 -*-
import numpy


#only accept order = numpy.nan or 0
def designmatrix( c, order ):
	
	if numpy.isnan(order):
		return numpy.array([],ndmin=2).reshape(c.shape[0],0), numpy.array([],ndmin = 2)
	else:
		n, nd = c.shape
		X = numpy.ones((n, 1))
		index = numpy.zeros((2, 1))
		if nd == 1:
			return X, index[0,:]
		else:
			return X, index
	
	# order = numpy.array([ [order, order] ])
	# if ~numpy.isnan( order[0][0] ) or ~numpy.isnan( order[0][1] ):
	# 	X = numpy.ones((n, 1))
	# 	index = numpy.zeros((2, 1))
	# if ~numpy.isnan( order[0][0] ):

