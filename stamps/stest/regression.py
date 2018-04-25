# -*- coding: utf-8 -*-
import numpy

from .designmatrix import designmatrix


def regression(c, z, order, K):

    X, index = designmatrix(c, order)

    index = index.T
    n, p = X.shape

    if X.size > 0:
        Xt = X.T
        if K.size == 0:
            invXtX = numpy.linalg.inv( Xt.dot(X) )
            best = invXtX.dot(Xt).dot(z)
            zest = X.dot(best)
            resi = z - zest
            s2 = ( resi.T.dot(resi) ) / float(n - p)
            Vbest = invXtX.dot( s2 )
        else:
            XtinvK = X.T.dot( numpy.linalg.inv(K) )
            invXtinvKX = numpy.linalg.inv( XtinvK.dot(X) )
            best = invXtinvKX.dot(XtinvK).dot(z)
            zest = X.dot(best)
            Vbest = invXtinvKX
    else: # set default size
        return numpy.array([],ndmin=2), numpy.array([],ndmin = 2),numpy.array([],ndmin=2), numpy.array([],ndmin = 2)


    return best, Vbest, zest, index
