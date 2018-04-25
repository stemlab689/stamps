# -*- coding: utf-8 -*-
import numpy

from .regression import regression
from .designmatrix import designmatrix


def localmeanBME( ck, ch, cs, zh, ms, vs,
                  Khh, Ksh, Kss, order ):

    nh = zh.shape[0]
    ns = ms.shape[0]
    mkest = 0.
    mhest = numpy.empty((nh, 1))*numpy.NaN
    msest = numpy.empty((ns, 1))*numpy.NaN
    vkest = 0.

    c = numpy.vstack((ch, cs))
    ms=ms.reshape((ns,1)) 
    vs=vs.reshape((ns,1))       
    Ksh=Ksh.reshape((ns,nh))
    Kss=Kss.reshape((ns,ns))
    #add '_' b/s we won't to change vs and Kss inplace
#    vs_ = numpy.tile( vs,(1,ns ) ) if len(vs) else vs
#    Kss_ = Kss + numpy.diag(vs_)
    K = numpy.vstack( (numpy.hstack((Khh, Ksh.T)), numpy.hstack((Ksh, Kss)) ) )
    z = numpy.vstack((zh, ms))

    best, Vbest, mm, index = regression(c, z, order, K)

    if mm.size > 0:
        mhest = mm[:nh,0:1]
        msest = mm[nh:,0:1]
    else:
        mhest[:] = 0.0 #numpy.array([])
        msest[:] = 0.0 #numpy.array([])

    x, index = designmatrix(ck, order)
    if x.size > 0:
        mkest = ( x.dot(best) )[0][0]
        vkest = ( x.dot(Vbest).dot(x.T) )[0][0]

    return mkest , mhest, msest, vkest
