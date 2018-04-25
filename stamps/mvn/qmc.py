import os

import numpy as np
from six import print_ as print

from .latticeseq_b2 import latticeseq_b2


def qmc(
    func, xmin, xmax, args=(), kwargs={},
    nshifts=8, pow2min=3, pow2max=20,
    chebyshevk=2, abserr=0, relerr=1e-06, maxeval=None,
    showinfo=True):
    """
    Lattice sequence approximation of multivariate integral
    (using latticeseq_b2).

    Inputs:
        func        function to integrate over the n-dimensional rectangular area,
                    if your function can compute multiple point at one time,
                    it should accept x of shape (npts, ndim) and
                    return a numpy 2d array of shape (npts, fdim)
                    otherwise it should accpet x as numpy 1d array of shape (ndim,) and
                    return a scalar or a numpy 1d array of shape (fdim,)

        xmin        whatever can convert to numpy 2d arrays of shape (1, n)
        xmax        whatever can convert to numpy 2d arrays of shape (1, n)
                    *note*
                        maximum number of dimensions for current generating vector
                        is 250
        args        arguments for the integrand function
        kwargs      keyword arguments for the integrand function
        nshifts     number of randomizations for the lattice seq to obtain stderr
        chebyshevk a multiple of the stderr is used to obtain an error estimate
                    the most conservative way is to use the Cheybshev inequality:
                       k  | confidence (= 1 - 1/k^2)
                     -----+------------
                       2  |     75%
                       3  |    ~88%
                       4  |    ~93%
                       5  |    ~96%
                       8  |    ~98%
                    since this holds for any distribution (of the results) with mean
                    and variance these estimates are normally very conservative and
                    so the actual confidence level might be a lot higher than indicated above
        maxeval     if set then we set pow2max such that 2**pow2max * nshifts is as close as possible
    

        example
            qmc(func, array([-3, -2, 1]), array([5, 2, 4]))
            then func should accept numpy array of dimension npts-by-3 or
            numpy 1d array of shape (3,)

            func is called like:
                func(x, a1, a2)    if args=(a1,a2)
                func(x, a1)        if args=(a1,)
                func(x, a1, k1=k1) if args=(a1,) and kwargs={'k1': k1}

        minimum evaluations is 2**pow2min * nshifts
        maximum evaluations is 2**pow2max * nshifts

    Outputs:
        Q           approximation to the integral
        estAbsErr   estimated absolute error: chebyshev_k * stderr(nshifts independent approximations)
        infos       some info as dictionary
            stdQ        standard deviation of Q
            nbFunEvals  number of function evaluate
    """

    if maxeval != None:
        pow2max = min(int(np.rint(np.log2(maxeval/nshifts))), 20)

    xmin = np.array(xmin, ndmin=2)
    xmax = np.array(xmax, ndmin=2)
    scale = xmax - xmin
    jacobian = np.prod(scale)

    ndim = xmin.shape[1] # get ndim
    latgen = latticeseq_b2(s=ndim)

    np.random.seed(1) # Mersenne Twister reset
    shifts = np.random.rand(nshifts, ndim)

    # test vectorized and get fdim
    try:
        fdim = func(np.vstack((xmin,xmax)), *args, **kwargs).shape[1]
        vectorized = True
    except Exception as e: #has a risk if func is for no vectorized but
                         #can accept 2d numpy array
        print('qmc warning: integration function is not vectorized.', e)
        val = func(xmin.ravel(), *args, **kwargs)
        if hasattr(val,'__len__'):
            fdim = len(val)
        else:
            fdim = 1
        vectorized = False
    acc = np.zeros((nshifts,fdim))
    m = 0

    for m in range(0, pow2max + 1):
        # generate an array of 2**(m-1) x s unless m=0, 1 x s
        x = latgen.calc_block(m)
        if vectorized == True:
            npts = x.shape[0]
            x_vec = np.tile(x, (nshifts, 1))
            sh_vec = np.repeat(shifts, npts, axis=0)
            x_sh_vec = (scale * ((x_vec + sh_vec) % 1) + xmin)
            v_vec = func(x_sh_vec, *args, **kwargs)
            acc += jacobian* v_vec.reshape((nshifts, npts, fdim)).sum(axis=1)
        else:
            for xk in x:
                for j in range(nshifts):
                    xshifted = (scale * ((xk + shifts[j,:]) % 1) + xmin).ravel()
                    #force convert to 1d array
                    fval = np.array(func(xshifted, *args, **kwargs), ndmin=1)
                    acc[i] += jacobian * fval
        Q = np.mean(acc/2**m, axis=0)
        stdQ = np.std(acc/2**m, axis=0) / np.sqrt(nshifts)
        if m < pow2min: continue
        if (stdQ * chebyshevk <= abserr).all(): break
        if (stdQ * chebyshevk <= relerr * np.abs(Q)).all(): break
    if showinfo:
        print("QMC Q=", Q, " std=", stdQ, " N=", 2**m * nshifts, \
              " s=", ndim, " relErrReq=", relerr, " absErrReq=", abserr)
    
    #Q = Q
    estAbsErr = chebyshevk*stdQ
    #stdQ = stdQ
    nbFunEvals = 2**m*nshifts
    infos = {'stdQ': stdQ, 'nbFunEvals': nbFunEvals}
    
    return Q, estAbsErr, infos


if __name__ == '__main__':
    import time

    radius = 1.
    xmin = np.array([0, 0, 0], np.float64)
    xmax = np.array([radius, 2*np.pi, np.pi], np.float64)
    def integrand_sphere(x_array):
        r, theta, phi = x_array
        return r**2*np.sin(phi)

    def integrand_sphere_vectorized(x_array):
        r = x_array[:,0:1]
        theta = x_array[:,1:2]
        phi = x_array[:,2:3]
        # r, theta, phi = x_array
        return r**2*np.sin(phi)
 
    # # TOO SLOW without vectorized...
    # v,e = qmc(integrand_sphere, xmin, xmax)
    # print(v,e)
    method = 'qmc'
    abserr = 0
    relerr = 10**-7

    print('Using                 :', method)
    sss = time.time()
    v, e, infos = \
        qmc(
            integrand_sphere_vectorized, xmin, xmax,
            abserr=abserr, relerr=relerr, pow2max=24, showinfo=True
            )
    finfo = {
        'thinksAbsErrReqReached': e < abserr,
        'thinksRelErrReqReached': e < np.abs(v) * relerr,
        'stdQ': infos['stdQ'],
        'nbFunEvals': infos['nbFunEvals'] }
    print('Time                  : %.3g sec' % (time.time() - sss))
    orgprintopts = np.get_printoptions()
    np.set_printoptions(formatter={'float': '{: 10.03e}'.format})
    print('Value                 :', v)
    print('EstAbsErr             :', e)
    for key, val in finfo.items():
        print("%-22s: %s" % (key, val))
    np.set_printoptions(orgprintopts)
    
    # print('sphere volume:', v)
    # print('sphere volume error:', e)
    print('exact sphere volume:', 4/3. * np.pi * radius**3)
