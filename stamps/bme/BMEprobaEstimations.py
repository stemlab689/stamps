# -*- coding: utf-8 -*-
import copy

from six.moves import range
from six import print_ as print
from six import iteritems
import numpy as np
import scipy.stats
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

from .softconverter import proba2stat
from .pystks_variable import get_standard_order, get_standard_soft_pdf_type
from .BMEoptions import BMEoptions
from .softconverter import pdf2cdf

from ..general.coord2K import coord2K, coord2Ksplit, coord2Kcombine
from ..stats.mepdf import maxentpdf_gc, maxentcondpdf_gc
from ..general.valstvgx import valstv2stg, valstg2stv
from ..general.neighbours import neighbours, neighbours_index_kd
from ..mvn.qmc import qmc

KHS_DICT = {'k': 0, 'h': 1, 's': 2}


def _bme_posterior_pdf(
    ck, ch=None, cs=None, zh=None, zs=None,
    covmodel=None, covparam=None, covmat=None,
    order=np.nan, options=None,
    general_knowledge='gaussian',
    #  specific_knowledge='unknown',  
    pdfk=None, pdfh=None, pdfs=None, hk_k=None, hk_h=None, hk_s=None,
    gui_args=None):

    def __get_zs_integration_limits(zs):
        '''get soft data integration limit''' 
        ranges = []
        for zsi in zs:
            pdftype = get_standard_soft_pdf_type(zsi[0])
            if pdftype in  [1, 2]: # nl, limi, probdens
                nl = zsi[1]
                limi = zsi[2]
                ranges.append((limi[0], limi[nl[0]-1]))
            elif pdftype in [10]:
                zm = zsi[1]
                zstd = np.sqrt(zsi[2])
                ranges.append((zm-3*zstd, zm+3*zstd))
        ranges = np.array(ranges)
        return ranges.copy()
    order = get_standard_order(order)
    nk = ck.shape[0]
    nh = ch.shape[0] if ch is not None else 0
    ns = cs.shape[0] if cs is not None else 0

    x_all_split = _get_x_all_split(nk, zh, zs)
    Xh = _get_x(x_all_split, 'h')
    mean_all_split = _get_mean_all_split(x_all_split, order)
    #get cov_all_split
    if covmat is None:
        cov_all_split = _get_cov_all_split(ck, ch, cs, covmodel, covparam)
        covmat = coord2Kcombine(cov_all_split)
    else:
        cov_all_split = np.vsplit(
            covmat, [nk, nk+nh, nk+nh+ns]
            )[:-1] #exclude final empty array
        cov_all_split = \
            [np.hsplit(c, [nk, nk+nh, nk+nh+ns][:-1])\
            for c in cov_all_split]

    #find cuplicated point
    if ns:
        dup_ck_cs_idx = np.array(
            np.all((ck[:, None, :] == cs[None, :, :]), axis=-1).nonzero()
            ).T
    else:
        dup_ck_cs_idx = np.array([[]])

    if general_knowledge == 'gaussian':
        
        def __get_fG_Xkh_each_ck(fG_Xkh_):
            if Xh is None:
                def fG_Xkh_each_ck(xk):
                    xk_origin_shape = xk.shape
                    xk = xk.flatten()
                    input_xk = xk.T
                    return fG_Xkh_(input_xk).reshape(xk_origin_shape)
            else:
                def fG_Xkh_each_ck(xk):
                    xk_origin_shape = xk.shape
                    xk = xk.flatten()
                    input_xk = np.vstack(
                        (xk, np.tile(Xh, xk.size))
                        ).T
                    return fG_Xkh_(input_xk).reshape(xk_origin_shape)
            return fG_Xkh_each_ck     
        
        #fG_Xh: const
        if nh == 0:
            fG_Xh = 1.
        else:
            fG_Xh = _get_multivariate_normal_pdf(
                x_all_split, mean_all_split, cov_all_split, 'h')(
                    _get_x(x_all_split, 'h').T)

        def __get_fG_Xs_gvn_Xh_all_ck():
            fG_Xs_gvn_Xh = _get_multivariate_normal_pdf(
                x_all_split, mean_all_split, cov_all_split, 's_h')
            def fG_Xs_gvn_Xh_all_ck(xs):
                xs_origin_shape = xs.shape
                xs = xs.reshape(-1, xs_origin_shape[-1])
                output = fG_Xs_gvn_Xh(xs)
                return output.reshape(xs_origin_shape[:-1]+(1,))
            return fG_Xs_gvn_Xh_all_ck
        fG_Xs_gvn_Xh = __get_fG_Xs_gvn_Xh_all_ck()

        zs_limits = __get_zs_integration_limits(zs)
        if options['integration method'] == 'qmc':
            fS_Xs = _get_fs(zs)
            def __qmc_int_fG_Xs_gvn_Xh__fS_Xs(x_array):
                return fG_Xs_gvn_Xh(x_array) * fS_Xs(x_array)
            xmin = zs_limits[:,0].copy()
            xmax = zs_limits[:,1].copy()

        elif options['integration method'] == 'qmc_F':
            fS_Xs = _get_fs(zs)
            Fsinv = _get_Fsinv(zs)

            def __qmc_int_fG_Xs_gvn_Xh__fS_Xs(Fx_array):
                return fG_Xs_gvn_Xh(Fsinv(Fx_array))
            xmin = np.zeros(zs_limits[:,0].shape)
            xmax = np.ones(zs_limits[:,1].shape)

        elif options['integration method'] == 'qmc_T':
            from scipy.special import erfinv
            m = _get_mean_a_given_b(x_all_split, mean_all_split,
                                cov_all_split, 's', 'h')
            v = _get_sigma_a_given_b(cov_all_split, 's', 'h')
            u,s,vh = np.linalg.svd(v)
            A = vh.T*np.sqrt(s)

            fS_Xs = _get_fs(zs)
            def __qmc_int_fG_Xs_gvn_Xh__fS_Xs(Fx_array):
                Fx_array[Fx_array==0.] = 10**-5
                Fx_array[Fx_array==1.] = 1 - 10**-5

                x_array = np.sqrt(2) * erfinv(2*Fx_array - 1)
                xs_array = (A.dot(x_array.T) + m).T
                return fS_Xs(xs_array)
            xmin = np.zeros(zs_limits[:,0].shape)
            xmax = np.ones(zs_limits[:,1].shape)

        int_fG_Xs_gvn_Xh__fS_Xs, e, info = qmc(
            __qmc_int_fG_Xs_gvn_Xh__fS_Xs,
            xmin, xmax, relerr=options[3,0], pow2min=10,
            showinfo=options['qmc_showinfo']
            )

        if int_fG_Xs_gvn_Xh__fS_Xs == 0: # NC is zero
            print('warning: normolized constant is equal to 0.')
            #int_fG_Xs_gvn_Xh__fS_Xs: const

        def __get_fSk_Xk_each_ck(zsk):
            def fSk_Xk_each_ck(xk):
                if zsk is None:
                    return np.ones(xk.shape)
                else:
                    xk_origin_shape = xk.shape
                    xk = xk.flatten()
                    pdf_type = get_standard_soft_pdf_type(zsk[0])
                    if pdf_type == 2:
                        nl = zsk[1][0]
                        limi = zsk[2]
                        probdens = zsk[3]
                        y_i = np.interp(
                            xk, limi[:nl], probdens[:nl],
                            left = 0., right = 0.)
                    elif pdf_type == 1:
                        nl = zs[1]
                        limi = zs[2]
                        probdens = zs[3]
                        idd = np.where(xk - limi[:nl] > 0)[0]
                        y_i = probdens[idd]
                    elif pdf_type == 10:
                        zm = zsk[1]
                        zstd = np.sqrt(zsk[2])
                        try:
                            y_i = scipy.stats.norm.pdf(
                                xk, loc=zm, scale=zstd)
                        except FloatingPointError:
                            import pdb
                            pdb.set_trace()
                      
                    return y_i.reshape(xk_origin_shape)
            return fSk_Xk_each_ck

        def __get_fG_Xs_gvn_Xkh_each_ck(ck_i, cs, zs,
            x_all_split_each_ck, mean_all_split_each_ck, cov_all_split_each_ck):
            idx_result = np.where(np.all(ck_i == cs, axis=1))[0]
            if idx_result.size == 0:
                x_all_split_each_ck_dup = x_all_split_each_ck
                mean_all_split_each_ck_dup = mean_all_split_each_ck
                cov_all_split_each_ck_dup = cov_all_split_each_ck
            elif idx_result.size == 1:
                x_all_split_each_ck_dup =\
                    x_all_split_each_ck[:2] +\
                    [np.delete(x_all_split_each_ck[2], idx_result, axis=0)]
                mean_all_split_each_ck_dup =\
                    mean_all_split_each_ck[:2] +\
                    [np.delete(mean_all_split_each_ck[2], idx_result, axis=0)]
                cov_all_split_each_ck_dup = copy.deepcopy(cov_all_split_each_ck)
                cov_all_split_each_ck_dup[0][2] =\
                    np.delete(cov_all_split_each_ck[0][2], idx_result, axis=1)
                cov_all_split_each_ck_dup[1][2] =\
                    np.delete(cov_all_split_each_ck[1][2], idx_result, axis=1)
                cov_all_split_each_ck_dup[2][0] =\
                    np.delete(cov_all_split_each_ck[2][0], idx_result, axis=0)
                cov_all_split_each_ck_dup[2][1] =\
                    np.delete(cov_all_split_each_ck[2][1], idx_result, axis=0)
                cov_all_split_each_ck_dup[2][2] =\
                    np.delete(cov_all_split_each_ck[2][2], idx_result, axis=0)
                #be careful below
                cov_all_split_each_ck_dup[2][2] =\
                    np.delete(cov_all_split_each_ck_dup[2][2], idx_result, axis=1)
            elif idx_result.size > 1: #strange
                raise ValueError('ck match cs twice. (strange)')
            def __get_fG_Xs_gvn_Xkh_each_ck_eack_xk(xk):
                fG_Xs_gvn_Xkh_container = []
                xk_origin_shape = xk.shape
                xk = xk.flatten()
                if options['integration method'] == 'qmc_T':
                    for xk_i in xk:
                        x_all_split_each_ck_dup[0] = np.array([[xk_i]])
                        m = _get_mean_a_given_b(
                            x_all_split_each_ck_dup,
                            mean_all_split_each_ck_dup,
                            cov_all_split_each_ck_dup, 's', 'kh')
                        v = _get_sigma_a_given_b(cov_all_split_each_ck_dup, 's', 'kh')
                        u, s, vh = np.linalg.svd(v)
                        A = vh.T*np.sqrt(s)

                        fG_Xs_gvn_Xkh_ = lambda xs, A=A, m=m: (A, m)

                        def fG_Xs_gvn_Xkh_each_ck_eack_xk(xs, ff):
                            xs_origin_shape = xs.shape
                            xs = xs.reshape(-1, xs_origin_shape[-1])
                            A, m = ff(xs)
                            xs[xs==0.] = 10**-5
                            xs[xs==1.] = 1 - 10**-5
                            xs2 = np.sqrt(2) * erfinv(2*xs - 1)
                            xs_array = (A.dot(xs2.T) + m).T
                            return xs_array

                        fG_Xs_gvn_Xkh_container.append(
                            lambda xs, ff=fG_Xs_gvn_Xkh_: fG_Xs_gvn_Xkh_each_ck_eack_xk(xs, ff))
                else:
                    for xk_i in xk:
                        x_all_split_each_ck_dup[0] = np.array([[xk_i]])
                        fG_Xs_gvn_Xkh_ = _get_multivariate_normal_pdf(
                        x_all_split_each_ck_dup,
                        mean_all_split_each_ck_dup,
                        cov_all_split_each_ck_dup, 's_kh')
                        def fG_Xs_gvn_Xkh_each_ck_eack_xk(xs, ff):
                            xs_origin_shape = xs.shape
                            xs = xs.reshape(-1, xs_origin_shape[-1])
                            output = ff(xs)
                            return output.reshape(xs_origin_shape[:-1]+(1,))
                        fG_Xs_gvn_Xkh_container.append(
                            lambda xs, ff=fG_Xs_gvn_Xkh_: fG_Xs_gvn_Xkh_each_ck_eack_xk(xs, ff))
                return np.array(
                    fG_Xs_gvn_Xkh_container).reshape(xk_origin_shape)
            return __get_fG_Xs_gvn_Xkh_each_ck_eack_xk

        def __get_fS_Xs_dup(ck_i, cs, zs):
            idx_result = np.where(np.all(ck_i == cs, axis=1))[0]
            if idx_result.size == 0:
                if options['integration method'] == 'qmc_F':
                    return Fsinv
                else:
                    return fS_Xs
            elif idx_result.size == 1:
                if options['integration method'] == 'qmc_F':
                    return _get_Fsinv(
                        [zs_i for i, zs_i in enumerate(zs) if i != idx_result])
                else:
                    return _get_fs(
                        [zs_i for i, zs_i in enumerate(zs) if i != idx_result])
            elif idx_result.size > 1: #strange
                raise ValueError('ck match cs twice. (strange)')

        def __get_pdf_each_ck(i):
            ck_i = ck[i]
            idx_result = np.where(np.all(ck_i == cs, axis=1))[0]
            if idx_result.size == 0:
                zs_dup = zs
            elif idx_result.size == 1:
                zs_dup = [zs_i for ii, zs_i in enumerate(zs) if ii != idx_result]
            elif idx_result.size > 1: #strange
                raise ValueError('ck match cs twice. (strange)')
            def _fK_Xk(xk):
                xk_origin_shape = xk.shape
                xk = xk.flatten()
                zs_limits = __get_zs_integration_limits(zs_dup)
                if options['integration method'] == 'qmc':
                    def __qmc_int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup(x_array):
                        G_Xs_gvn_Xkh_dup_i =\
                            np.hstack(
                                [fi(x_array) for fi in fG_Xs_gvn_Xkh_dup[i](xk)])
                        fS_Xs_dup_i = fS_Xs_dup[i](x_array)
                        return G_Xs_gvn_Xkh_dup_i * fS_Xs_dup_i
                    xmin = zs_limits[:,0].copy()
                    xmax = zs_limits[:,1].copy()

                elif options['integration method'] == 'qmc_F':
                    def __qmc_int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup(Fx_array):
                        G_Xs_gvn_Xkh_dup_i =\
                            np.hstack(
                                [fi(fS_Xs_dup[i](Fx_array)) for fi in fG_Xs_gvn_Xkh_dup[i](xk)])
                        return G_Xs_gvn_Xkh_dup_i
                    xmin = np.zeros(zs_limits[:,0].shape)
                    xmax = np.ones(zs_limits[:,1].shape)

                elif options['integration method'] == 'qmc_T':
                    def __qmc_int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup(Fx_array):
                        G_Xs_gvn_Xkh_dup_i =\
                            [fi(Fx_array) for fi in fG_Xs_gvn_Xkh_dup[i](xk)]
                        fS_Xs_dup_i =\
                            np.hstack(
                                [fS_Xs_dup[i](xs_array_i) for xs_array_i in G_Xs_gvn_Xkh_dup_i] )
                        return fS_Xs_dup_i
                    xmin = np.zeros(zs_limits[:,0].shape)
                    xmax = np.ones(zs_limits[:,1].shape)

                int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup , e, info = qmc(
                    __qmc_int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup,
                    xmin, xmax, relerr=options[3,0], pow2min=10,
                    showinfo=options['qmc_showinfo']
                    )
                int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup =\
                    int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup.reshape(xk_origin_shape)
                xk = xk.reshape(xk_origin_shape)
                if options['ck pdf debug']:
                    print('fG_Xkh[i](xk): {p}'.format(p=fG_Xkh[i](xk)))
                    print('fSk_Xk[i](xk): {p}'.format(p=fSk_Xk[i](xk)))
                    print('int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup: {p}'.format(p=int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup))
                    print('fG_Xh: {p}'.format(p=fG_Xh))
                    print('int_fG_Xs_gvn_Xh__fS_Xs: {p}'.format(p=int_fG_Xs_gvn_Xh__fS_Xs))
                    
                return (fG_Xkh[i](xk) * fSk_Xk[i](xk)
                    * int_fG_Xs_gvn_Xkh_dup__fS_Xs_dup
                    / fG_Xh / int_fG_Xs_gvn_Xh__fS_Xs)
            return _fK_Xk

        cov_hs_range = list(range(nk, nk+nh+ns))
        
        fG_Xkh = [] # a list contains each ck's fG_Xkh
        fSk_Xk = [] # a list contains each ck's fSk_Xk
        fS_Xs_dup = []
        fG_Xs_gvn_Xkh_dup = [] # a list contains each ck's fG_Xs_gvn_Xkh_dup
        fK_Xk = []
        for i in range(nk):
            #get x/mean/cov_all_split at each ck point
            x_all_split_each_ck =\
                [x_all_split[0][i:i+1,:]] + x_all_split[1:]
            mean_all_split_each_ck =\
                [mean_all_split[0][i:i+1,:]] + mean_all_split[1:]
            covmat_each_ck =\
                covmat[np.ix_(
                    [i]+cov_hs_range,
                    [i]+cov_hs_range
                    )]
            cov_all_split_each_ck = np.vsplit(
                covmat_each_ck, [1, 1+nh, 1+nh+ns]
                )[:-1] #exclude final empty array
            cov_all_split_each_ck = \
                [np.hsplit(c, [1, 1+nh, 1+nh+ns][:-1])\
                for c in cov_all_split_each_ck]

            # get fG_Xkh each ck part
            fG_Xkh_ = _get_multivariate_normal_pdf(
                x_all_split_each_ck,
                mean_all_split_each_ck,
                cov_all_split_each_ck, 'kh')
            fG_Xkh.append(
                __get_fG_Xkh_each_ck(fG_Xkh_))

            # get fSk_Xk
            idx_result = np.where(np.all(ck[i] == cs, axis=1))[0]
            if idx_result.size == 0:
                fSk_Xk.append(__get_fSk_Xk_each_ck(None))
            elif idx_result.size == 1:
                fSk_Xk.append(__get_fSk_Xk_each_ck(zs[idx_result[0]]))
            elif idx_result.size > 1: #strange
                raise ValueError('ck match cs twice. (strange)')

            # get fS_Xs_dup
            fS_Xs_dup.append(__get_fS_Xs_dup(ck[i], cs, zs))

            # get fG_Xs_gvn_Xkh
            fG_Xs_gvn_Xkh_dup.append(
                __get_fG_Xs_gvn_Xkh_each_ck(ck[i], cs, zs,
                    x_all_split_each_ck,
                    mean_all_split_each_ck,
                    cov_all_split_each_ck))

            fK_Xk.append(__get_pdf_each_ck(i))

        return np.array([fK_Xk]).reshape((-1, 1))
    else: #general knowledge is not gaussian
        pass

def _bme_posterior_moments(
    ck, ch=None, cs=None, zh=None, zs=None,
    covmodel=None, covparam=None, covmat=None,
    order=np.nan, options=None,
    general_knowledge='gaussian',
    #  specific_knowledge='unknown',
    pdfk=None, pdfh=None, pdfs=None, hk_k=None, hk_h=None, hk_s=None,
    gui_args=None, ck_cov_output=False):

    '''
        no neighbour considered, so spatial-temporal range
        should be transform first (no dmax support).

        covmat:
            covariance matrix, a numpy 2d array
            with shape (nk+nh+ns) by (nk+nh+ns)
        if covmat provieded, covmodel and covparam are simply skipped.
    '''

    if general_knowledge == 'gaussian':
        if zs:
            all_zs_type = np.array(
                map(get_standard_soft_pdf_type, [zsi[0] for zsi in zs])
                )
            if (all_zs_type==10).all(): # all soft type are gaussian
                # find cs in ck index
                dup_index = np.where((ck==cs[:,None]).all(-1))[1]
                if dup_index.size > 0: #has duplicated point
                    if not ck_cov_output:
                        mask = np.ones(len(dup_index), dtype=bool)
                        mask[dup_index] = False
                        mvs = np.empty((ck.shape[0],3))
                        mvs[:] = np.nan
                        ck_dup = ck[dup_index, :]
                        ck_no_dup = np.delete(ck, dup_index, axis=0)
                        mvs[dup_index, :] =\
                            _bme_proba_gaussian_dup(
                                ck_dup, ch, cs, zh, zs,
                                covmodel, covparam, covmat, order, options)
                        if ck_no_dup.size > 0:
                            mvs[mask, :] =\
                                _bme_proba_gaussian(
                                    ck_no_dup, ch, cs, zh, zs,
                                    covmodel, covparam, covmat, order, options)
                        return mvs
                    else:
                        raise ValueError('We can not do ck_cov_output with duplicated point.')
                else:
                    if not ck_cov_output:
                        mvs = _bme_proba_gaussian(
                            ck, ch, cs, zh, zs,
                            covmodel, covparam, covmat, order, options)
                        return mvs
                    else:
                        mvs, ckcov = _bme_proba_gaussian(
                            ck, ch, cs, zh, zs,
                            covmodel, covparam, covmat, order, options, ck_cov_output)
                        return mvs, ckcov
            else: # has non-gaussian
                nk = ck.shape[0]
                nh = ch.shape[0] if ch is not None else 0
                ns = cs.shape[0] if cs is not None else 0
                x_all_split = _get_x_all_split(nk, zh, zs)
                mean_all_split = _get_mean_all_split(x_all_split, order)
                if covmat is not None:
                    cov_all_split = np.vsplit(
                        covmat, [nk, nk+nh, nk+nh+ns]
                        )[:-1] #exclude final empty array
                    cov_all_split = \
                        [np.hsplit(c, [nk, nk+nh, nk+nh+ns][:-1])\
                        for c in cov_all_split]
                else:
                    cov_all_split = _get_cov_all_split(
                        ck, ch, cs, covmodel, covparam)

                fg_s_given_h = _get_multivariate_normal_pdf(
                    x_all_split, mean_all_split,
                    cov_all_split, 's_h')
                fs = _get_fs(zs)

                #split hard and soft of m_k_gvn_hs
                m_k = _get_mean(mean_all_split, 'k')
                Bm_sigma_inv_multi = (
                    _get_sigma(cov_all_split, 'k', 'hs').dot(
                        _get_sigma(cov_all_split, 'hs', 'hs', inv=True))
                    )
                m_hs = _get_mean(mean_all_split, 'hs')
                m_k_gvn_hs_part = m_k - Bm_sigma_inv_multi.dot(m_hs)
                x_hs = _get_x(x_all_split, 'hs') #put true Xs later

                sigma_k_given_hs = _get_sigma_a_given_b(
                    cov_all_split, 'k', 'hs')
                diag_sigma_k_given_hs =\
                    np.diag(sigma_k_given_hs)

                def func_moments(x_array):
                    nMon = 3
                    npts = x_array.shape[0]
                    res = np.empty(
                        (npts, nMon*nk + 1)
                        )
                    x_hs_npts = np.tile(x_hs, (1, npts))
                    x_hs_npts[nh:,:] = x_array.T # put true Xs part

                    m_k_gvn_hs_npts = (
                        m_k_gvn_hs_part + Bm_sigma_inv_multi.dot(x_hs_npts)
                        ).T

                    fg_fs = (
                        fg_s_given_h(x_array).reshape((npts,1)) * fs(x_array)
                        )
                    res[:, 0*nk:1*nk] = m_k_gvn_hs_npts * fg_fs
                    res[:, 1*nk:2*nk] = (res[:,:nk] * m_k_gvn_hs_npts)
                    res[:, 2*nk:3*nk] = (
                        3 * diag_sigma_k_given_hs * res[:,:nk]
                        - 2 * res[:,nk:2*nk] * m_k_gvn_hs_npts
                        )
                    res[:,-1:] = fg_fs
                    return res

                ranges = []
                for zsi in zs:
                    pdftype = get_standard_soft_pdf_type(zsi[0])
                    if pdftype in  [1, 2]: # nl, limi, probdens
                        nl = zsi[1]
                        limi = zsi[2]
                        ranges.append((limi[0], limi[nl[0]-1]))
                    elif pdftype in [10]:
                        zm = zsi[1]
                        zstd = np.sqrt(zsi[2])
                        ranges.append((zm-5*zstd, zm+5*zstd))
                ranges = np.array(ranges)
                xmin = ranges[:,0].copy()
                xmax = ranges[:,1].copy()

                Mon, e, info = qmc(
                    func_moments, xmin, xmax, relerr=options[3,0],
                    showinfo=options['qmc_showinfo'])

                Mon_NC = Mon[-1]
                Mon = Mon[:-1].reshape((-1, nk)).T # k by nmon
                Mon /= Mon_NC # for moments 1,2,3
                Mon[:,1:2] += (
                    diag_sigma_k_given_hs.reshape((-1, 1))
                    ) # for moments 2

                M1 = Mon[:, 0:1]
                M2 = Mon[:, 1:2]
                M3 = Mon[:, 2:3]

                mvs = np.empty(Mon.shape)
                mvs[:,0:1] = M1
                mvs[:,1:2] = M2 - M1**2
                mvs[:,2:3] = M3 - 3*M2*M1 + 2*M1**3

                if not ck_cov_output:
                    return mvs
                else:
                    zs_gau = []
                    for zsi in zs:
                        zs_gau_m, zs_gau_v = proba2stat(
                            zsi[0],
                            np.array([zsi[1]]),
                            np.array([zsi[2]]),
                            np.array([zsi[3]])
                            )
                        zs_gau.append((10, zs_gau_m, zs_gau_v))
                    mvs22, ckcov22 = _bme_proba_gaussian(
                        ck, ch, cs, zh, zs_gau,
                        covmodel, covparam, covmat, order, options, ck_cov_output)

                    #do something transform between cov and corr

                    return mvs, ckcov
        else:
            if not ck_cov_output:
                mvs = _bme_proba_gaussian(
                    ck, ch, cs, zh, zs, covmodel, covparam, covmat, order, options)
                return mvs
            else:
                mvs, ckcov = _bme_proba_gaussian(
                    ck, ch, cs, zh, zs,
                    covmodel, covparam, covmat, order, options, ck_cov_output)
                return mvs, ckcov
    else:
        raise ValueError("Now we can not consider non-gaussian GK." )

def _bme_proba_gaussian(
    ck, ch=None, cs=None, zh=None, zs=None,
    covmodel=None, covparam=None, covmat=None,
    order=np.nan, options=None, ck_cov_output=False):
    '''
    no neighbour consider, no data format transform.

    ch, cs, zh: np.2darray or None
    zs: new zs data or None, see softconverter.py for detail.
    ck_cov_output: if True, result will additionally return
        covariance between ck
    NOTE zs there should gaussian type e.g.
        zs = ((10, mean1, var1),...,(10, mean2, var2))
    '''

    nk, nh, ns = __get_khs_size(ck, ch, cs)
    x_all_split, mean_all_split, cov_all_split, covmat =\
        __get_all_split(
            ck, ch, cs, zh, zs, covmodel, covparam, covmat, order)
    
    has_remove_index, remove_index =\
        __get_covmat_remove_index(covmat)

    if has_remove_index: #need change data
        s_remove_index = remove_index-(nk+nh)
        if (s_remove_index >= 0).all():
            cs = np.delete(cs, s_remove_index, axis=0)
            zs = [zs_i for idx, zs_i in enumerate(zs) if idx not in s_remove_index]
            covmat = np.delete(covmat, nk+nh+s_remove_index, axis=0)
            covmat = np.delete(covmat, nk+nh+s_remove_index, axis=1)

            x_all_split, mean_all_split, cov_all_split, covmat =\
                __get_all_split(
                ck, ch, cs, zh, zs, covmodel, covparam, covmat, order)
        else: # need remove k or h, strange
            import pdb
            pdb.set_trace()
    nk, nh, ns = __get_khs_size(ck, ch, cs)

    if ns == 0 and nh == 0:
        mvs = np.empty((ck.shape[0],3))
        mvs[:] = np.nan
        return mvs
        #raise ValueError('hard and soft data can not both without input')

    if ns == 0: # only hard data
        mean_k_given_h = _get_mean_a_given_b(
            x_all_split, mean_all_split,
            cov_all_split, sub_a='k', sub_b='h')
        sigma_k_given_h = _get_sigma_a_given_b(
            cov_all_split, sub_a='k', sub_b='h')

        skewness = np.zeros(mean_k_given_h.shape)
        mvs = np.hstack(
            (mean_k_given_h, sigma_k_given_h.diagonal().reshape((-1,1)),
            skewness)
            )
        if ck_cov_output:
            return mvs, sigma_k_given_h
        else:
            return mvs
    else: # both hard and soft data (hard data can be empty)
        mean_k = _get_mean(mean_all_split, 'k')
        
        #check outlier data
        mean_s_given_h = _get_mean_a_given_b(
            x_all_split, mean_all_split,
            cov_all_split, sub_a='s', sub_b='h')
        sigma_s_given_h = _get_sigma_a_given_b(cov_all_split, 's', 'h')
        general_mean = mean_s_given_h.ravel()
        general_var = np.diag(sigma_s_given_h)
        del sigma_s_given_h
        soft_mean = np.array([zs_i[1] for zs_i in zs])
        soft_var = np.array([zs_i[2] for zs_i in zs])

        problem_bool = (np.abs(general_mean - soft_mean)\
            > 3*(np.sqrt(general_var) + np.sqrt(soft_var))
            )
        if problem_bool.any():
            problem_index = np.where(problem_bool)[0]
            print('warning: the soft data at index(es) {i} '\
                'are far from its general marginal pdf'.format(
                    i=str(problem_index)))
            #remove and re calculate
            problem_index
            cs = np.delete(cs, problem_index, axis=0)
            zs = [zs_i for idx, zs_i in enumerate(zs) if idx not in problem_index]
            covmat = np.delete(covmat, nk+nh+problem_index, axis=0)
            covmat = np.delete(covmat, nk+nh+problem_index, axis=1)

            x_all_split, mean_all_split, cov_all_split, covmat =\
                __get_all_split(
                ck, ch, cs, zh, zs, covmodel, covparam, covmat, order)
            nk, nh, ns = __get_khs_size(ck, ch, cs)

            mean_s_given_h = _get_mean_a_given_b(
                x_all_split, mean_all_split,
                cov_all_split, sub_a='s', sub_b='h')

        mean_hs = _get_mean(mean_all_split, 'hs')

        NC, useful_args = _get_int_fg_a_given_b_fs_s(
            x_all_split, mean_all_split,
            cov_all_split, zs,
            sub_multi = 's_h', sub_s='s'
            )

        NC = __check_normalized_constant(NC, options)
        if NC == 0:
            print('mvs set to NaN.')
            mvs = np.empty((nk,3))
            mvs = np.nan
            return mvs
        if np.isnan(NC) or np.isinf(NC):
            print('NC error... is NaN or Inf.')
            if 1 or options['debug']: #always stop
                import pdb
                pdb.set_trace()

        (sigma_t_prime, inv_sigma_s_given_h,
            inv_sigma_tilde_s, mean_tilde_s,
            alias_c, alias_fgfs1234, alias_mean_d, alias_sigma_d
            ) = useful_args
        hat_x_s = NC * sigma_t_prime.dot(
            inv_sigma_s_given_h.dot(mean_s_given_h)
            + inv_sigma_tilde_s.dot(mean_tilde_s)
            )
        if not nh:
            hat_x_hs = hat_x_s
        else:
            hat_x_h = _get_x(x_all_split, 'h') * NC
            hat_x_hs = np.vstack((hat_x_h, hat_x_s))
        sigma_k_hs = _get_sigma(
            cov_all_split, sub_a='k', sub_b='hs')
        inv_sigma_hs_hs = _get_sigma(
            cov_all_split, sub_a='hs', sub_b='hs', inv=True)
        cond_k_hs = sigma_k_hs.dot(inv_sigma_hs_hs)
        BME_mean_k_given_hs_a = cond_k_hs.dot(hat_x_hs)
        BME_mean_k_given_hs_b = mean_k - cond_k_hs.dot(mean_hs)
        BME_mean_k_given_hs =\
            BME_mean_k_given_hs_a/NC + BME_mean_k_given_hs_b

        sigma_k_given_hs = _get_sigma_a_given_b(
            cov_all_split, sub_a='k', sub_b='hs')
        mean_t = hat_x_hs/NC
        aa = np.zeros(inv_sigma_hs_hs.shape)
        aa[nh:nh+ns, nh:nh+ns] = sigma_t_prime*NC
        bb = (mean_t - mean_hs).dot((mean_t - mean_hs).T) * NC
        tt = cond_k_hs.dot(aa + bb).dot(cond_k_hs.T)
        
        if not ck_cov_output:
            sigma_k_given_hs_diag =\
                sigma_k_given_hs.diagonal().reshape((-1,1))
            tt_diag = tt.diagonal().reshape((-1,1))
            BME_var_k_given_hs = (
                sigma_k_given_hs_diag - BME_mean_k_given_hs**2 + mean_k**2
                - 2*mean_k * cond_k_hs.dot(mean_hs)
                + 2*mean_k * cond_k_hs.dot(hat_x_hs) / NC
                + tt_diag / NC
                )
        else:
            exm = cond_k_hs.dot(hat_x_hs).dot(mean_k.T)
            emm = cond_k_hs.dot(mean_hs).dot(mean_k.T)
            BME_var_k_given_hs_cov = (
                sigma_k_given_hs
                - BME_mean_k_given_hs.dot(BME_mean_k_given_hs.T)
                + mean_k.dot(mean_k.T)
                + 2*exm - 2*emm
                + tt / NC
                )
            BME_var_k_given_hs =\
                BME_var_k_given_hs_cov.diagonal().reshape((-1,1))
            
        skewness = np.zeros(BME_mean_k_given_hs.shape)
        mvs = np.hstack(
            (BME_mean_k_given_hs, BME_var_k_given_hs, skewness)
            )
        if ck_cov_output:
            return mvs, BME_var_k_given_hs_cov
        else:
            return mvs

def _bme_proba_gaussian_dup(
    ck, ch=None, cs=None, zh=None, zs=None,
    covmodel=None, covparam=None, covmat=None,
    order=np.nan, options=None, ck_cov_output=False):

    nk, nh, ns = __get_khs_size(ck, ch, cs)
    x_all_split, mean_all_split, cov_all_split, covmat =\
        __get_all_split(
            ck, ch, cs, zh, zs, covmodel, covparam, covmat, order)

    if ns == 0: #strange, should have ck duplicates with cs
        raise ValueError('no cs, strange.')
    else:
        NC, (sigma_d, inv_sigma_s_given_h,
            inv_sigma_tilde_s, mean_tilde_s,
            alias_c, alias_fgfs1234, alias_mean_d, alias_sigma_d
            ) =\
            _get_int_fg_a_given_b_fs_s(
                x_all_split, mean_all_split, cov_all_split, zs, 
                sub_multi = 's_h', sub_s='s'
                )

        # no used but give a warning   
        #NC = __check_normalized_constant(NC)
        if NC == 0:
            print('warning: NC found to be 0.')
            if options['debug']:
                import pdb
                pdb.set_trace()
        # find ck in cs index
        ck_dup_index, cs_dup_index = np.where((cs==ck[:,None]).all(-1))
        return np.hstack((
            alias_mean_d[cs_dup_index,:],
            np.diagonal(alias_sigma_d).reshape((-1,1))[cs_dup_index,:],
            np.zeros((nk, 1))
            ))

        # NC, (sigma_d, inv_sigma_s_given_h,
        #     inv_sigma_tilde_s, mean_tilde_s,
        #     alias_c, alias_fgfs1234, alias_mean_d, alias_sigma_d
        #     ) =\
        #     _get_int_fg_a_given_b_fs_s(
        #         x_all_split, mean_all_split, cov_all_split, zs, 
        #         sub_multi = 's_h', sub_s='s'
        #         )
        # NC = __check_normalized_constant(NC)
        
        # #get top part
        # #ns x 1
        # top_part2 = NC * alias_mean_d
        # top_part = alias_c * alias_fgfs1234 * alias_mean_d
        # import pdb
        # pdb.set_trace()
        # #get each nc_dup
        # dup_NC = np.ones(top_part.shape)
        # # find ck in cs index
        # ck_dup_index, cs_dup_index = np.where((cs==ck[:,None]).all(-1))
        # for ck_i, cs_i in zip(ck_dup_index, cs_dup_index):
        #     #get x/mean/cov_all_split at each ck by remove dup cs point
        #     x_all_split_each_ck =\
        #         x_all_split[:2]\
        #         + [np.delete(x_all_split[2], cs_i, axis=0)]
        #     mean_all_split_each_ck =\
        #         mean_all_split[:2]\
        #         + [np.delete(mean_all_split[2], cs_i, axis=0)]

        #     covmat_each_ck = np.delete(covmat, nk+nh+cs_i, axis=0)
        #     covmat_each_ck = np.delete(covmat_each_ck, nk+nh+cs_i, axis=1)
           
        #     cov_all_split_each_ck = np.vsplit(
        #         covmat_each_ck, [nk, nk+nh, nk+nh+ns-1]
        #         )[:-1] #exclude final empty array
        #     cov_all_split_each_ck = \
        #         [np.hsplit(c, [nk, nk+nh, nk+nh+ns-1][:-1])\
        #         for c in cov_all_split_each_ck]
        #     zs_each_ck = [z for idx_z, z in enumerate(zs) if idx_z != cs_i]

        #     dup_NC[cs_i, 0] = __check_normalized_constant(
        #         __get_normalized_constant(
        #             x_all_split_each_ck,
        #             mean_all_split_each_ck,
        #             cov_all_split_each_ck, zs_each_ck, 
        #             sub_multi = 's_h', sub_s='s'
        #             )
        #         )
        # return (top_part/dup_NC)[cs_dup_index,:]

def _get_x_all_split(nk, zh, zs):
    '''
        Create the "estimated" observed values 
    for the estimation and observations
        For now, zero is used for estimation points
    (which should be specified as NaN)
        mean values are used for soft data
    '''
    x_all_split = []
    xk = np.empty((nk, 1))  # will be replaced later
    x_all_split.append(xk)
    x_all_split.append(zh) # e.g. xh
    if zs:
        xs = np.empty((len(zs), 1))
        for i, zsi in enumerate(zs):
            if get_standard_soft_pdf_type(zsi[0]) == 10:  # gaussian/normal
                xs[i] = zsi[1] #z_mean
            else:
                xs[i], dummy_v = proba2stat(
                    zsi[0],
                    np.array([zsi[1]]),
                    np.array([zsi[2]]),
                    np.array([zsi[3]])
                    )
        x_all_split.append(xs)
    else:
        x_all_split.append(None)
    return x_all_split

def _get_mean_all_split(x_all_split, order):
    '''
    Obtain the trend estimations at the estimation and data locations based 
    upon the specified trend order
    '''
    if isinstance(order, np.ndarray):  # user defined general knowledge, (row x 1 2d array)
        start_i = 0
        mean_all_split = []
        for i in x_all_split:
            if i is not None:
                end_i = start_i + i.size
                mean_all_split.append(order[start_i:end_i, :])
                start_i = end_i
            else:
                mean_all_split.append(None)   
    elif order == 0:  # constant mean, exclude zk, average h and s
        xx=[x for x in x_all_split[1:] if x is not None]
        constant_mean_ = np.vstack(xx).mean()
        mean_all_split = []
        for i in x_all_split:
            if i is not None:
                mean_all_split.append(np.ones(i.shape)*constant_mean_)
            else:
                mean_all_split.append(None)
        #for means in mean_all_split:
        #  means[:] = constant_mean_
    elif np.isnan(order):  # zero mean: 
        mean_all_split = []
        for i in x_all_split:
            if i is not None:
                mean_all_split.append(np.zeros(i.shape))
            else:
                mean_all_split.append(None)
    return mean_all_split

def _get_cov_all_split(ck, ch, cs, covmodel, covparam):
  '''
  Obtain the covariance in the split ways.
  See coord2K
  '''
  return coord2Ksplit((ck, ch, cs), (ck, ch, cs),
                      covmodel, covparam)[0]

def _get_x(x_all_split, sub):
  '''
  Retrieve the estimated observed values given specified class, i.e., sub
  sub can be k, h, and s for estimation, hard, and soft data
  '''
  idx = [KHS_DICT[i] for i in sub]
  output=[x_all_split[i] for i in idx if x_all_split[i] is not None]
  if len(output)>0:
    return np.vstack(output)
  else:
    return None

def _get_mean(mean_all_split, sub):
  '''
  Retrieve the expected values given specified class, i.e., sub
  sub can be k, h, and s for estimation, hard, and soft data
  '''
  idx = [KHS_DICT[i] for i in sub]     
  output=[mean_all_split[i] for i in idx if mean_all_split[i] is not None]
  if len(output)>0:
    return np.vstack(output)
  else:
    return None

def _get_sigma(cov_all_split, sub_a, sub_b, inv=False):
  '''
  Retrieve the cross-covaiance between specified class, i.e., sub_a and sub_b
  sub_a and sub_b can be k, h, and s for estimation, hard, and soft data
  '''
  idx_a = [KHS_DICT[i] for i in sub_a]
  idx_b = [KHS_DICT[i] for i in sub_b]

  cov_a_b = []
  for i in idx_a:
    output=[cov_all_split[i][j] for j in idx_b if cov_all_split[i][j] is not None]
    if len(output)>0:
      cov_a_b.append(np.hstack(output))
    else:
      cov_a_b.append(None)
  if len(cov_a_b)>1:    
    output2=[x for x in cov_a_b if x is not None]   
    cov_a_b = np.vstack(output2)
  else:
    cov_a_b = np.vstack(cov_a_b)
  if not inv:
    return cov_a_b
  else:
    return np.linalg.pinv(cov_a_b)

def _get_mean_a_given_b(x_all_split, mean_all_split,
    cov_all_split, sub_a, sub_b):
    '''
    Obtain the conditonal mean a given b by using conditonal Gaussian formula
    '''      

    if 'k' not in sub_b:                    
        x_b = _get_x(x_all_split, sub_b)
        mean_a = _get_mean(mean_all_split, sub_a)
        mean_b = _get_mean(mean_all_split, sub_b)
        if mean_b is not None: # consider the case that data in sub_b does not exist
            sigma_a_b = _get_sigma(cov_all_split, sub_a, sub_b)
            inv_sigma_b_b = _get_sigma(cov_all_split, sub_b, sub_b, inv=True)
            output=mean_a + sigma_a_b.dot(inv_sigma_b_b).dot(x_b - mean_b)
        else:
           output=mean_a
        return output
    else:
        nlim=np.asarray(x_all_split[0]).size
        smtx=np.ones((1,nlim))
        if nlim>1:
            idx = [KHS_DICT[i] for i in sub_b if i is not 'k']
            xhs=np.vstack([x_all_split[i] for i in idx])
            x_b=[x_all_split[0].reshape((1,nlim)),xhs.dot(smtx)]
            x_b=np.vstack(x_b)
        else:            
           x_b = _get_x(x_all_split, sub_b)
        
        mean_a = _get_mean(mean_all_split, sub_a)
        mean_b = _get_mean(mean_all_split, sub_b)
        if mean_b is not None: # consider the case that data in sub_b does not exist
            sigma_a_b = _get_sigma(cov_all_split, sub_a, sub_b)
            inv_sigma_b_b = _get_sigma(cov_all_split, sub_b, sub_b, inv=True)
            output=mean_a + sigma_a_b.dot(inv_sigma_b_b).dot(x_b - mean_b)
        else:
            output=mean_a
        return output

def _get_sigma_a_given_b(cov_all_split, sub_a, sub_b):
  '''
  Obtain the conditional covariance a given b by using conditonal Gaussian 
  formula
  '''
  sigma_a_a = _get_sigma(cov_all_split, sub_a, sub_a)
  sigma_a_b = _get_sigma(cov_all_split, sub_a, sub_b)
  if sigma_a_b.size > 0:
    inv_sigma_b_b = _get_sigma(cov_all_split, sub_b, sub_b, inv=True)
    sigma_b_a = _get_sigma(cov_all_split, sub_b, sub_a)
    return sigma_a_a - sigma_a_b.dot(inv_sigma_b_b).dot(sigma_b_a)
  else:
    return sigma_a_a

def _get_multivariate_normal_pdf(x_all_split, mean_all_split,
    cov_all_split, sub_multi):
    '''
    Obtain multivariate Gaussian pdf or conditional multivariate Gaussian
    based upon the specified notations, i.e., sub_multi
    
    Note:
    sub_multi     string    h, s, and k for hard, soft and estimation locations
                            a_b represents a given b, e.g., k_h
    '''                                   
    if "_" in sub_multi:  # "given" type
        sub_a, sub_b = sub_multi.split('_')#x_sub.split('_') (temporarily change by HL)
        m = _get_mean_a_given_b(x_all_split, mean_all_split,
                                cov_all_split, sub_a, sub_b)
        v = _get_sigma_a_given_b(cov_all_split, sub_a, sub_b)
    else:  # single_sub
        sub_a = sub_multi
        m = _get_mean(mean_all_split, sub_a)
        v = _get_sigma(cov_all_split, sub_a, sub_a)
    return scipy.stats.multivariate_normal(m.T[0], v).pdf

def _get_fs(zs):
    '''
    the product of fs distributions
    x   ndim(e.g. npts) by ns
            x is a 2-D np array with the dimension of 
            ndim(number of samples at each integral) 
            by ns (the number of integrals, i.e., number 
            of soft data)      
    '''
    def fs(x):
        res = np.ones((x.shape[0],1))
        for idx_k, zsi in enumerate(zs):
            pdf_type = get_standard_soft_pdf_type(zsi[0])
            if pdf_type == 2:
                nl = zsi[1][0]
                limi = zsi[2]
                probdens = zsi[3]
                y_i = np.interp(
                    x[:,idx_k:idx_k+1], limi[:nl], probdens[:nl],
                    left = 0., right = 0.)
            elif pdf_type == 1:
                nl = zs[1]
                limi = zs[2]
                probdens = zs[3]
                idd = np.where(x[:, idx_k:idx_k+1] - limi[:nl] > 0)[0]
                y_i = probdens[idd]
            elif pdf_type == 10:
                zm = zsi[1]
                zstd = np.sqrt(zsi[2])
                try:
                    y_i = scipy.stats.norm.pdf(
                        x[:,idx_k:idx_k+1], loc=zm, scale=zstd)
                except FloatingPointError:
                    import pdb
                    pdb.set_trace()
              
            if not (y_i.all() or y_i.any()):
                return np.zeros((x.shape[0], 1))
            else:
                res *= y_i
        return res
    return fs

def _get_Fsinv(zs):

    Fs = pdf2cdf(zs)
    def Fsinv(x):  
        res = np.zeros(x.shape)
        for idx_k, Fsi in enumerate(Fs):
            pdf_type = get_standard_soft_pdf_type(Fsi[0])
            if pdf_type == 2:
                nl = Fsi[1][0]
                limi = Fsi[2]
                probdens = Fsi[3]
                probCDFs = Fsi[4]

                alpha = np.diff(probdens) / np.diff(limi)
                i = np.searchsorted(probCDFs[1:],x[:,idx_k:idx_k+1])
                D = np.abs(probdens[i]**2 + 2*alpha[i] * (x[:,idx_k:idx_k+1] - probCDFs[i]))
                y_i = limi[i] + (-probdens[i] + np.sqrt(D)) / alpha[i]
                
            res[:,idx_k:idx_k+1] = y_i
        return res
    return Fsinv

def _get_int_fg_a_given_b_fs_s(x_all_split, mean_all_split,
  cov_all_split, zs, sub_multi, sub_s='s'):
  '''
  The upper right part and lower right part of the last row of formula (1)
  The evaluation is based upon Eqns. (8) or (9) in the cases of s_h and s_kh 
  respectively
  '''

  #fg='s_kh', fs='s'
  sub_a, sub_b = sub_multi.split('_')
  sigma_a_given_b = _get_sigma_a_given_b(cov_all_split, sub_a, sub_b)
  try:
      inv_sigma_a_given_b = np.linalg.pinv(sigma_a_given_b)
  except np.linalg.LinAlgError as e:
      import pdb
      pdb.set_trace()
      raise e
  # mean_tilde_s = zs[1]  # mean
  # sigma_tilde_s = np.diag(zs[2].T[0])  # cov matrix
  mean_tilde_s = []
  sigma_tilde_s = []
  for zsi in zs:
      mean_tilde_s.append([zsi[1]])
      sigma_tilde_s.append(zsi[2])
  mean_tilde_s = np.array(mean_tilde_s) # mean
  sigma_tilde_s = np.diag(sigma_tilde_s)  # cov matrix
  try:
      inv_sigma_tilde_s = np.linalg.pinv(sigma_tilde_s)
  except np.linalg.LinAlgError as e:
      import pdb
      pdb.set_trace()
      raise e

  sigma_t = np.linalg.pinv(inv_sigma_a_given_b + inv_sigma_tilde_s)
  det_sigma_t = np.linalg.det(sigma_t)
  det_sigma_a_given_b = np.linalg.det(sigma_a_given_b)
  det_sigma_tilde_s = np.linalg.det(sigma_tilde_s)
  ns = mean_tilde_s.shape[0]
  alias_c = (
    np.sqrt(det_sigma_t)
    / np.sqrt(det_sigma_a_given_b * det_sigma_tilde_s)
    )
  fgfs_front = alias_c / (2*np.pi)**(ns/2.)
  alias_c = fgfs_front
  # fgfs_front = np.sqrt(det_sigma_t) /\
  #       ((2*np.pi)**(ns/2.) *
  #        np.sqrt(det_sigma_a_given_b * det_sigma_tilde_s)
  #        )

  mean_a_given_b = _get_mean_a_given_b(
        x_all_split, mean_all_split, cov_all_split, sub_a, sub_b)
  fgfs_1 = np.diag((mean_a_given_b.T).dot(
        inv_sigma_a_given_b).dot(mean_a_given_b))
  fgfs_2 = (mean_tilde_s.T).dot(inv_sigma_tilde_s).dot(mean_tilde_s)
  fgfs_3 = (mean_a_given_b.T).dot(inv_sigma_a_given_b) +\
        (mean_tilde_s.T).dot(inv_sigma_tilde_s)
  fgfs_4 = inv_sigma_tilde_s.dot(mean_tilde_s) +\
        inv_sigma_a_given_b.dot(mean_a_given_b)
  alias_sigma_d = sigma_t
  alias_mean_d = alias_sigma_d.dot(fgfs_4)
  # 1 x 1 array (scalar)
  alias_fgfs1234 = np.asscalar(
      (fgfs_1 + fgfs_2 - np.diag(fgfs_3.dot(sigma_t).dot(fgfs_4)))
      )
  fgfs_end = np.exp((-1/2.) * alias_fgfs1234)
  alias_fgfs1234 = fgfs_end
  # fgfs_end = np.exp(
  #       (-1/2.) * (fgfs_1 + fgfs_2 - np.diag(fgfs_3.dot(sigma_t).dot(fgfs_4))))

  # will check later
  # if fgfs_front * fgfs_end == 0:
  #       import pdb
  #       pdb.set_trace()

  return fgfs_front * fgfs_end, \
        (sigma_t, inv_sigma_a_given_b, inv_sigma_tilde_s, mean_tilde_s,
            alias_c, alias_fgfs1234, alias_mean_d, alias_sigma_d)

def _changetimeform(ck,ch=None,cs=None):
  '''
  Change the time format into float while it is in datetime format
  '''  
  
  if type(ck[0,-1])==np.datetime64:
    origin=ck[0,-1]
    ck[:,-1]=np.double(np.asarray(ck[:,-1],dtype='datetime64')-origin)
    ck=ck.astype(np.double)
    if ch is not None and ch.size>0:
      if (not type(ch[0,-1]==np.datetime64)):
        print('Time format of ch is not consistent with ck (np.datetime64)')
        raise
      ch[:,-1]=np.double(np.asarray(ch[:,-1],dtype='datetime64')-origin)
      ch=ch.astype(np.double)
    if cs is not None and cs.size>0:
      if (not type(cs[0,-1]==np.datetime64)):
        print('Time format of cs is not consistent with ck (np.datetime64)')
        raise
      cs[:,-1]=np.double(np.asarray(cs[:,-1],dtype='datetime64')-origin)
      cs=cs.astype(np.double)
      
  return ck,ch,cs

def _set_nh_ns(ck,ch,cs,nhmax,nsmax,dmax):
  '''
  Set the size of nhmax and nsmax that limits the size of matrix to be allocated
  it can be important for an efficient S/T estimation
  '''
  if dmax is not None and np.all(dmax):
    nhmax = int(nhmax)
    nsmax = int(nsmax)
    dmax = np.array(dmax,ndmin=2)
    return nhmax,nsmax,dmax
  
  if ck[0,:].size<3:
    if dmax is None:
      dmax_=0
      if ch is not None:
        ch=np.array(ch,ndmin=2)
        maxd_h=pdist(ch).max()
        dmax_=np.max([dmax_,maxd_h])
      if cs is not None:
        cs=np.array(cs,ndmin=2)
        maxd_s=pdist(cs).max()
        dmax_=np.max([dmax_,maxd_s])
      dmax=np.array(dmax_).reshape(1,1)

    if nhmax is None:
      if ch is not None:
        nhmax=ch.shape[0]
      else:
        nhmax=0

    if nsmax is None:
      if cs is not None:
        nsmax=cs.shape[0]
      else:
        nsmax=0

    
  else:
    maxd=0
    maxt=0
    if dmax is None:
      if ch is not None:
        dummy=np.random.rand(ch.shape[0],1)
        _,cMS_h,tME_h,_=valstv2stg(ch,dummy)
        if nhmax is None:
          nhmax=cMS_h.shape[0]*3
        maxd_h=pdist(cMS_h).max()
        maxt_h=pdist(tME_h.reshape((tME_h.size,1))).max()
      else:
        maxd_h=0
        maxt_h=0
        nhmax=0
      maxd=np.max([maxd_h,maxd]) 
      maxt=np.max([maxt_h,maxt])
      if cs is not None: 
        dummy=np.random.rand(cs.shape[0],1)
        _,cMS_s,tME_s,_=valstv2stg(cs,dummy)
        if nsmax is None:
          if zs[0]==10 or zs[0] is 'gaussian':
            nsmax=cMS_s.shape[0]*3
          else:
            nsmax=3
        maxd_s=pdist(cMS_s).max()
        maxt_s=pdist(tME_s.reshape((tME_s.size,1))).max()
        maxd=np.max([maxd_s,maxd])
        maxt=np.max([maxt_s,maxt])
      else:
        nsmax=0
        maxd_s=0
        maxt_s=0
      maxd=np.max([maxd_s,maxd])
      maxt=np.max([maxt_s,maxt])
    dmax=np.array([maxd,maxt,np.nan]).reshape(1,3)

  return nhmax,nsmax,dmax

def _stratio(covparam):
  '''
  Estimate the S/T ratio for dmax
  '''
  nm=len(covparam)
  sills= np.array([covparam[k][0] for k in range(nm)])  
  hrange = np.array([covparam[k][1][0] for k in range(nm)]) 
  idx0 = np.where([hrange[k] is not None for k in range(nm)])[0]
  idx=np.where(sills[idx0]==sills.max())[0]
  ratio=covparam[idx0[idx]][1][0]/covparam[idx0[idx]][2][0]
  return ratio

def _bme_posterior_prepare(
    ck, ch=None, cs=None, zh=None, zs=None,
    covmodel=None, covparam=None, covmat=None,
    order=np.nan, options=None,
    nhmax=None, nsmax=None, dmax=None,
    general_knowledge='gaussian',
    #  specific_knowledge='unknown',  
    pdfk=None,pdfh=None,pdfs=None,hk_k=None,hk_h=None,hk_s=None,
    gui_args=None):

    '''
    check and configure arguments and 
        find neighbor ckhs index for bme posterior calculation

    ckhs_idx_list:  [ck_idx, ch_idx, cs_idx] represents
        these ck have the same neighbors ch and cs

    return (output_arguments, configured_arguments):
        a tuple contain arguments
    '''
    print('preparing...', end='')
    if covmat is None:
        if (covmodel is None) or (covparam is None):
            raise ValueError(
                'Covariance model and their associated parameters '\
                'should be specified if no covarinace matrix provided.')


    dk = ck.shape[1]
    nk = ck.shape[0]
    nh = ch.shape[0] if ch is not None else 0
    ns = cs.shape[0] if cs is not None else 0

    ck, ch, cs = _changetimeform(ck, ch, cs)
    nhmax, nsmax, dmax = _set_nh_ns(ck, ch, cs, nhmax, nsmax, dmax)
    if dmax.size == 3 and np.isnan(dmax[0][2]):
        dmax[0][2] = _stratio(covparam)
    stratio = dmax[0][2] if dk == 3 else 1.
      
    if options is None:
        options = BMEoptions()

    if gui_args:
        qpgd = gui_args[0]    
    
    if general_knowledge == 'gaussian':
        order = get_standard_order(order)

        if covmat is not None: #consider covmat if exists, not distance, no need to calculate distance
            ckhs_idx_list = []
            if nh != 0:
                covmat_k_h = covmat[:nk, nk:nk+nh] # slice cov(k x h)
                # sort from big to small, clip with nhmax
                k_by_h_idx = (-covmat_k_h).argsort(axis=1)[:, :nhmax]
                # sort h index (inplace)
                k_by_h_idx.sort(axis=1)
                # make ch_ck_dict
                ch_ck_dict = {}
                for k_idx, h_idx in enumerate(k_by_h_idx):
                    tuple_h_idx = tuple(h_idx)
                    if tuple_h_idx not in ch_ck_dict.keys():
                        k_idx_mul = np.where(
                            np.all(h_idx == k_by_h_idx, axis=1)
                            )[0]
                        ch_ck_dict[tuple_h_idx] = list(k_idx_mul)
                    else:
                        continue #skip duplicated row
                if ns != 0:
                    # slice cov(k x s)
                    covmat_k_s = covmat[:nk, nk+nh:nk+nh+ns]
                    for ch_idx, ck_idx in iteritems(ch_ck_dict):
                        picked_covmat_k_s = covmat_k_s[ck_idx, :]
                        # sort from big to small, clip with nsmax
                        picked_k_by_s_idx =\
                            (-picked_covmat_k_s).argsort(axis=1)[:, :nsmax]
                        # sort s index (inplace)
                        picked_k_by_s_idx.sort(axis=1)
                        # make ch_ck_dict
                        cs_ck_dict = {}
                        for picked_k_idx, s_idx in enumerate(picked_k_by_s_idx):
                            tuple_s_idx = tuple(s_idx)
                            if tuple_s_idx not in cs_ck_dict.keys():
                                picked_k_idx_mul = np.where(
                                    np.all(s_idx == picked_k_by_s_idx, axis=1)
                                    )[0]
                                cs_ck_dict[tuple_s_idx] = list(picked_k_idx_mul)
                            else:
                                continue #skip duplicated row
                        for cs_idx, ck2_idx in iteritems(cs_ck_dict):
                            ck_idx = np.array(ck_idx)
                            ckhs_idx_list.append(
                                [ck_idx[ck2_idx,], ch_idx, cs_idx]
                                )
                else: # ns = 0
                    for ch_idx, ck_idx in iteritems(ch_ck_dict):
                        ckhs_idx_list.append([ck_idx, ch_idx, ()])
            elif nh == 0 and ns != 0: # nh = 0, ns != 0
                covmat_k_s = covmat[:nk, nk+nh:nk+nh+ns] # slice cov(k x s)
                # sort from big to small, clip with nsmax
                k_by_s_idx = (-covmat_k_s).argsort(axis=1)[:, :nsmax]
                # sort s index (inplace)
                k_by_s_idx.sort(axis=1)
                # make cs_ck_dict
                cs_ck_dict = {}
                for k_idx, s_idx in enumerate(k_by_s_idx):
                    tuple_s_idx = tuple(s_idx)
                    if tuple_s_idx not in cs_ck_dict.keys():
                        k_idx_mul = np.where(
                            np.all(s_idx == k_by_s_idx, axis=1)
                            )[0]
                        cs_ck_dict[tuple_s_idx] = list(k_idx_mul)
                    else:
                        continue #skip duplicated row
                for cs_idx, ck_idx in iteritems(cs_ck_dict):
                    ckhs_idx_list.append([ck_idx, (), cs_idx])
            else: # nh = 0, ns = 0
                raise ValueError("nh and ns shouldn't be both 0.")
        else:
            #aggregate ck for same hard data and soft data
            # chs = np.vstack(ch, cs)
            ck_norm = np.copy(ck)
            ck_norm[:, -1] = ck_norm[:, -1] * stratio
            if dk == 3:
                dmax_norm = (dmax[0][0]**2 + (dmax[0][1] * stratio)**2)**0.5
            else:
                dmax_norm = dmax[0][0]

            if isinstance(ch, np.ndarray) and nhmax != 0:
                ch_norm = np.copy(ch)
                ch_norm[:, -1] = ch_norm[:, -1] * stratio
                ch_tree = cKDTree(ch_norm)
            if isinstance(cs, np.ndarray) and nsmax != 0:
                cs_norm = np.copy(cs)
                cs_norm[:, -1] = cs_norm[:, -1] * stratio
                cs_tree = cKDTree(cs_norm)

            ckhs_idx_list = []
            if isinstance(ch, np.ndarray) and nhmax != 0: #has harddata
                ch_ck_dict =\
                    neighbours_index_kd(ck_norm, ch_tree, nhmax, dmax_norm)
                for ch_idx, ck_idx in iteritems(ch_ck_dict):
                    if isinstance(cs, np.ndarray) and nsmax != 0: #both hard and soft
                        picked_ck_norm = ck_norm[ck_idx, :]
                        cs_ck_dict =\
                            neighbours_index_kd(
                                picked_ck_norm, cs_tree, nsmax, dmax_norm
                                )
                        for cs_idx, ck2_idx in iteritems(cs_ck_dict):
                            ck_idx = np.array(ck_idx)
                            ckhs_idx_list.append(
                                [ck_idx[ck2_idx,], ch_idx, cs_idx]
                                )
                    else: #only harddata
                        ckhs_idx_list.append([ck_idx, ch_idx, ()])
            elif isinstance(cs, np.ndarray) and nsmax != 0: #only softdata
                cs_ck_dict =\
                    neighbours_index_kd(ck_norm, cs_tree, nsmax, dmax_norm)
                for cs_idx, ck_idx in iteritems(cs_ck_dict):
                    ckhs_idx_list.append([ck_idx, (), cs_idx])
    else:
        raise ValueError("Now we can not consider non-gaussian GK." )

    configured_arguments =\
        (ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, options,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
        gui_args)
    output_arguments = \
        (ckhs_idx_list,)
    print('done')
    return (output_arguments, configured_arguments)

def __get_khs_size(k,h,s):
    if k.shape[0] == 0:
        raise ValueError('ck can not be empty.')
    else:
        return map(__get_coord_size,[k, h, s])

def __get_coord_size(c):
    n = c.shape[0] if c is not None else 0
    return n

def __get_all_split(ck, ch, cs, zh, zs, covmodel, covparam, covmat, order):
    nk, nh, ns = __get_khs_size(ck, ch, cs)
    x_all_split = _get_x_all_split(nk, zh, zs)
    mean_all_split = _get_mean_all_split(x_all_split, order)
    if covmat is not None:
        cov_all_split = np.vsplit(
            covmat, [nk, nk+nh, nk+nh+ns]
            )[:-1] #exclude final empty array
        cov_all_split = \
            [np.hsplit(c, [nk, nk+nh, nk+nh+ns][:-1])\
            for c in cov_all_split]
    else:
        cov_all_split =\
            _get_cov_all_split(ck, ch, cs, covmodel, covparam)
        covmat = coord2Kcombine(cov_all_split)
    return x_all_split, mean_all_split, cov_all_split, covmat

def __get_normalized_constant(
    x_all_split, mean_all_split, cov_all_split, zs, 
    sub_multi = 's_h', sub_s='s'):

    NC, useful_args = _get_int_fg_a_given_b_fs_s(
        x_all_split, mean_all_split,
        cov_all_split, zs, 
        sub_multi = 's_h', sub_s='s'
        )
    return NC

def __get_covmat_remove_index(covmat):
    if np.linalg.matrix_rank(covmat) < covmat.shape[0]:
        # print('warning: matrix rank less than covmat row')
        has_remove_index = True
        diff_n = covmat.shape[0] - np.linalg.matrix_rank(covmat)
        corrmat = np.corrcoef(covmat)
        remove_index = np.vstack(
            np.unravel_index(
                (np.abs(corrmat).ravel()).argsort(),
                covmat.shape
                )
            ).T
        remove_index = remove_index[
            remove_index[:, 0] < remove_index[:, 1]
            ]
        remove_index_res = np.unique(remove_index[:, 1][-diff_n:])
        remove_index_len = remove_index_res.size
        i = 1
        while remove_index_len < diff_n:
            remove_index_res = np.unique(remove_index[:, 1][-(diff_n+i):])
            remove_index_len = remove_index_res.size
            i += 1
        # print('warning: remove_index_res:', remove_index_res)
    else:
        has_remove_index = False
        remove_index_res = np.array([])
    return has_remove_index, remove_index_res

def __check_normalized_constant(NC, options):
    if NC == 0:
        print('Warning NC is equals to zero.')
        if options['debug']:
            import pdb
            pdb.set_trace()
    elif np.isnan(NC):
        print('NC is equals to NaN.')
        if options['debug']:
            import pdb
            pdb.set_trace()
    elif np.isinf(NC):
        print('NC is equals to Inf.')
        if options['debug']:
            import pdb
            pdb.set_trace()
    return NC

def BMEPosteriorMoments(
    ck, ch=None, cs=None, zh=None, zs=None,
    covmodel=None, covparam=None, covmat=None,
    order=np.nan, options=None,
    nhmax=None, nsmax=None, dmax=None,
    general_knowledge='gaussian',
    #  specific_knowledge='unknown',  
    pdfk=None,pdfh=None,pdfs=None,hk_k=None,hk_h=None,hk_s=None,
    gui_args=None, ck_cov_output=False):
    '''
    ck: n by d numpy 2d array
        the estimated data coordinate, usually, d = 3 for spatial-temporal
        coordinate, first two column for spatial, e.g. x, y, and last column
        for temporal, e.g. t.

    ch: n by d numpy 2d array
        the hard data coordinate.

    cs: n by d numpy 2d array
        the soft data coordinate.

    zh: n by 1 numpy 2d array
        the hard data measurement.

    zs: a sequence of soft data record, e.g. (zs1, zs2, zs3, ..., zsn)
        each zsi(i=1~n) is a sequence of data arguments,
        first item should be softpdftype to determind
        the other rest arguments format, e.g.
        syntax: (softdata_type, *softdata_args)
            zs1 = (1, nl, limi, probdens)
            zs2 = (10, zm, zstd)
        if element is emtpy, put None
            e.g. (zs1, zs2, None, ..., zsn)

    covmodel: 

    covmat:
        covariance matrix, a np 2d array
        with shape (nk+nh+ns) by (nk+nh+ns)
    if covmat provieded, covmodel and covparam are simply skipped.
    
    #SI = integrate (fg_s_given_kh * fs_s) dx_s
    #NC = integrate (fg_s_given_h * fs_s) dx_s
    #pdf_k = (fg_kh * SI) / (fg_h * NC)  # eq.1
    #exp_k = ... # eq.2
    #exp_kp = ... # eq.3
    #if general_knowledge == gaussian and specific_knowledge == unknown
    #exp_k = ... # eq.4
    #var_k = ... # eq.5

    gui_args: a tuple with gui arguments

    
    return 
    '''
    (output_arguments, configured_arguments) = _bme_posterior_prepare(
        ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, options,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
        gui_args)
    (ckhs_idx_list,) = output_arguments

    (ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, options,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
        gui_args) = configured_arguments

    if isinstance(order, np.ndarray):
        has_user_defined_general_knowledge = True
    elif order == 0 or np.isnan(order):
        has_user_defined_general_knowledge = False
    else:
        raise ValueError('order type error')

    nk = ck.shape[0]
    nh = ch.shape[0] if ch is not None else 0
    ns = cs.shape[0] if cs is not None else 0
    zk = np.empty((nk,3)) #to 3rd moments
    cur_cnt = 0
    cum_cnt = 0
    
    if general_knowledge == 'gaussian':
        for ck_idx, ch_idx, cs_idx in ckhs_idx_list:
            ck_idx = np.array(ck_idx, dtype=int)
            ch_idx = np.array(ch_idx, dtype=int)
            cs_idx = np.array(cs_idx, dtype=int)
            picked_ch =\
                ch[ch_idx, :] if isinstance(ch, np.ndarray) else None
            picked_cs =\
                cs[cs_idx, :] if isinstance(cs, np.ndarray) else None
            picked_zh =\
                zh[ch_idx, :] if isinstance(zh, np.ndarray) else None
            picked_zs =\
                [zs[cs_idx_i] for cs_idx_i in cs_idx] if zs is not None else None
            ck_count = ck_idx.shape[0]
            split_count = np.ceil(ck_count/250.)
            for ck_idx_piece in np.array_split(ck_idx, split_count):
                picked_ck = ck[ck_idx_piece, :]
                if covmat is not None:
                    covidx = np.hstack(
                        (ck_idx_piece, nk+ch_idx, nk+nh+cs_idx)
                        )
                    picked_covmat = covmat[np.ix_(covidx, covidx)]
                else:
                    picked_covmat = covmat

                #picked order for specified general knowledge
                if has_user_defined_general_knowledge:
                    order_idx = np.hstack(
                        (ck_idx_piece, nk+ch_idx, nk+nh+cs_idx)
                        )
                    picked_order = order[order_idx, :]
                else:
                    picked_order = order

                try:
                    picked_mvs = _bme_posterior_moments(
                        picked_ck, picked_ch, picked_cs,
                        picked_zh, picked_zs,
                        covmodel, covparam, picked_covmat,
                        picked_order, options, general_knowledge,
                        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
                        gui_args, ck_cov_output)
                except np.linalg.LinAlgError as e:
                    import pdb
                    pdb.set_trace()
                    raise e

                
                zk[ck_idx_piece, :] = picked_mvs
                if gui_args:
                    qpgd = gui_args[0]
                    if qpgd.wasCanceled(): #cancel by user
                        return False
                    else:
                        qpgd.setValue(qpgd.value()+ck_idx_piece.size)
                else:
                    cur_cnt += ck_idx_piece.size
                    if cur_cnt - cum_cnt >= 2000:
                        print(cur_cnt, '/', nk)
                        cum_cnt = cur_cnt
        print(cur_cnt, '/', nk)
        return zk
    else:
      nk=len(pdfk)
      moments=np.empty((nk,3))

      for k in range(nk):
        print('BME MOMENTS:' + str(k+1) + '/' + str(nk))
        
        cklocal=ck[k:k+1,:]
        pdfk_local=[pdfk[k]]
        hk_k_local=[hk_k[k]]
        
        pdf_k=BMEPosteriorPDF(cklocal, ch, cs, zh, zs, covmodel, covparam,
              order, options, nhmax, nsmax, dmax, general_knowledge,
              pdfk=pdfk_local,pdfh=pdfh,pdfs=pdfs,
              hk_k=hk_k_local,hk_h=hk_h,hk_s=hk_s)[0]
          
        zmin=hk_k[k][0]-6*np.sqrt(hk_k[k][1])
        zmax=hk_k[k][0]+6*np.sqrt(hk_k[k][1])
        
        xxx=np.linspace(zmin,zmax,100)
        aaa=pdf_k(xxx,0)

        maxpts = options[2][0]
        aEps = 0
        rEps = options[3][0]

        from cubature import cubature

        mon1_for_cubature = lambda x_array: x_array[:,0] * pdf_k(x_array[:,0],0)[:,0]
        mon1,mon1_err = cubature(
            func=mon1_for_cubature, ndim=1, fdim=1, xmin=np.array([zmin]),
            xmax=np.array([zmax]), adaptive='h', maxEval = maxpts,
            abserr = 0, relerr = rEps, vectorized = True)
        mon2_for_cubature = lambda x_array: x_array[:,0]**2 * pdf_k(x_array[:,0],0)[:,0]
        mon2,mon2_err = cubature(
            func=mon2_for_cubature,ndim=1, fdim=1, xmin=np.array([zmin]),
            xmax=np.array([zmax]), adaptive='h', maxEval = maxpts,
            abserr = 0, relerr = rEps, vectorized = True)  
        mon3_for_cubature = lambda x_array: x_array[:,0]**3 * pdf_k(x_array[:,0],0)[:,0]
        mon3,mon3_err = cubature(
            func=mon3_for_cubature,ndim=1, fdim=1, xmin=np.array([zmin]),
            xmax=np.array([zmax]), adaptive='h', maxEval = maxpts,
            abserr = 0, relerr = rEps, vectorized = True)

        moments[k,0]=mon1
        moments[k,1]=mon2-mon1**2
        moments[k,2]=mon3-3*mon1*mon2-mon1**3

      return moments[:,0], moments[:,1], moments[:,2]

def BMEPosteriorPDF(
    ck, ch=None, cs=None, zh=None, zs=None,
    covmodel=None, covparam=None, covmat=None,
    order=np.nan, options=None,
    nhmax=None, nsmax=None, dmax=None,
    general_knowledge='gaussian',
    #  specific_knowledge='unknown',  
    pdfk=None,pdfh=None,pdfs=None,hk_k=None,hk_h=None,hk_s=None,
    gui_args=None):

    (output_arguments, configured_arguments) = _bme_posterior_prepare(
        ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, options,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
        gui_args)
    (ckhs_idx_list,) = output_arguments

    (ck, ch, cs, zh, zs,
        covmodel, covparam, covmat,
        order, options,
        nhmax, nsmax, dmax,
        general_knowledge,
        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
        gui_args) = configured_arguments
   
    if isinstance(order, np.ndarray):
        has_user_defined_general_knowledge = True
    elif order == 0 or np.isnan(order):
        has_user_defined_general_knowledge = False
    else:
        raise ValueError('order type error')
    nk = ck.shape[0]
    nh = ch.shape[0] if ch is not None else 0
    ns = cs.shape[0] if cs is not None else 0
    zk = np.empty((ck.shape[0],1), dtype=object) # to 1 pdf function
    if general_knowledge == 'gaussian':
        for ck_idx, ch_idx, cs_idx in ckhs_idx_list:
            ck_idx = np.array(ck_idx, dtype=int)
            ch_idx = np.array(ch_idx, dtype=int)
            cs_idx = np.array(cs_idx, dtype=int)
            picked_ch =\
                ch[ch_idx, :] if isinstance(ch, np.ndarray) else None
            picked_cs =\
                cs[cs_idx, :] if isinstance(cs, np.ndarray) else None
            picked_zh =\
                zh[ch_idx, :] if isinstance(zh, np.ndarray) else None
            picked_zs =\
                [zs[cs_idx_i] for cs_idx_i in cs_idx] if zs is not None else None
            ck_count = ck_idx.shape[0]
            split_count = np.ceil(ck_count/250.)
            for ck_idx_piece in np.array_split(ck_idx, split_count):
                picked_ck = ck[ck_idx_piece, :]
                if covmat is not None:
                    covidx = np.hstack(
                        (ck_idx_piece, nk+ch_idx, nk+nh+cs_idx)
                        )
                    picked_covmat = covmat[np.ix_(covidx, covidx)]
                else:
                    picked_covmat = covmat

                #picked order for specified general knowledge
                if has_user_defined_general_knowledge:
                    order_idx = np.hstack(
                        (ck_idx_piece, nk+ch_idx, nk+nh+cs_idx)
                        )
                    picked_order = order[order_idx, :]
                else:
                    picked_order = order

                try:
                    picked_mvs = _bme_posterior_pdf(
                        picked_ck, picked_ch, picked_cs,
                        picked_zh, picked_zs,
                        covmodel, covparam, picked_covmat,
                        picked_order, options, general_knowledge,
                        pdfk, pdfh, pdfs, hk_k, hk_h, hk_s,
                        gui_args)
                except Exception as e:
                    print(e)
                    import pdb
                    pdb.set_trace()
                    raise e
                zk[ck_idx_piece, :] = picked_mvs
                if gui_args:
                    if qpgd.wasCanceled(): #cancel by user
                        return False
                    else:
                        qpgd.setValue(qpgd.value()+ck_idx_piece.size)
        return zk
