# -*- coding: utf-8 -*-

# Creating crossbasis matrix from r package: dlnm
import pandas as pd
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
except Exception as e:
    print ('Cannot import modeul "rpy2.robjects", try to install "rpy2" first.')
    raise e
pandas2ri.activate()

rdlnm = importr('dlnm')
rmgcv = importr('mgcv')


def ConvertArgsType(func):
    """Converts function arguments to Pandas type"""
    def inner_function(*args, **kwargs):
        for arg in args:
            if not isinstance(arg, pd.Series):
                arg = pd.Series(arg)
            return(func)
        for kwarg in kwargs:
            if kwarg == None:
                kwarg = ro.NULL
        return(inner_function)

# @ConvertArgsType
def crossbasis(x,
    lag,
    argvar,
    arglag,
    group=ro.NULL):
    crossbasisObject = rdlnm.crossbasis(x, lag, argvar, arglag, group)
    return crossbasisObject

def gam(formula,
    data=ro.r('list()'),
    control=ro.r('list()'),
    family=ro.r('gaussian()'),
    na_action=ro.r('na.omit'),
    optimizer=["outer","newton"],
    method='GCV.Cp',
    subset=ro.NULL,
    drop_unused_levels=True,
    fit=True,
    select=False,
    gamma=1.,
    scale=0.,
    min_sp=ro.NULL,
    in_out=None,
    knots=ro.NULL,
    G=ro.NULL,
    H=ro.NULL,
    offset=ro.NULL,
    paraPen=ro.NULL,
    sp=ro.NULL,
    weights=ro.NULL,
    drop_intercept=ro.NULL):
    
    if not in_out:
        model = rmgcv.gam(
            formula, family, data=data, control=control, 
            weights=weights, subset=subset, na_action=na_action,
            offset=offset, method=method, optimizer=optimizer, scale=scale, 
            select=select, knots=knots, sp=sp, min_sp=min_sp, 
            H=H, gamma=gamma, fit=fit, paraPen=paraPen, G=G, 
            drop_unused_levels=drop_unused_levels,
            drop_intercept=drop_intercept
            )
    else:
        model = rmgcv.gam(
            formula, family, data=data, control=control, 
            weights=weights, subset=subset, na_action=na_action,
            offset=offset, method=method, optimizer=optimizer, scale=scale, 
            select=select, knots=knots, sp=sp, min_sp=min_sp, 
            H=H, gamma=gamma, fit=fit, paraPen=paraPen, G=G, 
            drop_unused_levels=drop_unused_levels,
            drop_intercept=drop_intercept, in_out=in_out
            )
    return model
