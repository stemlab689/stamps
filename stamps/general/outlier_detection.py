# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist as scipy_cdist

from ..general.neighbours import neighbours_index_kd
from ..stest.idw import idw_est_coord_value


# need fix this function #
def idw_st(c, z, c_est, n_max=100,
    s_range=None, t_range=None, st_ratio=None, power=2):

    def check_arg(c, n_max, s_range, t_range, st_ratio):
        def get_max_distance(c):
            from np import nanmax
            from scipy.spatial.distance import pdist
            print 'cal pdist...'
            d = pdist(c)
            return nanmax(d)
        n_max = 100 if n_max is 100 else int(n_max)
        s_range = get_max_distance(c[:,:-1]) if s_range is None else s_range
        t_range = get_max_distance(c[:,-1:]) if t_range is None else t_range
        st_ratio = s_range / t_range if st_ratio is None else st_ratio
        return n_max, s_range, t_range, st_ratio

    def make_tree(c, st_ratio):
        c_norm = np.copy(c)
        c_norm[:, -1] = c_norm[:, -1] * st_ratio
        c_tree = cKDTree(c_norm)
        return c_norm, c_tree

    n_max, s_range, t_range, st_ratio =\
        check_arg(c, n_max, s_range, t_range, st_ratio)
    print 'make c tree...'
    c_norm, c_tree = make_tree(c, st_ratio)
    print 'make c est tree...'
    c_est_norm, c_est_tree = make_tree(c_est, st_ratio)
    d_max_norm = (s_range**2 + (t_range * st_ratio)**2)**0.5

    z_est = np.empty((c_est.shape[0],1))
    est_list = [] # c_idx, est_idx

    print 'start neighbouring...'
    # try:
    #     est_dict = unpickleDictionary('est_dict.cpkl')
    # except Exception, e:
    #     print e
    est_dict = neighbours_index_kd(c_est_norm, c_tree, n_max, d_max_norm)
    #     pickleDictionary(est_dict, 'est_dict.cpkl')
    # import pdb
    # pdb.set_trace()
    
    print 'start idw...'
    print c_est.shape 
    cc = 0
    old_cc = 0
    for c_idx, est_idx in est_dict.iteritems():
        est_idx = np.array(est_idx, dtype=int)
        c_idx = np.array(c_idx, dtype=int)

        picked_c = c[c_idx, :]
        picked_z = z[c_idx, :]

        if len(c_idx) == 0:
            print 'no neighbors... set it to np.nan.'
            z_est[c_idx,:] = np.nan
            est_count = est_idx.shape[0]
            cc+=est_count
            if cc - old_cc >= 2000:
                print cc ,'/', c_est.shape
                old_cc = cc
            continue
            # import pdb
            # pdb.set_trace()
        
        est_count = est_idx.shape[0]
        cc+=est_count
        if cc - old_cc >= 2000:
            print cc ,'/', c_est.shape
            old_cc = cc

        split_count = np.ceil(est_count/250.)
        for est_idx_piece in np.array_split(est_idx, split_count):
            picked_c_est = c_est[est_idx_piece, :]
            picked_z_est =\
                idw_est_coord_value(picked_c, picked_z, picked_c_est, power = 2)
            if pd.isnull(picked_z_est).any():
                print 'est has nan result... stop.'
                import pdb
                pdb.set_trace()

            z_est[est_idx_piece, :] = picked_z_est
    print cc ,'/', c_est.shape
    return z_est

# need fix this function #
def outlier_detection(
    data_frame,
    id_column=None,
    spatial_column=None,
    temporal_column=None,
    value_column=None,
    normalized=True,
    time_group_frequency=None,
    search_parameters={
        'bandwith_start': 1.5,
        'bandwith_step': 0.1,
        'bandwith_end': 2,
        'max_count': 100
    },
    inverse_distance_weighted_parameters=None):
    '''
    data_frame: pd dataframe
    coords: lon, lat, time
    values: values for each column
    normalized: zscore at each id
    '''
    data_frame = data_frame.dropna(subset=value_column)
    id_coord = data_frame[[id_column]+spatial_column]\
        .drop_duplicates()\
        .set_index(id_column)

    if normalized: #need to get mean, std at each id(location)
        grp = data_frame.groupby(id_column)
        agg_dict = {}
        for vc in value_column:
            agg_dict[vc] = [np.mean, np.std]
        id_zscore = grp.agg(agg_dict)
        id_zscore.columns = id_zscore.columns.map(('{0[0]}_{0[1]}'.format))
        for vc in value_column:
            id_zscore.loc[id_zscore[vc+'_std'].isnull(),[vc+'_std']] = 1. #trick one point
            id_zscore.loc[id_zscore[vc+'_std']==0,[vc+'_std']] = 1. #trick 0 std

    data_frame =\
        pd.merge(
            data_frame,
            id_zscore,
            how='inner', left_on=id_column, right_index=True,
            sort=True, copy=True
            ) #use inner to check matched length

    data_frame['time_int'] =\
        (data_frame[temporal_column] - data_frame[temporal_column].min())\
        / np.timedelta64(1,'s')

    for vc in value_column:
        #zscore data
        data_frame[vc+'_zs'] =\
            (data_frame[vc] - data_frame[vc+"_mean"]) / data_frame[vc+"_std"]
        #setting outlier flag
        data_frame['outlier_'+vc+'_idw'] = False
        data_frame[vc+'_zs_idw'] = np.nan

    #do idw, but only use past data
    gp_time = data_frame.groupby(
        pd.Grouper(key=temporal_column, freq=time_group_frequency,
            )
        )
    gp_len = len(gp_time)
    for idx, (name, gp_df) in enumerate(gp_time):
        # if idx >= 2500:
        #     break
        if len(gp_df) > 0: #has data in group
            for vc in value_column:
                filtered_data = []
                orig_bs = search_parameters['bandwith_start']
                while (len(filtered_data) < search_parameters['max_count'])\
                    and (search_parameters['bandwith_start'] <= search_parameters['bandwith_end']):

                    search_parameters['bandwith_start'] += search_parameters['bandwith_step']
                    filtered_data =\
                        data_frame[
                            (data_frame[temporal_column] < gp_df[temporal_column].min())
                            & (data_frame[temporal_column] >
                               gp_df[temporal_column].min() - pd.Timedelta(search_parameters['bandwith_start'], 'h'))
                        ] #get filtered data
                    filtered_data = filtered_data[~filtered_data['outlier_'+vc+'_idw']] #remove outlier at vc
                if filtered_data.size == 0: # no valid data
                    data_frame.loc[gp_df.index, vc+"_zs_idw"] = np.nan
                else:
                    filtered_data_2 = filtered_data
                    # filtered_data_2 = pd.merge(
                    #     filtered_data, id_coord,
                    #     how='inner', left_on=id_column, right_index=True,
                    #     sort=True, copy=True
                    #     )
                    gp_df_2 = gp_df
                    # gp_df_2 = pd.merge(
                    #     gp_df, id_coord,
                    #     how='inner', left_on=id_column, right_index=True,
                    #     sort=True, copy=True
                    #     )

                    # if idx == 11:
                    #     import pdb
                    #     pdb.set_trace()
                    rr = idw_st(
                        filtered_data_2[spatial_column + ['time_int']].values,
                        filtered_data_2[[vc+"_zs"]].values,
                        gp_df_2[spatial_column + ['time_int']].values, n_max=100,
                        s_range=10000, t_range=2*3600, st_ratio=10000/3600., power=2) # 10km/1hr
                    data_frame.loc[gp_df.index, vc+"_zs_idw"] = rr
                    outlier_bool =\
                        (data_frame.loc[gp_df.index, vc+"_zs"] - data_frame.loc[gp_df.index, vc+"_zs_idw"]).abs() > 3
                    data_frame.loc[outlier_bool.index[outlier_bool],'outlier_'+vc+'_idw'] = True
                print '{f} -- {a}/{b}, data count: {c}, hour: {d}'.format(
                    f=vc, a=idx+1, b=gp_len,
                    c=filtered_data.shape[0], d=search_parameters['bandwith_start'])
                search_parameters['bandwith_start'] = orig_bs
        else:
            continue
    return data_frame

def time_series_outlier_detection(
    time_series, estimation_datetime_range=None, group_frequency=None,
    standard_deviation_mutiplier=3,
    maximum_outlier_duration=None,
    maximum_outlier_count=0,
    search_parameters={
        'bandwith':
            {'start': '6 hours',
             'step': '6 hours',
             'end': '6 hours'},
        'max_count': 100
        },
    inverse_distance_weighted_parameters={
        'power': 1.5
        },
    print_step=2000):

    '''
    outlier detection for time series

    time_series:
        pandas time series
    estimation_datetime_range: 
        a list of estimation datetime range, e.x. ['2016-03-04', '2017-03-01']
        if None, use time_series
    group_frequency:
        pandas timedelta obejct or any string can be convert to
        used for time series to groupby
        if not use, set it to the data's minimum time unit, like '1 s'
        it is very useful for large dataset with larger group frequency,
        it will give a fast result at a glance
        e.x. '30 minites'
    standard_deviation_mutiplier:
        a number,
        threshold of determind whether data is outlier or not.
    maximum_outlier_duration:
        pandas timedelta obejct or any string can be convert to
        it used for correct the detection, if all outlier are continually
        exists for a duration larger than maximum outlier duration,
        it will set all these outliers to False
        (not outlier, e.g. detection error)
    search_parameters:
        e.x. search_parameters={
            'bandwith':
                {'start': '6 hours',
                 'step': '1 hours',
                 'end': '12 hours'},
            'max_count': 100
            }
        to do...
    inverse_distance_weighted_parameters:
        a dictionary for idw function
    '''

    def __convert_param(search_parameters):
        search_parameters['bandwith']['start'] =\
            pd.Timedelta(search_parameters['bandwith']['start'])
        search_parameters['bandwith']['step'] =\
            pd.Timedelta(search_parameters['bandwith']['step'])
        search_parameters['bandwith']['end'] =\
            pd.Timedelta(search_parameters['bandwith']['end'])
        return search_parameters

    #get estimation datetime
    if estimation_datetime_range:
        estimation_time_series =\
            time_series[
                estimation_datetime_range[0]:estimation_datetime_range[1]]
    else:
        estimation_time_series = time_series


    outlier = pd.Series(0, estimation_time_series.index, name='outlier')
    maximum_outlier_duration = pd.Timedelta(maximum_outlier_duration)
    search_parameters = __convert_param(search_parameters)

    gp = estimation_time_series.groupby(pd.Grouper(freq=group_frequency))
    gp_len = len(gp)
    for idx, (i, ss) in enumerate(gp):
        if len(ss) > 0: #has data in time slice

            #outlier detection
            current_bandwith = search_parameters['bandwith']['start']
            picked_data = time_series[
                    (time_series.index >= i - current_bandwith)\
                        & (time_series.index < i + pd.Timedelta(group_frequency))]

            #get outlier in picked data, remove it
            outlier_in_picked =\
                 outlier[outlier.index.isin(picked_data.index)]
            outlier_in_picked_index =\
                outlier_in_picked[outlier_in_picked == 1].index
            picked_data = picked_data.drop(index=outlier_in_picked_index)

            value = picked_data.values.reshape((-1,1))
            value_mean = value.mean()
            value_std = value.std()
            #avg mean
            outlier_bool =\
                np.abs(ss.values - value_mean)\
                > (standard_deviation_mutiplier * value_std)
            if outlier_bool.any():
                try:
                    for ii in ss[outlier_bool].index:
                        outlier.loc[ii] = 1
                except AttributeError, e:
                    print e
                    import pdb
                    pdb.set_trace()
            else:
                pass

            #event detection
            # current_bandwith = search_parameters['bandwith']['start']
            # picked_data = time_series[
            #         (time_series.index >= i - current_bandwith)\
            #             & (time_series.index < i)]
            # outlier_in_picked =\
            #     outlier[outlier.index.isin(picked_data.index)]

            # if maximum_outlier_duration:
            #     outlier_in_duration = outlier_in_picked[
            #         outlier_in_picked.index.max() - maximum_outlier_duration:\
            #         outlier_in_picked.index.max()]
            #     if outlier_in_picked.shape[0]\
            #         and (outlier_in_duration==1).all()\
            #         and outlier_in_duration.size >= maximum_outlier_count:

            #         outlier_in_picked[outlier_in_duration.index] = 2
            #         outlier[outlier_in_duration.index] = 2

            # outlier_in_picked_index =\
            #     outlier_in_picked[outlier_in_picked == 1].index
            # # if outlier_in_picked_index.shape[0]:
            # #     import pdb
            # #     pdb.set_trace()
            # picked_data = picked_data.drop(index=outlier_in_picked_index)
            # while picked_data.size < search_parameters['max_count']\
            #     and current_bandwith < search_parameters['bandwith']['end']:
            #     current_bandwith += search_parameters['bandwith']['step']
            #     #get data
            #     picked_data = time_series[
            #         (time_series.index >= i - current_bandwith)\
            #             & (time_series.index < i)]
            #     outlier_in_picked =\
            #         outlier[outlier.index.isin(picked_data.index)]
            #     outlier_in_picked_index =\
            #         outlier_in_picked[outlier_in_picked == 1].index
            #     picked_data = picked_data.drop(index=outlier_in_picked_index)
                
            # if picked_data.size <= 100:
            #     outlier[ss.index] = 0
            # else:
            #     value = picked_data.values.reshape((-1,1))
            #     value_mean = value.mean()
            #     value_std = value.std()
                
            #     #avg mean
            #     outlier_bool =\
            #         np.abs(ss.values - value_mean)\
            #         > (standard_deviation_mutiplier * value_std)
            #     #idw mean
            #     # coord = picked_data.index\
            #     #     .astype('int64')\
            #     #     .astype('float')\
            #     #     .values.reshape((-1,1))
            #     # coord_est = ss.index\
            #     #     .astype('int64')\
            #     #     .astype('float')\
            #     #     .values.reshape((-1,1))
            #     # res = idw_est_coord_value(
            #     #     coord, value, coord_est,
            #     #     **inverse_distance_weighted_parameters)
            #     # outlier_bool =\
            #     #     np.abs(ss.values - res.flatten())\
            #     #     > (standard_deviation_mutiplier * value_std)

            #     outlier.loc[ss[outlier_bool].index] = 1
        else:
            pass
        if (idx+1) % print_step ==0:
            print '{f} -- {a}/{b}, data count: {c}, hour: {d}'.format(
                    f=time_series.name, a=idx+1, b=gp_len,
                    c=picked_data.shape[0], d=current_bandwith)
    print '{f} -- {a}/{b}, data count: {c}, hour: {d}'.format(
        f=time_series.name, a=idx+1, b=gp_len,
        c=picked_data.shape[0], d=current_bandwith)
    return outlier

def spatial_data_outlier_detection(
    spatial_data, estimation_boundary=['xmin', 'ymin', 'xmax', 'ymax'],
    standard_deviation_mutiplier=3,
    search_parameters={
        'bandwith': 1000,
        'max_count': 100
        },
    inverse_distance_weighted_parameters={
        'power': 1.5
        },
    print_step=2000):

    if estimation_boundary:
        estimation_spatial_data = spatial_data.loc[
        (spatial_data['x'] >= estimation_boundary[0]) &
        (spatial_data['x'] <= estimation_boundary[2]) &
        (spatial_data['y'] >= estimation_boundary[1]) &
        (spatial_data['y'] <= estimation_boundary[3])].copy()
    else:
        estimation_spatial_data = spatial_data.copy()
    estimation_spatial_data['outlier'] = False

    spatial_data_coord = spatial_data[['x', 'y']].values
    estimation_spatial_data_coord =\
        estimation_spatial_data[['x' ,'y']].values

    est_dict = neighbours_index_kd(
        estimation_spatial_data_coord,
        spatial_data_coord,
        nmax=search_parameters['max_count'],
        dmax=search_parameters['bandwith'])

    c_count = 0
    old_c_count = 0
    has_show_no_neighbor = False
    for c_idx, est_idx in est_dict.iteritems():
        # if 5852 in est_idx or 5936 in est_idx:
        #     import pdb
        #     pdb.set_trace()
        est_idx = np.array(est_idx, dtype=int)
        c_idx = np.array(c_idx, dtype=int)
        picked_c = spatial_data_coord[c_idx, :]
        picked_z = spatial_data[['z']].values[c_idx, :]
        
        if picked_c.shape[0] and np.unique(picked_c, axis=0).shape[0] == 1:
            if not has_show_no_neighbor:
                print 'some estimated point has no neighbors... default to False.'
                has_show_no_neighbor = True
            # estimation_spatial_data.loc[est_idx, 'z_est'] = np.nan
            # import pdb
            # pdb.set_trace()
            est_count = est_idx.shape[0]
            c_count+=est_count
            if c_count - old_c_count >= print_step:
                print c_count ,'/', estimation_spatial_data.shape[0]
                old_c_count = c_count
            continue
        else:
            value_mean = picked_z.mean()
            value_std = picked_z.std()

            picked_z_est = estimation_spatial_data[['z']].values[est_idx, :]
            # outlier_bool =\
            #     np.abs(picked_z_est.flatten() - value_mean)\
            #     > (standard_deviation_mutiplier * value_std)

            # idw mean
            picked_c_est = estimation_spatial_data_coord[est_idx, :]
            res = idw_est_coord_value(
                picked_c, picked_z, picked_c_est,
                **inverse_distance_weighted_parameters)
            if pd.isnull(res).any():
                print 'est has nan result... stop.'
                import pdb
                pdb.set_trace()
            outlier_bool =\
                np.abs(picked_z_est.flatten() - res.flatten())\
                > (standard_deviation_mutiplier * value_std)
            estimation_spatial_data.iloc[
                est_idx[outlier_bool],
                estimation_spatial_data.columns.get_loc('outlier')] = True
            est_count = est_idx.shape[0]
            c_count+=est_count
            if c_count - old_c_count >= print_step:
                print c_count ,'/', estimation_spatial_data.shape[0]
                old_c_count = c_count

    print c_count ,'/', estimation_spatial_data_coord.shape[0]
    return estimation_spatial_data
    
def space_time_outlier_detection(
    space_time_dataframe,
    id_column=None,
    spatial_column=None,
    temporal_column=None,
    value_column=None,
    space_group_frequency=None,
    space_standard_deviation_mutiplier=3,
    space_search_parameters={
        'bandwith': 10000,
        'max_count': 100},
    space_print_step=2000,
    time_group_frequency=None,
    time_standard_deviation_mutiplier=3,
    time_search_parameters={
        'bandwith':
            {'start': '2 hours',
             'step': '2 hours',
             'end': '2 hours'},
        'max_count': 100},
    time_print_step=2000,
    time_maximum_outlier_duration='1 hours',
    time_maximum_outlier_count=12,
    inverse_distance_weighted_parameters=None):

    def __time_series_outlier_detection(gpdf):
        print gpdf.name, '(', __time_series_outlier_detection.i+1,\
            '/', __time_series_outlier_detection.len, ')'
        ts = pd.Series(
            data=gpdf[value_column].values,
            index=gpdf[temporal_column],
            name=value_column)
        gg =\
        time_series_outlier_detection(
            time_series = ts, estimation_datetime_range = None,#['2016-12-30', '2017-3-1'],
            group_frequency=time_group_frequency,
            standard_deviation_mutiplier=3,
            maximum_outlier_duration=time_maximum_outlier_duration,
            maximum_outlier_count=time_maximum_outlier_count,
            search_parameters=time_search_parameters,
            inverse_distance_weighted_parameters=\
                inverse_distance_weighted_parameters,
            print_step=time_print_step)
        gpdf['temporal_outlier'] = gg.values
        __time_series_outlier_detection.i += 1
        return gpdf

    def __spatial_data_outlier_detection(gpdf):
        print gpdf.name, '(', __spatial_data_outlier_detection.i,\
            '/', __spatial_data_outlier_detection.len, ')'

        # {{for debug}}
        # if gpdf.name == pd.Timestamp('2016-10-20 07:10:00'):
        #     import ipdb
        #     ipdb.set_trace()
        # else:
        #     return gpdf
        gpdf = gpdf.rename(
            columns={
                spatial_column[0]: 'x',
                spatial_column[1]: 'y',
                value_column: 'z'})
        if gpdf.shape[0] == 0: #empty
            print 'no data in this group... skipped.'
            gpdf = pd.concat(
                [gpdf, pd.DataFrame(columns=['spatial_outlier'])]
                )
        else:
            gg =\
            spatial_data_outlier_detection(
                spatial_data = gpdf, estimation_boundary=None, #['xmin', 'ymin', 'xmax', 'ymax'],
                standard_deviation_mutiplier=3,
                search_parameters=space_search_parameters,
                inverse_distance_weighted_parameters=\
                    inverse_distance_weighted_parameters,
                print_step=space_print_step)
            gpdf['spatial_outlier'] = gg['outlier']
        __spatial_data_outlier_detection.i += 1
        return gpdf.rename(
            columns={
                'x': spatial_column[0],
                'y': spatial_column[1],
                'z': value_column})

    stdf = space_time_dataframe

    print 'start spatial data outlier...'
    #make space data outlier
    sdgp = stdf.groupby(pd.Grouper(
        key=temporal_column, freq=space_group_frequency))
    __spatial_data_outlier_detection.len = len(sdgp)
    __spatial_data_outlier_detection.i = 0
    stdf = sdgp.apply(__spatial_data_outlier_detection)

    print 'start time series outlier...'
    #make time series outlier
    tsgp = stdf.groupby(id_column)
    __time_series_outlier_detection.len = len(tsgp)
    __time_series_outlier_detection.i = 0
    stdf = tsgp.apply(__time_series_outlier_detection)

    stdf = stdf\
        .reset_index(drop=True)\
        .rename(
            columns={
                'x': spatial_column[0],
                'y': spatial_column[1],
                'z': value_column})
    stdf['outlier'] = stdf['temporal_outlier'] & stdf['spatial_outlier']
    #NOTE!!
    '''
    and tests whether both expressions are logically True
    while & (when used with True/False values) tests if both are True.
    '''


    # # set temporal event to False
    # stdf.loc[
    #     stdf['temporal_outlier']==2,
    #     "outlier"
    #     ] = False
    return stdf
    

    
    
