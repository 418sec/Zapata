'''
Data retrieve and resampling module
===================================
'''

import os, sys, re
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import yaml, glob

import zapata.lib as lib
import zapata.data_drivers as zdrv

xr.set_options(keep_attrs=True)

def read_xarray(dataset=None, var=None, period=None, level=None, season=None, region=None, verbose=False):
    '''
    Load into a DataArray the requested variable from dataset source.

    Parameters
    ----------
    dataset : string
        Name of dataset
    var : string
         variable name
    period : list
        A two element list with initial and final years
    level : float
        level value (if not in levels list the closest one will be used)
    season : string
        Month ('JAN'), season ('DJF', 'AMJ') or annual ('ANN')
    region : list
        Region corners [LonMax, LonMin, LatMax, LatMin]
    verbose : Boolean
        True/False -- Tons of Output

    Returns
    -------
    out : DataArray
        extracted data

    Examples
    --------

    >>> da = read_xarray(dataset='ERA5_MM',var='T',period=[2000 2010], level='500')
    >>> da = read_xarray(dataset='C-GLORSv7', var='votemper', period=[2000 2010], season='DJF')
    '''
    datacat = inquire_catalogue(dataset)

    out = load_dataarray(datacat, var, level, period)
 
    # temporal sampling
    if season is not None:
        out = da_time_mean(out, season)

    # horizontal sampling (need test)
    if region is not None:
        out = out.sel(lon = slice(region[0],region[1]), lat = slice(region[2],region[3]))

    # vertical sampling
    if level is not None and 'lev' in out.coords.keys():
        lev_sel = []
        for lev in level:
            if lev in datacat['levels']:
                lev_sel.append(lev)
            else:
                # find closest level if not in levels list
                idx = np.abs(out.lev.values - lev).argmin().min()
                lev_sel.append(out.lev.values[idx])
                print ('Warning: approximate requested level %s to nearest one %s' % (str(level),str(lev_sel[-1])))

        out = out.sel(lev = lev_sel)

    return out


def da_time_mean(da, sample):
    '''
    Sample datarray based on month/season and compute average over timewindows

    Parameters
    ----------
    da : DataArray
        Input data
    sample : string
        Identifier of temporal sampling (e.g., JAN, FEB, ...,  ANN, DJF, MAM ...)

    Returns
    -------
    out : DataArray
        Time sampled DataArrray

    Examples
    --------

    >>> da = da_time_mean(da, 'JJA')
    >>> da = da_time_mean(da, 'ANN')
    '''
    indexes = None

    # admissible time groups
    time_grp ={'DJF': ['Q-NOV', [0, 4]], 'MAM':['Q-NOV', [1, 4]], 'JJA':['Q-NOV', [2, 4]], 'SON':['Q-NOV', [3, 4]],
        'JFM':['Q-DEC', [0, 4]], 'AMJ':['Q-DEC', [1, 4]], 'JAS':['Q-DEC', [2, 4]], 'OND':['Q-DEC', [3, 4]],
        'ANN':['A', [0, 1]], 
        'JAN':[1,], 'FEB':[2,], 'MAR': [3,], 'APR':[4,], 'MAY':[5,], 'JUN':[6,],
        'JUL':[7,], 'AUG':[8,], 'SEP': [9,], 'OCT':[10,], 'NOV':[11,], 'DEC':[12,]}

    # reduce data to months
    da = da.resample(time='M').mean(dim='time')

    if sample in time_grp.keys():
        if len(time_grp[sample]) > 1:
            weights = subyear_weights(da.time, time_grp[sample][0])
            da = (da * weights).resample(time=time_grp[sample][0]).sum(dim='time')
            idx = time_grp[sample][1]
            da = da.isel(time=slice(idx[0], None, idx[1]))
            da.attrs.update({'time_resample':sample})
        else:
            months = ( da.time.dt.month == time_grp[sample])
            da = da.sel(time=months)
            da.attrs.update({'time_resample':sample})

    else:
        print('requested temporal sampling' + sample + ' is not in admissible time groups.')
        sys.exit(1)

    return da


def subyear_weights(time, freq):
    '''
    Compute month weights according to frequency anchored offsets (season/year)

    Parameters
    ----------
    time : DataArray
        Time array of input data
    freq : string
        Identifier of pandas temporal anchored offsets (e.g., A, Q-DEC)

    Returns
    -------
    weights : DataArray
        weights to be applied at data before subyear time averaging

    '''

    weights = None 
    months = time.dt.days_in_month
    monbyfreq = pd.PeriodIndex(time.data,  freq=freq)

    cummon = months * 0. 
    for ii in monbyfreq.unique():
        idx = (monbyfreq == ii)
        asum = months.sel(time=idx).sum(dim='time').data
        cummon.data[idx] = asum
    
    weights = months / cummon

    return weights


def load_dataarray(dataset, var, level, period):
    '''
    Read requested data into an xarray DataArray using dataset driver

    Parameters
    ----------
    dataset : dict
        Dataset informative structure
    var : string
         variable name
    level : list
        vertical levels float value
    period : list
        Might be None or a two element list with initial and final year

    Returns
    -------
    out : DataArray
        Output data from dataset

    '''
    data_driver = dataset['driver']

    if data_driver == 'default':

        # get files list to read
        files = get_data_files(dataset, var, level, period)

        # open files as a dataset
        ds = xr.open_mfdataset(files['files'], engine='netcdf4', combine = 'by_coords', coords='minimal', compat='override')
        out = ds[files['var']]
        out.attrs['realm'] = files['component']

        # rename dimensions and coordinates if mapping provided
        if 'coord_map' in files.keys():
            out = fix_coords(out, files['coord_map'])

    else:
        # check if driver external driver exist 
        if data_driver in dir(zdrv):
            out = getattr(zdrv, data_driver)(dataset, var, level, period)
        else:
            print('Driver %s not defined in data_drivers.py.' % data_driver)
            sys.exit(1)
            
    return out


def fix_coords(da, coord_map):
    '''
    Rename dimensions and coordinates to common names using mapping provided in dataset definition

    Parameters
    ----------
    da : dataArray
        Xarray data structure
    coord_map : dict
         Dictionary to rename coordinate and dimensions

    Returns
    -------
    da: dataArray
        Xarray data structure with rename features

    '''
    for coord in coord_map.keys():
        if coord_map[coord] in da.dims:
            da = da.rename({coord_map[coord]:coord})

    return da

def get_data_files(dataset, var, level, period):
    '''
    Retrieve list of input files for requested dataset/variable pair.

    Parameters
    ----------
    dataset : dict
        Dataset informative structure
    var : string
         variable name
    level : float
        vertical levels float value
    period : list
        Might be None or a two element list with initial and final year

    Returns
    -------
    files: dict
        Dataset input files, variable name, requested period

    '''
    if dataset is None:
        print('No dataset provided.')
        sys.exit(1)

    var_info = dataset_request_var(dataset, var, level, period)

    # compose list of files
    datapath = dataset['path']
    datatree = dataset['subtree']
    filename = dataset['components'][var_info[0]]['filename']

    if period is None:
        period = dataset['year_bounds']

    if datatree is not None:
        # check if variable is arranged by levels
        islevel = True if re.search('<lev>',datatree) else False
        if len(level) > 1:
            print('File list over explicit multiple levels not allowed.')
            print('Use a loop to call get_data_files for each level with a specific data driver.')
            sys.exit(1)
        level = level[0]
        # check if year in subtree
        subyear = True if re.search('year',datatree) else False
        #TODO do we need to handle months in subtree?
        if re.search('mon',datatree):
            print('Cannot handle dataset subtree with months')
            sys.exit(1)
    else:
        islevel = False
        subyear = False
        datatree = ''

    nameyear = True if re.search('year',filename) else False
        
    # standard set of wildcards
    wildcards={'var':var, 'lev':str(level), 'mon':'*', 'comp':var_info[0], 'data_stream':var_info[1]}
    for ii in wildcards.keys():
        datatree = datatree.replace('<' + ii +'>',wildcards[ii])
        filename = filename.replace('<' + ii +'>',wildcards[ii])
    
    # compose files list
    in_files=[]
    for yy in np.arange(period[0], period[1]+1):
        thispath = '/'.join([datapath, datatree])
        if subyear:
            thispath = thispath.replace('<year>',str(yy))
        thisname = filename
        if nameyear:
            thisname = thisname.replace('<year>',str(yy))
        tmpfile = sorted(glob.glob('/'.join([thispath, thisname])))
        in_files.extend(tmpfile)
        del thispath, thisname, tmpfile
    
    if not in_files:
        print('Input files not found for ' + dataset['name'] + ' located in ' + datapath)
        sys.exit(1)

    # create output dictionary
    files={}
    files['files'] = in_files
    files['var'] = var
    files['period'] = period
    files['islevel'] = islevel
    files['component'] = var_info[0]

    # coordinate mapping
    data_stream = dataset['components'][var_info[0]]['data_stream'][var_info[1]]
    if 'coord_map' in data_stream.keys():
        files['coord_map'] = data_stream['coord_map']

    return files


def dataset_request_var(dataset, var, level, period):
    '''
    Perform consistency control on user dataset requests and find matching variable.

    Parameters
    ----------
    dataset : dict
        Dataset informative structure
    var : string
         variable name
    level : list
        vertical levels float value
    period : list
        Might be None or a two element list with initial and final year

    Returns
    -------
    var_info : list
        requested variable information: [component, data stream, type]

    '''
    # check for level bounds
    level_bnd = [min(dataset['levels']), max(dataset['levels'])]
    if level is not None and not isinstance(level[0], str):
        for lev in level:
            if lev < level_bnd[0] or lev > level_bnd[1]:
                print('Requested level ' + str(lev) + ' is not within dataset bounds [%s, %s]' % tuple(level_bnd))
                sys.exit(1)

    # check for time bounds
    time_bnd = dataset['year_bounds']
    if period is not None and len(time_bnd) > 1:
           if period[0] < time_bnd[0] or period[1] > time_bnd[1]:
               print('Requested time period is not within dataset bounds [%s, %s]' % tuple(time_bnd))
               sys.exit(1)

    # find matching variable
    var_match=[]
    for cc in dataset['components'].keys():
        for dd in dataset['components'][cc]['data_stream'].keys():
             for xy in dataset['components'][cc]['data_stream'][dd].keys():
                 thevars = dataset['components'][cc]['data_stream'][dd][xy].keys()
                 if var in thevars:
                     var_match.append([cc, dd, xy])
    del cc, dd, xy, thevars

    if len(var_match) > 1:
        print('Requested variable ' + var + ' is available from multiple data streams. Something is wrong.')
        sys.exit(1)
    elif len(var_match) == 0:
        print('Requested variable ' + var + ' is not available in the dataset %s', dataset['name'])
        sys.exit(1)
    else:
        print('Retrieve variable ' + var + ' from component %s of data stream %s as %s field' % tuple(var_match[0]))
        var_info = var_match[0]


    return var_info


def inquire_catalogue(dataset=None, info=False):
    '''
    Retrieve requested dataset informative structure from general catalogue (YAML file).

    If no dataset is requested, print a list of available datasets name and description.

    If info is True print additional details without loading dataset if provided

    Parameters
    ----------
    dataset : string
        Name of data set
    info: boolean
        Print additional details for all or selected dataset

    Returns
    -------
    out : dict
        requested dataset informative structure

    Examples
    --------

    >>> datacat = inquire_catalogue(dataset='C-GLORSv7')
    >>> inquire_catalogue()
        List of datasets in catalogue:
        C-GLORSv7 : Ocean Global Reanalyses at 1/4° resolution monthly means
        ERA5_MM : ERA5 Monthly Mean on Pressure Levels
    >>> inquire_catalogue(dataset='ERA5_MM', info=True)
        Access dataset ERA5_MM
        atm component [IFS]
         Data Stream monthly 3D variables:
         - U : Zonal Wind
         - V : Meridional Wind
         - W : Vertical Velocity
         ...
    '''
    out = None
    pwd = os.path.dirname(os.path.abspath(__file__))

    # Load catalogue
    catalogue = yaml.load(open(pwd + '/catalogue.yml'), Loader=yaml.FullLoader)

    # Print list of available datasets    
    if dataset is None:
        print('List of datasets in catalogue:\n')
        for cat in catalogue.keys():
            print('%s : %s' % tuple([cat, catalogue[cat]['description']]))
        print('\n')
        return

    # retrieve dataset information
    if dataset not in catalogue.keys():
        print('Requested dataset ' + dataset + ' is not available in catalogue.')
        sys.exit(1)
    else:
        print('Access dataset ' + dataset )
        out = catalogue[dataset]
        out['name'] = dataset

    if info:
       for comp in out['components']:
           thecomp = out['components'][comp]
           print( comp + ' component [' +thecomp['model'] + ']')
           for ss in thecomp['data_stream'].keys():
               print('\nData Stream : ' + ss)
               for grp in thecomp['data_stream'][ss].keys():
                   if grp not in ['coord_map',]:
                       print(' ' + grp  +' variables')
                       for vv in thecomp['data_stream'][ss][grp].keys():
                           print(' - ' + vv + ' : ' + thecomp['data_stream'][ss][grp][vv])

    print('\n')

    return out
