'''
**Data retrieve and resampling module**

This module contains the two main functions of the data interface (see below for usage details):

- :meth:`inquire_catalogue<zapata.data.inquire_catalogue>` : Retrieve requested dataset informative structure from catalogue
- :meth:`read_data<zapata.data.read_data>` : Load into a DataArray the requested variable from specified dataset

The data extraction from each dataset is performed by the function :meth:`load_dataarray<zapata.data.load_dataarray>` that contains the `default` driver for extraction operations and also handles the call to specific data drivers, which are contained in module :meth:`data_drivers.py<zapata.data_drivers>`.

The maintained dataset catalogue is located within the zapata library, named `catalogue.yml` (YAML format), while users can add their own datasets by editing the file `user_catalogue.yml` located in the root path of zapata.

A new dataset can be included by editing the catalogue YAML files (either main or user), by means of a python dictionary structured as in the following:

.. code-block:: python

   <DATASET_NAME>:
       remote: logical                         # used to indentify data on remote filesystem (True) ot not (False)
       path: /path_to_data/                    # path of data without the subtree elements (see next item)
       subtree: <t_card>(/<t_card>)            # wildcards used to define data organization on temporal basis, availables cards
                                               # <year>: YYYY, <month>:MM (leading zero), <mon>:MM (no leading zero), 
                                               # <var>: variable name; <lev>: vertical level value
       source_url: http://www.adress/          # reference webpage of the dataset (if any)
       description: string                     # short description of the dataset (max 125 characters)
       contact: string                         # data originator name and mail contact
       year_bounds: list                       # Initial and final years of the dataset time extension, e.g. [0111, 1900]
       driver: string                          # 'default' handles NetCDF files that includes coordinates, while 
                                               # 'driver_name' refers to the specific intake procedure defined in `data_drivers.py`
       levels: list                            # Python list object with vertical reference levels of data
       components:
           <comp_name>:                        # dataset component name, identifing data realm among atm, ocn, lnd, ice, ocnbgc
               source: string                  # name of the originator, e.g. model name or satellite platform
               filename: string                # generalized data filename structure with the inclusion of wildcards to parse time and type
                                               # For example, Dummy_Model_<year>_<data_stream>.nc
                                               # where <data_stream> identifies the data type spcified in the section below
               data_stream:
                   <output_group>:             # group of output files with the same structure over the time
                       3D: dict                # 3-dimensional variables name stored in files and corresponding long_name, e.g. {'temp':'Temperature'}
                       2D: dict                # same as above but for 2-dimensional variables
                       coords: dict            # mapping of 2-dimensional coordinate names to library standard lon and lat names.
                                               # Example: {'lon':'glamt', 'lat':'gphit'}
                       coord_map: dict         # mapping of dimensions names to library standard lon,lat,lev,time. 
                                               # Example: {'lon':'x', 'lat':'y', 'time':'time_centered', 'lev':'deptht'}
                       mask: string            # name of mask variable to be applied during data intake, if mask is provided in metrics section below,
                                               # Mask is expected to follow this convention, 0:remove, 1:retain (see also `mask_data` function)
       metrics:
            mask: dict                         # Two element dictionary to identify input mask file, with
                                               #   'file': full path of filename 
                                               #   'coord_map': mapping of dimensions names, as described above       

===================================
'''

import os, sys, re
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import yaml, glob

import zapata.data_drivers as zdrv

xr.set_options(keep_attrs=True)

def inquire_catalogue(dataset=None, info=False):
    '''
    Retrieve requested dataset informative structure from general catalogue (YAML file).

    If no dataset is requested, print a compact list of available datasets (name and description).

    If info is True print additional details without loading dataset if provided

    Parameters
    ----------
    dataset : string
        Name of dataset
    info: boolean
        Print additional details for the selected dataset

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
    catalogue = yaml.load(open(pwd + '/catalogue.yml'), Loader=yaml.SafeLoader)
 
    # if user defined catalogue exists, update generale one
    user_catalogue = pwd + '/../user_catalogue.yml'
    if os.path.isfile(user_catalogue):
       tmp_dict = yaml.load(open(user_catalogue), Loader=yaml.SafeLoader)
       if (tmp_dict is not None):
          catalogue.update(tmp_dict)
          print('Append user defined lists of datasets to catalogue:\n')

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
       print(out['description'])
       print('(Contact: ' + out['contact'] + ', URL: ' + out['source_url'] + ')')
       tmp_val = [str(x) for x in out['year_bounds']]
       print('Time window: ' + '-'.join(tmp_val) + '\nLocation: ' + out['path'] + '\n')
       if 'levels' in out.keys():
           tmp_val = [str(x) for x in out['levels']]
           print('Vertical levels: ' + ', '.join(tmp_val) + '\n')
       for comp in out['components']:
           thecomp = out['components'][comp]
           print( comp + ' component [' +thecomp['source'] + ']')
           for ss in thecomp['data_stream'].keys():
               print('\nData Stream : ' + ss)
               for grp in thecomp['data_stream'][ss].keys():
                   if grp not in ['coords', 'coord_map', 'mask']:
                       print(' ' + grp  +' variables')
                       for vv in thecomp['data_stream'][ss][grp].keys():
                           print(' - ' + vv + ' : ' + thecomp['data_stream'][ss][grp][vv])
       print('\n')
       return

    print('\n')

    return out


def read_data(dataset=None, var=None, period=None, level=None, season=None, region=None, verbose=False):
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
    level : list  
        level value as float (if not matching any dataset level, it takes the closest one)
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

    out = load_dataarray(datacat, var, level, period, season)
 
    # temporal sampling
    if season is not None:
        out = da_time_mean(out, season)

    # horizontal sampling
    #TODO  need test with NEMO grid as coordinate are not associated to dimensions (maybe a dedicated function)
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


def load_dataarray(dataset, var, level, period, season):
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
    season : string
        Month ('JAN'), season ('DJF', 'AMJ') or annual ('ANN')

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
        ds = xr.open_mfdataset(files['files'], engine='netcdf4', combine = 'by_coords', coords='minimal', compat='override', parallel=True)
        out = ds[files['var']]
        out.attrs['realm'] = files['component']

        # rename dimensions and coordinates if mapping provided
        if 'coord_map' in files.keys():
            out = fix_coords(out, files['coord_map'])

        # apply mask to data if provided
        if 'mask' in files.keys():
            out = mask_data(out, files['mask']['name'], files['mask']['file'],files['mask']['coord_map'])

    else:
        # check if external driver exist
        if data_driver in dir(zdrv):
            out = getattr(zdrv, data_driver)(dataset, var, level, period, season)
        else:
            print('Driver %s not defined in data_drivers.py.' % data_driver)
            sys.exit(1)

    out = roll_longitude(out)

    out = check_nptime(out)

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

    time_frames = define_time_frames(sample)

    # reduce data to months
    da = da.resample(time='M').mean(dim='time')

    if sample in time_frames.keys():
        if len(time_frames[sample]) > 1:
            weights = subyear_weights(da.time, time_frames[sample][0])
            da = (da * weights).resample(time=time_frames[sample][0]).sum(dim='time')
            idx = time_frames[sample][1]
            da = da.isel(time=slice(idx[0], None, idx[1]))
            da.attrs.update({'time_resample':sample})
        else:
            months = ( da.time.dt.month == time_frames[sample])
            da = da.sel(time=months)
            da.attrs.update({'time_resample':sample})

    else:
        print('requested temporal sampling' + sample + ' is not in admissible time groups.')
        sys.exit(1)

    return da


def define_time_frames(sample):
    '''
    Define time frames handled by the library

    Parameters
    ----------
    sample : string
        Identifier of temporal sampling (e.g., JAN, FEB, ...,  ANN, DJF, MAM ...)

    Returns
    -------
    time_frames : dict
        dictionary with allowed time frames (short name and attributes)
    '''

    time_frames={'DJF': ['Q-NOV', [0, 4], [12, 1, 2]], 'MAM':['Q-NOV', [1, 4], [3, 4, 5]], 
        'JJA':['Q-NOV', [2, 4], [6, 7, 8]], 'SON':['Q-NOV', [3, 4], [9, 10, 11]],
        'JFM':['Q-DEC', [0, 4], [1, 2, 3]], 'AMJ':['Q-DEC', [1, 4], [4, 5, 6]], 
        'JAS':['Q-DEC', [2, 4], [7, 8, 9]], 'OND':['Q-DEC', [3, 4], [10, 11, 12]],
        'ANN':['A', [0, 1], range(1,13)],
        'JAN':[1,], 'FEB':[2,], 'MAR': [3,], 'APR':[4,], 'MAY':[5,], 'JUN':[6,],
        'JUL':[7,], 'AUG':[8,], 'SEP': [9,], 'OCT':[10,], 'NOV':[11,], 'DEC':[12,]}

    if sample not in time_frames.keys():
        print('define_time_frames: requested temporal sampling' + sample + ' is not in admissible time groups.')
        sys.exit(1)

    return time_frames

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


def roll_longitude(da):
    '''
    Roll longitude coordinate between 0..360. Note that longitude dimension name must be 'lon'.

    Parameters
    ----------
    da : dataArray
        Xarray data structure

    Returns
    -------
    da: dataArray
        Xarray data structure

    '''
    coord = da.coords
    # 1D coordinate
    if 'lon' in coord:
        if np.min(da.lon) < 0.:
            da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
    # 2D coordinate for NEMO default name
    elif 'nav_lon' in coord:
        if np.min(da.nav_lon) < 0.:
            da = da.assign_coords(nav_lon=(da.nav_lon % 360))

    return da


def check_nptime(da):
    '''
    Check if dataArray time axis format is numpy datetime64 and apply conversion if not.

    Parameters
    ----------
    da : dataArray
        Xarray data structure

    Returns
    -------
    da: dataArray
        Xarray data structure

    '''
    if not np.issubdtype(da['time'].dtype, np.datetime64):
        try:
            from xarray.coding.times import cftime_to_nptime
            da['time'] = cftime_to_nptime(da['time'])
        except ValueError as e:
            print(e)
            pass

    return da


def mask_data(da, mask_name, mask_file, coord_map):
    '''
    Mask dataarray using a [0-1] mask file and the following convention, 0:remove, 1:retain

    Parameters
    ----------
    dataset : dict
        Dataset informative structure
    mask_name : string
         mask variable name
    mask_file : string
         input file full path where mask variable is contained
    coord_map : dict
         Dictionary to rename dimensions

    Returns
    -------
    da: dataArray
        Xarray data structure with rename features
    '''
    dm = xr.open_dataset(mask_file)
    dm = fix_coords(dm[mask_name], coord_map)
    dm = dm.squeeze()

    # drop lev dimension if data is 2D
    if da.ndim < 4 and 'time' in da.dims:
       if 'lev' in dm.dims:
           dm = dm.sel(lev=1)

    da = da.where(dm == 1)

    return da


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

    months = ['01',]
    if datatree is not None:
        # check if variable is arranged by levels
        islevel = True if re.search('<lev>',datatree) else False
        if len(level) > 1:
            print('File list over explicit multiple levels not allowed.')
            print('Use a loop to call get_data_files for each level with a specific data driver.')
            sys.exit(1)
        level = level[0]
        # check if month in subtree
        if re.search('<month>',datatree) :
            months = [str(item).zfill(2) for item in range(1,13)]
        elif re.search('<mon>',datatree) :
            months = [str(item) for item in range(1,13)]
        #TODO do we need to handle dayss in subtree?
        if re.search('<day>',datatree):
            print('Cannot handle dataset subtree with days')
            sys.exit(1)
    else:
        islevel = False
        datatree = ''

    nameyear = True if re.search('year',filename) else False
        
    # standard set of wildcards
    wildcards={'var':var, 'lev':str(level), 'comp':var_info[0], 'data_stream':var_info[1]}
    for ii in wildcards.keys():
        datatree = datatree.replace('<' + ii +'>',wildcards[ii])
        filename = filename.replace('<' + ii +'>',wildcards[ii])

    # check month format in filename
    if re.search('<month>',filename):
        months = [str(item).zfill(2) for item in range(1,13)]
    elif re.search('<mon>',filename):
        months = [str(item) for item in range(1,13)]
    
    # compose files list
    in_files=[]
    for yy in np.arange(period[0], period[1]+1):
        for mm in months:
            thispath = '/'.join([datapath, datatree])
            #subtree replace
            thispath = thispath.replace('<year>',str(yy))
            thispath = thispath.replace('<mon>',str(mm))
            thispath = thispath.replace('<month>',str(mm))
            #filename replace
            thisname = filename
            thisname = thisname.replace('<year>',str(yy))
            thisname = thisname.replace('<mon>',str(mm))
            thisname = thisname.replace('<month>',str(mm))
            #list files
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

    # coordinates from file
    if 'coords' in data_stream.keys():
        if 'coords' in dataset['metrics'].keys():
            files['coords'] = data_stream['coords']
            files['coords'].update({'file':dataset['metrics']['coords']})
        else:
            print('Coordinates file not available within metrics files')
            sys.exit(1)

    # mask to be applied at the input data fields
    if 'mask' in data_stream.keys():
        if 'mask' in dataset['metrics'].keys():
            files['mask'] = {'name':data_stream['mask']}
            files['mask'].update(dataset['metrics']['mask'])
        else:
            print('Mask file not available within metrics files (maskname is ' + data_stream['mask'] + ')')
            sys.exit(1)

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
    if level is not None:
        if 'levels' in dataset.keys():
            if not isinstance(level[0], str):
                level_bnd = [min(dataset['levels']), max(dataset['levels'])]
                for lev in level:
                    if lev < level_bnd[0] or lev > level_bnd[1]:
                        print('Requested level ' + str(lev) + ' is not within dataset bounds [%s, %s]' % tuple(level_bnd))
                        sys.exit(1)
        else:
            print('Level was requested, but the dataset has no levelsi defined')
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
                 if xy not in ['coords', 'coord_map', 'mask']:
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

