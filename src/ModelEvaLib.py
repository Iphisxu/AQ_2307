# some functions to calculate results of model outputs
# Evan, 2023-03-28

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import shapely.geometry as sgeom
from shapely.prepared import prep
import matplotlib.pyplot as plt
from matplotlib import rcParams

# silence the warning note
import warnings
warnings.filterwarnings("ignore")

def polygon_to_mask(polygon, x, y):
    '''
    Generate a mask array of points falling into the polygon
    '''
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    mask = np.zeros(x.shape, dtype=bool)

    # if each point falls into a polygon, without boundaries
    prepared = prep(polygon)
    for index in np.ndindex(x.shape):
        point = sgeom.Point(x[index], y[index])
        if prepared.contains(point):
            mask[index] = True

    return mask

def get_var(ncfile, var, shpfile, xlsfile, varch):
    '''
    Get the variable data for evaluation
    
    @PARAM ncfile: 'your_directory/model_output.nc'
    @PARAM var: the variable to be evaluated, where NetCDF file contains
    @PARAM shpfile: 'your_directory/shapefile.shp'
    @PARAM xlsfile: 'your_directory/obs_output.xlsx'
    @PARAM varch: the variable in Chinese, consistent with var
    
    '''
    simdata = xr.open_dataset(ncfile)
    simvar = simdata[var][:,0,:,:] # extract surface data
    print('Get the ' + str(var) + ' in shape of ' + str(simvar.shape))
    
    print('Extract the data within the boundary ...')
    shp = gpd.read_file(shpfile)
    for i in range(np.size(simvar.latitude,0)):
        for j in range(np.size(simvar.latitude,1)):
            if polygon_to_mask(shp.geometry[0],simvar.longitude[i,j],simvar.latitude[i,j])==False:
                simvar[:,i,j] = np.nan
    
    sim = np.nanmean(np.nanmean(simvar,1),1)
    print('Finish')
    
    obsdata = pd.read_excel(xlsfile)
    obs = obsdata[varch]
    print('Get the ' + str(varch) + ' in shape of ' + str(obs.shape))
    if obs.shape == sim.shape:
        df = pd.DataFrame({'sim':sim, 'obs':obs})
        print('data is ready')
        return df, sim, obs
    else:
        print('the shape of ' + str(varch) + ' is not consistent with ' + str(var) +', please check.')
        return None

def linechart(sim, obs, start_date, end_date):
    '''
    plot the line chart using data prepared
    
    @PARAM sim: simulation data from get_var function
    @PARAM obs: observation data from get_var function
    @PARAM start_date: format{%y-%m-%d-%h}
    @PARAM end_date: format{%y-%m-%d-%h}
    
    '''
    config = {
        "font.family":'Times New Roman',
        "mathtext.fontset":'stix',
        "font.serif": ['Times New Roman'],
    }
    rcParams.update(config)
    
    timelength=np.size(pd.date_range(start_date,end_date,freq='h'))
    date = pd.date_range(start_date,end_date,freq='D')
    time = np.arange(0,timelength)
    
    if sim.shape == pd.date_range(start_date,end_date,freq='h').shape:
        fig = plt.figure(figsize=(10,4),dpi=300)
        ax = fig.subplots()
        ax.plot(time,sim,label='simulation')
        ax.plot(time,obs,label='observation')
        ax.set_xticks(np.arange(0,timelength,24))
        ax.set_xticklabels(date.strftime('%m-%d'),size=6)
        return ax
    else:
        print('length of data is not consistent with that of date')
        return None

def cal_IOA(obsList, simList):
    '''
    calculate Willmott's Index of Agreement, so-called WIA or IOA
    '''
    if len(obsList) != len(simList):
        raise Exception("length of sim is not consistent with that of obs")
    # calculate the numerator and denominator for IOA
    numerator = np.sum((simList - obsList) ** 2)
    mean_obsList = np.mean(obsList)
    denominator = np.sum((np.abs(simList - mean_obsList) + np.abs(obsList - mean_obsList)) ** 2)
    ioa = 1 - (numerator / denominator)

    return ioa

def cal_RMSE(obsList, simList):
    """
    calculat Root Mean Square Error, so-called RMSE
    """
    if len(obsList) != len(simList):
        raise Exception("length of sim is not consistent with that of obs")
    rmse = np.sqrt(np.mean((simList - obsList) ** 2))
    
    return rmse

def evaluation_frame(obs, sim, df):
    '''
    print evaluation results
    
    '''
    dfout = pd.DataFrame([['obs mean', np.nanmean(obs)],
                          ['sim mean', np.nanmean(sim)],
                          ['R', df.corr().iloc[0,1]],
                          ['MB', np.nanmean(sim)-np.nanmean(obs)],
                          ['RMSE', cal_RMSE(obs,sim)],
                          ['IOA', cal_IOA(obs,sim)]],
                         columns=['param','value'])
    
    return dfout

