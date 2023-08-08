import numpy as np

def findpoint(in_lon, in_lat, ncfile):
    """
    in_lon: longitude of the station
    in_lat: latitude of the station
    ncfile: xarray dataarray that contains coordinates 'longitude' and 'latitude'
    nearest_point: output of the dataarray
    """

    # Compute the distances between each grid point and the specified [lon, lat] location
    distances = ((ncfile.longitude - in_lon)**2 + (ncfile.latitude - in_lat)**2)**0.5

    # Find the minimum distance and corresponding index in the flattened array
    min_distance = distances.min()
    min_index = distances.argmin()

    # Convert the flattened index to 2D index
    y_index, x_index = np.unravel_index(min_index, ncfile[0,0,:,:].shape)

    # Get the value of the DataArray at the nearest grid point
    # nearest_point = ncfile.isel(x=x_index, y=y_index)
    
    return x_index, y_index



def findpoint_test(in_lon, in_lat, nlon, nlat):
    """
    in_lon: longitude of the station
    in_lat: latitude of the station
    nlon: longitude of the model domain (2D Var)
    nlat: latitude of the model domain (2D Var)
    out_x: the nearest position of the station in the model domain in x direction
    out_y: the nearest position of the station in the model domain in y direction
    """
       
    xnum, ynum = nlon.shape
    gridnum = xnum * ynum
    
    nlon = nlon.reshape(gridnum, 1)
    nlat = nlat.reshape(gridnum, 1)
    
    out_x = np.nan
    out_y = np.nan
    
    index = np.argmin((nlon - in_lon) ** 2 + (nlat - in_lat) ** 2)
    out_y = np.ceil((index + 1) / xnum)
    out_x = index + 1 - xnum * (out_y - 1)
    
    return out_x, out_y
