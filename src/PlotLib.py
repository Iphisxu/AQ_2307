import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.io.shapereader import Reader
import os

from matplotlib import rcParams
config = {
"font.family":'Times New Roman',
"mathtext.fontset":'stix',
"font.serif": ['SimSun'],
}
rcParams.update(config)

def calc_mda8(inputdata):
    '''
    
    '''
    if str(type(inputdata))=="<class 'xarray.core.dataarray.DataArray'>":
        print('calculating '+ inputdata.name +' mda8 data as xarray.dataarray')
        mda8 = inputdata.rolling(time=8).mean().resample({'time':'D'}).max(dim='time')
        
        return mda8
        
    elif str(type(inputdata))=="<class 'pandas.core.frame.DataFrame'>":
        print('calculating obs mda8 data as pandas.DataFrame')
        mda8 = inputdata.rolling(8).mean().resample('D').max()
        
        return mda8
        
    else:
        raise Exception("The data type does not match")
    
        
def calc_ave(inputdata):
    '''
    
    '''
    if str(type(inputdata))=="<class 'xarray.core.dataarray.DataArray'>":
        print('calculating '+ inputdata.name +' average data as xarray.dataarray')
        ave = inputdata.resample({'time':'D'}).mean(dim='time')
        
        return ave
    
    elif str(type(inputdata))=="<class 'pandas.core.frame.DataFrame'>":
        print('calculating obs average data as pandas.DataFrame')
        ave = inputdata.resample('D').mean()
    
        return ave    

    else:
        raise Exception("The data type does not match")


def get_simvar(chemfile, metfile, varname, level1, level2):
    '''
    
    '''
    chem = xr.open_dataset(chemfile)
    met = xr.open_dataset(metfile)
    
    lat = chem.latitude
    lon = chem.longitude
    
    var_L1 = np.squeeze(chem[varname][:,level1,:,:])
    var_L2 = np.squeeze(chem[varname][:,level2,:,:])
    
    uw_L1 = np.squeeze(met.uwind[:,level1,:,:])
    vw_L1 = np.squeeze(met.vwind[:,level1,:,:])

    uw_L2 = np.squeeze(met.uwind[:,level2,:,:])
    vw_L2 = np.squeeze(met.vwind[:,level2,:,:])
    
    print('get ' + str(varname) + ' in shape of ' + str(var_L1.shape))
    
    if varname == 'O3':
        var_L1 = calc_mda8(var_L1)
        var_L2 = calc_mda8(var_L2)
    else:
        var_L1 = calc_ave(var_L1)
        var_L2 = calc_ave(var_L2)
        
    uuL1 = calc_ave(uw_L1)
    uuL2 = calc_ave(uw_L2)
    vvL1 = calc_ave(vw_L1)
    vvL2 = calc_ave(vw_L2)
    
    return lat, lon, var_L1, var_L2, uuL1, uuL2, vvL1, vvL2

def get_obsvar(dir_path, varname):
    '''
    
    '''
    # create an empty dataframe to store the data
    df = pd.DataFrame()

    # loop through each file in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith('.xlsx'): # make sure it's an Excel file
            # read the Excel file into a dataframe and set the first column as the index
            file_path = os.path.join(dir_path, filename)
            temp_df = pd.read_excel(file_path, index_col=0)
            
            # extract the var column and add it to the main dataframe
            col_name = os.path.splitext(filename)[0] # use the file name as the column name
            df[col_name] = temp_df[varname]
    
    if varname == 'O3':
        df = calc_mda8(df)
    else:
        df = calc_ave(df)
               
    return df


# =============================================
#                 spatial plots                
# =============================================

def oneplot_spatial_d03(date, colormax, lat, lon, var_L1, var_L2, uuL1, uuL2, vvL1, vvL2):
    '''
    
    '''
    cmaplevel=np.arange(0,colormax,3)
    proj=ccrs.PlateCarree()

    fig = plt.figure(figsize=(9,4),dpi=300)
    ax = fig.subplots(1,2,subplot_kw={'projection':proj})

    shp_urban = Reader('F:/Data/case_anqing/shapefile/Anqing_urban/urban.shp')
    shp_rural = Reader('F:/Data/case_anqing/shapefile/Anqing_rural/rural.shp')
    shp_pro = Reader('F:/shapefile/shp_for_ArcGis/ChinaAdminDivisonSHP-master/2. Province/province.shp')
    for i in range(2):
        ax[i].add_feature(cfeat.ShapelyFeature(shp_pro.geometries(),proj,edgecolor='gray',facecolor='None'), linewidth=0.8)
        ax[i].add_feature(cfeat.ShapelyFeature(shp_urban.geometries(),proj,edgecolor='k',facecolor='None'), linewidth=0.8)
        ax[i].add_feature(cfeat.ShapelyFeature(shp_rural.geometries(),proj,edgecolor='k',facecolor='None'), linewidth=0.8)

        gl=ax[i].gridlines(
            xlocs=np.arange(-180, 180 + 1, 1), ylocs=np.arange(-90, 90 + 1, 1),
            draw_labels=True, x_inline=False, y_inline=False,
            linewidth=0, linestyle='--', color='gray')
        gl.top_labels = False
        gl.right_labels =False
        gl.rotate_labels=False
        
        ax[i].set_extent([114.8, 118.8, 29.1, 32.6],ccrs.PlateCarree()) # d03

        if i>0:
            gl.left_labels=False

    xgrid=np.size(uuL1,2)
    ygrid=np.size(vvL1,1)
    ngrid=8

    # contour map
    cmap=ax[0].contourf(lon,lat,var_L1[date-1,:,:],transform=proj,cmap='Spectral_r',levels=cmaplevel,extend='both')
    cmap=ax[1].contourf(lon,lat,var_L2[date-1,:,:],transform=proj,cmap='Spectral_r',levels=cmaplevel,extend='both')

    # wind quiver
    ax[0].quiver(lon[0:ygrid:ngrid,0:xgrid:ngrid],lat[0:ygrid:ngrid,0:xgrid:ngrid],
            uuL1[date-1,0:ygrid:ngrid,0:xgrid:ngrid],vvL1[date-1,0:ygrid:ngrid,0:xgrid:ngrid],
            transform=proj,color='k',alpha=1,scale=150,headwidth=3)
    ax[1].quiver(lon[0:ygrid:ngrid,0:xgrid:ngrid],lat[0:ygrid:ngrid,0:xgrid:ngrid],
            uuL2[date-1,0:ygrid:ngrid,0:xgrid:ngrid],vvL2[date-1,0:ygrid:ngrid,0:xgrid:ngrid],
            transform=proj,color='k',alpha=1,scale=150,headwidth=3)
    
    # share colorbar
    fig.subplots_adjust(right=0.9,wspace=0.02)
    position= fig.add_axes([0.92,0.15,0.015,0.7])
    cbar=fig.colorbar(cmap,cax=position)
    cbar.set_ticks(np.arange(0,colormax+1,30))
    cbar.set_label('$\mu$$g$/$m^3$')

    ax[0].set_title('(a) 1000hPa',loc='left')
    ax[1].set_title('(b) 850hPa',loc='left')
    plt.suptitle(pd.to_datetime('2023-02-'+str(date)).strftime('%Y-%m-%d'),size=14)

    plt.show()
    
def oneplot_spatial_aq(date, colormax, lat, lon, var_L1, var_L2,
                       uuL1, uuL2, vvL1, vvL2, sites, obs):
    '''
    
    '''
    cmaplevel=np.arange(0,colormax,3)
    proj=ccrs.PlateCarree()

    fig = plt.figure(figsize=(9,4),dpi=300)
    ax = fig.subplots(1,2,subplot_kw={'projection':proj})

    shp_aq = Reader('F:/Data/case_anqing/shapefile/Anqing_district/anqing_district.shp')
    shp_pro = Reader('F:/shapefile/shp_for_ArcGis/ChinaAdminDivisonSHP-master/2. Province/province.shp')
    for i in range(2):
        ax[i].add_feature(cfeat.ShapelyFeature(shp_pro.geometries(),proj,edgecolor='gray',facecolor='None'), linewidth=0.8)
        ax[i].add_feature(cfeat.ShapelyFeature(shp_aq.geometries(),proj,edgecolor='k',facecolor='None'), linewidth=0.8)
        
        gl=ax[i].gridlines(
            xlocs=np.arange(-180, 180 + 1, 0.5), ylocs=np.arange(-90, 90 + 1, 0.5),
            draw_labels=True, x_inline=False, y_inline=False,
            linewidth=0, linestyle='--', color='gray')
        gl.top_labels = False
        gl.right_labels =False
        gl.rotate_labels=False
        
        ax[i].set_extent([115.6, 117.4, 29.7, 31.4],ccrs.PlateCarree()) # anqing

        if i>0:
            gl.left_labels=False

    xgrid=np.size(uuL1,2)
    ygrid=np.size(vvL1,1)
    ngrid=5

    # contour map
    cmap=ax[0].contourf(lon,lat,var_L1[date-1,:,:],transform=proj,cmap='Spectral_r',levels=cmaplevel,extend='both')
    cmap=ax[1].contourf(lon,lat,var_L2[date-1,:,:],transform=proj,cmap='Spectral_r',levels=cmaplevel,extend='both')

    # wind quiver
    ax[0].quiver(lon[0:ygrid:ngrid,0:xgrid:ngrid],lat[0:ygrid:ngrid,0:xgrid:ngrid],
            uuL1[date-1,0:ygrid:ngrid,0:xgrid:ngrid],vvL1[date-1,0:ygrid:ngrid,0:xgrid:ngrid],
            transform=proj,color='k',alpha=1,scale=150,headwidth=3)
    ax[1].quiver(lon[0:ygrid:ngrid,0:xgrid:ngrid],lat[0:ygrid:ngrid,0:xgrid:ngrid],
            uuL2[date-1,0:ygrid:ngrid,0:xgrid:ngrid],vvL2[date-1,0:ygrid:ngrid,0:xgrid:ngrid],
            transform=proj,color='k',alpha=1,scale=150,headwidth=3)
    
    # station obs
    for m in range(np.size(obs.columns)):
        sitename = sites['站点名称'][m]
        ax[0].scatter(sites['经度'][m],sites['纬度'][m],transform=proj,marker='o',s=20,
                  c=obs[sitename][date-1],cmap='Spectral_r',vmin=0,vmax=colormax,edgecolors='k',linewidth=0.5)

    # share colorbar
    fig.subplots_adjust(right=0.9,wspace=0.02)
    position= fig.add_axes([0.92,0.15,0.015,0.7])
    cbar=fig.colorbar(cmap,cax=position)
    cbar.set_ticks(np.arange(0,colormax+1,30))
    cbar.set_label('$\mu$$g$/$m^3$')

    ax[0].set_title('(a) 1000hPa',loc='left')
    ax[1].set_title('(b) 850hPa',loc='left')
    plt.suptitle(pd.to_datetime('2023-02-'+str(date)).strftime('%Y-%m-%d'),size=14)

    plt.show()
    
def batchplots_spatial_d03(dates, colormax, lat, lon, var_L1, var_L2, uuL1, uuL2, vvL1, vvL2):
    '''
    
    '''
    cmaplevel=np.arange(0,colormax,3)
    proj=ccrs.PlateCarree()
    
    for date in dates:
        fig = plt.figure(figsize=(9,4),dpi=300)
        ax = fig.subplots(1,2,subplot_kw={'projection':proj})

        shp_urban = Reader('F:/Data/case_anqing/shapefile/Anqing_urban/urban.shp')
        shp_rural = Reader('F:/Data/case_anqing/shapefile/Anqing_rural/rural.shp')
        shp_pro = Reader('F:/shapefile/shp_for_ArcGis/ChinaAdminDivisonSHP-master/2. Province/province.shp')
        for i in range(2):
            ax[i].add_feature(cfeat.ShapelyFeature(shp_pro.geometries(),proj,edgecolor='gray',facecolor='None'), linewidth=0.8)
            ax[i].add_feature(cfeat.ShapelyFeature(shp_urban.geometries(),proj,edgecolor='k',facecolor='None'), linewidth=0.8)
            ax[i].add_feature(cfeat.ShapelyFeature(shp_rural.geometries(),proj,edgecolor='k',facecolor='None'), linewidth=0.8)

            gl=ax[i].gridlines(
                xlocs=np.arange(-180, 180 + 1, 1), ylocs=np.arange(-90, 90 + 1, 1),
                draw_labels=True, x_inline=False, y_inline=False,
                linewidth=0, linestyle='--', color='gray')
            gl.top_labels = False
            gl.right_labels =False
            gl.rotate_labels=False
        
            ax[i].set_extent([114.8, 118.8, 29.1, 32.6],ccrs.PlateCarree()) # d03

            if i>0:
                gl.left_labels=False

        xgrid=np.size(uuL1,2)
        ygrid=np.size(vvL1,1)
        ngrid=8

        # contour map
        cmap=ax[0].contourf(lon,lat,var_L1[date-1,:,:],transform=proj,cmap='Spectral_r',levels=cmaplevel,extend='both')
        cmap=ax[1].contourf(lon,lat,var_L2[date-1,:,:],transform=proj,cmap='Spectral_r',levels=cmaplevel,extend='both')

        # wind quiver
        ax[0].quiver(lon[0:ygrid:ngrid,0:xgrid:ngrid],lat[0:ygrid:ngrid,0:xgrid:ngrid],
                uuL1[date-1,0:ygrid:ngrid,0:xgrid:ngrid],vvL1[date-1,0:ygrid:ngrid,0:xgrid:ngrid],
                transform=proj,color='k',alpha=1,scale=150,headwidth=3)
        ax[1].quiver(lon[0:ygrid:ngrid,0:xgrid:ngrid],lat[0:ygrid:ngrid,0:xgrid:ngrid],
                uuL2[date-1,0:ygrid:ngrid,0:xgrid:ngrid],vvL2[date-1,0:ygrid:ngrid,0:xgrid:ngrid],
                transform=proj,color='k',alpha=1,scale=150,headwidth=3)
    
        # share colorbar
        fig.subplots_adjust(right=0.9,wspace=0.02)
        position= fig.add_axes([0.92,0.15,0.015,0.7])
        cbar=fig.colorbar(cmap,cax=position)
        cbar.set_ticks(np.arange(0,colormax+1,30))
        cbar.set_label('$\mu$$g$/$m^3$')

        ax[0].set_title('(a) 1000hPa',loc='left')
        ax[1].set_title('(b) 850hPa',loc='left')
        plt.suptitle(pd.to_datetime('2023-02-'+str(date)).strftime('%Y-%m-%d'),size=14)
        
        plt.savefig('D:/Download/d03_'+pd.to_datetime('2023-02-'+str(date)).strftime('%Y-%m-%d'))
        print('saving plot on '+pd.to_datetime('2023-02-'+str(date)).strftime('%Y-%m-%d'))
        plt.close()
    print('Running batch plot successfully')
        

def batchplots_spatial_aq(dates, colormax, lat, lon, var_L1, var_L2,
                          uuL1, uuL2, vvL1, vvL2, sites, obs):
    '''
    
    '''
    cmaplevel=np.arange(0,colormax,3)
    proj=ccrs.PlateCarree()
    
    for date in dates:
        fig = plt.figure(figsize=(9,4),dpi=300)
        ax = fig.subplots(1,2,subplot_kw={'projection':proj})

        shp_aq = Reader('F:/Data/case_anqing/shapefile/Anqing_district/anqing_district.shp')
        shp_pro = Reader('F:/shapefile/shp_for_ArcGis/ChinaAdminDivisonSHP-master/2. Province/province.shp')
        for i in range(2):
            ax[i].add_feature(cfeat.ShapelyFeature(shp_pro.geometries(),proj,edgecolor='gray',facecolor='None'), linewidth=0.8)
            ax[i].add_feature(cfeat.ShapelyFeature(shp_aq.geometries(),proj,edgecolor='k',facecolor='None'), linewidth=0.8)
            
            gl=ax[i].gridlines(
                xlocs=np.arange(-180, 180 + 1, 0.5), ylocs=np.arange(-90, 90 + 1, 0.5),
                draw_labels=True, x_inline=False, y_inline=False,
                linewidth=0, linestyle='--', color='gray')
            gl.top_labels = False
            gl.right_labels =False
            gl.rotate_labels=False
            
            ax[i].set_extent([115.6, 117.4, 29.7, 31.4],ccrs.PlateCarree()) # anqing

            if i>0:
                gl.left_labels=False

        xgrid=np.size(uuL1,2)
        ygrid=np.size(vvL1,1)
        ngrid=5

        # contour map
        cmap=ax[0].contourf(lon,lat,var_L1[date-1,:,:],transform=proj,cmap='Spectral_r',levels=cmaplevel,extend='both')
        cmap=ax[1].contourf(lon,lat,var_L2[date-1,:,:],transform=proj,cmap='Spectral_r',levels=cmaplevel,extend='both')

        # wind quiver
        ax[0].quiver(lon[0:ygrid:ngrid,0:xgrid:ngrid],lat[0:ygrid:ngrid,0:xgrid:ngrid],
                uuL1[date-1,0:ygrid:ngrid,0:xgrid:ngrid],vvL1[date-1,0:ygrid:ngrid,0:xgrid:ngrid],
                transform=proj,color='k',alpha=1,scale=150,headwidth=3)
        ax[1].quiver(lon[0:ygrid:ngrid,0:xgrid:ngrid],lat[0:ygrid:ngrid,0:xgrid:ngrid],
                uuL2[date-1,0:ygrid:ngrid,0:xgrid:ngrid],vvL2[date-1,0:ygrid:ngrid,0:xgrid:ngrid],
                transform=proj,color='k',alpha=1,scale=150,headwidth=3)
        
        # station obs
        for m in range(np.size(obs.columns)):
            sitename = sites['站点名称'][m]
            ax[0].scatter(sites['经度'][m],sites['纬度'][m],transform=proj,marker='o',s=20,
                    c=obs[sitename][date-1],cmap='Spectral_r',vmin=0,vmax=colormax,edgecolors='k',linewidth=0.5)

        # share colorbar
        fig.subplots_adjust(right=0.9,wspace=0.02)
        position= fig.add_axes([0.92,0.15,0.015,0.7])
        cbar=fig.colorbar(cmap,cax=position)
        cbar.set_ticks(np.arange(0,colormax+1,30))
        cbar.set_label('$\mu$$g$/$m^3$')

        ax[0].set_title('(a) 1000hPa',loc='left')
        ax[1].set_title('(b) 850hPa',loc='left')
        plt.suptitle(pd.to_datetime('2023-02-'+str(date)).strftime('%Y-%m-%d'),size=14)
        
        plt.savefig('D:/Download/aq_'+pd.to_datetime('2023-02-'+str(date)).strftime('%Y-%m-%d'))
        print('saving plot on '+pd.to_datetime('2023-02-'+str(date)).strftime('%Y-%m-%d'))
        plt.close()
    print('Running batch plot successfully')
    
def batchplots_aqwithsites(dates, colormax, lat, lon, var_L1,
                          uuL1, vvL1, sites, obs):
    '''
    
    '''
    cmaplevel=np.arange(0,colormax,3)
    proj=ccrs.PlateCarree()
    
    for date in dates:
        fig = plt.figure(figsize=(9,4),dpi=300)
        ax = fig.subplots(1,2,subplot_kw={'projection':proj})

        shp_aq = Reader('F:/Data/case_anqing/shapefile/Anqing_district/anqing_district.shp')
        shp_pro = Reader('F:/shapefile/shp_for_ArcGis/ChinaAdminDivisonSHP-master/2. Province/province.shp')
        for i in range(2):
            ax[i].add_feature(cfeat.ShapelyFeature(shp_pro.geometries(),proj,edgecolor='gray',facecolor='None'), linewidth=0.8)
            ax[i].add_feature(cfeat.ShapelyFeature(shp_aq.geometries(),proj,edgecolor='k',facecolor='None'), linewidth=0.8)
            
            gl=ax[i].gridlines(
                xlocs=np.arange(-180, 180 + 1, 1), ylocs=np.arange(-90, 90 + 1, 1),
                draw_labels=True, x_inline=False, y_inline=False,
                linewidth=0, linestyle='--', color='gray')
            gl.top_labels = False
            gl.right_labels =False
            gl.rotate_labels=False

        ax[0].set_extent([115., 118.5, 29.1, 32.6],ccrs.PlateCarree()) # d03
        ax[1].set_extent([115.6, 117.4, 29.7, 31.4],ccrs.PlateCarree()) # anqing
        
        xgrid=np.size(uuL1,2)
        ygrid=np.size(vvL1,1)
        ngrid=5

        # contour map
        cmap=ax[0].contourf(lon,lat,var_L1[date-1,:,:],transform=proj,cmap='Spectral_r',levels=cmaplevel,extend='both')
        cmap=ax[1].contourf(lon,lat,var_L1[date-1,:,:],transform=proj,cmap='Spectral_r',levels=cmaplevel,extend='both')

        # wind quiver
        ax[0].quiver(lon[0:ygrid:ngrid,0:xgrid:ngrid],lat[0:ygrid:ngrid,0:xgrid:ngrid],
                uuL1[date-1,0:ygrid:ngrid,0:xgrid:ngrid],vvL1[date-1,0:ygrid:ngrid,0:xgrid:ngrid],
                transform=proj,color='k',alpha=1,scale=150,headwidth=3)
        ax[1].quiver(lon[0:ygrid:ngrid,0:xgrid:ngrid],lat[0:ygrid:ngrid,0:xgrid:ngrid],
                uuL1[date-1,0:ygrid:ngrid,0:xgrid:ngrid],vvL1[date-1,0:ygrid:ngrid,0:xgrid:ngrid],
                transform=proj,color='k',alpha=1,scale=150,headwidth=3)
        
        # station obs
        for m in range(np.size(obs.columns)):
            sitename = sites['站点名称'][m]
            ax[1].scatter(sites['经度'][m],sites['纬度'][m],transform=proj,marker='o',s=20,
                    c=obs[sitename][date-1],cmap='Spectral_r',vmin=0,vmax=colormax,edgecolors='k',linewidth=0.5)

        # share colorbar
        fig.subplots_adjust(right=0.9,wspace=0.12)
        position= fig.add_axes([0.92,0.15,0.015,0.7])
        cbar=fig.colorbar(cmap,cax=position)
        cbar.set_ticks(np.arange(0,colormax+1,30))
        cbar.set_label('$\mu$$g$/$m^3$')

        ax[0].set_title('(a)',loc='left')
        ax[1].set_title('(b)',loc='left')
        plt.suptitle(pd.to_datetime('2023-02-'+str(date)).strftime('%Y-%m-%d'),size=14)
        
        plt.savefig('D:/Download/aq_'+pd.to_datetime('2023-02-'+str(date)).strftime('%Y-%m-%d'))
        print('saving plot on '+pd.to_datetime('2023-02-'+str(date)).strftime('%Y-%m-%d'))
        plt.close()
    print('Running batch plot successfully')