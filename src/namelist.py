
# basic information
#! the path changed when using different computer
# progdir = 'F:/Data/Project_Anqing/' #! YF-G15
progdir = 'D:/data/Project_Anqing/' #! A275-Desktop
datadir = progdir + '202307/'
runID   = 'AQ_2307'

# simulation data file
# gridfile = progdir + 'GRIDCRO2D_2023141.nc' #! YF-G15
gridfile  = progdir + 'GRIDCRO2D_2023141.nc' #! A275-Desktop
mcipfile  = datadir + runID + '_mcip.nc'
cmaqfile  = datadir + runID + '_chem.nc'
pafile    = datadir + runID + '_pa.nc'
isamfile1 = datadir + runID + '_isam1.nc'
isamfile2 = datadir + runID + '_isam2.nc'

# observation data file
obsall   = datadir + 'obsdata/allsite.xlsx'
obsurban = datadir + 'obsdata/urban.xlsx'
obsother = datadir + 'obsdata/others.xlsx'

# shapefile
shpall   = progdir + 'shapefile/Anqing/Anqing.shp'
shpurban = progdir + 'shapefile/Anqing_urban/urban.shp'
shprural = progdir + 'shapefile/Anqing_rural/rural.shp'
shpmap   = progdir + 'shapefile/Anqing_district/anqing_district.shp'

# time range
timestart = '2023-07-01T00'
timeend   = '2023-07-31T23'