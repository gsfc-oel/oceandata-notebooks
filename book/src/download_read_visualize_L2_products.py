# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-trusted
#     notebook_metadata_filter: -all,jupytext,language_info.name
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   language_info:
#     name: python
# ---

# _Author: Guoqing Wang (guoqing.wang@nasa.gov), OB.DAAC Scientist, NASA GSFC; SSAI_
#
# **<ins>Download, read, and visualize level-2 ocean color data</ins>**
#
# This tutorial shows an example of downloading, reading, and plot OB.DAAC standard level 2 ocean color file.

# +
# # !pip install cartopy
# # !pip install netcdf4
# -

import warnings
warnings.filterwarnings("ignore")

# +
from matplotlib import pyplot as plt
import pandas as pd

import numpy as np
import urllib.request
import netCDF4 as nc
import cartopy  #!pip install cartopy
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# -

# **1. Download file**
# <br>In this example, I will use the MODIS_AQUA L2 OC file on April 30, 2023 (AQUA_MODIS.20221007T175001.L2.OC.nc).
#
# - Specify your **appkey**
# <br>[Generate appkey for your Earthdata login credentials](https://oceandata.sci.gsfc.nasa.gov/appkey/) and update the variable "key" correspondingly.

# +
key = 'abcd1234' # paste your appkey here.
filename = 'AQUA_MODIS.20221007T175001.L2.OC.nc' # put here the file name to download
urls = r'https://oceandata.sci.gsfc.nasa.gov/ob/getfile/%s?appkey=%s'%(filename, key)

urllib.request.urlretrieve(urls, filename) # download the file to the current path
# -

# **2. Loading the Level-2 Dataset** <br>
# NASA ocean color data are stored in NetCDF files. They can be read with a few different Python modules. The most popular are netCDF4 and gdal. For this script we’ll focus on netCDF4.<br>
# Loading a dataset is simple, just pass a NetCDF file path to netCDF4.Dataset().

# open file and print the metadata
f = nc.Dataset(filename, 'r')

# A NetCDF file consists of three fundamental components: metadata, dimensions, and variables. Variables encompass both metadata and data. The netCDF4 library enables us to retrieve the metadata and data linked to a NetCDF file.
#
# **3. Retrieve Metadata**
# <br>When we print the dataset f, it provides details about the variables present in the file as well as the groups of variables.

print (f)

# Above you can see information for the file format, data source, data version, citation, dimensions, and variables. In L2 ocean color data, the variables are put in different groups, the ones that we are interested in, such as Rrs, Chla, etc. are in "geophysical_data", and lat, lon are in "navigation_data"

# print grouped variables
print(f.groups.keys())

# **4. Dimensions** 
# <br>Accessing dimensions is akin to accessing file metadata. Each dimension is stored as a dimension class that holds relevant information. To retrieve metadata for all dimensions, one can loop through all the available dimensions, as demonstrated below.

# +
for dim in f.dimensions.values():
    print(dim)
    
# Individual dimensions are accessed like so: f.dimensions['x'].
# -

# **5. Variable Metadata** <br>
# Access variable metadata in the groups of "geophysical_data" and "navigation_data". 

#print(f.groups['geophysical_data'].variables)
# if you just want to see the variable names, use the following code
print(f.groups['geophysical_data'].variables.keys())

# **6. Access Data Values**
# <br>The actual precipitation data values are accessed by array indexing, and a numpy array is returned. All variable data is returned as follows:

# metadata of variable: chlor_a
print(f.groups['geophysical_data'].variables['chlor_a'])

# value of chlor_a
chlor_a = f.groups['geophysical_data'].variables['chlor_a'][:]
print(chlor_a)

# + scrolled=true
# read information of lat, lon
print(f.groups['navigation_data'].variables.keys())
lat = f.groups['navigation_data'].variables['latitude'][:]
lon = f.groups['navigation_data'].variables['longitude'][:]
print(lat.shape)
print(lon.shape)
print(chlor_a.shape)
chlor = np.log10(chlor_a)

# -

# **7. Visualization and mapping**
# <br>Here, we use cartopy for basemap creation and matplotlib to plot data onto the created map.
# The combination of Matplotlib and Cartopy provides a powerful toolkit for creating geospatial visualizations.

# +
# Plot Chlor_a data
fig = plt.figure(figsize=(10, 7))
map_projection = cartopy.crs.PlateCarree()
ax = plt.axes(projection=map_projection)

im = ax.pcolormesh(lon, lat,np.squeeze(chlor),cmap = 'jet', vmin=-2, vmax=1.3)
ax.coastlines()
ax.add_feature(cartopy.feature.STATES, linewidth=0.5)

ax.set_xticks(np.linspace(-90, -52.5, 5), crs=map_projection)
ax.set_yticks(np.linspace(20, 50, 5), crs=map_projection)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

plt.colorbar(im, label='log10(Chlorophyll-a)',)
# +


