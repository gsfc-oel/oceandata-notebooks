---
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# OCI CLOUD Products

**Authors:** Chamara (NASA, SSAI), Andy (NASA, UMBC), Kirk (NASA), Meng (NASA, SSAI), Sean (NASA, MSU)

[edl]: https://urs.earthdata.nasa.gov/
[oci-data-access]: https://oceancolor.gsfc.nasa.gov/resources/docs/tutorials/notebooks/oci_data_access/

## Summary

This notebook summarizes how to access OCI cloud products (CLDMASK and CLOUD) and visualize the variables they contain.
Note that this notebook is based on an early, preliminary version of the product and is therefore subject to future optimizations and changes.

## Learning Objectives
By the end of this notebook, you will understand:

- How to acquire OCI CLDMASK and CLOUD L2 data
- Available variables in the products
- How to visualize variables

## Contents

1. [Setup](#1.-Setup)
2. [Get Level-2 Data](#2.-Get-Level-2-Data)
3. [Visualizing L2 CLDMASK Variables](#3.-Visualizing-L2-CLDMASK-Variables)
4. [Visualizing L2 CLOUD Variables](#4.-Visualizing-L2-CLOUD-Variables)

+++

## 1. Setup

Begin by importing all of the packages used in this notebook. If your kernel uses an environment defined following the guidance on the [tutorials] page, then the imports will be successful.

[tutorials]: https://oceancolor.gsfc.nasa.gov/resources/docs/tutorials/

```{code-cell} ipython3
from pathlib import Path
from textwrap import wrap

import cartopy.crs as ccrs
import earthaccess
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import requests
import xarray as xr

plt.style.use("seaborn-v0_8-notebook")
```

```{code-cell} ipython3
auth = earthaccess.login(persist=True)
fs = earthaccess.get_fsspec_https_session()
```

[back to top](#Contents)

+++

## 2. Get Level-2 Data

You can find the appropriate product "short name" for CLOUD_MASK and CLOUD products as follows.

```{code-cell} ipython3
results = earthaccess.search_datasets(instrument="oci")
for item in results:
    summary = item.summary()
    print(summary["short-name"])
```

```{code-cell} ipython3
results = earthaccess.search_data(
    short_name="PACE_OCI_L2_CLOUD_MASK_NRT",
    temporal=("2025-07-02", "2025-07-02"),
    bounding_box=(-90, -15, -89, -14),  # (west, south, east, north) if desired
    count=1,
)
paths_cldmask = earthaccess.open(results)
```

```{code-cell} ipython3
paths_cldmask
```

```{code-cell} ipython3
datatree_cldmask = xr.open_datatree(paths_cldmask[0])
# datatree
```

Here we merge all the data group together for convenience in data manipulations.

```{code-cell} ipython3
dataset_cldmask = xr.merge(datatree_cldmask.to_dict().values())
# dataset
```

```{code-cell} ipython3
results = earthaccess.search_data(
    short_name="PACE_OCI_L2_CLOUD_NRT",
    temporal=("2025-07-02", "2025-07-02"),
    bounding_box=(-90, -15, -89, -14),  # (west, south, east, north) if desired
    count=1,
)
paths_cloud = earthaccess.open(results)
```

```{code-cell} ipython3
paths_cloud
```

```{code-cell} ipython3
datatree_cloud = xr.open_datatree(paths_cloud[0])
# datatree
```

```{code-cell} ipython3

dataset_cloud = xr.merge(datatree_cloud.to_dict().values())
# dataset
```

[back to top](#Contents)

+++

## 3. Visualizing L2 CLDMASK Variables

The product includes two cloud flags. The first, cloud_flag, classifies each pixel as either cloudy or clear. The second, cloud_flag_dilated, provides a more conservative classification: it labels pixels as either cloud-free and not adjacent to cloudy pixels, or as cloudy or adjacent to cloudy pixels.

```{code-cell} ipython3
transform = ccrs.PlateCarree(central_longitude=0)
projection = ccrs.PlateCarree()

def geo_axis_tags(ax, crs=ccrs.PlateCarree(central_longitude=0)):
    gl = ax.gridlines(crs=crs, draw_labels=True)
    ax.coastlines()
    # gl = ax.gridlines(draw_labels=True,linestyle='--',\
    #     xlocs=mticker.FixedLocator(np.arange(-180,180.1,10)),ylocs=mticker.FixedLocator(np.arange(-90,90.1,10)))
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 12, "color": "k"}
    gl.ylabel_style = {"size": 12, "color": "k"}

    ax.coastlines()
    return gl
def plot_cloud_flag(dataset,cloud_flag_option,fig,ax,transform):
    '''
    To plot projected maps of cloud_flag
    parameters:
        dataset: xarray.core.dataset.Dataset
        cloud_flag_option: 'cloud_flag' or 'cloud_flag_dilated'
        fig,ax : fig,ax = plt.subplots(figsize=(8,6), subplot_kw={'projection':projection=ccrs.PlateCarree(central_longitude=0)})
        transform : ccrs.PlateCarree(central_longitude=0)
    '''
    values = [0,1,2]
    cloud_flag = dataset[cloud_flag_option].values.copy()
    cloud_flag[np.isnan(cloud_flag)] = 2
    tab10_colors = plt.get_cmap("tab20",3).colors # First 3 colors
    colors = list(tab10_colors) + [tab10_colors[-1]]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(values + [max(values) + 1], cmap.N)

    ctf = ax.pcolormesh(dataset.longitude,dataset.latitude,cloud_flag,cmap=cmap,norm=norm,transform=transform)
    cbar = fig.colorbar(ctf, ax=ax, orientation="vertical")
    cbar.set_ticks(np.array(values)+0.5)
    cbar.set_ticklabels(['clear','cloud','invalid'])
    geo_axis_tags(ax)
```

```{code-cell} ipython3
fig1, ax1 = plt.subplots(figsize=(10, 6), subplot_kw={"projection": projection})
plot_cloud_flag(dataset_cldmask,'cloud_flag',fig1,ax1,transform)
ax1.set_title('cloud_flag')
```

```{code-cell} ipython3
fig1, ax1 = plt.subplots(figsize=(10, 6), subplot_kw={"projection": projection})
plot_cloud_flag(dataset_cldmask,'cloud_flag_dilated',fig1,ax1,transform)
ax1.set_title('cloud_flag_dilated')
```

[back to top](#Contents)

+++

## 4. Visualizing L2 CLOUD Variables

```{code-cell} ipython3
transform = ccrs.PlateCarree(central_longitude=0)
projection = ccrs.PlateCarree()
def extremes_removed_ids(x):
    """
    Returns indices of array x after removing the extreme values
    """
    q3 = np.percentile(x, 75)
    q1 = np.percentile(x, 25)
    xmin = q1 - 1.5 * (q3 - q1)
    xmax = q3 + 1.5 * (q3 - q1)
    return (xmin <= x) * (x < xmax)


def geo_axis_tags(ax, crs=ccrs.PlateCarree(central_longitude=0)):
    gl = ax.gridlines(crs=crs, draw_labels=True)
    ax.coastlines()
    # gl = ax.gridlines(draw_labels=True,linestyle='--',\
    #     xlocs=mticker.FixedLocator(np.arange(-180,180.1,10)),ylocs=mticker.FixedLocator(np.arange(-90,90.1,10)))
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 12, "color": "k"}
    gl.ylabel_style = {"size": 12, "color": "k"}

    ax.coastlines()
    return gl
def plot_cld_phase_21(dataset,fig,ax,transform):
    '''
    Plot cloud phase 2.1 um projected map
    ----------------------
    parameters:
        dataset: xarray.core.dataset.Dataset 
        fig,ax : fig,ax = plt.subplots(figsize=(8,6), subplot_kw={'projection':projection=ccrs.PlateCarree(central_longitude=0)})
        transform : ccrs.PlateCarree(central_longitude=0)
    '''
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    values = [0, 1, 2, 3, 4]
    cmap = mcolors.ListedColormap(colors)
    # Define the boundaries and create a BoundaryNorm
    boundaries = [i - 0.5 for i in range(len(values) + 1)]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ph21 = dataset.cld_phase_21.copy()
    ctf  = ax.pcolormesh(dataset.longitude,dataset.latitude,ph21,cmap=cmap,norm=norm,transform=transform)
    geo_axis_tags(ax)
    cbar = fig.colorbar(ctf, ax=ax, orientation="vertical",label="\n".join(wrap('(1: Clear, 2: Water, 3: Ice, 4: Undetermined)',45)))
    #cbar = add_cb(fig,ctf,ax,orientation='vertical',label="\n".join(wrap('(1: Clear, 2: Water, 3: Ice, 4: Undetermined)',45)))
    cbar.set_ticks([i for i in range(len(values))])
    cbar.set_ticklabels(values)
```

```{code-cell} ipython3
var = "cer_21"
fig1, ax1 = plt.subplots(figsize=(10, 6), subplot_kw={"projection": projection})
cmap = plt.get_cmap("viridis", 20)
_arr = dataset_cloud[var].values
arr_tes = np.ma.masked_array(_arr, mask=np.isnan(_arr))
# vmin, vmax = har_l2_tes.read_from_file('retrievals/%s'%variable_list[i],attrs_local='valid_min'), har_l2_tes.read_from_file('retrievals/%s'%variable_list[i],attrs_local='valid_max')
vmin, vmax = (
    arr_tes.compressed()[extremes_removed_ids(arr_tes.compressed())].min(),
    arr_tes.compressed()[extremes_removed_ids(arr_tes.compressed())].max(),
)
ctf = ax1.pcolormesh(
    dataset_cloud.longitude,
    dataset_cloud.latitude,
    arr_tes,
    transform=transform,
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
)
ax1.set_title(var, size=12)
gl = geo_axis_tags(ax1, crs=transform)
# plt.colorbar(pm, ax=ax_map, orientation="vertical", pad=0.1, label=label)
fig1.colorbar(
    ctf, ax=ax1, orientation="vertical", label="%s [%s]" % (var, dataset_cloud[var].units)
)
```

```{code-cell} ipython3
var = "cot_22"
fig1, ax1 = plt.subplots(figsize=(10, 6), subplot_kw={"projection": projection})
cmap = plt.get_cmap("viridis", 20)
_arr = dataset_cloud[var].values
arr_tes = np.ma.masked_array(_arr, mask=np.isnan(_arr))
# vmin, vmax = har_l2_tes.read_from_file('retrievals/%s'%variable_list[i],attrs_local='valid_min'), har_l2_tes.read_from_file('retrievals/%s'%variable_list[i],attrs_local='valid_max')
vmin, vmax = (
    arr_tes.compressed()[extremes_removed_ids(arr_tes.compressed())].min(),
    arr_tes.compressed()[extremes_removed_ids(arr_tes.compressed())].max(),
)
ctf = ax1.pcolormesh(
    dataset_cloud.longitude,
    dataset_cloud.latitude,
    arr_tes,
    transform=transform,
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
)
ax1.set_title(var, size=12)
gl = geo_axis_tags(ax1, crs=transform)
# plt.colorbar(pm, ax=ax_map, orientation="vertical", pad=0.1, label=label)
fig1.colorbar(
    ctf, ax=ax1, orientation="vertical", label="%s [%s]" % (var, dataset_cloud[var].units)
)
```

```{code-cell} ipython3
fig1, ax1 = plt.subplots(figsize=(10, 6), subplot_kw={"projection": projection})
plot_cld_phase_21(dataset_cloud,fig1,ax1,transform)
ax1.set_title('cld_phase_21')
```

[back to top](#Contents)

```{code-cell} ipython3

```
