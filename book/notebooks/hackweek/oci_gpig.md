---
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Applying Gaussian Pigment (GPig) Phytoplankton Community Composition Algorithm to OCI data

**Authors:** Anna Windle (NASA, SSAI), Max Danenhower (Bowdoin College), Ali Chase (University of Washington) <br>
Last updated: July 28, 2025

<div class="alert alert-success" role="alert">

The following notebooks are **prerequisites** for this tutorial.

- Learn with OCI: [Data Access][oci-data-access]

</div>

<div class="alert alert-info" role="alert">

An [Earthdata Login][edl] account is required to access data from the NASA Earthdata system, including NASA ocean color data. 

</div>

[edl]: https://urs.earthdata.nasa.gov/
[oci-data-access]: https://oceancolor.gsfc.nasa.gov/resources/docs/tutorials/notebooks/oci_data_access/

## Summary

This notebook applies the inversion algorithm described in [Chase et al., 2017][Chase-et-al] to estimate phytoplankton pigment concentrations from PACE OCI Rrs data. This algorithm, called Gaussian Pigment (GPig), is currently being implemented into OBPG's OCSSW software. This work was originally developed in MatLab by Ali Chase and can be found here: https://github.com/alisonpchase/Rrs_inversion_pigments. <br>
It was translated to Python by Max Danenhower, Charles Stern, and Ali Chase, and can be found here: https://github.com/max-danenhower/pace-rrs-inversions-pigments/tree/main. This tutorial demonstrates how to apply the Python GPig algortithm to Level-2 (L2) and Level-3 Mapped (L3M) PACE OCI data. <br>

[Chase-et-al]: https://doi.org/10.1002/2017JC012859

## Learning Objectives

At the end of this notebook you will know:

- How to use a packaged Python project 
- How to run the GPig Algorithm on PACE OCI L2 and L3 data

## Contents

1. [Setup](#1.-Setup)
2. [Run GPig on L2 data](#2.-Run-GPig-on-Level-2-(L2)-data)
3. [Run GPig on L3M data](#3.-Run-GPig-on-L3-mapped-(L3M)-data)

+++

## 1. Setup

The GPig Python code has been packaged to allow it to be easily installed, imported, and reused. We will use pip install to get the current packaged code from Github. You will need to restart the kernel after running this to use udpated packages.

```{code-cell} ipython3
---
collapsed: true
jupyter:
  outputs_hidden: true
---
%pip install git+https://github.com/max-danenhower/pace-rrs-inversions-pigments.git
```

Next, we'll import all of the packages used in this notebook. 

**TODO: Figure out how to use `from gpig import *`, I think this requires editing the `__init__.py`?**

```{code-cell} ipython3
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import earthaccess
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from gpig import L2_utils, L3_utils
```

Set (and persist to your user profile on the host, if needed) your Earthdata Login credentials.

```{code-cell} ipython3
auth = earthaccess.login(persist=True)
```

[back to top](#Contents)

+++

# 2. Run GPig on L2 data

+++

Let's first download the data using the `L2_utils.load()` function. Let's take a look at the docstring:

```{code-cell} ipython3
?L2_utils.load_data
```

Running `L2_utils.load_data()` downloads several data files from NASA EarthData: <br>

* Rrs: PACE OCI Level-2 AOP data products 
* Sea surface salinity: JPL SMAP-SSS V5.0 CAP, 8-day running mean, level 3 mapped, sea surface salinity (SSS) product from the NASA Soil Moisture Active Passive (SMAP) observatory 
* Sea surface temperature: Group for High Resolution Sea Surface Temperature (GHRSST) Level 4 sea surface temperature

Let's run `L2_utils.load_data()` to download data collected on May 5, 2025 corresponding to a bounding box off the coast of Washington, U.S.

```{code-cell} ipython3
rrs, sss, sst = L2_utils.load_data(("2025-05-01", "2025-05-01"), (-127, 40, -126, 41))
```

You should see 3 new folders in your working directory called 'L2_data', 'sal_data', and 'temp_data', with one data file downloaded in each. Let's take a quick look at Rrs at 500 nm:

```{code-cell} ipython3
datatree = xr.open_datatree(rrs)
dataset = xr.merge(datatree.to_dict().values())
dataset = dataset.set_coords(("longitude", "latitude"))

data = dataset["Rrs"].sel({"wavelength_3d": 500})

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})

im = data.plot(
    x="longitude",
    y="latitude",
    vmin=0,
    vmax=0.008,
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs={"label": "Rrs (sr⁻¹)"},
)
ax.set_extent([-135, -115, 35, 55], crs=ccrs.PlateCarree())
ax.coastlines(resolution="10m")

gl = ax.gridlines(
    draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
)
gl.top_labels = False
gl.right_labels = False

plt.show()
```

Now, we can use `L2_utils.estimate_inv_pigments` to calculate phytoplankton pigment concentrations for this data. Let's take a look at the docstring for this function:

```{code-cell} ipython3
?L2_utils.estimate_inv_pigments
```

You can see that this function requires user input to create a boundary box for processing. We will select the bounding box between these coordinates:
45 N, 44 S, -125 E, -126 W. This can take awhile depending on size of bounding box.

```{code-cell} ipython3
l2_pigments = L2_utils.estimate_inv_pigments(rrs, sss, sst)
l2_pigments
```

Let's plot the phytoplankton pigment concentrations:

```{code-cell} ipython3
variables = ["chla", "chlb", "chlc", "ppc"]
titles = ["Chlorophyll-a", "Chlorophyll-b", "Chlorophyll-c", "PPC"]

fig, axs = plt.subplots(2,2, figsize=(8, 8),
    subplot_kw={"projection": ccrs.PlateCarree()},
    gridspec_kw={"wspace": 0.02, "hspace": 0.05},
    constrained_layout=True,
)

for ax, var, title in zip(axs.flat, variables, titles):
    data = l2_pigments[var].values
    lon = l2_pigments["longitude"].values
    lat = l2_pigments["latitude"].values

    data = np.where(data > 0, np.log10(data), np.nan)

    im = ax.pcolormesh(
        lon, lat, data, transform=ccrs.PlateCarree(), cmap="viridis", shading="auto"
    )

    ax.set_title(title)
    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label(f"log₁₀({var})")

plt.show()
```

# 3. Run GPig on L3M data

+++

We can also run GPig on L3M data. Let's look at the `L3_utils.load_data()` docstring:

```{code-cell} ipython3
?L3_utils.load_data
```

We'll use this to download 4km L3M Rrs data, global SSS, and global SST data for June 12, 2024:

```{code-cell} ipython3
rrs, sss, sst = L3_utils.load_data(("2024-06-12", "2024-06-12"), "4km")
```

Let's quickly look at L3M Rrs at 500 nm:

```{code-cell} ipython3
dataset = xr.open_dataset('rrs_data/PACE_OCI.20240612.L3m.DAY.RRS.V3_0.Rrs.4km.nc')
data = dataset["Rrs"].sel({"wavelength": 500})

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})

im = data.plot(
    x="lon",
    y="lat",
    vmin=0,
    vmax=0.008,
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs={"label": "Rrs (sr⁻¹)"},
)
#ax.set_extent([-135, -115, 35, 55], crs=ccrs.PlateCarree())
ax.coastlines(resolution="10m")

gl = ax.gridlines(
    draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
)
gl.top_labels = False
gl.right_labels = False

plt.show()
```

Let's look at what `L3_utils.estimate_inv_pigments` requires as input:

```{code-cell} ipython3
?L3_utils.estimate_inv_pigments
```

We'll provide the Rrs, SSS, and SST file and input a bounding box corresponding to the coast of Washington, U.S.

```{code-cell} ipython3
bbox = (-127, 40, -126, 41)
l3_pigments = L3_utils.estimate_inv_pigments(rrs, sss, sst, bbox)
l3_pigments
```

Let's plot all phytoplankton pigment concentrations:

```{code-cell} ipython3
fig, axs = plt.subplots(2, 2, figsize=(8, 8),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True,
)

variables = ["chla", "chlb", "chlc", "ppc"]
titles = ["Chlorophyll-a", "Chlorophyll-b", "Chlorophyll-c", "PPC"]

for ax, var, title in zip(axs.flat, variables, titles):

    data = l3_pigments[var]
    data_log = np.log10(data.where(data > 0))  

    im = data_log.plot(
        ax=ax,
        transform=ccrs.PlateCarree(), 
        cmap="viridis",
        add_colorbar=False,
    )
    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, label=f"log₁₀({var})")

plt.show()
```

[back to top](#Contents)
