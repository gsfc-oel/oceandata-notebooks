---
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Run Level-2 Generator (l2gen) OCSSW program on OCI data

**Authors:** Anna Windle (NASA, SSAI), Jeremy Werdell (NASA) <br>
Last updated: July 23, 2025

<div class="alert alert-success" role="alert">

The following notebooks are **prerequisites** for this tutorial.

- Learn with OCI: [Data Access][oci-data-access]
- Learn with OCI: [Installing and Running OCSSW Command-line Tools][ocssw_install]

</div>

<div class="alert alert-info" role="alert">

An [Earthdata Login][edl] account is required to access data from the NASA Earthdata system, including NASA ocean color data.

</div>

<div class="alert alert-info" role="alert">

This notebook was desgined to use cloud-enabled OCSSW programs, which are available in OCSSW tag ___ or higher. Cloud-enabled OCSSW programs can only be run on an AWS EC2 instance, such as CryoCloud.  **TODO: Add more info here on setting AWS credentials for this to work...**

</div>

[edl]: https://urs.earthdata.nasa.gov/
[oci-data-access]: https://oceancolor.gsfc.nasa.gov/resources/docs/tutorials/notebooks/oci_data_access/
[ocssw_install]: https://oceancolor.gsfc.nasa.gov/resources/docs/tutorials/notebooks/oci_ocssw_install/

## Summary

The [OceanColor Science SoftWare][ocssw] (OCSSW) repository is the operational data processing software package of NASA's Ocean Biology Distributed Active Archive Center (OB.DAAC). OCSSW is publically available through the OB.DAAC's [Sea, earth, and atmosphere Data Analysis System][seadas] (SeaDAS), which provides a complete suite of tools to process, display and analyze ocean color data. SeaDAS is a desktop application that provides GUI access to OCSSW, but command line interfaces (CLI) also exist, which we can use to write processing scripts and notebooks without SeaDAS.

The Level-2 Generator (`l2gen`) program included in OCSSW is used to generate aquatic Level-2 (L2) data products from calibrated top-of-atmosphere (TOA) radiances. Specifically, `l2gen` atmospherically corrects spectral TOA Level-1B (L1B) radiances to obtain geophysical products, such as spectral remote-sensing reflectances (Rrs) and near-surface concentrations of the photosynthetic pigment chlorophyll-a. More information on `l2gen` methods can be found in the [Rrs Algorithm Theoretrical Basis Document]. 

This tutorial will demonstrate how to process PACE OCI L1B data through the `l2gen` default settings to retrieve the standard L2 ocean color 'OC' data suite. Done right, this data should be *exactly* what you would download from the NASA EarthData Cloud. This tutorial will also demonstrate how to modify the operation of `l2gen` configurations based on your research needs. 

[seadas]: https://seadas.gsfc.nasa.gov/
[ocssw]: https://oceandata.sci.gsfc.nasa.gov/ocssw
[Rrs Algorithm Theoretrical Basis Document]: https://www.earthdata.nasa.gov/apt/documents/rrs/v1.1

## Learning Objectives

At the end of this notebook you will know:

- How to navigate and open files within the OCSSW directory
- How to process L1B data to L2 using `l2gen` with geographical boundaries
- How to extract geographic regions from a L2 file and create a new file
- How to make modifications to `l2gen` based on your research needs

## Contents

1. [Setup](#1.-Setup)
2. [Search and access L1B Data ](#2.-Search-and-access-L1B-data)
3. [Run l2gen with default configurations](#3.-Run-l2gen-with-default-configurations)
4. [Compare the newly generated file with a standard OB.DAAC file](#4.-Compare-the-newly-generated-file-with-a-standard-OB.DAAC-file)
5. [Run l2gen with modifications to configurations](#5.-Run-l2gen-with-modifications-to-configurations)

+++

# 1. Setup

Begin by importing all of the packages used in this notebook. If your kernel uses an environment defined following the guidance on the [tutorials] page, then the imports will be successful.

[tutorials]: https://oceancolor.gsfc.nasa.gov/resources/docs/tutorials/

```{code-cell} ipython3
import csv
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import earthaccess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm
```

Next, we'll set up the OCSSW programs. 

<div class="alert alert-warning">

OCSSW programs are typically run using the Bash CLI.  Here, we employ a Bash terminal-in-a-cell using the [IPython magic][magic] command `%%bash`. In the specific case of OCSSW programs, the required Bash environment and environmental variables needed for that cell must be set up by loading `$OCSSWROOT/OCSSW_bash.env`.

</div>

Every `%%bash` cell that calls an OCSSW program needs to `source` this environment definition file shipped with OCSSW, as the environment and environmental variables it establishes are not retained from one cell to the next. We can, however, define the `OCSSWROOT` environmental variable in a way that effects every `%%bash` cell.

[magic]: https://ipython.readthedocs.io/en/stable/interactive/magics.html#built-in-magic-commands

```{code-cell} ipython3
ocsswroot = os.environ.setdefault("OCSSWROOT", "/opt/ocssw")
assert os.path.isdir(ocsswroot)
```

In every cell where we wish to call an OCSSW command, several lines of code need to appear first to establish the required environment. These are shown below, followed by a call to the OCSSW program `install_ocssw` to see which version (tag) of OCSSW is installed.

```{code-cell} ipython3
:scrolled: true

%%bash
source $OCSSWROOT/OCSSW_bash.env

install_ocssw --installed_tag
```

V tags are operational tags while T tags are testing tags. To install to a different tag, ______.

<b>TODO: Should we include information on how to install a different tag? Maybe not because they won't be able to in CC?

+++

## Setting up AWS S3 credentials

<b> TODO: Ian- can you add more information on why these are required, please? and why they only work for one hour?

```{code-cell} ipython3
credentials = earthaccess.get_s3_credentials(provider="OB_CLOUD")
os.environ.update(
    {
        "AWS_ACCESS_KEY_ID": credentials["accessKeyId"],
        "AWS_SECRET_ACCESS_KEY": credentials["secretAccessKey"],
        "AWS_SESSION_TOKEN": credentials["sessionToken"],
    }
)
```

## Navigating OCSSW

+++

Within the OCSSW directory, there are sub-directories that contain files that control OCSSW processing and set default parameterizations of the various codes. Let's look at the files in the `~/occsw/share/common` directory, which includes the files available to all satellite instruments considered in the OCSSW ecosystem:

```{code-cell} ipython3
:scrolled: true

%%bash
source $OCSSWROOT/OCSSW_bash.env

ls $OCSSWROOT/share/common
```

Clearly, there are an intimidatingly large number of files required to process satellite ocean color data!  The most valuable to the scientific end user are the .par (for parameter) files. .par files are plain text files used to store configuration settings for OCSSW programs. They typically define the inputs, outputs, and the different options one can modify for each program.

Tip: You can click on the blue vertical bar on the left side of an output to collapse it.

+++

<div class="alert alert-warning">
    
Fun fact: `l2gen` used to be named "Multi Sensor Level 1 to 2", or MSL12. That is why many .par files start with 'msl12'. The OB.DAAC renamed most programs with names like 'l2gen', or 'l3mapgen' to more clearly identify their purpose. But, the .par files for use with l2gen still have names that begin with 'msl12'. Just remember msl12 = l2gen.

</div>

+++

Now, let's look at the PACE OCI-specific files within the `~/ocssw/share/oci` directory:

```{code-cell} ipython3
:scrolled: true

%%bash
source $OCSSWROOT/OCSSW_bash.env

ls $OCSSWROOT/share/oci
```

Let's open 'msl12_defaults.par', where the `l2gen` default parameters for OCI are defined:

```{code-cell} ipython3
:scrolled: true

%%bash
source $OCSSWROOT/OCSSW_bash.env

cat $OCSSWROOT/share/oci/msl12_defaults.par
```

This .par file lists the default configuration for standard `l2gen` processing. If nothing is modified from this .par file, `l2gen` will process a L1B file to L2 containing the 'OC' data suite products and it will be *exactly* the same as the L2 data file that OBPG processes and ingests into NASA EarthData.

+++

<div class="alert alert-warning">
    
<b>Here are some examples of other aquatic data suites that may be of interest:
* AOP: Apparent Optical Properties 
* BGC: Biogeochemical Propeties 
* IOP: Inherent Optical Properties
* PAR: Photosynthetic available radiation

</b>

</div>

+++

Tip: You can also see OCSSW parameter options by running 'l2gen --help':

```{code-cell} ipython3
---
collapsed: true
jupyter:
  outputs_hidden: true
---
%%bash
source $OCSSWROOT/OCSSW_bash.env

l2gen --help
```

## Writing OCSSW parameter files

+++

User-generated parameter files provide a convenient way to control `l2gen` processing. 

For example, from a CLI, providing `l2gen` the names of the input L1B and output L2 files can be accomplished via:<br>
`l2gen ifile=data/PACE_OCI.20250507T170659.L1B.V3.nc ofile=data/PACE_OCI.20250507T170659.L2.V3.nc`

Alternatively, a user-defined par file, say <i>l2gen.par</i>, can be created with the following two lines of content:
<pre>ifile=data/PACE_OCI.20250507T170659.L1B.V3.nc
ofile=data/PACE_OCI.20250507T170659.L2.V3.nc</pre>

and, `l2gen` can now be called as:<br>
`l2gen par=l2gen.par`

You can imagine that the .par file option becomes far more convenient when many changes from default are desired.

+++

We will define a function to help write OCSSW parameter files, which is needed several times in this tutorial. To write the results in the format understood by OCSSW, this function uses the csv.writer from the Python Standard Library. Instead of writing comma-separated values, however, we specify a non-default delimiter to get equals-separated values. Not something you usually see in a data file, but it's better than writing our own utility from scratch!

```{code-cell} ipython3
def write_par(path, par):
    """Prepare a parameter file to be read by one of the OCSSW tools.

    Using a parameter file (a.k.a. "par file") is equivalent to specifying
    parameters on the command line.

    Parameters
    ----------
    path
        where to write the parameter file
    par
        the parameter names and values included in the file
    """
    with open(path, "w") as file:
        writer = csv.writer(file, delimiter="=")
        writer.writerows(par.items())
```

[back to top](#Contents)

+++

# 2. Search and access L1B data 

Let's use the `earthaccess` Python package to search and download a L1B and a L2 file.

Set (and persist to your user profile on the host, if needed) your Earthdata Login credentials.

```{code-cell} ipython3
auth = earthaccess.login(persist=True)
```

```{code-cell} ipython3
tspan = ("2025-05-07", "2025-05-07")
bbox = (-76.75, 36.97, -74.74, 39.01)

results = earthaccess.search_data(
    short_name="PACE_OCI_L1B_SCI",
    temporal=tspan,
    bounding_box=bbox,
)
results[0]
```

```{code-cell} ipython3
:scrolled: true

l1b_paths = earthaccess.open(results)
l1b_paths
```

Now let's do the same for the corresponding L2 file, which we'll use later.

```{code-cell} ipython3
results = earthaccess.search_data(
    short_name="PACE_OCI_L2_AOP_NRT",
    temporal=tspan,
    bounding_box=bbox,
)
results[0]
```

```{code-cell} ipython3
:scrolled: true

l2_paths = earthaccess.open(results)
l2_paths
```

Next, let's define variables with the path of the input file to process using `l2gen`and a corresponding L2 output file that we'll create.

+++

And let's do a quick plot of a `rhot_red` wavelength to see what the data looks like:

<b> TODO: This plotting takes a long time, is there a better way to do this?

```{code-cell} ipython3
dataset = xr.open_datatree(l1b_paths[0])
dataset = xr.merge(dataset.to_dict().values())
dataset = dataset.set_coords(("longitude", "latitude"))
plot = dataset["rhot_red"].sel({"red_bands": 100}).plot()
```

[back to top](#Contents)

+++

# 3. Run `l2gen` with default configurations

+++

Let's now run `l2gen` using its default configuration. 

Before we do this, however, there is one additional step required to <i>exactly</i> replicate an OB.DAAC-generated L2 file. The algorithms within `l2gen` require ancillary data such as meterological information, ozone concentrations, and sea surface temperatures. To use the corresponding ancillary data for the given date and region, we need to run the `getanc` OCSSW function. If `getanc` is not used, `l2gen` uses climatological data found in `~/occsw/share/common`. 

We can run `getanc -h` to see options for the program:

```{code-cell} ipython3
---
collapsed: true
jupyter:
  outputs_hidden: true
---
%%bash
source $OCSSWROOT/OCSSW_bash.env

getanc -h
```

Let's run it on our L1B file:

```{code-cell} ipython3
%%bash
source $OCSSWROOT/OCSSW_bash.env

getanc s3://ob-cumulus-prod-public/PACE_OCI.20250507T170659.L1B.V3.nc -u 
```

You'll notice that a file named "PACE_OCI.202507T170659.L1B.V3.nc.anc" now appears in your working directory. Reading this file, you can see that ancillary files are saved in the `~/ocssw/var/anc/` directory.  Note that this file also provides text in the correct format for use in a .par file.  

Now, we'll make a .par file that has an ifile, ofile, corresponding ancillary data, and latitude and longitude boundaries. Trust us ... subsetting the L1B file to a smaller region makes processing time faster for this demo!

<b> TODO: explain that turning off uncertaintities makes it goes faster, but have to take out Rrs_unc in l2prod... want to discuss if this is worth it or not

Let's first make a folder called 'data' to store the outputs:

```{code-cell} ipython3
os.makedirs("data", exist_ok=True)
```

```{code-cell} ipython3
par = {
    "ifile": "s3://ob-cumulus-prod-public/PACE_OCI.20250507T170659.L1B.V3.nc",
    "ofile": "data/PACE_OCI.20250507T170659.L2.V3.nc",
    "north": 39,
    "south": 35,
    "west": -76,
    "east": -74.5,
    "icefile": "/opt/ocssw/var/anc/2025/127/20250507120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc",
    "met1": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T170000.MET.nc",
    "met2": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T180000.MET.nc",
    "met3": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T180000.MET.nc",
    "ozone1": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T170000.MET.nc",
    "ozone2": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T180000.MET.nc",
    "ozone3": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T180000.MET.nc",
    "sstfile": "/opt/ocssw/var/anc/2025/127/20250507120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc",
    "l2prod": "Rrs,aot_865,angstrom,avw,nflh",
    "proc_uncertainty": 0,
}
write_par("l2gen.par", par)
```

A file named "l2gen.par" should now appear in your working directory.

Now, let's run l2gen using this new .par file. This can take several minutes.

```{code-cell} ipython3
---
collapsed: true
jupyter:
  outputs_hidden: true
---
%%bash
source $OCSSWROOT/OCSSW_bash.env

l2gen par=l2gen.par 2>/dev/null
```

You'll know `l2gen` processing is finished when you see "Processing Completed" at the end of the cell output. 

Let's open up this new L2 data using XArray's open_datatree function:

```{code-cell} ipython3
dat = xr.open_datatree(par["ofile"], decode_timedelta=True)
dat = xr.merge(dat.to_dict().values())
dat = dat.set_coords(("longitude", "latitude"))
dat
```

We can see the variables included in the standard OC data suite. Let's do a quick plot of Rrs at 550 nm:

```{code-cell} ipython3
dat["Rrs"].sel({"wavelength_3d": 550}).plot(vmin=0, vmax=0.008)
```

[back to top](#Contents)

+++

# 4. Compare the newly generated file with a standard OB.DAAC file

+++

Remember the OB.DAAC L2 file we previously downloaded?  Let's see how it compares with the L2 file we generated ourselves.

Note, however, that the L2 file we downloaded includes the full granule, whereas our homegrown L2 file only includes the geographic bounding box of 35 to 39 N and -76 to -74.5 W. So, let's pause briefly to learn how to extract a geographic region from a L2 file. OCSSW provides the tools to do so and the process includes two steps.

First, we'll use the program `lonlat2pixline` to identify the scan line and pixel boundaries that correspond to our latitude and longitude coordinates within the full L2 granule.  Recall that you can see all the options for OCSSW programs by calling them without any arguments.

```{code-cell} ipython3
---
collapsed: true
jupyter:
  outputs_hidden: true
---
%%bash
source $OCSSWROOT/OCSSW_bash.env

# the order to input coordinates is west, south, east, north
lonlat2pixline s3://ob-cumulus-prod-public/PACE_OCI.20250507T170659.L2.OC_AOP.V3_0.NRT.nc -76.0 35.0 -74.5 39.0
```

This output gets fed into `l2extract` to create a new, smaller file that only includes our defined geographic boundaries. The arguments are input file, start pixel, end pixel, start line, end line, sampling substep for pixels and lines (where 1 = every pixel), and output file.

```{code-cell} ipython3
%%bash
source $OCSSWROOT/OCSSW_bash.env

l2extract s3://ob-cumulus-prod-public/PACE_OCI.20250507T170659.L2.OC_AOP.V3_0.NRT.nc 175 310 1270 1651 1 1 data/PACE_OCI.20250507T170659.L2.OC_AOP.V3_0.NRT_sub.nc
```

This should have created a file named "PACE_OCI.20250507T170659.L2.OC_AOP.V3_0.NRT_sub.nc" in the data subdirectory.

Let's open it and see how it compares with the L2 file we generated.

```{code-cell} ipython3
:scrolled: true

dat_sub = xr.open_datatree("data/PACE_OCI.20250507T170659.L2.OC_AOP.V3_0.NRT_sub.nc", decode_timedelta=True)
dat_sub = xr.merge(dat_sub.to_dict().values())
dat_sub = dat_sub.set_coords(("longitude", "latitude"))
dat_sub
```

```{code-cell} ipython3
dat_sub["Rrs"].sel({"wavelength_3d": 550}).plot(vmin=0, vmax=0.008)
```

The two maps of Rrs(550) look extremely similar.  But, let's compare the data in a scatter plot to be sure.

```{code-cell} ipython3
fig, ax = plt.subplots()

x = dat["Rrs"].sel({"wavelength_3d": 550})
y = dat_sub["Rrs"].sel({"wavelength_3d": 550})

ax.scatter(x, y, s=20)
ax.set_xlabel("OB.DAAC Rrs(550)")
ax.set_ylabel("User produced Rrs(550)")
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="black")
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)

plt.show()
```

Other than a negligible number of oddball points, the data are identical. This shows that an end-user can exactly reproduce the data distributed by the OB.DAAC!

<b> TODO: Sometimes I run it, and it's 1:1 with a few oddball points. Other times there is more noise. Not sure why this is happening... </b>

+++

[back to top](#Contents)

+++

# 5. Run l2gen with modifications to configurations

+++

## Selecting biogeochemical data products

+++

Let's say you want to run `l2gen` to retrieve biogeochemical data products, such as chlorophyll a concentrations. There are two ways to do so. First, you can assign a specific product suite.  For chlorophyll, this is done by adding "suite=BGC" to your .par file.  Second, you can explicitly define your output products in a list using the "l2prod" keyword. Consider this example:

<pre>l2prod=Rrs_vvv,chlor_a,poc,l2flags</pre>

"Rrs_vvv" indicates all visible Rrs, "chlor_a" is chlorophyll-a, "poc" is particulate organic carbon, and "[l2flags][l2flags]" is the bitwise operator that identifies processing flags assigned to each pixel (you <b>always</b> want to include [l2flags][l2flags] as an output product!). 

Tip: You can run `get_product_info l sensor=oci` to see the many many products l2gen can produce. 


Let's write a new .par file named "l2gen_mod.par" to define the L2 products listed above and rerun `l2gen`. 

[l2flags]: https://oceancolor.gsfc.nasa.gov/resources/atbd/ocl2flags/

```{code-cell} ipython3
par = {
    "ifile": "s3://ob-cumulus-prod-public/PACE_OCI.20250507T170659.L1B.V3.nc",
    "ofile": "data/PACE_OCI.20250507T170659.L2_mod.V3.nc",
    "l2prod": "Rrs_vvv,chlor_a,poc,l2_flags",
    "north": 39,
    "south": 35,
    "west": -76,
    "east": -74.5,
    "icefile": "/opt/ocssw/var/anc/2025/127/20250507120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc",
    "met1": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T170000.MET.nc",
    "met2": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T180000.MET.nc",
    "met3": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T180000.MET.nc",
    "ozone1": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T170000.MET.nc",
    "ozone2": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T180000.MET.nc",
    "ozone3": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T180000.MET.nc",
    "sstfile": "/opt/ocssw/var/anc/2025/127/20250507120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc",
}
write_par("l2gen_mod.par", par)
```

```{code-cell} ipython3
---
collapsed: true
jupyter:
  outputs_hidden: true
---
%%bash
source $OCSSWROOT/OCSSW_bash.env

l2gen par=l2gen_mod.par
```

A new L2 file should have appeared in your data folder.  Let's open it using XArray again and plot the chlorophyll-a product:

```{code-cell} ipython3
dat_mod = xr.open_datatree(par["ofile"], decode_timedelta=True)
dat_mod = xr.merge(dat_mod.to_dict().values())
dat_mod = dat_mod.set_coords(("longitude", "latitude"))
dat_mod.chlor_a.plot(norm=LogNorm(vmin=0.01, vmax=2))
```

For fun, let's plot chlor_a again, but with some additional plotting functions.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})

dat_mod.chlor_a.plot(
    ax=ax,
    x="longitude",
    y="latitude",
    transform=ccrs.PlateCarree(),
    cmap="ocean_r",
    norm=LogNorm(vmin=0.01, vmax=1.5),
    cbar_kwargs={"label": "Log Chlorophyll-a (mg m⁻³)"},
)

ax.coastlines(resolution="10m")
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.gridlines(draw_labels=True)

plt.show()
```

## Disabling BRDF correction

+++

As a final example of the far-reaching utility that `l2gen` provides an end user, let's exercise one more example where we disable the standard bidirectional reflectance distribution function (BRDF) correction and see how it changes the retrieved chlor_a values. The default BRDF is 'brdf_opt=7', which is Morel f/Q + Fresnel solar + Fresnel sensor.

While a rather simple case-study, we hope it will introduce the practioner to an improved understanding of `l2gen`'s operation and the sensitivity of derived reflectances (and, therefore, biogeochemical variables) to choices made within the standard processing scheme.

```{code-cell} ipython3
par = {
    "ifile": "s3://ob-cumulus-prod-public/PACE_OCI.20250507T170659.L1B.V3.nc",
    "ofile": "data/PACE_OCI.20250507T170659.L2_brdf.V3.nc",
    "l2prod": "Rrs_vvv,chlor_a,poc,l2_flags",
    "brdf_opt": 0,
    "north": 39,
    "south": 35,
    "west": -76,
    "east": -74.5,
    "icefile": "/opt/ocssw/var/anc/2025/127/20250507120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc",
    "met1": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T170000.MET.nc",
    "met2": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T180000.MET.nc",
    "met3": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T180000.MET.nc",
    "ozone1": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T170000.MET.nc",
    "ozone2": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T180000.MET.nc",
    "ozone3": "/opt/ocssw/var/anc/2025/127/GMAO_MERRA2.20250507T180000.MET.nc",
    "sstfile": "/opt/ocssw/var/anc/2025/127/20250507120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc",
}
write_par("l2gen_brdf.par", par)
```

```{code-cell} ipython3
---
collapsed: true
jupyter:
  outputs_hidden: true
---
%%bash
source $OCSSWROOT/OCSSW_bash.env

l2gen par=l2gen_brdf.par
```

A new L2 file should have appeared in your data folder.  Let's open it using XArray again and plot the chlorophyll-a product:

```{code-cell} ipython3
dat_brdf = xr.open_datatree(par["ofile"], decode_timedelta=True)
dat_brdf = xr.merge(dat_brdf.to_dict().values())
dat_brdf = dat_brdf.set_coords(("longitude", "latitude"))
dat_brdf
```

```{code-cell} ipython3
dat_brdf.chlor_a.plot(norm=LogNorm(vmin=0.01, vmax=2))
```

This figure looks similar to what we produced in Section 3, but let's make a scatter plot to be sure ...

```{code-cell} ipython3
fig, ax = plt.subplots()

x = dat_mod.chlor_a
y = dat_brdf.chlor_a

ax.scatter(x, y, s=20)
ax.set_xlabel("default Chl a")
ax.set_ylabel("disabled BRDF Chl a")
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="black")
ax.set_xscale("log")
ax.set_yscale("log")

plt.tight_layout()
plt.show()
```

<b> TODO: Figure out why points aren't different? I think they should be different...

+++

[back to top](#Contents)

<div class="alert alert-info" role="alert">

You have completed the notebook on "Running the Level-2 Generator (l2gen) OCSSW program on OCI data". We suggest looking at the notebook on "Running l2gen's Generalized Inherent Optical Property (GIOP) model on OCI data" tutorial to learn more about deriving IOP products from PACE OCI data. 

</div>
