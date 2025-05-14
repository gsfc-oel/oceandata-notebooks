# Processing with OCSSW Command Line Interface (CLI)

**Authors:** Carina Poulin (NASA, SSAI), Ian Carroll (NASA, UMBC), Anna Windle (NASA, SSAI)

<div class="alert alert-success" role="alert">

The following notebooks are **prerequisites** for this tutorial.

- Learn with OCI: [Data Access][oci-data-access]
- Learn with OCI: [Installing and Running OCSSW Command-line Tools][ocssw_install]

</div>

<div class="alert alert-info" role="alert">

An [Earthdata Login][edl] account is required to access data from the NASA Earthdata system, including NASA ocean color data.

</div>

[edl]: https://urs.earthdata.nasa.gov/
[oci-data-access]: https://oceancolor.gsfc.nasa.gov/resources/docs/tutorials/notebooks/oci_data_access/
[ocssw_install]: https://oceancolor.gsfc.nasa.gov/resources/docs/tutorials/notebooks/oci_ocssw_install/

## Summary

[SeaDAS][seadas] is the official data analysis sofware of NASA's Ocean Biology Distributed Active Archive Center (OB.DAAC); used to process, display and analyse ocean color data. SeaDAS is a dektop application that includes the Ocean Color Science Software (OCSSW) libraries. There is also a command line interface (CLI) for the OCSSW libraries, which we can use to write processing scripts and notebooks.

This tutorial will show you how to process PACE OCI data using the sequence of OCSSW programs `l2gen`, `l2bin`, and `l3mapgen`.

[seadas]: https://seadas.gsfc.nasa.gov/

## Learning Objectives

At the end of this notebok you will know:
* How to process Level-1B data to Level-2 with `l2gen`
* How to merge two images with `l2bin`
* How to create a map with `l3mapgen`

## Contents

1. [Setup](#1.-Setup)
2. [Get L1B Data](#2.-Get-L1B-Data)
3. [Process L1B Data with `l2gen`](#3.-Process-L1B-Data-with-l2gen)
4. [Composite L2 Data with `l2bin`](#4.-Composite-L2-Data-with-l2bin)
5. [Make a Map from Binned Data with `l3mapgen`](#5.-Make-a-Map-from-Binned-Data-with-l3mapgen)

+++

## 1. Setup

Begin by importing all of the packages used in this notebook. If your kernel uses an environment defined following the guidance on the [tutorials] page, then the imports will be successful.

[tutorials]: https://oceancolor.gsfc.nasa.gov/resources/docs/tutorials/

```{code-cell}
import csv
import os

import cartopy.crs as ccrs
import earthaccess
import matplotlib.pyplot as plt
import xarray as xr
from IPython.display import display
```

+++ {"lines_to_next_cell": 2}

We are also going to define a function to help write OCSSW parameter files, which
is needed several times in this tutorial. To write the results in the format understood
by OCSSW, this function uses the `csv.writer` from the Python Standard Library. Instead of
writing comma-separated values, however, we specify a non-default delimiter to get
equals-separated values. Not something you usually see in a data file, but it's better than
writing our own utility from scratch!

```{code-cell}
def write_par(path, par):
    """Prepare a "par file" to be read by one of the OCSSW tools.

    Using a parameter file is equivalent to specifying parameters
    on the command line.

    Args:
        path (str): where to write the parameter file
        par (dict): the parameter names and values included in the file

    """
    with open(path, "w") as file:
        writer = csv.writer(file, delimiter="=")
        writer.writerows(par.items())
```

The Python docstring (fenced by triple quotation marks in the function definition) is not
essential, but it helps describe what the function does.

```{code-cell}
help(write_par)
```

[back to top](#Contents)

+++

## 2. Get L1B Data


Set (and persist to your user profile on the host, if needed) your Earthdata Login credentials.

```{code-cell}
auth = earthaccess.login(persist=True)
```

We will use the `earthaccess` search method used in the OCI Data Access notebook. Note that Level-1B (L1B) files
do not include cloud coverage metadata, so we cannot use that filter. In this search, the spatial filter is
performed on a location given as a point represented by a tuple of latitude and longitude in decimal degrees.

```{code-cell}
tspan = ("2024-04-27", "2024-04-27")
location = (-56.5, 49.8)
```

The `search_data` method accepts a `point` argument for this type of location.

```{code-cell}
results = earthaccess.search_data(
    short_name="PACE_OCI_L1B_SCI",
    temporal=tspan,
    point=location,
)
```

```{code-cell}
results[0]
```

Download the granules found in the search.

```{code-cell}
paths = earthaccess.download(results, local_path="L1B")
```

While we have the downloaded location stored in the list `paths`, store one in a variable we won't overwrite for future use.

```{code-cell}
l2gen_ifile = paths[0]
```

The Level-1B files contain top-of-atmosphere reflectances, typically denoted as $\rho_t$.
On OCI, the reflectances are grouped into blue, red, and short-wave infrared (SWIR) wavelengths. Open
the dataset's "observatin_data" group in the netCDF file using `xarray` to plot a "rhot_red"
wavelength.

```{code-cell}
dataset = xr.open_dataset(l2gen_ifile, group="observation_data")
plot = dataset["rhot_red"].sel({"red_bands": 100}).plot()
```

This tutorial will demonstrate processing this Level-1B granule into a Level-2 granule. Because that can
take several minutes, we'll also download a couple of Level-2 granules to save time for the next step of compositing multiple Level-2 granules into a single granule.

```{code-cell}
location = [(-56.5, 49.8), (-55.0, 49.8)]
```

Searching on a location defined as a line, rather than a point, is a good way to get granules that are
adjacent to eachother. Pass a list of latitude and longitude tuples to the `line` argument of `search_data`.

```{code-cell}
results = earthaccess.search_data(
    short_name="PACE_OCI_L2_BGC_NRT",
    temporal=tspan,
    line=location,
)
```

```{code-cell}
for item in results:
    display(item)
```

```{code-cell}
paths = earthaccess.download(results, "L2_BGC")
paths
```

While we have the downloaded location stored in the list `paths`, write the list to a text file for future use.

```{code-cell}
paths = [str(i) for i in paths]
with open("l2bin_ifile.txt", "w") as file:
    file.write("\n".join(paths))
```

[back to top](#Contents)

+++

## 3. Process L1B Data with `l2gen`

At Level-1, we neither have geophysical variables nor are the data projected for easy map making. We will need to process the L1B file to Level-2 and then to Level-3 to get both of those. Note that Level-2 data for many geophysical variables are available for download from the OB.DAAC, so you often don't need the first step. However, the Level-3 data distributed by the OB.DAAC are global composites, which may cover more Level-2 scenes than you want. You'll learn more about compositing below. This section shows how to use `l2gen` for processing the L1B data to L2 using customizable parameters.

<div class="alert alert-warning">

OCSSW programs are run from the command line in Bash, but we can have a Bash terminal-in-a-cell using the [IPython magic][magic] command `%%bash`. In the specific case of OCSSW programs, the Bash environment created for that cell must be set up by loading `$OCSSWROOT/OCSSW_bash.env`.

</div>

Every `%%bash` cell that calls an OCSSW program needs to `source` the environment
definition file shipped with OCSSW, because its effects are not retained from one cell to the next.
We can, however, define the `OCSSWROOT` environment variable in a way that effects every `%%bash` cell.

[magic]: https://ipython.readthedocs.io/en/stable/interactive/magics.html#built-in-magic-commands

```{code-cell}
os.environ.setdefault("OCSSWROOT", "/tmp/ocssw")
```

Then we need a couple lines, which will appear in multiple cells below, to begin a Bash cell initiated with the `OCSSW_bash.env` file.

Using this pattern, run the `l2gen` command with the single argument `help` to view the extensive list of options available. You can find more information about `l2gen` and other OCSSW functions on the [seadas website](https://seadas.gsfc.nasa.gov/help-8.3.0/processors/ProcessL2gen.html)

```{code-cell}
:scrolled: true

%%bash
source $OCSSWROOT/OCSSW_bash.env

l2gen help
```

To process a L1B file using `l2gen`, at a minimum, you need to set an infile name (`ifile`) and an outfile name (`ofile`). You can also indicate a data suite; in this example, we will proceed with the biogeochemical (BGC) suite that includes chlorophyll *a* estimates.

Parameters can be passed to OCSSW programs through a text file. They can also be passed as arguments, but writing to a text file leaves a clear processing record. Define the parameters in a dictionary, then send it to the `write_par` function
defined in the [Setup](#1.-Setup) section.

```{code-cell}
par = {
    "ifile": l2gen_ifile,
    "ofile": str(l2gen_ifile).replace("L1B", "L2_BGC"),
    "suite": "SFREFL",
    "atmocor": 0,
}
write_par("l2gen.par", par)
```

With the parameter file ready, it's time to call `l2gen` from a `%%bash` cell.

```{code-cell}
:scrolled: true

%%bash
source $OCSSWROOT/OCSSW_bash.env

l2gen par=l2gen.par
```

If successful, the `l2gen` program created a netCDF file at the `ofile` path. The contents should include the `chlor_a` product from the `BGC` suite of products. Once this process is done, you are ready to visualize your "custom" L2 data. Use the `robust=True` option to ignore outlier chl a values.

```{code-cell}
dataset = xr.open_dataset(par["ofile"], group="geophysical_data")
plot = dataset["rhos"].sel({"wavelength_3d": 25}).plot(cmap="viridis", robust=True)
```

Feel free to explore `l2gen` options to produce the Level-2 dataset you need! The documentation
for `l2gen` is kind of interactive, because so much depends on the data product being processed.
For example, try `l2gen ifile=granules/PACE_OCI.20240427T161654.L1B.nc dump_options=true` to get
a lot of information about the specifics of what the `l2gen` program generates.

The next step for this tutorial is to merge multiple Level-2 granules together.

[back to top](#Contents)

+++

## 4. Composite L2 Data with `l2bin`

It can be useful to merge adjacent scenes to create a single, larger image. The OCSSW program that performs merging, also known as "compositing" of remote sensing images, is called `l2bin`. Take a look at the program's options.

```{code-cell}
:scrolled: true

%%bash
source $OCSSWROOT/OCSSW_bash.env

l2bin help
```

Write a parameter file with the previously saved list of Level-2 files standing in
for the usual "ifile" value. We can leave the datetime out of the "ofile" name rather than extracing a
time period from the granules chosen for binning.

```{code-cell}
ofile = "granules/PACE_OCI.L3B.nc"
par = {
    "ifile": "l2bin_ifile.txt",
    "ofile": ofile,
    "prodtype": "regional",
    "resolution": 9,
    "flaguse": "NONE",
    "rowgroup": 2000,
}
write_par("l2bin.par", par)
```

Now run `l2bin` using your chosen parameters:

```{code-cell}
:scrolled: true

%%bash
source $OCSSWROOT/OCSSW_bash.env

l2bin par=l2bin.par
```

[back to top](#Contents)

+++

## 5. Make a Map from Binned Data with `l3mapgen`

The `l3mapgen` function of OCSSW allows you to create maps with a wide array of options you can see below:

```{code-cell}
:scrolled: true

%%bash
source $OCSSWROOT/OCSSW_bash.env

l3mapgen help
```

Run `l3mapgen` to make a 9km map with a Plate Carree projection.

```{code-cell}
ifile = "granules/PACE_OCI.L3B.nc"
ofile = ifile.replace(".L3B.", ".L3M.")
par = {
    "ifile": ifile,
    "ofile": ofile,
    "projection": "platecarree",
    "resolution": "9km",
    "interp": "bin",
    "use_quality": 0,
    "apply_pal": 0,
}
write_par("l3mapgen.par", par)
```

```{code-cell}
:scrolled: true

%%bash
source $OCSSWROOT/OCSSW_bash.env

l3mapgen par=l3mapgen.par
```

Open the output with XArray, note that there is no group anymore.

```{code-cell}
dataset = xr.open_dataset(par["ofile"])
dataset
```

Now that we have projected data, we can make a map with coastines and gridlines.

```{code-cell}
fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plot = dataset["chlor_a"].plot(x="lon", y="lat", cmap="viridis", robust=True, ax=ax)
ax.gridlines(draw_labels={"left": "y", "bottom": "x"}, linewidth=0.3)
ax.coastlines(linewidth=0.5)
```

[back to top](#Contents)

<div class="alert alert-info" role="alert">

You have completed the notebook on using OCCSW to process PACE data. More notebooks are comming soon!

</div>
