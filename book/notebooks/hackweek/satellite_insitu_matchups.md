---
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Matchups of in situ data with satellite data

**Demo Leads:** James Allen (NASA, MSU), Anna Windle (NASA, SSAI)

## Summary

In this example we will conduct matchups of in situ AERONET-OC Rrs data with PACE OCI Rrs data. 

The Aerosol Robotic Network (AERONET) was developed to sustain atmospheric studies at various scales with measurements from worldwide distributed autonomous sun-photometers. This has been extended to support marine applications, called AERONET – Ocean Color [(AERONET-OC)](https://aeronet.gsfc.nasa.gov/new_web/ocean_levels_versions.html), and provides the additional capability of measuring the radiance emerging from the sea (i.e., water-leaving radiance) with modified sun-photometers installed on offshore platforms like lighthouses, oceanographic and oil towers. AERONET-OC is instrumental in satellite ocean color validation activities.

In this tutorial, we will be collecting Rrs data from the [AAOT](https://aeronet.gsfc.nasa.gov/cgi-bin/data_display_seaprism_v3?site=AAOT&nachal=2&level=2&place_code=10) AERONET-OC site located at 45.3N, 12.5E off the coast of Venice, Italy.

We will extract PACE OCI Rrs data in a 5x5 pixel window around the AERONET-OC site and compare it to in situ AERONET-OC Rrs observations.

## Learning Objectives

At the end of this notebook you will be able to:

- Access and parse data from a specific AERONET-OC site and date range
- Access and filter PACE OCI Rrs data from a specific region and date range
- Match satellite and in situ data spatially and temporally
- Visualize matchup performance with statistical plots (Bland-Altman and scatter)

Throughout, you'll gain experience with Python tools including `pandas`, `xarray`, and `matplotlib`, as well as real-world remote sensing workflows.

## Contents

1. [Setup](#1.-Setup)
2. [Process AERONET-OC data](#2.-Process-AERONET-OC-data)
3. [Download PACE OCI granules](#3.-Download-PACE-OCI-granules)
4. [Apply matchup code](#4.-Apply-matchup-code)
5. [Make plots](#5.-Make-plots)

+++

## 1. Setup

We begin by loading a set of utility functions that work behind the scenes to do the majority of the work for us.

**Collapse this cell and just run it** — but feel free to explore it later to see how these functions are implemented. Understanding what's under the hood will deepen your skills!

<div class="alert alert-block alert-warning">
Note: The `get_f0` function requires an OCI sensor data file from the OCSSW shared files. Make sure your path is set up to be able to read the OCSSWROOT environment variable! If not, you can always download the F0 (TSIS-1 0.1nm) file from https://oceancolor.gsfc.nasa.gov/resources/docs/rsr_tables/
</div>

```{code-cell} ipython3
---
lines_to_end_of_cell_marker: 0
lines_to_next_cell: 1
tags: [hide-cell]
jupyter:
  source_hidden: true
---
"""Helper functions for PACE Hackweek Validation Tutorial.

Authors:
    James Allen and Anna Windle
"""

import datetime
import os
import re
from pathlib import Path

import earthaccess
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.ticker import FuncFormatter
from scipy import odr, stats

# AERONET-OC Download Constants
# Valid AERONET-OC site list
DF_AERONET_SITES = pd.read_csv(
    "https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt",
    delimiter=",",
    skiprows=1
    )
AERONET_SITES = list(DF_AERONET_SITES["Site_Name"].sort_values())
OCEAN_SITES = [
    "AAOT",
    "Abu_Al_Bukhoosh",
    "ARIAKE_TOWER",
    "Bahia_Blanca",
    "Banana_River",
    "Blyth_NOAH",
    "Casablanca_Platform",
    "Chesapeake_Bay",
    "COVE_SEAPRISM",
    "Galata_Platform",
    "Gloria",
    "GOT_Seaprism",
    "Grizzly_Bay",
    "Gustav_Dalen_Tower",
    "Helsinki_Lighthouse",
    "Ieodo_Station",
    "Irbe_Lighthouse",
    "Kemigawa_Offshore",
    "Lake_Erie",
    "Lake_Okeechobee",
    "Lake_Okeechobee_N",
    "LISCO",
    "Lucinda",
    "MVCO",
    "Palgrunden",
    "PLOCAN_Tower",
    "RdP-EsNM",
    "Sacramento_River",
    "San_Marco_Platform",
    "Section-7_Platform",
    "Socheongcho",
    "South_Greenbay",
    "Thornton_C-power",
    "USC_SEAPRISM",
    "Venise",
    "WaveCIS_Site_CSI_6",
    "Zeebrugge-MOW1",
]

# Get subset of AERONET columns to make it a bit more manageable (also rename)
AOC_KEEP_COLS = [
    "AERONET_Site",
    "field_datetime",
    "Site_Latitude(Degrees)",
    "Site_Longitude(Degrees)",
    "Solar_Zenith_Angle[400nm]",
]
COLUMN_RENAME = {
    "Site_Latitude(Degrees)": "field_latitude",
    "Site_Longitude(Degrees)": "field_longitude",
    "AERONET_Site": "field_site",
    "Solar_Zenith_Angle[400nm]": "field_solar_zenith",
}

# Bland-Altman/Scatterplot Constants
# Plot colors, font sizes
COLOR_PALETTE = sns.color_palette("colorblind")
COLOR_SCATTER = COLOR_PALETTE[0]
COLOR_LINE = "black"  # Was "black"
COLOR_LOA = COLOR_PALETTE[2]  # Was "green"
COLOR_FITLINE = COLOR_PALETTE[1]  # Was "magenta"
SIZE_TITLE = 24
SIZE_AXLABEL = 20
SIZE_TEXTLABEL = 14
SHOW_LEGEND = False

# Update some defaults
plt.rcParams.update({"figure.dpi": 300})
sns.set_style("ticks", rc={"figure.dpi": 300})
sns.set_context("notebook", font_scale=1.45)

# Satellite Matchup Constants
# Short names for earthaccess lookup
SAT_LOOKUP = {
    "PACE_AOP": "PACE_OCI_L2_AOP",
    "PACE_IOP": "PACE_OCI_L2_IOP",
    "PACE_BGC": "PACE_OCI_L2_BGC",
    "AQUA": "MODISA_L2_OC",
    "TERRA": "MODIST_L2_OC",
    "NOAA-20": "VIIRSJ1_L2_OC",
    "NOAA-21": "VIIRSJ2_L2_OC",
    "SUOMI-NPP": "VIIRSN_L2_OC",
}

# List l2 flags, then build them into a dict
l2_flags_list = [
    "ATMFAIL",
    "LAND",
    "PRODWARN",
    "HIGLINT",
    "HILT",
    "HISATZEN",
    "COASTZ",
    "SPARE",
    "STRAYLIGHT",
    "CLDICE",
    "COCCOLITH",
    "TURBIDW",
    "HISOLZEN",
    "SPARE",
    "LOWLW",
    "CHLFAIL",
    "NAVWARN",
    "ABSAER",
    "SPARE",
    "MAXAERITER",
    "MODGLINT",
    "CHLWARN",
    "ATMWARN",
    "SPARE",
    "SEAICE",
    "NAVFAIL",
    "FILTER",
    "SPARE",
    "BOWTIEDEL",
    "HIPOL",
    "PRODFAIL",
    "SPARE",
]
L2_FLAGS = {flag: 1 << idx for idx, flag in enumerate(l2_flags_list)}

# Bailey and Werdell 2006 exclusion criteria
EXCLUSION_FLAGS = [
    "LAND",
    "HIGLINT",
    "HILT",
    "STRAYLIGHT",
    "CLDICE",
    "ATMFAIL",
    "LOWLW",
    "FILTER",
    "NAVFAIL",
    "NAVWARN",
]

# OCSSW Dataroot folder for tables
OCDATAROOT = Path(os.environ.get("OCSSWROOT")).resolve() / "share"
OCI_SENSOR_FILE = (OCDATAROOT / "oci/msl12_sensor_info.dat").resolve()

##---------------------------------------------------------------------------##
#                              General Utilities                              #
##---------------------------------------------------------------------------##


def get_f0(wavelengths=None, window_size=10):
    """Load the OCI sensor file and return F0.

    Defaults to returning the full table. Input obs_time to correct for the
    Earth-Sun distance.

    Parameters
    ----------
    sensor_file : str or pathlib.Path
        Path to the OCI satellite sensor file containing wavelengths and F0.
    wavelengths : array-like, optional
        Wavelengths at which to compute the average irradiance.
        If None, returns the full wavelength and irradiance table.
    window_size : int, optional
        Bandpass filter size for mean filtering to selected wavelengths, in nm.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing:
        - f0_spectra : np.ndarray
            The extraterrestrial solar irradiance, in uW/cm^2/nm.
        - f0_wave : np.ndarray
            The corresponding wavelengths, in nm.

    """
    with open(OCI_SENSOR_FILE, "r") as file_in:
        for line in file_in:
            if "Nbands" in line:
                (key, nbands) = line.split("=")
                break

    wl = np.zeros(int(nbands), dtype=float)
    f0 = np.zeros(int(nbands), dtype=float)
    with open(OCI_SENSOR_FILE, "r") as file_in:
        for line in file_in:
            if "=" in line:
                (key, value) = line.split("=")
                if "Lambda" in key:
                    idx = re.findall(r"\d+", key)
                    wvlidx = int(idx[0]) - 1
                    wl[wvlidx] = float(value)
                if "F0" in key:
                    idx = re.findall(r"\d+", key)
                    wvlidx = int(idx[1]) - 1
                    f0[wvlidx] = float(value)

    if wavelengths is not None:
        f0_wave = np.array(wavelengths)
        f0_spectra = bandpass_avg(f0, wl, window_size, f0_wave)
    else:
        f0_wave = wl
        f0_spectra = f0

    return f0_spectra, f0_wave


def bandpass_avg(
        data,
        input_wavelengths,
        window_size=10,
        target_wavelengths=None
        ):
    """Apply a band-pass filter to the data.

    Parameters
    ----------
    data : np.ndarray
        1D or 2D array containing the spectral data (samples x wavelengths).
        If 1D, it's assumed to be a single sample.
    input_wavelengths : np.ndarray
        1D array of wavelength values corresponding to the columns of data.
    window_size : int, optional
        Size of the window to use for averaging. Default is 10 nm.
    target_wavelengths : np.ndarray, optional
        1D array of target wavelengths for filtered values.
        If None, the input wavelengths are used.

    Returns
    -------
    np.ndarray
        1D or 2D array containing the band-pass filtered data.

    """
    data = np.atleast_2d(data)
    half_window = window_size / 2
    num_samples, num_input_wavelengths = data.shape
    if target_wavelengths is None:
        target_wavelengths = input_wavelengths

    filtered_data = np.empty((num_samples, len(target_wavelengths))) * np.nan

    for idx, target_wl in enumerate(target_wavelengths):
        start = target_wl - half_window
        end = target_wl + half_window
        cols_in_range = np.where(
            (input_wavelengths >= start) & (input_wavelengths <= end)
        )[0]
        if cols_in_range.size > 0:
            filtered_data[:, idx] = np.nanmean(data[:, cols_in_range], axis=1)

    return filtered_data if num_samples > 1 else filtered_data.flatten()


def get_column_prods(df, type_prefix):
    """Process a dataframe to create a dictionary of data products.

    Parameters
    ----------
    df : pandas DataFrame
        Extracted dataframes from read_extract_file
    type_prefix : str
        Prefix to identify the product columns, e.g. "aoc"

    Returns
    -------
    data_dict
        dictionary mapping data product with their wavelengths and columns.

    """
    data_dict = {}
    pattern = rf"{type_prefix}_(\w+?)(\d*\.?\d+)?$"

    for col in df.columns:
        match = re.match(pattern, col)
        if match:
            product = match.group(1)
            wavelength = match.group(2) if match.group(2) else None
            if product not in data_dict:
                data_dict[product] = {"wavelengths": [], "columns": []}
            data_dict[product]["columns"].append(col)
            if wavelength:
                if "." in wavelength:
                    data_dict[product]["wavelengths"].append(float(wavelength))
                else:
                    data_dict[product]["wavelengths"].append(int(wavelength))
    return data_dict


def read_sb(filename_sb):
    """Read SeaBASS file and returns just the data.

    Input
    -----
    filename_sb : str
        path to seabass file

    Output
    ------
    data : pandas dataframe object
        seabass data from file
    """
    with open(filename_sb, "r") as file:
        lines = [line.rstrip() for line in file]

    # Parse headers, get index where they end
    idx_endheader = [index for index, value in enumerate(lines)
                     if value == "/end_header"]
    header_lines = lines[1:idx_endheader[0]]
    headers = dict()
    comments = []
    for header_line in header_lines:
        if header_line.startswith("!"):
            # Separate out the comments
            comments.append(header_line)
        else:
            # Split the header and add to the dictionary
            key, value = header_line.split("=", 1)
            headers[key[1:]] = value  # Remove leading "/" from key

    # Pull data into pandas dataframe
    data = pd.read_csv(filename_sb,
                       skiprows=idx_endheader[0]+1,
                       names=headers["fields"].split(","),
                       na_values=headers["missing"])

    # Index by datetime
    get_sb_datetime(data)

    return data


def get_sb_datetime(df):
    """Parse datetime from different combinations of dates and times."""
    if all(col in df.columns for col in ["year", "month", "day",
                                         "hour", "minute", "second"]):
        df["datetime"] = pd.to_datetime(df[["year", "month", "day",
                                            "hour", "minute", "second"]])
    elif all(col in df.columns for col in ["year", "month", "day", "time"]):
        df["datetime"] = pd.to_datetime(
            df["year"].astype(str) + df["month"].astype(str).str.zfill(2)
            + df["day"].astype(str).str.zfill(2) + ' ' + df["time"])
    elif all(col in df.columns for col in ["date", "time"]):
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + ' ' + df["time"])
    elif all(col in df.columns for col in ["year", "month", "day"]):
        df["datetime"] = pd.to_datetime(df[["year", "month", "day"]])
    elif all(col in df.columns for col in ["date", "hour",
                                           "minute", "second"]):
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + ' ' + df["hour"].astype(str).str.zfill(2)
            + ':' + df["minute"].astype(str).str.zfill(2) + ':'
            + df["second"].astype(str).str.zfill(2))
    else:
        print("Unrecognized date/time format in DataFrame columns."
              "\nMay be a profile, but doublecheck.")
        return

    # Reindex the dataframe with the new datetime
    df.set_index("datetime", inplace=True)


##---------------------------------------------------------------------------##
#                            AERONET_OC Utilities                             #
##---------------------------------------------------------------------------##


def construct_url(aeronet_site, data_level, start_date, end_date):
    """Craft the AERONET data URL for Lwn (AERONET-OC) or SDA (AERONET) data.

    Parameters
    ----------
    aeronet_site : str, optional
        Specific AERONET (or -OC) site
    start_date : datetime object
        Beginning of Aeronet data to run.
    end_date : datetime object, optional
        End of Aeronet data to run.
    data_level : int, {10, 15, 20}
        data quality; 20 (highest quality), 15, or 10.

    Returns
    -------
    str
        url to API pull

    """
    # Validate inputs
    if aeronet_site not in AERONET_SITES:
        raise ValueError(
            f"{aeronet_site} is not an AERONET site. Available "
            f"sites are: {', '.join(AERONET_SITES)}"
        )

    if aeronet_site in OCEAN_SITES:
        url = (
            "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?"
            f"AVG=10&LWN{data_level}=1&year={start_date.year}"
            f"&month={start_date.month}&day={start_date.day}"
            f"&if_no_html=1&year2={end_date.year}&month2={end_date.month}"
            f"&day2={end_date.day}&site={aeronet_site}"
        )
    else:
        url = (
            "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?"
            f"AVG=10&SDA{data_level}=1&year={start_date.year}"
            f"&month={start_date.month}&day={start_date.day}"
            f"&if_no_html=1&year2={end_date.year}&month2={end_date.month}"
            f"&day2={end_date.day}&site={aeronet_site}"
        )

    return url


def get_data_dict(df, search_str=None):
    """Process a dataframe to create a dict and ndarray of products and waves.

    Parameters
    ----------
    df : pandas DataFrame
        Extracted dataframes from read_extract_file

    Returns
    -------
    wavelengths
        numpy array of wavelengths of the data
    column_map
        dict of the dataframe columns associated with each wavelength

    """
    if search_str is None:
        search_str = "Lwn_IOP"
    wavelengths = []
    column_map = {}
    pattern = re.compile(rf"{search_str}\[(\d+)nm\]")

    for col in df.columns:
        match = pattern.search(col)
        if match:
            wavelength = int(match.group(1))
            wavelengths.append(wavelength)
            column_map[wavelength] = col
    return np.array(wavelengths), column_map


def process_aeronet(
    aeronet_site="AAOT", start_date="2024-03-01", end_date=None, data_level=15
):
    """Download and process AERONET-OC data for matchups.

    Parameters
    ----------
    aeronet_site : str, optional
        Specific AERONET site (else AAOT by default)
    start_date : datetime or str, optional
        Beginning of Aeronet data to run. Defaults to "2024-03-01"
    end_date : datetime or str, optional
        End of Aeronet data to run. Defaults to today.
    data_level : int, {10, 15, 20}
        data quality; 20 (highest, but fewest), 15 (autochecked), or 10.

    Returns
    -------
    pandas DataFrame object
        Dataframe of downloaded AERONET data, semi-processed if AERONET-OC, or
        full SDA data if AERONET

    """
    # Set up processing
    if end_date is None:
        end_date = datetime.now()
    start_date = pd.to_datetime(start_date, errors="raise")
    end_date = pd.to_datetime(end_date, errors="raise")

    # Make url
    url_aeronet = construct_url(aeronet_site, data_level, start_date, end_date)

    # Download data (skip the 5 header rows)
    try:
        df_full = pd.read_csv(
            url_aeronet, delimiter=",", na_values=-999, skiprows=5
            )
    except Exception as e:
        raise Exception(
            f"Error: {e}: Could not download data. Try another station, "
            f"reduce the data_level, or expand the times."
        )

    # Drop empty columns
    df_full.dropna(axis=1, how="all", inplace=True)

    # Parse datetimes
    date_col = [col for col in df_full.columns if col.startswith("Date")][0]
    time_col = [col for col in df_full.columns if col.startswith("Time")][0]
    df_full["field_datetime"] = pd.to_datetime(
        df_full[date_col] + " " + df_full[time_col],
        format="%d:%m:%Y %H:%M:%S"
    ).dt.tz_localize("UTC")

    if aeronet_site in OCEAN_SITES:
        # Get subset of Lwn_f/Q columns (ignore the count columns)
        # Alternatively, could pull Lwn_IOP for L11 BRDF
        subset_lwn = [
            col
            for col in df_full.columns
            if "Lwn_f/Q" in col and "N[Lwn_f/Q" not in col
        ]
        lwn_iop = df_full[subset_lwn].values

        # Now get array of wavelengths from columns
        wavelengths, _ = get_data_dict(df_full[subset_lwn], "Lwn_f/Q")

        # Lwn need to be normalized by F0
        # Note: Lwn_IOP already accounts for the Earth-Sun Distance, BRDF, and
        # atmosphere transmittance
        f0_spectra, _ = get_f0(wavelengths)

        # Normalize to get Rrs
        rrs = lwn_iop / f0_spectra[None, :]

        # Generate new column names and make the rrs dataframe
        rrs_cols = [f"field_rrs{wavelength}" for wavelength in wavelengths]
        df_rrs = pd.DataFrame(rrs, columns=rrs_cols)

        # Now combine with the subset of the full dataframe
        df_aeronet = pd.concat([df_full[AOC_KEEP_COLS], df_rrs], axis=1)
    else:
        df_aeronet = df_full

    # Do some final cleanup
    df_aeronet.rename(columns=COLUMN_RENAME, inplace=True)

    return df_aeronet


##---------------------------------------------------------------------------##
#                             Satellite Utilities                             #
##---------------------------------------------------------------------------##


def parse_quality_flags(flag_value):
    """Parse bitwise flag into a list of flag names.

    Parameters
    ----------
    flag_value : int
        The integer representing the combined bitwise quality flags.

    Returns
    -------
    list of str
        List of flag names that are set in the flag_value.

    """
    return [
        flag_name for flag_name, value in L2_FLAGS.items()
        if (flag_value & value) != 0
    ]


def get_fivebyfive(file, latitude, longitude, rrs_wavelengths):
    """Get stats on 5x5 box around station coordinates of a satellite granule.

    This checks l2flags and runs statistics on valid pixels and returns their
    valid count, the coefficient of variance (cv), and the Rrs values.

    Parameters
    ----------
    file : earthaccess granule object
        Satellite granule from earthaccess.
    latitude : float
        In decimal degrees for Aeronet-OC site for matchups
    longitude : float
        In decimal degrees (negative West) for Aeronet-OC site for matchups
    rrs_wavelengths ; numpy array
        Rrs wavelengths (from wavelength_3d for OCI)

    Returns
    -------
    dict
        A dictionary of the processed 5x5 box with:
            - "sat_datetime": pd.datetime
                Datetime of the overall granule start time
            - "sat_cv": float
                Median coefficient of variation of Rrs(405nm - 570nm)
            - "sat_latitude": float
                Latitude of center pixel
            - "sat_longitude": float
                Longitude of center pixel
            - "sat_pixel_valid": float
                Number of valid pixels in 5x5 box based on l2 flags

    Notes
    -----
    This is set to use just Rrs data for the demo. As an exercise, make this
    function more generalized by adding an input for the desired product and
    removing the wavelength dependency (if not needed) as well as the cv
    calculation. This will also require refactoring the `match_data` function.
    """
    with xr.open_dataset(file, group="navigation_data") as ds_nav:
        sat_lat = ds_nav["latitude"].values
        sat_lon = ds_nav["longitude"].values

    # Calculate the Euclidean distance for 2D lat/lon arrays
    distances = np.sqrt((sat_lat - latitude) ** 2 + (sat_lon - longitude) ** 2)

    # Find the index of the minimum distance
    # Dimensions are (lines, pixels)
    min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
    center_line, center_pixel = min_dist_idx

    # Get indices for a 5x5 box around the center pixel
    line_start = max(center_line - 2, 0)
    line_end = min(center_line + 2 + 1, sat_lat.shape[0])
    pixel_start = max(center_pixel - 2, 0)
    pixel_end = min(center_pixel + 2 + 1, sat_lat.shape[1])

    # Extract the data
    # NOTE: This is hard-coded to Rrs from an L2 AOP file.
    with xr.open_dataset(file, group="geophysical_data") as ds_data:
        rrs_data = (
            ds_data["Rrs"].isel(
                number_of_lines=slice(line_start, line_end),
                pixels_per_line=slice(pixel_start, pixel_end),
            ).values
        )
        flags_data = (
            ds_data["l2_flags"].isel(
                number_of_lines=slice(line_start, line_end),
                pixels_per_line=slice(pixel_start, pixel_end),
            ).values
        )

    # Calculate the bitwise OR of all flags in EXCLUSION_FLAGS to get a mask
    exclude_mask = sum(L2_FLAGS[flag] for flag in EXCLUSION_FLAGS)

    # Create a boolean mask
    # True means the flag value does not contain any of the EXCLUSION_FLAGS
    valid_mask = np.bitwise_and(flags_data, exclude_mask) == 0

    # Get stats and averages
    if valid_mask.any():
        rrs_valid = rrs_data[valid_mask]
        rrs_std_initial = np.std(rrs_valid, axis=0)
        rrs_mean_initial = np.mean(rrs_valid, axis=0)

        # Exclude spectra > 1.5 stdevs away
        std_mask = np.all(
            np.abs(rrs_valid - rrs_mean_initial) <= 1.5 * rrs_std_initial,
            axis=1
        )
        rrs_std = np.std(rrs_valid[std_mask], axis=0)
        rrs_mean = np.mean(rrs_valid[std_mask], axis=0).flatten()

        # Matchup criteria uses cv as median of 405-570nm
        rrs_cv = rrs_std / rrs_mean
        rrs_cv_median = np.median(
            rrs_cv[(rrs_wavelengths >= 405) & (rrs_wavelengths <= 570)]
        )
    else:
        rrs_cv_median = np.nan
        rrs_mean = np.nan * np.empty_like(rrs_wavelengths)

    # Put in dictionary of the row
    row = {
        "sat_datetime": pd.to_datetime(
            file.granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"],
            utc=0
        ),
        "sat_cv": rrs_cv_median,
        "sat_latitude": sat_lat[center_line, center_pixel],
        "sat_longitude": sat_lon[center_line, center_pixel],
        "sat_pixel_valid": np.sum(valid_mask),
    }

    # Add mean spectra to the row dictionary
    for wavelength, mean_value in zip(rrs_wavelengths, rrs_mean):
        key = f"sat_rrs{int(wavelength)}"
        row[key] = mean_value

    return row


def get_sat_ts_matchups(
    start_date,
    end_date,
    latitude,
    longitude,
    sat="PACE_AOP",
    selected_dates=None
):
    """Make satellite timeseries of matchups from single station.

    Caution: If the date or coordinates aren't formatted correctly, it might
    pull a huge granule list and take forever to run. If it takes more than 45
    seconds to print the number of granules, just kill the process.

    Uses the earthaccess package. Defaults to the PACE OCI L2 IOP datasets,
    but other satellites can be used if they have a corresponding short_name
    in the SAT_LOOKUP dictionary.

    Workflow:
        1. Get list of matchup granules
        2. Loop through each file and:
            2a. Find closest pixel to station, extract 5x5 pixel box
            2b. Exclude pixels based on l2_flags
            2c. Filtered mean to get single spectra
            2d. Compute statistics and save data row
        3. Organize output pandas dataframe

    Parameters
    ----------
    start_date : datetime or str
        Beginning of Aeronet data to run.
    end_date : datetime or str, optional
        End of Aeronet data to run.
    latitude : float
        In decimal degrees for Aeronet-OC site for matchups
    longitude : float
        In decimal degrees (negative West) for Aeronet-OC site for matchups
    sat : str
        Name of satellite to search. Must be in SAT_LOOKUP dict constant.
    selected_dates : list of str, optional
        If given, only pull granules if the dates are in this list

    Returns
    -------
    pandas DataFrame object
        Flattened table of all satellite granule matchups.

    """
    # Look up short name from constants
    if sat not in SAT_LOOKUP.keys():
        raise ValueError(
            f"{sat} is not in the lookup dictionary. Available "
            f"sats are: {', '.join(SAT_LOOKUP)}"
        )
    short_name = SAT_LOOKUP[sat]

    # Format search parameters
    time_bounds = (f"{start_date}T00:00:00", f"{end_date}T23:59:59")

    # Run Earthaccess data search
    results = earthaccess.search_data(
        point=(longitude, latitude),
        temporal=time_bounds,
        short_name=short_name
    )
    if selected_dates is not None:
        filtered_results = [
            result
            for result in results
            if result["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:10]
            in selected_dates
        ]
        print(f"Filtered to {len(filtered_results)} Granules.")
        files = earthaccess.open(filtered_results)
    else:
        files = earthaccess.open(results)

    # Pull out Rrs wavelengths for easier processing
    with xr.open_dataset(files[0], group="sensor_band_parameters") as ds_bands:
        rrs_wavelengths = ds_bands["wavelength_3d"].values

    # Loop through files and process
    sat_rows = []
    for idx, file in enumerate(files):
        granule_date = pd.to_datetime(
            file.granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        )
        print(f"Running Granule: {granule_date}")
        row = get_fivebyfive(file, latitude, longitude, rrs_wavelengths)
        sat_rows.append(row)

    return pd.DataFrame(sat_rows)


##---------------------------------------------------------------------------##
#                              Matchup Utilities                              #
##---------------------------------------------------------------------------##


def match_data(
    df_sat,
    df_field,
    cv_max=0.15,
    senz_max=60.0,
    min_percent_valid=55.0,
    max_time_diff=180,
    std_max=1.5,
):
    """Create matchup dataframe based on selection criteria.

    Parameters
    ----------
    df_sat : pandas dataframe
        Satellite data from flat validation file.
    df_field : pandas dataframe
        Field data from flat validation file.
    cv_max : float, default 0.15
        Maximum coefficient of variation (stdev/mean) for sat data.
    senz_max : float, default 60.0
        Maximum sensor zenith for sat data.
    min_percent_valid : float, default 55.0
        Minimum percentage of valid satellite pixels.
    max_time_diff : int, default 180
        Maximum time difference (minutes) between sat and field matchup.
    std_max : float, default 1.5
        If multiple valid field matchups, select within std_max stdevs of mean.

    Returns
    -------
    pandas dataframe of matchups for product

    Notes
    -----
    This is hard-coded to match on Rrs for the demo. For other products, take
    out the cv parameter and make the row product column search more generic.
    """
    # Setup
    time_window = pd.Timedelta(minutes=max_time_diff)
    df_match_list = []

    # Filter Field data based on Solar Zenith
    df_field_filtered = df_field[df_field["field_solar_zenith"] <= senz_max]

    # Filter satellite data based on cv threshold
    df_sat_filtered = df_sat[df_sat["sat_cv"] <= cv_max]

    # Filter satellite data based on percent good pixels
    df_sat_filtered = df_sat_filtered[
        df_sat_filtered["sat_pixel_valid"] >= min_percent_valid * 25 / 100
    ]

    for _, sat_row in df_sat_filtered.iterrows():
        # Filter field data based on time difference and coordinates
        time_diff = abs(
            df_field_filtered["field_datetime"] - sat_row["sat_datetime"]
            )
        time_mask = time_diff <= time_window
        lat_mask = 0.2 >= abs(
            df_field_filtered["field_latitude"] - sat_row["sat_latitude"]
        )
        lon_mask = 0.2 >= abs(
            df_field_filtered["field_longitude"] - sat_row["sat_longitude"]
        )
        field_matches = df_field_filtered[time_mask & lat_mask & lon_mask]

        if field_matches.shape[0] > 5:
            # Filter by Standard Deviation for rrs columns
            rrs_cols = [
                col for col in field_matches.columns
                if col.startswith("field_rrs")
                and int(col.rsplit("_rrs")[1]) >= 400
                and int(col.rsplit("_rrs")[1]) <= 700
            ]
            if rrs_cols:
                mean_spectra = field_matches[rrs_cols].mean(axis=0)
                std_spectra = field_matches[rrs_cols].std(axis=0)
                within_std = (
                    abs(field_matches[rrs_cols] - mean_spectra) <= std_max * std_spectra
                )
                field_matches = field_matches[within_std.all(axis=1)]

        if not field_matches.empty:
            # Select the best match based on time delta
            time_diff = abs(
                field_matches["field_datetime"] - sat_row["sat_datetime"]
                )
            best_match = field_matches.loc[time_diff.idxmin()]
            df_match_list.append({**best_match.to_dict(), **sat_row.to_dict()})

    df_match = pd.DataFrame(df_match_list)
    return df_match


##---------------------------------------------------------------------------##
#                              Plotting Utilities                             #
##---------------------------------------------------------------------------##


def compute_bland_altman_metrics(xx, yy, xx_unc_modl, yy_unc_modl):
    """Compute metrics for Bland-Altman plot.

    Parameters
    ----------
    xx : array
        Array of X data values.
    yy : array
        Array of Y data values.
    xx_unc_modl : float
        Uncertainty in X.
    yy_unc_modl : float
        Uncertainty in Y.

    Returns
    -------
    dict
        Dictionary of Bland-Altman metrics.

    """
    jj = (xx + yy) / 2
    kk = (yy - xx) / np.sqrt((xx_unc_modl**2) + (yy_unc_modl**2))

    meanbias = np.mean(kk)
    stdbias = np.std(kk)
    LOAlow = meanbias - stdbias
    LOAhgh = meanbias + stdbias

    ba_stat, ba_p = stats.spearmanr(jj, kk)
    ba_independ = ba_p > 0.05

    return {
        "count": kk.shape[0],
        "jj": jj,
        "kk": kk,
        "meanbias": meanbias,
        "LOAlow": LOAlow,
        "LOAhgh": LOAhgh,
        "ba_stat": ba_stat,
        "ba_p": ba_p,
        "ba_independ": ba_independ,
    }


def compute_regression_metrics(xx, yy, is_type2=False):
    """Compute regression metrics using specified type.

    Parameters
    ----------
    xx : array
        Array of X data values.
    yy : array
        Array of Y data values.
    is_type2 : bool, optional
        Whether to use Type 2 regression (orthogonal distance regression).
        Default is False, for Type 1 regression (ordinary least squares).

    Returns
    -------
    dict
        Dictionary of regression metrics.

    """
    if is_type2:
        # Perform Type 2 regression (orthogonal distance regression)
        def linear_model(B, x):
            """Linear function y = m*x + b.

            B is a vector of the parameters.
            x is an array of the current x values.
            x is in the same format as the x passed to Data or RealData.
            Return an array in the same format as y passed to Data or RealData.
            """
            return B[0] * x + B[1]

        # Create a model instance
        linear = odr.Model(linear_model)

        # Create a RealData object using the data
        data = odr.RealData(xx, yy)

        # Set up ODR with the model and data
        odr_instance = odr.ODR(data, linear, beta0=[1.0, 0.0])

        # Run the regression
        odr_result = odr_instance.run()
        slope = odr_result.beta[0]
        intercept = odr_result.beta[1]
    else:
        # Perform Type 1 regression (ordinary least squares)
        regress_result = stats.linregress(xx, yy)
        slope = regress_result.slope
        intercept = regress_result.intercept

    spearman_r = stats.spearmanr(xx, yy)
    pearson_r = stats.pearsonr(xx, yy)
    rmse_all = np.sqrt(np.mean((yy - xx) ** 2))
    mae_all = np.mean(np.abs(yy - xx))

    return {
        "count": len(xx),
        "slope": slope,
        "intercept": intercept,
        "r_spear": spearman_r.correlation,
        "r_pear": pearson_r[0],
        "rmse": rmse_all,
        "mae": mae_all,
    }


def add_text_annotations(ax, text_lines, position="top right", fontsize=SIZE_TEXTLABEL):
    """Add text annotations to the plot.

    Parameters
    ----------
    ax : Axes
        The axis to add text to.
    text_lines : list of str
        List of strings to be displayed as text.
    position : str, default 'top right'
        Position of the text on the plot.
    fontsize : int, default 12
        Font size of the text.

    """
    if position == "top right":
        x = 0.95
        y = 0.95
        ha = "right"
        va = "top"
    elif position == "top left":
        x = 0.05
        y = 0.95
        ha = "left"
        va = "top"
    elif position == "bottom left":
        x = 0.05
        y = 0.05
        ha = "left"
        va = "bottom"
    elif position == "bottom right":
        x = 0.95
        y = 0.05
        ha = "right"
        va = "bottom"

    text = "\n".join(text_lines)
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )


def setup_plot(label):
    """Set up the plot with titles and labels.

    Parameters
    ----------
    label : str
        Title of the plot.

    Returns
    -------
    tuple
        Figure and axes of the plot.

    """
    style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    fig.suptitle(label, fontsize=22)
    return fig, ax1, ax2


def format_ticks(ax):
    """Format the tick labels on the axes to be more readable."""
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3g}"))
    ax.tick_params(axis="both", which="major", width=2, length=6)
    ax.spines["top"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)


def plot_bland_altman(
    ax1,
    metrics,
    binscale,
    xx_unc_modl,
    x_label="x",
    y_label="y"
):
    """Plot Bland-Altman plot.

    Parameters
    ----------
    ax1 : Axes
        Axis for the Bland-Altman plot.
    metrics : dict
        Bland-Altman metrics.
    binscale : float
        Scaling factor for bin size.
    xx_unc_modl : float
        Uncertainty in X.
    x_label : string, default "x"
        String for labels for x data
    y_label : string, default "y"
        String for labels for y data

    """
    jj = metrics["jj"]
    kk = metrics["kk"]
    npoints = metrics["count"]
    meanbias = metrics["meanbias"]
    LOAlow = metrics["LOAlow"]
    LOAhgh = metrics["LOAhgh"]
    ba_independ = metrics["ba_independ"]
    ba_stat = metrics["ba_stat"]

    min_kk = meanbias - 5 * np.std(kk)
    max_kk = meanbias + 5 * np.std(kk)
    min_jj = np.min(jj)
    max_jj = np.max(jj)
    lineclr, loaclr, fitclr = (COLOR_LINE, COLOR_LOA, COLOR_FITLINE)
    ax1.scatter(jj, kk, color=COLOR_SCATTER)
    ax1.set_xlim([min_jj, max_jj])
    ax1.set_ylim([min_kk, max_kk])

    ax1.set_title("Bland-Altman plot", fontsize=SIZE_TITLE)
    ylabel = (
        "Uncertainty normalized bias"
        if xx_unc_modl != np.sqrt(0.5)
        else f"Bias, ${y_label}-{x_label}$"
    )
    ax1.set_ylabel(ylabel, fontsize=SIZE_AXLABEL)
    ax1.set_xlabel(
        f"Paired mean, $({x_label}+{y_label})/2$", fontsize=SIZE_AXLABEL
        )
    ax1.plot(
        [min_jj, max_jj], [0, 0],
        color=lineclr, linestyle="solid", linewidth=4.0
        )

    if ba_independ:
        ax1.plot(
            [min_jj, max_jj],
            [meanbias, meanbias],
            color=fitclr,
            linestyle="dashed",
            linewidth=3.0,
            label="Mean Bias",
        )
        ax1.plot(
            [min_jj, max_jj],
            [LOAlow, LOAlow],
            color=loaclr,
            linestyle="dashed",
            linewidth=2.0,
            label="Lower LOA",
        )
        ax1.plot(
            [min_jj, max_jj],
            [LOAhgh, LOAhgh],
            color=loaclr,
            linestyle="dashed",
            linewidth=2.0,
            label="Upper LOA",
        )
        ax1.fill_between(
            [min_jj, max_jj], LOAlow, LOAhgh,
            color=loaclr, alpha=0.1
            )
    else:
        ba_regress_result = stats.linregress(jj, kk)
        ba_min_fit_yy = ba_regress_result.slope * min_jj + ba_regress_result.intercept
        ba_max_fit_yy = ba_regress_result.slope * max_jj + ba_regress_result.intercept
        ax1.plot(
            [min_jj, max_jj],
            [ba_min_fit_yy, ba_max_fit_yy],
            color=fitclr,
            linestyle="dashed",
            linewidth=3.0,
            label="Linear Fit",
        )
    if SHOW_LEGEND:
        ax1.legend()
    ax1.grid(True)
    format_ticks(ax1)

    text_lines = [
        f"Number of Points: {npoints}",
        f"Mean Bias: {meanbias:.2e}",
        f"Limits of Agreement: [{LOAlow:.2e}, {LOAhgh:.2e}]",
        f"Rank Correlation: {ba_stat:.3f}",
        "Bias Independent" if ba_independ else "Bias Dependent",
    ]
    add_text_annotations(ax1, text_lines, position="bottom right")


def plot_scatter(
    ax2, xx, yy, regress_metrics, binscale, x_label="x", y_label="y"
):
    """Plot scatter plot with regression line.

    Parameters
    ----------
    ax2 : Axes
        Axis for the scatter plot.
    xx : array
        Array of X data values.
    yy : array
        Array of Y data values.
    regress_metrics : dict
        Regression metrics.
    binscale : float
        Scaling factor for bin size.
    x_label : string, default "x"
        String for labels for x data
    y_label : string, default "y"
        String for labels for y data

    """
    min_val = min(np.min(xx), np.min(yy))
    max_val = max(np.max(xx), np.max(yy))

    ax2.scatter(xx, yy, color=COLOR_SCATTER)
    ax2.set_xlim([min_val, max_val])
    ax2.set_ylim([min_val, max_val])

    ax2.set_title("Scatterplot", fontsize=SIZE_TITLE)
    ax2.set_xlabel(f"${x_label}$", fontsize=SIZE_AXLABEL)
    ax2.set_ylabel(f"${y_label}$", fontsize=SIZE_AXLABEL)
    ax2.plot(
        [min_val, max_val],
        [min_val, max_val],
        color=COLOR_LINE,
        linestyle="solid",
        linewidth=4.0,
    )

    slope = regress_metrics["slope"]
    intercept = regress_metrics["intercept"]
    min_fit_yy = slope * min_val + intercept
    max_fit_yy = slope * max_val + intercept
    ax2.plot(
        [min_val, max_val],
        [min_fit_yy, max_fit_yy],
        color=COLOR_FITLINE,
        linestyle="dashed",
        linewidth=3.0,
        label="Regression Line",
    )
    if SHOW_LEGEND:
        ax2.legend()
    ax2.grid(True)
    format_ticks(ax2)

    text_lines = [
        f"Slope: {slope:.3f}",
        f"Intercept: {intercept:.2e}",
        f"Linear Correlation: {regress_metrics['r_pear']:.3f}",
        f"Rank Correlation: {regress_metrics['r_spear']:.3f}",
        f"RMSE: {regress_metrics['rmse']:.2e}",
        f"MAE: {regress_metrics['mae']:.2e}",
    ]
    add_text_annotations(ax2, text_lines, position="bottom right")


def plot_BAvsScat(
    x_input,
    y_input,
    label="",
    saveplot=None,
    binscale=1.0,
    xx_unc_modl=np.sqrt(0.5),
    yy_unc_modl=np.sqrt(0.5),
    x_label="x",
    y_label="y",
    is_type2=True,
):
    """Routine to plot paired data as Bland-Altman and scatter plot.

    Parameters
    ----------
    x_input : array-like
        Array of X data values.
    y_input : array-like
        Corresponding array of Y data values.
    label : string, default ''
        Text label for plotting.
    saveplot : string, default None
        Set to save plot in ../output/ with the string as the filename.
    binscale : float, default 1.0
        Scaling factor for how many bins to include in a 2D histogram.
    xx_unc_modl : float, default np.sqrt(0.5)
        Uncertainty in X.
    yy_unc_modl : float, default np.sqrt(0.5)
        Uncertainty in Y.
    x_label : string, default "x"
        String for labels for x data
    y_label : string, default "y"
        String for labels for y data

    Returns
    -------
    dict
        Dictionary of computed statistics.

    """
    xx = np.asarray(x_input)
    yy = np.asarray(y_input)
    valid_indices = (
        np.isfinite(x_input)
        & np.isfinite(y_input)
        & (x_input != -999)
        & (y_input != -999)
    )
    xx = x_input[valid_indices]
    yy = y_input[valid_indices]

    ba_metrics = compute_bland_altman_metrics(xx, yy, xx_unc_modl, yy_unc_modl)
    regress_metrics = compute_regression_metrics(xx, yy, is_type2=is_type2)

    fig, ax1, ax2 = setup_plot(label)
    plot_bland_altman(ax1, ba_metrics, binscale, xx_unc_modl, x_label, y_label)
    plot_scatter(ax2, xx, yy, regress_metrics, binscale, x_label, y_label)

    if saveplot is not None:
        figpath = Path("../output") / saveplot
        fig.savefig(figpath)
        print("Saved figure to:", figpath)

    plt.show()

    return {
        "Number_of_Points": ba_metrics["count"],
        "Scale_Independence": ba_metrics["ba_independ"],
        "Mean_Bias": ba_metrics["meanbias"],
        "Limits_of_Agreement_low": (
            ba_metrics["LOAlow"] if ba_metrics["ba_independ"] else float("nan")
        ),
        "Limits_of_Agreement_high": (
            ba_metrics["LOAhgh"] if ba_metrics["ba_independ"] else float("nan")
        ),
        "Linear_Slope": regress_metrics["slope"],
        "Linear_Intercept": regress_metrics["intercept"],
        "Linear_Correlation": regress_metrics["r_pear"],
        "Rank_Correlation": regress_metrics["r_spear"],
        "RMSE": regress_metrics["rmse"],
        "MAE": regress_metrics["mae"],
    }
```

For those starting this demonstration fresh, we'll double-check that we've got our earthaccess credentials set up!

```{code-cell} ipython3
auth = earthaccess.login(strategy="interactive", persist=True)
```

## 2. Process AERONET-OC data

We will use the function `process_aeronet` to download and process AERONET-OC data from the 'AAOT' site. We will filter Level 1.5 data from the dates June 1, 2024 to July 31, 2024. This function will output a pandas dataframe of every AERONET-OC record between the dates.

There are three "levels" of AERONET-OC data in terms of data quality: 1, 1.5, and 2. If a complete measurement sequence with the instruments is able to be performed, it is collected and stored as Level 1. These data are then passed through an automated quality control system and stored as Level 1.5 if they pass all tests. Finally, Level 2 data are data from Level 1.5 that are subsequently screened by an experienced scientist and validated. We'll be using Level 1.5 data to pull as much good quality data as possible without the time lag for manual validation. More information on AERONET-OC levels can be found in [Zibordi et al., 2009.](https://doi.org/10.1175/2009JTECHO654.1)

```{code-cell} ipython3
df_aeronet = process_aeronet(
    aeronet_site="AAOT",
    start_date="2024-03-01",
    end_date="2025-06-30",
    data_level=15,
)
df_aeronet.head()
```

## 3. Download PACE OCI granules

We will use the function `get_sat_ts_matchups` to search for `PACE_OCI_L2_AOP_NRT` data using `earthaccess` within the specified time range and at the (lat,lon) coordinate of the AAOT AERONET-OC site. This function finds the closest pixel and extracts all data within a 5x5 pixel window, excludes pixels based on L2 flags, calculates the mean to retrive a single Rrs spectra, and computes matchup statistics. The function outputs a pandas dataframe of every `PACE_OCI_L2_AOP` Rrs spectra for the specified time range. We'll also include an optional list of unique date strings from the AERONET-OC dataframe to "skip" the granules that don't have any field data associated with them.

<div class="alert alert-block alert-warning">
Note: This section will actually take quite a while to pull enough granules for the plotting section to give us robust stats (at least 35 valid matchups are needed), so we'll be skipping this part in favor of a pre-made dataset for the demo.
</div>

```{code-cell} ipython3
:scrolled: true
:tags: [scroll-output]

# Pull out coordinates
aeronet_lat = df_aeronet["field_latitude"][0]
aeronet_lon = df_aeronet["field_longitude"][0]

# Pull out unique days
unique_days = df_aeronet["field_datetime"].dt.date.unique()
unique_days_str = [day.strftime("%Y-%m-%d") for day in unique_days]

df_satellite = get_sat_ts_matchups(
    start_date="2024-03-01",
    end_date="2024-03-05",
    latitude=aeronet_lat,
    longitude=aeronet_lon,
    sat="PACE_AOP",
    selected_dates=unique_days_str,
)
```

## 3b. Precooked Download PACE OCI granules

Since the previous section takes quite a while to pull the granules we need, here's a premade data pull of PACE OCI data (for AAOT). 

```{code-cell} ipython3
df_satellite = pd.read_csv("/home/jovyan/shared-public/pace-hackweek/oci_quickpull_aaot.csv")
df_satellite["sat_datetime"] = pd.to_datetime(df_satellite["sat_datetime"], utc=True)
```

```{code-cell} ipython3
df_satellite.head()
```

## 4. Apply matchup code

We will use the function `match_data` to create a matchup dataframe based on selection criteria. This function defaults to using the [Bailey and Werdell 2006](https://oceancolor.gsfc.nasa.gov/staff/jeremy/bailey_and_werdell_2006_rse.pdf) matchup criteria, which reduces the measurements made at a given station to one representative sample for validating against the satellite spectra. Data are filtered based on the solar zenith angle, their noise level, and the time difference (here 180 minutes from the satellite overpass). Potential satellite matchups are also reduced based on the signal to noise level of the 5x5 pixel aggregation.

```{code-cell} ipython3
:scrolled: true
:tags: [scroll-output]

?match_data
```

```{code-cell} ipython3
:scrolled: true
:tags: [scroll-output]

matchups = match_data(
    df_satellite,
    df_aeronet,
    cv_max=0.60,
    senz_max=60.0,
    min_percent_valid=55.0,
    max_time_diff=180,
    std_max=1.5,
)
matchups
```

Pull out wavelengths and Rrs data from matchups

```{code-cell} ipython3
dict_aoc = get_column_prods(matchups, "field")
waves_aoc = np.array(dict_aoc["rrs"]["wavelengths"])
rrs_aoc = matchups[dict_aoc["rrs"]["columns"]].to_numpy()

dict_sat = get_column_prods(matchups, "sat")
waves_sat = np.array(dict_sat["rrs"]["wavelengths"])
rrs_sat = matchups[dict_sat["rrs"]["columns"]].to_numpy()
```

## 4. Make plots

We will use the function `plot_BAvsScat` to plot the paired matchup data as Bland_Altman and scatter plots. The Bland-Altman plots provide insights into the bias and precision of the satellite measurements compared to field measurements. A mean difference close to zero indicates good agreement, while the spread of differences (limits of agreement) puts the bias within the context of the variability of the field data. Additionally, a check is done to assess the scale dependency of the bias, such as errors increasing when the magnitude of the observations increases. If a scale dependency exists, the limits of agreement are replaced with a regression line showing its direction and magnitude. Scatter plots complement Bland-Altman plots by showing the strength of the linear relationship between the two datasets, with high correlation coefficients and low RMSE values indicating strong agreement and high accuracy of the satellite-derived measurements.

```{code-cell} ipython3
:scrolled: true
:tags: [scroll-output]

?plot_BAvsScat
```

```{code-cell} ipython3
MATCH_WAVES = np.array([400, 412, 443, 490, 510, 560, 620, 667])

# Loop through matchup wavelengths
stats_list = []
for idx, match_wave in enumerate(MATCH_WAVES):
    # Find matching OCI columns
    idx_sat = np.where(np.abs(waves_sat - match_wave) <= 5)[0]
    match_sat = np.nanmean(rrs_sat[:, idx_sat], axis=1)

    # Find matching AOC columns
    idx_aoc = np.where(np.abs(waves_aoc - match_wave) <= 5)[0]
    match_aoc = np.nanmean(rrs_aoc[:, idx_aoc], axis=1)

    valid_indices = np.isfinite(match_sat) & np.isfinite(match_aoc)
    if np.sum(valid_indices) > 5:
        fig_label = f"Rrs({match_wave}), sr" + r"$\mathregular{^{-1}}$"
        dict_stats = plot_BAvsScat(
            match_aoc[valid_indices],
            match_sat[valid_indices],
            label=fig_label,
            saveplot=None,
            x_label="AeronetOC",
            y_label="PaceOCI",
            is_type2=True,
        )
        dict_stats["wavelength"] = match_wave
        stats_list.append(dict_stats)

# Organize stats DataFrame
df_stats = pd.DataFrame(stats_list)
df_stats.set_index("wavelength", inplace=True)
df_stats = df_stats.fillna(-999)
df_stats
```

```{code-cell} ipython3

```
