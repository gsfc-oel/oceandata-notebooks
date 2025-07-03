---
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Landsat Commercial Cloud Data Access from CryoCloud

**Authors:** Ian Carroll (NASA, UMBC)

## Summary

Landsat Collection 2 Level-1 data, Level-2 and Level-3 scene-based products, and Landsat U.S. Analysis Ready Data (ARD)
have a [secondary archive and distribution location][lcc] in the AWS s3://usgs-landsat requester pays bucket within the us-west-2 region.

[lcc]: https://www.usgs.gov/landsat-missions/landsat-commercial-cloud-data-access

## Learning objectives

By the end of this tutorial you will be able to:

- stream Landsat data

## Contents

1. [Setup](#1.-Setup)

+++

## 1. Setup

```{code-cell} ipython3
from matplotlib.pyplot import imshow
from pystac_client import Client
import rasterio as rio
import boto3
import xarray as xr
import rioxarray
```

## 2. Search

```{code-cell} ipython3
catalog = Client.open("https://landsatlook.usgs.gov/stac-server")
list(catalog.get_all_collections())
```

```{code-cell} ipython3
results = catalog.search(
    bbox=(-73.21, 43.99, -73.12, 44.05),
    datetime=("2019-01-01T00:00:00Z", "2019-01-02T00:00:00Z"),
    collections=("landsat-c2l2-sr",),
)
for item in results.items():
    display(item)
```

## 3. Access

```{code-cell} ipython3
item = item.to_dict()
href = item["assets"]["swir16"]["alternate"]["s3"]["href"]
href
```

```{code-cell} ipython3
session = rio.session.AWSSession(boto3.Session(), requester_pays=True)
with rio.Env(session):
    with rio.open(href) as src:
        profile = src.profile
        arr = src.read(1)
```

```{code-cell} ipython3
imshow(arr)
```
