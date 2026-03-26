
import ee
import os
import requests
import rasterio
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import sys

# arguments
if len(sys.argv) != 4:
    print("Usage: <GROUP_ID> <START_DATE> <END_DATE>")
    sys.exit(1)

GROUP_ID = int(sys.argv[1])
START_DATE = pd.to_datetime(sys.argv[2])
END_DATE = pd.to_datetime(sys.argv[3])

# initialize earth engine
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

# specify output folder to hold all of the .nc files
output_folder = "/insert/path/here"
os.makedirs(output_folder, exist_ok=True)

'''
load site info from parquet file
make sure to use the proper naming conventions for each file to match the group ID
'''
parquet_pattern = f"/specify/path/here/group_{GROUP_ID}*.parquet"  # modify to fit chosen file naming convention
files = glob.glob(parquet_pattern)
if not files:
    raise FileNotFoundError(f"No Parquet found for group {GROUP_ID}")

df = pd.read_parquet(files[0])
df["time"] = pd.to_datetime(df["date"])
df = df[(df["time"] >= START_DATE) & (df["time"] <= END_DATE)].reset_index(drop=True)

if len(df) == 0:
    print("No data in date range — exiting.")
    sys.exit(0)

# COORDINATE CLEANUP
df["latitude"] = (
    df["latitude"]
    .fillna(df.get("latitude_x"))
    .fillna(df.get("latitude_y"))
    .fillna(df.get("latitude_csv"))
)

df["longitude"] = (
    df["longitude"]
    .fillna(df.get("longitude_x"))
    .fillna(df.get("longitude_y"))
    .fillna(df.get("longitude_csv"))
)

# select discharge column
possible_discharge_cols = [c for c in df.columns if "discharge" in c.lower() and not c.startswith("mean_")]
possible_mean_cols = [c for c in df.columns if "mean_discharge" in c.lower()]

if not possible_discharge_cols or not possible_mean_cols:
    raise ValueError("Discharge columns not found.")

discharge_col = possible_discharge_cols[0]
mean_col = possible_mean_cols[0]

location_qmean = (
    df.groupby(["latitude", "longitude"])[mean_col]
    .mean()
    .reset_index()
    .rename(columns={mean_col: "q_mean"})
)

df = df.merge(location_qmean, on=["latitude", "longitude"], how="left")


#Specify desired modis product and spatial extent

PRODUCT_NAME = "MOD09GA"
PIXEL_SIZE = 500
REGION_SIZE_PIXELS = 64
REGION_SIZE_M = PIXEL_SIZE * REGION_SIZE_PIXELS

#specify bands
MODIS_BANDS = [
    'sur_refl_b03', 'sur_refl_b04', 'sur_refl_b01',
    'sur_refl_b02', 'sur_refl_b06', 'sur_refl_b07',
    'state_1km'
]

#rename bands
MODIS_BAND_NAMES = [
    'blue', 'green', 'red',
    'nir', 'swir1', 'swir2',
    'state'
]

# function to specify the exact spatial region
def get_region(lon, lat):
    half_lat = (REGION_SIZE_M / 2) / 111320
    half_lon = half_lat / np.cos(np.radians(lat))
    return ee.Geometry.Rectangle([
        lon - half_lon, lat - half_lat,
        lon + half_lon, lat + half_lat
    ])

def mask_modis_clouds(image):
    qa = image.select('state_1km')
    cloud_state = qa.bitwiseAnd(3)
    mask = cloud_state.lte(1)
    return image.updateMask(mask)

def cloud_percentage(image, region):
    total = image.select(0).reduceRegion(
        ee.Reducer.count(), region, PIXEL_SIZE, maxPixels=1e9
    ).values().get(0)

    clear = mask_modis_clouds(image).select(0).reduceRegion(
        ee.Reducer.count(), region, PIXEL_SIZE, maxPixels=1e9
    ).values().get(0)

    return ee.Number(1).subtract(ee.Number(clear).divide(total)).multiply(100)

# crop to the center reference point
def download_and_crop(url):
    temp_tif = "temp_image.tif"
    try:
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code != 200:
            print("Download failed")
            return None

        with open(temp_tif, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

        with rasterio.open(temp_tif) as src:
            array = src.read() / 10000.0  # (bands, y, x)

        if array.shape[0] != 7:
            print(f"Wrong band count: {array.shape}")
            return None

        _, h, w = array.shape
        if h < 64 or w < 64:
            print(f"Too small: {array.shape}")
            return None

        y0 = (h - 64) // 2
        x0 = (w - 64) // 2
        array = array[:, y0:y0+64, x0:x0+64]

        return array.astype(np.float32)

    except Exception as e:
        print(f"Raster error: {e}")
        return None
    finally:
        if os.path.exists(temp_tif):
            os.remove(temp_tif)

def save_as_netcdf(array, product_name, site_id, date_str, row):
    safe_site = str(site_id).replace("/", "_").replace("\\", "_")
    filename = f"{product_name}_{safe_site}_{date_str.replace('-', '')}.nc"
    filepath = os.path.join(output_folder, filename)

    ds = xr.Dataset(
        {"reflectance": (("band", "y", "x"), array)},
        coords={"band": MODIS_BAND_NAMES},
        attrs={
            "acquisition_date": date_str,
            "site_id": site_id,
            "observed_discharge": float(row.get(discharge_col, np.nan)),
            "latitude": float(row.get("latitude", np.nan)),
            "longitude": float(row.get("longitude", np.nan)),
            "width": float(row.get("width", np.nan)),
            "pixel_size_m": PIXEL_SIZE
        }
    )

    ds.to_netcdf(filepath)
    print(f"Saved: {filepath}")

def try_download_image(row):
    date = row["time"]
    start = (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    region = get_region(row["longitude"], row["latitude"])
    site_id = row.get("site_id", "unknown")

    try:
        coll = (
            ee.ImageCollection('MODIS/006/MOD09GA')
            .filterDate(start, end)
            .filterBounds(region)
            .map(lambda img: img.set('CLOUD_PERCENTAGE', cloud_percentage(img, region)))
            .filter(ee.Filter.lt('CLOUD_PERCENTAGE', 10))
            .map(mask_modis_clouds)
            .select(MODIS_BANDS)
            .sort('system:time_start')
        )

        size = coll.size().getInfo()
        if size == 0:
            return False

        for i in range(size):
            img = ee.Image(coll.toList(size).get(i))
            date_str = img.date().format('YYYY-MM-dd').getInfo()

            safe_site = str(site_id).replace("/", "_").replace("\\", "_")
            out_name = f"{PRODUCT_NAME}_{safe_site}_{date_str.replace('-', '')}.nc"
            out_path = os.path.join(output_folder, out_name)
            if os.path.exists(out_path):
                return True

            url = img.clip(region).getDownloadURL({
                "scale": PIXEL_SIZE,
                "region": region,
                "crs": "EPSG:4326",
                "format": "GEO_TIFF"
            })

            array = download_and_crop(url)
            if array is not None:
                save_as_netcdf(array, PRODUCT_NAME, site_id, date_str, row)
                return True

    except Exception as e:
        print(f"Failed for {site_id} on {date.date()}: {e}")
        return False

print(f"Processing {len(df)} rows")

# main loop
for _, row in tqdm(df.iterrows(), total=len(df)):
    try_download_image(row)

print("complete")











