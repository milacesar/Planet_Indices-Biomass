# -*- coding: utf-8 -*-
"""
funcs.py - spectral index functions

Author:
    Dr. Camila CESAR
    Toi Hangarau | Geospatial Research Institute
    Te Whare Wananga o Waitaha | University of Canterbury
    Ōtautahi, Aotearoa | Christchurch, New Zealand

"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import rasterio
from rasterio.mask import mask
import pyproj
from shapely.geometry import mapping
from shapely.ops import transform



def band_stats(arr: np.ndarray) -> dict:
    """
    Return mean / median / standard deviation (std) / valid-pixel-count for a 2d float array
    """
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "n_pixels": 0}
    return {
        "mean":     float(np.mean(valid)),
        "median":   float(np.median(valid)),
        "std":      float(np.std(valid)),
        "n_pixels": int(valid.size),
    }


def extract_band_stats(tif_path: str, aoi_geom, band_index, crs_wgs84="EPSG:4326") -> dict | None:
    """
    Open TIF, clip to aoi_geom (WGS-84), return per-band zone stats
    Returns None if image cannot be read or has no valid pixels
    """
    try:
        with rasterio.open(tif_path) as src:
            # Reproject AOI to the TIF's CRS once per file
            project = pyproj.Transformer.from_crs(
                crs_wgs84, src.crs.to_epsg() or src.crs.to_string(),
                always_xy=True
            ).transform
            aoi_proj = transform(project, aoi_geom)

            try:
                out_image, _ = mask(src, [mapping(aoi_proj)], crop=True)
            except ValueError:
                # AOI does not overlap raster extent
                return None

            n_bands = out_image.shape[0]
            if n_bands < 8:
                print(f"  WARNING: only {n_bands} bands in {tif_path}; expected 8. Skipping.")
                return None

            result = {}
            for band_name, idx in band_index.items():
                arr = out_image[idx].astype("float32")
                arr[arr == 0] = np.nan          # mask no-data (DN=0 in SR products)
                stats = band_stats(arr)
                for stat_name, val in stats.items():
                    result[f"{band_name}_{stat_name}"] = val

            # If every pixel in every band is NaN the crop is empty
            if all(result[f"{b}_n_pixels"] == 0 for b in band_index):
                return None

            return result

    except rasterio.errors.RasterioIOError as e:
        print(f"  Skipping corrupted raster: {tif_path}  ({e})")
        return None


# ----------
#  filsystem funcs
# ----------

def find_folder_upward(foldername: str, start_dir=None) -> Path:
    """Walk upward from start_dir until a folder named foldername is found."""
    start_dir = Path(start_dir or Path.cwd()).resolve()
    for parent in [start_dir] + list(start_dir.parents):
        if parent.name == foldername:
            return parent
    raise FileNotFoundError(f"Folder '{foldername}' not found above {start_dir}")


# ----------
#  paddocks funcs
# ----------
def load_paddocks(paddock_path) -> gpd.GeoDataFrame:
    paddock_path = Path(paddock_path)

    if paddock_path.suffix.lower() == ".gpkg":
        gdf = gpd.read_file(paddock_path)
    elif paddock_path.suffix.lower() == ".pkl":
        gdf = pd.read_pickle(paddock_path)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    elif paddock_path.suffix.lower() == ".parquet":
        gdf = gpd.read_parquet(paddock_path)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    else:
        raise ValueError(f"Unsupported file format: {paddock_path.suffix}")

    gdf = gdf.rename(columns=str.strip)
    gdf.columns = gdf.columns.str.replace(" ", "_")
    return gdf


def load_shapefile(shapefile_path) -> gpd.GeoDataFrame:
    shapefile_path = Path(shapefile_path)
    if shapefile_path.suffix.lower() == ".gpkg":
        return gpd.read_file(shapefile_path)
    raise ValueError(f"Unsupported file format: {shapefile_path.suffix}")


# -------------
#  spectral indexes funcs
# -------------

def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Vegetation Index
    """
    return (nir - red) / (nir + red + 1e-10)

def compute_ndvi2(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    ndvi using a from 
    """
    return (nir - 4 * red) / (nir + 4 * red + 1e-10)

def compute_ndre(red_edge: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Red-Edge ndvi : more sensitive to chlorophyll at high biomass densities (<2500)
    less prone to saturation than ndvi
    """
    return (nir - red_edge) / (nir + red_edge + 1e-10)

def compute_ndwi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Water Index
    """
    return (nir - swir) / (nir + swir + 1e-10)

def compute_savi(nir, red, L=0.5):
    """
    Soil Adjusted Vegetation Index (standard L=0.5)
    reflectance in range 0–1
    """
    denom = nir + red + L
    denom = np.where(denom < 0.001, np.nan, denom)
    return (nir - red) / denom * (1.0 + L)

def compute_evi(red, blue, nir, L=1.0, C1=6.0, C2=7.5, G=2.5):
    """
    Enhanced Vegetation Index
    reflectance in rnage 0–1
    result clipped to [-1, 2] to avoid denominator blowup
    """
    denom = nir + C1 * red - C2 * blue + L
    # Avoid division by near-zero denominator
    denom = np.where(np.abs(denom) < 0.001, np.nan, denom)
    evi = G * (nir - red) / denom
    return np.clip(evi, -1.0, 2.0)


'''
def compute_evi_0(
    red: np.ndarray,
    blue: np.ndarray,
    nir: np.ndarray,
    L: float = 1.0,
    C1: float = 6.0,
    C2: float = 7.5,
    G: float = 2.5,
) -> np.ndarray:
    return G * (nir - red) / (nir + C1 * red - C2 * blue + L + 1e-10)

def compute_savi_0(nir: np.ndarray, red: np.ndarray, L: float = 0.725) -> np.ndarray:
    return (nir - red) / (nir + red + L) * (1.0 + L)
'''



# -----------
# biomass funcs
# -----------
def biomass_from_ndvi(ndvi, a=3830.0, b=-600.0):
    """
    linear model: Biomass (kg DM/ha) = a * NDVI + b
    Default Canterbury: 3830 * NDVI - 600 
    Default coefficients are from Amies et al. (2021) National Mapping of New Zealand Pasture Productivity Using Temporal Sentinel-2 Data. (https://doi.org/10.3390/rs13081481)
    """
    return a * ndvi + b


def biomass_from_ndvi_0(ndvi: float | np.ndarray, a: float = 38.3, b: float = -6.0) -> float | np.ndarray:
    """
    linear model:  Biomass (kg DM/ha) = a * NDVI + b
    Default coefficients are from Amies et al. (2021) National Mapping of New Zealand Pasture Productivity Using Temporal Sentinel-2 Data. (https://doi.org/10.3390/rs13081481)
    """
    return a * ndvi + b



'''
# ------------
#  local biomass calib with RPM
# -------------
def fit_biomass_calibration(
    rpm_series: pd.Series,
    ndvi_series: pd.Series,
    index: str = "ndvi",
) -> dict:
    """
    Local biomass calibration using field measurments - Rising Plate Meter (RPM) 
    
    pasture biomass (kg DM/ha) = slope * RPM reading + intercept
    
    using paired RPM readings (converted to kg DM/ha) and satellite index values.

    Parameters
    ----------
    rpm_series : RPM plate-meter readings (clicks or pre-converted kg DM/ha)
    ndvi_series : Corresponding mean NDVI (or NDRE, EVI etc) values
    index : Label for the index used (stored in returned dict)

    Returns
    -------
    dict with keys: a, b, r2, n, index
    """
    from scipy.stats import linregress

    mask = ~(np.isnan(ndvi_series) | np.isnan(rpm_series))
    x = np.asarray(ndvi_series)[mask]
    y = np.asarray(rpm_series)[mask]

    if len(x) < 3:
        raise ValueError("Need at least 3 paired observations to calibrate.")

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    result = {
        "index":     index,
        "a":         slope,
        "b":         intercept,
        "r2":        r_value ** 2,
        "p_value":   p_value,
        "std_err":   std_err,
        "n":         int(mask.sum()),
    }
    print(
        f"Calibration [{index}]: y = {slope:.2f}x + {intercept:.2f}  "
        f"R²={r_value**2:.3f}  n={result['n']}"
    )
    return result


def predict_biomass(ndvi: float | np.ndarray, calibration: dict) -> float | np.ndarray:
    """
    apply calibration coefficients for new biomass (kg DM/ha)
    """
    return calibration["a"] * ndvi + calibration["b"]


# -------------
#  CH4 proxy
# -------------

def estimate_methane(
    biomass_kg_dm_ha: float | np.ndarray,
    area_ha: float,
    cows_per_ha: float,
    dmi_fraction: float = 0.90,
    ch4_per_kg_dmi: float = 20.7,
) -> dict:
    """
    Estimate daily enteric CH4 for a paddock based on biomass

    Uses the Hammond et al. (2016) NZ-specific coefficient:
        CH₄ (g/cow/day) = 20.7 × DMI (kg DM/day)

    The daily dry-matter intake (DMI) is estimated as:
        DMI = biomass_available × dmi_fraction / grazing_days

    This is a PROXY — treat results as relative, not absolute, until
    calibrated with actual intake or respiration chamber data.

    Parameters
    ----------
    biomass_kg_dm_ha : Standing biomass above residual (kg DM/ha)
    area_ha          : Paddock area (ha)
    cows_per_ha      : Stocking rate
    dmi_fraction     : Fraction of available biomass consumed (default 0.90 for NZ dairy)
    ch4_per_kg_dmi   : g CH₄ per kg DM intake (Hammond et al. 2016 NZ ryegrass default)

    Returns
    -------
    dict: {
        dmi_cow_day   : Estimated DMI per cow per day (kg DM)
        ch4_cow_day   : CH₄ per cow per day (g)
        ch4_herd_day  : CH₄ for whole paddock herd per day (g)
        ch4_herd_day_kg: Same in kg
    }
    """
    NZ_RESIDUAL_KG_DM_HA = 1_500   # typical post-grazing residual target

    available_biomass = np.maximum(
        np.asarray(biomass_kg_dm_ha, dtype=float) - NZ_RESIDUAL_KG_DM_HA, 0.0
    )
    total_available_kg = available_biomass * area_ha

    n_cows = cows_per_ha * area_ha
    if n_cows <= 0:
        raise ValueError("cows_per_ha × area_ha must be > 0")

    # Assume single grazing event (1 day) — caller can divide by rotation length
    dmi_cow_day   = (total_available_kg * dmi_fraction) / n_cows
    ch4_cow_day   = ch4_per_kg_dmi * dmi_cow_day          # g CH₄ / cow / day
    ch4_herd_day  = ch4_cow_day * n_cows                   # g CH₄ / herd / day

    return {
        "available_biomass_kg_ha": float(np.mean(available_biomass)),
        "dmi_cow_day_kg":          float(dmi_cow_day),
        "ch4_cow_day_g":           float(ch4_cow_day),
        "ch4_herd_day_g":          float(ch4_herd_day),
        "ch4_herd_day_kg":         float(ch4_herd_day / 1_000),
        "n_cows":                  float(n_cows),
    }

'''