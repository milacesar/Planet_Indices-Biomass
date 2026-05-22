# -*- coding: utf-8 -*-
"""
Author:
    Dr. Camila CESAR
    Toi Hangarau | Geospatial Research Institute
    Te Whare Wananga o Waitaha | University of Canterbury
    Ōtautahi, Aotearoa | Christchurch, New Zealand

Description:
    Planet (8b SR) imagery stats band extraction per-paddock per-image combo. 
    Input : {PlanetImageID}_3B_AnalyticMS_SR_8b_clip.tif
    Output : band_stats_all_paddocks.parquet

    Re-running the script only processes new (paddock x image) combinations, handling any new images added to the Planet folder (more recent or new backdated) for every paddock listed.

    Make sure to check the config area for modification of paths/files. 

Notes:
Planet SuperDove (PSB.SD) 8-band SR (https://docs.planet.com/data/imagery/planetscope/):
    1  Coastal Blue (443 nm)
    2  Blue (490 nm)
    3  Green 1 (531 nm)
    4  Green (565 nm)
    5  Yellow (610 nm)
    6  Red (665 nm)
    7  Red Edge (705 nm)
    8  NIR (865 nm)
"""

import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely.ops import transform

import funcs


# ---------
#  CONFIG
# ---------

print("== Config ==")

working_dir = Path(os.getcwd())

data_dir = working_dir / "data"
input_dir = data_dir / "input"
output_dir = data_dir / "output"
#output_dir  = (working_dir / "output").resolve()

planet_dir  = Path("E:/NZ_PSScene_ortho_udm2")
metadata_path = planet_dir.parent / "NZ_PSScene_ortho_udm2_metadata.gpkg"

out_data_dir = output_dir / "Planet"
out_data_dir.mkdir(parents=True, exist_ok=True)

# Output Parquet – one row per (paddock_name, tif_id)
band_stat_parquet = data_dir / "band_stats_all_paddocks.parquet"


# Planet SuperDove 8-band: rasterio uses 0-based indexing in out_image
# Band name  →  0-based index in out_image array
band_index = {
    "coastal_blue": 0,   # band 1
    "blue":         1,   # band 2
    "green_i":      2,   # band 3
    "green":        3,   # band 4
    "yellow":       4,   # band 5
    "red":          5,   # band 6
    "red_edge":     6,   # band 7
    "nir":          7,   # band 8
}

tif_suffix = "_3B_AnalyticMS_SR_8b_clip.tif"

# Reduces I/O overhead
save_step = 5

# ---------
#  loa paddocks
# ---------
print("\n== Bounds ==")

start_time = time.time()

NTF_paddocks_parquet = input_dir / "NTF_Paddocks_sorted.parquet"
NTF_paddocks = funcs.load_paddocks(NTF_paddocks_parquet)

NTF_boundaries = input_dir / "ntf_boundaries.gpkg"
ntf_bounds = funcs.load_shapefile(NTF_boundaries)


'''
# ---------
#  Functions
# ---------
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
def extract_band_stats(tif_path: str, aoi_geom, crs_wgs84="EPSG:4326") -> dict | None:
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
            for band_name, idx in BAND_INDEX.items():
                arr = out_image[idx].astype("float32")
                arr[arr == 0] = np.nan          # mask no-data (DN=0 in SR products)
                stats = band_stats(arr)
                for stat_name, val in stats.items():
                    result[f"{band_name}_{stat_name}"] = val

            # If every pixel in every band is NaN the crop is empty
            if all(result[f"{b}_n_pixels"] == 0 for b in BAND_INDEX):
                return None

            return result

    except rasterio.errors.RasterioIOError as e:
        print(f"  Skipping corrupted raster: {tif_path}  ({e})")
        return None
'''


# ---------
#  Planet metadata  (cached Parquet for speed)
# ---------
cached_meta_path = out_data_dir / "planet_meta_cached.parquet"

if cached_meta_path.exists():
    planet_meta = gpd.read_parquet(cached_meta_path)
    print("Loaded planet_meta from cached Parquet")
else:
    #planet_meta = gpd.read_file(metadata_path, columns=["id", "acquired", "geometry", "path"])
    
    try:
        planet_meta = gpd.read_file(
            metadata_path,
            columns=["id", "acquired", "geometry", "path"],
            engine="pyogrio",
        )
        print("Loaded via pyogrio")

    # SQLite fallback - load attrs then join geometry
    except Exception as e:
        print(f"pyogrio failed ({e}), falling back to SQLite load")
        import sqlite3
        con = sqlite3.connect(metadata_path)
        layer = pd.read_sql("Selcte table_name FROM gpkg_contents", con)["table_name"].iloc[0]
        df_raw = pd.read_sql(f"Select id, acquired, path, geom FROM '{layer}'", con)
        con.close()

        # Fix seconds=60 timestamps
        df_raw["acquired"] = df_raw["acquired"].str.replace(
            r"(\d{2}):60(\b|Z)", r"\g<1>:59\2", regex=True
        )
        # Reconstruct geometry from WKB blob
        from shapely import wkb
        df_raw["geometry"] = df_raw["geom"].apply(
            lambda b: wkb.loads(bytes(b), include_srid=True) if b else None
        )
        planet_meta = gpd.GeoDataFrame(
            df_raw[["id", "acquired", "path", "geometry"]],
            geometry="geometry", crs="EPSG:4326"
        )
    
    planet_meta.to_parquet(cached_meta_path)
    #print("Loaded planet_meta from GPKG and cached as Parquet")
    print(f"Cached {len(planet_meta):,} rows to Parquet")

# Build spatial index once - reused for every paddock
planet_sindex = planet_meta.sindex
print(f"Bounds loaded in {time.time() - start_time:.1f}s")


# ---------
#  tif lookup  {image_id: tif_path}
# ---------
print("== Planet TIF lookup ==")

tif_lookup: dict[str, str] = {}
for root, dirs, files in os.walk(planet_dir):
    for f in files:
        if f.endswith(f"{tif_suffix}"):
            tif_id   = f.replace(f"{tif_suffix}", "")
            tif_path = os.path.join(root, f)
            tif_lookup[tif_id] = tif_path

print(f"  {len(tif_lookup)} TIF files indexed on disk")


# ---------
#  CRS transform
# ---------
_proj_cache: dict[str, object] = {}

def get_transformer(target_crs_wkt: str):
    """
    Return and cache a transform function from EPSG:4326 to target_crs
    """
    if target_crs_wkt not in _proj_cache:
        _proj_cache[target_crs_wkt] = pyproj.Transformer.from_crs(
            "EPSG:4326", target_crs_wkt, always_xy=True
        ).transform
    return _proj_cache[target_crs_wkt]


# ---------
#  Checking existing (paddock x image) pairs and those needing processing
# ---------
print("\n== Checking existing results ==")

if band_stat_parquet.exists():
    existing_df = pd.read_parquet(band_stat_parquet)
    # Set of already-processed (paddock_name, tif_id) pairs
    processed_keys: set[tuple[str, str]] = set(
        zip(existing_df["paddock_name"], existing_df["tif_id"])
    )
    all_results = existing_df.to_dict("records")
    print(f"  {len(all_results)} existing records loaded.")
else:
    processed_keys = set()
    all_results = []
    print("  No existing Parquet - starting fresh.")


# ---------
#  Extract band stats for every new (paddock x image) pair
# ---------
print("\n== Extracting band statistics ==")

new_records   = 0
paddocks_done = 0

for _idx, paddock in NTF_paddocks.iterrows():
    paddock_poly = paddock.geometry
    paddock_name = paddock.Short_Name

    # Images that spatially intersect this paddock AND exist on disk
    intersecting = planet_meta[planet_meta.intersects(paddock_poly)].copy()
    intersecting["filepath"] = intersecting["id"].map(tif_lookup)
    intersecting = intersecting.dropna(subset=["filepath"])

    # Filter to only new (paddock x tif_id) pairs
    todo = intersecting[
        ~intersecting["id"].apply(lambda tid: (paddock_name, tid) in processed_keys)
    ]

    if todo.empty:
        continue  # nothing new for this paddock

    print(f"\n  Paddock '{paddock_name}': {len(todo)} new image(s) to process "
          f"({len(intersecting) - len(todo)} already done).")

    for _, row in todo.iterrows():
        tif_id   = row["id"]
        tif_path = row["filepath"]

        stats = funcs.extract_band_stats(tif_path, paddock_poly, band_index)
        if stats is None:
            continue

        record = {
            "paddock_name": paddock_name,
            "tif_id":       tif_id,
            "date":         row["acquired"],
            **stats,
        }
        all_results.append(record)
        processed_keys.add((paddock_name, tif_id))
        new_records += 1

    paddocks_done += 1

    # Incremental save every N paddocks
    if paddocks_done % save_step == 0:
        pd.DataFrame(all_results).to_parquet(band_stat_parquet)
        print(f"  [checkpoint] Saved {len(all_results)} records ({new_records} new).")

# Final save
pd.DataFrame(all_results).to_parquet(band_stat_parquet)
print(f"\n\nExtraction complete. {new_records} new records added.")
print(f"Band stats Parquet: {band_stat_parquet}")
