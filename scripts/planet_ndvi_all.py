# -*- coding: utf-8 -*-
"""
Author:
    Dr. Camila CESAR
    Toi Hangarau | Geospatial Research Institute
    Te Whare Wananga o Waitaha | University of Canterbury
    Ōtautahi, Aotearoa | Christchurch, New Zealand

planet_ndvi_all.py

re-run any time new Planet imagery arrives

"""

import os
import sys
import time
import pyproj
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
from shapely.geometry import mapping
from shapely.ops import transform
from rasterio.mask import mask

import funcs

tqdm.monitor_interval = 0   # prevents threading glitches in IPython


#print("" + os.chdir(Path(__file__).resolve().parent))
#sys.exit()

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
tif_suffix = "_3B_AnalyticMS_SR_8b_clip.tif"
# Planet SuperDove 8-band order (0-indexed)
expected_bands = 8
band_index = {
    "coastal_blue": 0,
    "blue":         1,
    "green_i":      2,
    "green":        3,
    "yellow":       4,
    "red":          5,
    "red_edge":     6,
    "nir":          7,
}


out_data_dir = output_dir / "Planet"
out_data_dir.mkdir(parents=True, exist_ok=True)

parquet_path = out_data_dir / "ndvi_all_paddocks.parquet"


# ---------
#  loa paddocks
# ---------
print("\n== Bounds ==")

start_time = time.time()

NTF_paddocks_parquet = input_dir / "NTF_Paddocks_sorted.parquet"
NTF_paddocks = funcs.load_paddocks(NTF_paddocks_parquet)

NTF_boundaries = input_dir / "ntf_boundaries.gpkg"
ntf_bounds = funcs.load_shapefile(NTF_boundaries)

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
#  load existing results 
# ---------
print("== Loading existing results ==")

all_results: list[dict] = []
processed_set: set[tuple[str, str]] = set()   # {(paddock_name, image_id)}

if parquet_path.exists():
    existing_df = pd.read_parquet(parquet_path)
    all_results = existing_df.to_dict("records")
    # Build the processed set from whatever is already stored
    if "image_id" in existing_df.columns:
        processed_set = set(zip(existing_df["paddock_name"], existing_df["image_id"]))
    else:
        print("  WARNING: existing file has no image_id column — results will be merged on re-run.")
    print(f"  Loaded {len(existing_df):,} existing rows, {len(processed_set):,} unique (paddock, image) pairs")
else:
    print("  No existing Parquet found — starting fresh")


# ---------
#  ndvi loop
# ---------
print("\n== Calculating spectral indices ==")

paddocks = NTF_paddocks   # swap to .head(N) for testing

for idx, paddock in paddocks.iterrows():
    paddock_poly  = paddock.geometry
    paddock_name  = paddock.Short_Name
    print(f"\n[{idx}] Paddock: {paddock_name}")

    # --- fast spatial pre-filter with sindex, then exact geometry check ---
    candidate_idx  = list(planet_sindex.intersection(paddock_poly.bounds))
    intersecting   = planet_meta.iloc[candidate_idx]
    intersecting   = intersecting[intersecting.intersects(paddock_poly)].copy()

    # --- resolve TIF paths on disk ---
    intersecting["filepath"] = intersecting["id"].map(tif_lookup)
    intersecting = intersecting.dropna(subset=["filepath"])

    total  = len(intersecting)
    to_do  = [(r["id"], r["filepath"]) for _, r in intersecting.iterrows()
               if (paddock_name, r["id"]) not in processed_set]

    print(f"  {total} rasters intersect | {len(to_do)} not yet processed")

    if not to_do:
        continue

    for image_id, tif_path in to_do:
        try:
            with rasterio.open(tif_path) as src:
                # Guard: skip non-8-band files silently
                if src.count != expected_bands:
                    print(f"  SKIP (not 8-band, got {src.count}): {os.path.basename(tif_path)}")
                    continue

                nodata = src.nodata if src.nodata is not None else 0

                # Reproject paddock geometry to raster CRS
                project  = get_transformer(src.crs.to_wkt())
                aoi_proj = transform(project, paddock_poly)

                try:
                    out_image, _ = mask(src, [mapping(aoi_proj)], crop=True)
                except ValueError:
                    continue  # paddock outside raster extent

                def band(name: str) -> np.ndarray:
                    arr = out_image[band_index[name]].astype("float32")
                    arr[arr == nodata] = np.nan
                    return arr

                blue      = band("blue")
                green     = band("green")
                red       = band("red")
                red_edge  = band("red_edge")
                nir       = band("nir")

                red_r  = red  / 10000.0
                blue_r = blue / 10000.0
                nir_r  = nir  / 10000.0
                re_r   = red_edge / 10000.0
                
                ndvi = funcs.compute_ndvi(red_r, nir_r)
                ndre = funcs.compute_ndre(re_r, nir_r)
                evi  = funcs.compute_evi(red_r, blue_r, nir_r)
                savi = funcs.compute_savi(nir_r, red_r)

                # # --- All indices in one pass ---
                # ndvi  = funcs.compute_ndvi(red, nir)
                # ndre  = funcs.compute_ndvi(red_edge, nir)   # Red-Edge NDVI
                # evi   = funcs.compute_evi(red, blue, nir)
                # savi  = funcs.compute_savi(nir, red)

                # Skip if NDVI has no valid data
                if ndvi.size == 0 or np.isnan(ndvi).all():
                    continue

                # Acquire date from metadata row
                acquired_date = intersecting.loc[intersecting["id"] == image_id, "acquired"].values[0]

                all_results.append({
                    "paddock_name":  paddock_name,
                    "image_id":      image_id,           # key for deduplication
                    "date":          acquired_date,
                    # NDVI
                    "ndvi_mean":     float(np.nanmean(ndvi)),
                    "ndvi_median":   float(np.nanmedian(ndvi)),
                    "ndvi_std":      float(np.nanstd(ndvi)),
                    # Red-Edge NDVI (better at high biomass)
                    "ndre_mean":     float(np.nanmean(ndre)),
                    "ndre_median":   float(np.nanmedian(ndre)),
                    "ndre_std":      float(np.nanstd(ndre)),
                    # EVI
                    "evi_mean":      float(np.nanmean(evi)),
                    "evi_median":   float(np.nanmedian(evi)),
                    "evi_std":       float(np.nanstd(evi)),
                    # SAVI
                    "savi_mean":     float(np.nanmean(savi)),
                    "savi_median":   float(np.nanmedian(savi)),
                    "savi_std":      float(np.nanstd(savi)),
                })

                processed_set.add((paddock_name, image_id))

        except rasterio.errors.RasterioIOError as e:
            print(f"  SKIP corrupted raster: {os.path.basename(tif_path)} ({e})")
            continue

    # --- Save incrementally after each paddock ---
    pd.DataFrame(all_results).to_parquet(parquet_path)

print("\n\nVegetation OIndices precomputation complete.")
print(f"Results saved to: {parquet_path}")
