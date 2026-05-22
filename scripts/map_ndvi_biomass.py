# -*- coding: utf-8 -*-
'''
Created on Fri May 15 10:49:57 2026

Author: 
    Dr. Camila CESAR
    Toi Hangarau | Geospatial Research Institute
    Te Whare Wananga o Waitaha | University of Canterbury
    Ōtautahi, Aotearoa | Christchurch, New Zealand


Description: 
    
map_ndvi_biomass.py

reads band_stats_all_paddocks.parquet
plots ndvi and biomass (median) maps for that date

'''

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# ==== config ====
PARQUET = Path(r"..\data\output\Planet\band_stats_all_paddocks.parquet")
PADDOCK_FILE = Path(r"..\data\input\NTF_Paddocks_sorted.parquet")
OUTDIR = Path(r"..\data\output\maps\test4")
OUTDIR.mkdir(parents=True, exist_ok=True)
data_parquet = Path(r"..\data\output\ndvi_biomass_data.parquet")

# canterbury linear biomass model: y (kg DM / ha) = 38.3 * NDVI - 6
BIOMASS_A = 38.3
BIOMASS_B = -6.0

DATE_START = "2024-08-01"   # "YYYY-MM-DD" or None
DATE_END = "2024-09-05"   # "YYYY-MM-DD" or None

# min paddock coverage
MIN_COVERAGE_FRACTION = 0.0   # eg 0.5 = only dates with > 50%, 0.0 -> plot everything)

# paddock geometry name column
NAME_COL = "Short Name"

# ==== OSM BASEMAP ====
# set to True to add an OpenStreetMap tile underlay behind each map panel
USE_OSM = True
PADDOCK_ALPHA = 1

if USE_OSM:
    try:
        import contextily as ctx
    except ImportError:
        raise ImportError(
            "contextily is required for the OSM basemap.\n"
            "Install with: conda install -c conda-forge contextily\n"
        )

#fig options        
dpi = 200
ndvi_cmap = plt.cm.Blues  #RdYlGn
bm_cmap = plt.cm.Greens #YlGn
sns_cm = "mako"

calc_data = False
save_data = False
savefig = True


# ==== loading data ====
print("Loading Planet stats parquet ...")
df = pd.read_parquet(PARQUET)
df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
df["date_only"] = df["date"].dt.date

# compute ndvi from band stats
if "ndvi_median" not in df.columns:
    eps = 1e-10
    df["ndvi_median"] = (df["nir_median"] - df["red_median"]) / (df["nir_median"] + df["red_median"] + eps)
    df["ndvi_mean"]   = (df["nir_mean"]   - df["red_mean"])   / (df["nir_mean"]   + df["red_mean"]   + eps)

# Biomass from median NDVI
df["biomass_median"] = BIOMASS_A * df["ndvi_median"] + BIOMASS_B

df_data = df[['paddock_name', 'tif_id', 'date', 'date_only', 'ndvi_median', 'biomass_median', 'ndvi_mean']].copy()
df_data["biomass_mean"] = BIOMASS_A * df["ndvi_mean"] + BIOMASS_B
if save_data == True: 
    df_data.to_parquet(data_parquet)

# ==== date covergae ====
total_paddocks = df["paddock_name"].nunique()
coverage = (
    df.groupby("date_only")["paddock_name"]
    .nunique()
    .reset_index()
    .rename(columns={"paddock_name": "n_paddocks"})
    .sort_values("date_only")
)
coverage["fraction"] = coverage["n_paddocks"] / total_paddocks
 
print(f"Total unique paddocks in parquet : {total_paddocks}")
print(f"Total dates with any data        : {len(coverage)}")
 
# date range filter
if DATE_START:
    coverage = coverage[coverage["date_only"] >= pd.to_datetime(DATE_START).date()]
if DATE_END:
    coverage = coverage[coverage["date_only"] <= pd.to_datetime(DATE_END).date()]
 
# minimum coverage filter
coverage = coverage[coverage["fraction"] >= MIN_COVERAGE_FRACTION]

print(f"Dates to plot (after filters)    : {len(coverage)}")
print(coverage.to_string(index=False))


# ==== loading paddocks geom ====
print("\nLoading paddock geometries ...")
gdf = gpd.read_parquet(PADDOCK_FILE)
#print(gdf.crs)
epsg_code = gdf.crs.to_epsg()
print(f"EPSG: {epsg_code}")

paddocks_wgs84 = gdf.to_crs("EPSG:4326")
paddocks_merc = gdf.to_crs("EPSG:3857")

# ==== figure layout ====
# contextily needs mercator
if USE_OSM:
    paddocks_plot = paddocks_merc
else:
    paddocks_plot = paddocks_wgs84

bounds = paddocks_plot.total_bounds # [minx, miny, maxx, maxy]
lon_ext = bounds[2] - bounds[0]
lat_ext = bounds[3] - bounds[1]


MAP_W = 12                              # inches wide
MAP_H = MAP_W * (lat_ext / lon_ext)    # preserves true aspect
FIG_W = MAP_W + 0.5                    # + colorbar room
FIG_H = MAP_H * 2 + 1.8               # two stacked panels + suptitle


# Fix colorbar scales across all dates so maps are comparable
ndvi_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
bm_global = df["biomass_median"].dropna()
bm_norm   = mcolors.Normalize(
    vmin=max(0.0, float(np.nanpercentile(bm_global, 2))),
    vmax=float(np.nanpercentile(bm_global, 98)),
)
 
 
fill_alpha = PADDOCK_ALPHA if USE_OSM else 1.0
 
 
# ==== plot loop for each date ====
 
for _, row in coverage.iterrows():
    date = row["date_only"]
    n_pads = int(row["n_paddocks"])
    date_str = str(date)
 
    day_df = df[df["date_only"] == date][["paddock_name", "ndvi_median", "biomass_median"]].copy()
    gdf = paddocks_plot.merge(day_df, left_on=NAME_COL, right_on="paddock_name", how="left")
 
    gdf_data = gdf[gdf["ndvi_median"].notna()]
    gdf_nodata = gdf[gdf["ndvi_median"].isna()]
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H))
    fig.suptitle(
        f"{date_str}", #" - {n_pads}/{total_paddocks} paddocks",
        fontsize=13, fontweight="bold", y=0.99,
    )
 
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
 
    # ==== ndvi subfig ====
    if USE_OSM:
        ctx.add_basemap(ax1, crs=paddocks_plot.crs, source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto")

    gdf_data.plot(ax=ax1, column="ndvi_median", cmap=ndvi_cmap, norm=ndvi_norm,
                  edgecolor="white", linewidth=0.3)
    if not gdf_nodata.empty:
        gdf_nodata.plot(ax=ax1, color="#d0d0d0", edgecolor="white", linewidth=0.3)
 
    ax1.set_title("median NDVI", fontsize=11)
    #ax1.set_xlabel("Longitude", fontsize=9)
    ax1.set_ylabel("Latitude",  fontsize=9)
    ax1.tick_params(labelsize=8)
 
    # Colorbar attached to the axes — same height as the map, no excess whitespace
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes("right", size="3%", pad=0.08)
    plt.colorbar(ScalarMappable(norm=ndvi_norm, cmap=ndvi_cmap),
                 cax=cax1, label="NDVI")
    cax1.tick_params(labelsize=8)
 
    # ==== biomass subfig ====
    if USE_OSM:
        ctx.add_basemap(ax2, crs=paddocks_plot.crs, source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto")
 
    gdf_data.plot(ax=ax2, column="biomass_median", cmap=bm_cmap, norm=bm_norm,
                  edgecolor="white", linewidth=0.3)
    if not gdf_nodata.empty:
        gdf_nodata.plot(ax=ax2, color="#d0d0d0", edgecolor="white", linewidth=0.3)
 
    ax2.set_title("Biomass", fontsize=11)
    ax2.set_xlabel("Longitude", fontsize=9)
    ax2.tick_params(labelsize=8)
 
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="3%", pad=0.08)
    plt.colorbar(ScalarMappable(norm=bm_norm, cmap=sns_cm),
                 cax=cax2, label="kg DM ha⁻¹")
    cax2.tick_params(labelsize=8)
 
    plt.tight_layout(rect=[0, 0, 1, 0.97])
 
    out_path = OUTDIR / f"{date_str}_ndvi_biomass_map.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
 
print(f"\nDone. {len(coverage)} maps saved to {OUTDIR}")
