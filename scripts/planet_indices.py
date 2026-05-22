# -*- coding: utf-8 -*-
"""
Author:
    Dr. Camila CESAR
    Toi Hangarau | Geospatial Research Institute
    Te Whare Wananga o Waitaha | University of Canterbury
    Ōtautahi, Aotearoa | Christchurch, New Zealand

Description:
    run after planet_bands_all.py
    Reads the band-stats Parquet produced by planet_bands_all.py 

    Output: output/Planet/indices_all_paddocks.parquet

---------
  index refs  (all use mean reflectance per band per paddock-image pair)
---------

  NDVI    Normalised Difference Vegetation Index
          (NIR - Red) / (NIR + Red)
          Saturates at high biomass (LAI > 3).

  NDRE    Normalised Difference Red Edge Index
          (NIR - RedEdge) / (NIR + RedEdge)
          Less saturation than NDVI at high biomass (<2500kg DM)
          Requires red-edge band (SuperDove band 7)

  GNDVI   Green NDVI
          (NIR - Green) / (NIR + Green)
          Correlates well with chlorophyll content, less saturation than NDVI.

  EVI     Enhanced Vegetation Index
          2.5 * (NIR - Red) / (NIR + 6·Red - 7.5·Blue + 1)
          Reduced atmospheric + soil noise,better in dense canopy

  SAVI    Soil-Adjusted Vegetation Index  (L = 0.5)
          ((NIR - Red) / (NIR + Red + L)) * (1 + L)
          Reduces bare-soil influence, useful in paddocks with mixed cover

  CIre    Chlorophyll Index Red Edge
          (NIR / RedEdge) - 1
          Strongly correlated with canopy chlorophyll content. Use SuperDove band 7 

  LAI_proxy  Simplified LAI estimate from NDRE
          Empirical approximation

  Note on band scaling:
    Planet SR products are scaled as integer DN (typically 0-10000 for SR,
    sometimes 0-65535 for 16-bit). If your data still has the raw integer
    scale (not converted to 0-1 reflectance), the ratio-based indices (NDVI,
    NDRE, GNDVI, CIre) are unaffected, but EVI and SAVI denominators assume
    0-1 reflectance. Set REFLECTANCE_SCALE below to match your data.

"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------
#  CONFIG
# ---------

working_dir = Path(os.getcwd())
output_dir  = (working_dir / "output").resolve()
data_dir    = output_dir / "Planet"

band_stat_parquet = data_dir / "band_stats_all_paddocks.parquet"
indexes_parquet    = data_dir / "indices_all_paddocks.parquet"

# Planet SR integer scale factor.
#   10000  → divide by 10000 to get 0–1 reflectance (common for PSScene SR)
#   1      → values are already 0–1 reflectance
reflectance_scale = 10000


# ---------
#  load stats
# ---------
print("== Loading band stats ==")

if not band_stat_parquet.exists():
    raise FileNotFoundError(
        f"Band stats Parquet not found: {band_stat_parquet}\n"
        "Run planet_bands_all.py first."
    )

df = pd.read_parquet(band_stat_parquet)
print(f"  {len(df)} records, {df['paddock_name'].nunique()} paddocks, "
      f"{df['tif_id'].nunique()} unique images.")

# Parse acquisition date
df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)


# ---------
#  scaled reflectance func
# ---------
S = reflectance_scale

def col(band: str, stat: str = "mean") -> pd.Series:
    """Return a band-stat column scaled to 0–1 reflectance"""
    return df[f"{band}_{stat}"] / S


# ---------
#  index calc
# ---------
print("\n== Computing spectral indices ==")

out = df[["paddock_name", "tif_id", "date"]].copy()

# --- Retrieve scaled reflectance for each band (mean only for index math) ----
NIR  = col("nir")
RED  = col("red")
RE   = col("red_edge")
GRN  = col("green")
BLU  = col("blue")

eps = 1e-6   # avoid divide-by-zero


# --------- NDVI ---------
out["ndvi"] = (NIR - RED) / (NIR + RED + eps)

# --------- NDRE ---------
out["ndre"] = (NIR - RE) / (NIR + RE + eps)

# --------- GNDVI ---------
out["gndvi"] = (NIR - GRN) / (NIR + GRN + eps)

# --------- EVI  (assumes reflectance 0–1) ---------
out["evi"] = 2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLU + 1 + eps)

# --------- SAVI  (L = 0.5) ---------
L = 0.5
out["savi"] = ((NIR - RED) / (NIR + RED + L + eps)) * (1 + L)

# --------- CIre ---------
out["ci_re"] = (NIR / (RE + eps)) - 1

# --------- LAI proxy from NDRE (empirical, relative use only) ---------
ndre = out["ndre"].clip(-1, 1)
out["lai_proxy"] = np.exp(2.5 * ndre)   # simple exponential proxy

# --------- Pixel count (from NIR – representative of valid coverage) ---------
out["n_pixels"] = df["nir_n_pixels"]


# ---------
#  QUICK SANITY CHECK
# ---------
idx_cols = ["ndvi", "ndre", "gndvi", "evi", "savi", "ci_re", "lai_proxy"]
print("\n  Summary statistics:")
print(out[idx_cols].describe().round(4).to_string())

n_invalid = out[idx_cols].isna().any(axis=1).sum()
if n_invalid:
    print(f"\n  WARNING: {n_invalid} rows have at least one NaN index "
          " retained in output")


# ---------
#  SAVE
# ---------
out = out.sort_values(["paddock_name", "date"]).reset_index(drop=True)
out.to_parquet(indexes_parquet)
print(f"\nIndices saved: {indexes_parquet}")
print(f"Columns: {list(out.columns)}")
print("\nDone.")
