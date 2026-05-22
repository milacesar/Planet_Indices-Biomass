# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 12:10:02 2025

Author: Dr. Camila CESAR
        Toi Hangarau | Geospatial Research Institute
        Te Whare Wananga o Waitaha | University of Canterbury
        Ōtautahi, Aotearoa | Christchurch, New Zealand

"""

import zipfile
import geopandas as gpd
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import re


# --- functions ---

def parse_short_name(s):
    """Extract numeric parts (farm, pivot, paddock) as sortable tuple"""
    match = re.match(r"F(\d+)-P(\d+)-(\d+)", str(s))
    if match:
        return tuple(map(int, match.groups()))
    else:
        return (999, 999, 999)  # fallback for malformed names

def parse_description(desc):
    """Extract structured fields from the HTML <Description> block"""
    if not isinstance(desc, str) or desc.strip() == "":
        return {}
    soup = BeautifulSoup(desc, "html.parser")
    data = {}
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) == 2:
            key = cells[0].get_text(strip=True)
            value = cells[1].get_text(strip=True)
            data[key] = value
    return data



# ---end functions ---




# Path to your KMZ
kmz_path = Path("TWH_All_Paddocks_Aug2025.kmz")

extracted_name = "NTF_All_Aug2025.kml"
extracted_path = kmz_path.parent / extracted_name


# --- Check if .kml already exists ---
if extracted_path.exists():
    print(f"KML already exists → using {extracted_path}")
else:
    # --- Extract .kml from .kmz ---
    with zipfile.ZipFile(kmz_path, 'r') as kmz:
        for name in kmz.namelist():
            if name.endswith('.kml'):
                print(f"Extracting {name} from {kmz_path.name}")
                kmz.extract(name, kmz_path.parent)
                extracted = kmz_path.parent / name
                # Rename safely (delete old if something went wrong previously)
                if extracted_path.exists():
                    extracted_path.unlink()
                extracted.rename(extracted_path)
                break


# --- Load KML with GeoPandas ---
ntf_raw = gpd.read_file(extracted_path, driver='KML')
print("Loaded:", len(ntf_raw), "features")

# --- Parse the <Description> field into structured metadata ---

# Apply to all rows
meta = ntf_raw["Description"].apply(parse_description)
meta_df = pd.DataFrame(meta.tolist())

# Combine metadata with the geometry
ntf_parsed = pd.concat([ntf_raw.drop(columns=["Description"], errors="ignore"), meta_df], axis=1)

# --- Select wanted columns ---
wanted_cols = [
    'Farm Name', 
    'Short Name', # Farm#-Pivot##-Paddock##
    'Area (ha)',
    'geometry', 
    'Farm Region',
    'Farm Code','Feature Name',
    'Feature Type', 'Category Type', 'Land Management Unit'
]

ntf_select = ntf_parsed[[c for c in wanted_cols if c in ntf_parsed.columns]]

# --- Filter paddocks only ---
paddocks = ntf_select[ntf_select["Feature Type"].str.contains("Paddock", case=False, na=False)]
print(f"Filtered to {len(paddocks)} paddocks")

# --- Sort and reindex ---
paddocks = paddocks.assign(sort_key=paddocks["Short Name"].apply(parse_short_name))
paddocks = paddocks.sort_values(by="sort_key").drop(columns="sort_key")
# paddocks = paddocks.sort_values(by="Short Name")
paddocks = paddocks.reset_index(drop=True)
paddocks.index += 1

# --- Save as GeoPackage ---
output_path = kmz_path.with_name("NTF_Paddocks.gpkg")
paddocks.to_file(output_path, driver="GPKG")
print("Saved to:", output_path)

paddocks.to_parquet("NTF_Paddocks_sorted.parquet")
paddocks.to_pickle("NTF_Paddocks_sorted.pkl")
print('Saved to parquet and pickle filesformats')