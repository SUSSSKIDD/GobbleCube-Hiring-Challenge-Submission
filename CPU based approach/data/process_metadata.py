import os
import zipfile
import pandas as pd
import geopandas as gpd
from urllib.request import urlretrieve
from pathlib import Path

DATA_DIR = Path(__file__).parent
ZIP_PATH = DATA_DIR / "taxi_zones.zip"
EXTRACT_DIR = DATA_DIR / "taxi_zones"
OUTPUT_CSV = DATA_DIR / "taxi_zone_meta.csv"

SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"

def main():
    # 1. Download Lookup
    print("Downloading taxi zone lookup...")
    lookup_path = DATA_DIR / "taxi_zone_lookup.csv"
    urlretrieve(LOOKUP_URL, lookup_path)
    
    # 2. Download Shapefile for centroids
    print("Downloading taxi zone shapefile...")
    urlretrieve(SHAPEFILE_URL, ZIP_PATH)
    
    print("Extracting shapefile...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    
    print("Calculating centroids...")
    # Read shapefile - find the .shp file
    shp_file = list(EXTRACT_DIR.rglob("*.shp"))[0]
    gdf = gpd.read_file(shp_file)
    
    # Convert to Lat/Lon (WGS84)
    gdf = gdf.to_crs(epsg=4326)
    
    # Calculate centroids
    gdf['lon'] = gdf.geometry.centroid.x
    gdf['lat'] = gdf.geometry.centroid.y
    
    # Merge with lookup to get boroughs
    lookup = pd.read_csv(lookup_path)
    
    # LocationID in lookup matches LocationID in shapefile
    meta = gdf[['LocationID', 'lat', 'lon']].copy()
    meta = meta.merge(lookup, on='LocationID', how='outer')
    
    # Save to CSV
    meta.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved metadata to {OUTPUT_CSV}")
    
    # Cleanup
    os.remove(ZIP_PATH)
    import shutil
    shutil.rmtree(EXTRACT_DIR)

if __name__ == "__main__":
    main()
