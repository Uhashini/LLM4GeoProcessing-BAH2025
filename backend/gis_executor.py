import geopandas as gpd
from shapely.geometry import Point

def load_data(file_path):
    return gpd.read_file(file_path)

def create_buffer(gdf, distance_km):
    gdf = gdf.to_crs(epsg=32643)  # Project to metric CRS (UTM Zone 43N for Tamil Nadu)
    gdf['geometry'] = gdf.buffer(distance_km * 1000)
    return gdf.to_crs(epsg=4326)  # Convert back to WGS84

def intersect_layers(layer1, layer2):
    return gpd.overlay(layer1, layer2, how='intersection')

def save_output(gdf, path="output/result.geojson"):
    gdf.to_file(path, driver="GeoJSON")
