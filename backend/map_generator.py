import folium
import geopandas as gpd
from shapely.geometry import Point
from geocoder_utils import get_coordinates
import osmnx as ox


def generate_sample_map(place_name="Chennai", buffer_km=10):
    location = get_coordinates(place_name)
    if not location:
        print("❌ Unable to geocode location.")
        return None

    lat, lon, city = location["lat"], location["lon"], location["name"]

    # Create point geometry
    point = Point(lon, lat)
    gdf_point = gpd.GeoDataFrame([{"city": city}], geometry=[point], crs="EPSG:4326")

    # Project to UTM for buffer (Zone 43N default for India)
    gdf_projected = gdf_point.to_crs(epsg=32643)
    gdf_buffered = gdf_projected.buffer(buffer_km * 1000)  # in meters

    # Reproject buffer to WGS84 for map display
    gdf_buffer_wgs = gpd.GeoDataFrame(geometry=gdf_buffered).set_crs(epsg=32643).to_crs(epsg=4326)

    # Create Folium map centered on location
    fmap = folium.Map(location=[lat, lon], zoom_start=11, control_scale=True)

    # Add location marker
    folium.Marker(location=[lat, lon], popup=city, icon=folium.Icon(color="red")).add_to(fmap)

    # Add buffer zone
    folium.GeoJson(gdf_buffer_wgs.geometry.__geo_interface__, name="Buffer Zone", style_function=lambda x: {
        "fillColor": "orange", "color": "orange", "weight": 2, "fillOpacity": 0.3
    }).add_to(fmap)

        # Download roads and hospitals from OpenStreetMap
    try:
        buffer_polygon = gdf_buffer_wgs.unary_union
        roads = ox.geometries_from_polygon(buffer_polygon, tags={"highway": True})
        hospitals = ox.geometries_from_polygon(buffer_polygon, tags={"amenity": "hospital"})

        print(f"✅ Roads: {len(roads)}, Hospitals: {len(hospitals)}")

        # Add roads to map
        roads_wgs = roads.to_crs(epsg=4326)
        folium.GeoJson(roads_wgs, name="OSM Roads", style_function=lambda x: {
            "color": "gray", "weight": 1
        }).add_to(fmap)

        # Add hospitals to map
        hospitals_wgs = hospitals.to_crs(epsg=4326)
        for _, row in hospitals_wgs.iterrows():
            if row.geometry.geom_type == "Point":
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup="Hospital",
                    icon=folium.Icon(color="green", icon="plus-sign")
                ).add_to(fmap)

    except Exception as e:
        print(f"⚠️ OSM data error: {e}")


    # Add NASA GIBS WMS Flood Layer (MODIS)
    folium.raster_layers.WmsTileLayer(
        url="https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi",
        name="NASA Flood (MODIS)",
        layers="MODIS_Terra_SurfaceReflectance_Bands143",
        format="image/png",
        transparent=True,
        attr="NASA GIBS",
        opacity=0.6,
    ).add_to(fmap)

    folium.LayerControl().add_to(fmap)

        # 🌧 Rainfall (GPM IMERG precipitation)
    folium.raster_layers.WmsTileLayer(
        url="https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi",
        name="Precipitation (GPM)",
        layers="GPM_3IMERGDL_06_precipitationCal",
        format="image/png",
        transparent=True,
        attr="NASA GIBS",
        opacity=0.5
    ).add_to(fmap)

    # 🗻 Elevation (SRTM)
    folium.raster_layers.WmsTileLayer(
        url="https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi",
        name="Elevation (SRTM)",
        layers="SRTM_Color_Index",
        format="image/png",
        transparent=True,
        attr="NASA GIBS",
        opacity=0.4
    ).add_to(fmap)

    output_file = "output_map.html"
    fmap.save(output_file)
    print(f"✅ Map with global NASA flood layer saved to {output_file}")
    return output_file
