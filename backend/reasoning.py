import folium
import geopandas as gpd
import osmnx as ox
import requests
from shapely.geometry import Point
from shapely import union_all
import json
import os
import re
import datetime
import matplotlib.pyplot as plt
import numpy as np
import ee
from IPython.display import Image, display
import geemap
from geemap import foliumap

# ------------------------ CONFIG --------------------------------------
BEARER_TOKEN = "<YOUR_NASA_TOKEN_HERE>"
NASA_DOWNLOAD_DIR = "nasa_downloads"
os.makedirs(NASA_DOWNLOAD_DIR, exist_ok=True)

# Initialize Google Earth Engine
try:
    ee.Initialize(project="light-client-464917-h3")
    print("✅ Google Earth Engine initialized successfully")
except Exception as e:
    print(f"❌ GEE initialization error: {e}")
    print("Please run: ee.Authenticate() first, then ee.Initialize()")

# ------------------------ EMOJI LEGEND --------------------------------
EMOJI_ICONS = {
    "temperature": "🌡️",
    "humidity": "💧",
    "precipitation": "🌧️",
    "rain": "🌧️",
    "hospital": "🏥",
    "road": "🛣️",
    "satellite": "🛰️",
    "hurricane": "🌀",
    "flood": "🌊",
    "fire": "🔥",
    "ndvi": "🌿",
    "urban": "🏙️"
}

# ------------------------ UTILITIES --------------------------------------
def fix_quotes_and_json(text):
    text = text.replace("\u201c", "\"").replace("\u201d", "\"").replace("\u2018", "'").replace("\u2019", "'")
    json_match = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
    return json_match.group(0) if json_match else "[]"

def get_real_data_from_api(data_type, lat, lon):
    try:
        today = datetime.date.today()
        start = today - datetime.timedelta(days=180)
        end = today

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": data_type,
            "timezone": "auto"
        }

        response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
        data = response.json()

        daily_data = data["daily"].get(data_type, [])
        dates = data["daily"].get("time", [])

        month_values = {}
        for date, value in zip(dates, daily_data):
            if value is not None:
                month = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%b")
                month_values.setdefault(month, []).append(value)

        avg_per_month = {
            k: sum(x for x in v if x is not None) / max(1, len([x for x in v if x is not None]))
            for k, v in month_values.items()
        }
        return avg_per_month
    except Exception as e:
        print(f"❌ Real data error: {e}")
        return None

def generate_query_graph(data_type="rainfall", location="Chennai"):
    location_data = get_coordinates(location)
    if not location_data:
        return

    real_data_type_map = {
        "rainfall": "precipitation_sum",
        "precipitation": "precipitation_sum",
        "temperature": "temperature_2m_max",
        "humidity": "relative_humidity_2m_max",
        "windspeed": "wind_speed_10m_max"
    }

    query_field = real_data_type_map.get(data_type.lower())
    if not query_field:
        print(f"⚠️ No real-time data field found for '{data_type}'")
        return

    values = get_real_data_from_api(query_field, location_data['lat'], location_data['lon'])
    if not values:
        return

    plt.figure(figsize=(10, 5))
    plt.title(f"{data_type.capitalize()} Trend in {location}", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel(f"{data_type.capitalize()} Level")

    plt.plot(list(values.keys()), list(values.values()), marker='o', linestyle='-', color='blue')
    plt.grid(True)
    plt.tight_layout()

    output_path = "query_graph.png"
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Graph saved to {output_path}")
    return output_path

# ------------------------ Google Earth Engine Functions --------------------------------
def get_hurricane_dates(hurricane_name, year):
    """Get hurricane dates from known major hurricanes"""
    hurricane_dates = {
        "milton": {"year": 2024, "landfall": "2024-10-09", "start": "2024-10-07", "end": "2024-10-11"},
        "ian": {"year": 2022, "landfall": "2022-09-28", "start": "2022-09-26", "end": "2022-09-30"},
        "michael": {"year": 2018, "landfall": "2018-10-10", "start": "2018-10-08", "end": "2018-10-12"},
        "irma": {"year": 2017, "landfall": "2017-09-10", "start": "2017-09-08", "end": "2017-09-12"},
        "matthew": {"year": 2016, "landfall": "2016-10-07", "start": "2016-10-05", "end": "2016-10-09"}
    }
    return hurricane_dates.get(hurricane_name.lower(), None)

def get_flooding_dates(location):
    """Get historical flooding dates for major cities"""
    flooding_dates = {
        "chennai": [
            {"event": "2015 Chennai Floods", "start": "2015-11-01", "end": "2015-12-31"},
            {"event": "2023 Chennai Floods", "start": "2023-11-01", "end": "2023-12-31"},
            {"event": "2021 Chennai Floods", "start": "2021-11-01", "end": "2021-12-31"}
        ],
        "mumbai": [
            {"event": "2005 Mumbai Floods", "start": "2005-07-26", "end": "2005-08-26"},
            {"event": "2017 Mumbai Floods", "start": "2017-08-29", "end": "2017-09-29"},
            {"event": "2019 Mumbai Floods", "start": "2019-07-01", "end": "2019-08-01"}
        ],
        "bangalore": [
            {"event": "2022 Bangalore Floods", "start": "2022-08-30", "end": "2022-09-30"}
        ],
        "hyderabad": [
            {"event": "2020 Hyderabad Floods", "start": "2020-10-13", "end": "2020-11-13"}
        ]
    }
    return flooding_dates.get(location.lower(), [])

def get_satellite_imagery(location, date_range, imagery_type="optical", off_nadir=False):
    """Get satellite imagery from Google Earth Engine with improved error handling"""
    try:
        # Get location coordinates
        location_data = get_coordinates(location)
        if not location_data:
            return None
            
        lat, lon = location_data['lat'], location_data['lon']
        
        # Create area of interest
        if location.lower() == "florida":
            # Florida bounding box
            aoi = ee.Geometry.Rectangle([-87.6, 24.5, -80.0, 31.0])
        else:
            # Create buffer around point
            point = ee.Geometry.Point([lon, lat])
            aoi = point.buffer(50000)  # 50km buffer
        
        # Parse date range
        start_date = date_range["start"]
        end_date = date_range["end"]
        
        print(f"🔍 Searching for imagery from {start_date} to {end_date}")
        
        # Select satellite collection based on imagery type
        if imagery_type == "optical":
            if off_nadir:
                # Use Sentinel-2 for off-nadir capability
                collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                    .filterDate(start_date, end_date) \
                    .filterBounds(aoi) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                
                # Filter for off-nadir images (viewing angle > 0)
                collection = collection.filter(ee.Filter.gt('MEAN_INCIDENCE_ZENITH_ANGLE_B8', 0))
                print(f"📡 Searching Sentinel-2 off-nadir collection...")
            else:
                # Try Sentinel-2 first, then Landsat
                collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                    .filterDate(start_date, end_date) \
                    .filterBounds(aoi) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                
                print(f"📡 Searching Sentinel-2 collection...")
                
                # If no Sentinel-2, try Landsat
                if collection.size().getInfo() == 0:
                    print("⚠️ No Sentinel-2 images found, trying Landsat...")
                    collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                        .filterDate(start_date, end_date) \
                        .filterBounds(aoi) \
                        .filter(ee.Filter.lt('CLOUD_COVER', 50))
                    print(f"📡 Searching Landsat collection...")
        
        elif imagery_type == "radar":
            # Use Sentinel-1 SAR
            collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterDate(start_date, end_date) \
                .filterBounds(aoi) \
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
            print(f"📡 Searching Sentinel-1 SAR collection...")
        
        # Check if collection has any images
        collection_size = collection.size().getInfo()
        print(f"🔢 Found {collection_size} images in collection")
        
        if collection_size == 0:
            print(f"❌ No imagery found for {location} in date range {start_date} to {end_date}")
            return None
        
        # Get the most recent image
        image = collection.sort('system:time_start', False).first()
        
        # Verify image is not null
        try:
            image_info = image.getInfo()
            if image_info is None:
                print(f"❌ Retrieved image is null")
                return None
        except Exception as e:
            print(f"❌ Error getting image info: {e}")
            return None
        
        print(f"✅ Successfully retrieved image from collection")
        
        return {
            "image": image,
            "aoi": aoi,
            "collection": collection,
            "imagery_type": imagery_type,
            "off_nadir": off_nadir,
            "collection_size": collection_size
        }
        
    except Exception as e:
        print(f"❌ Satellite imagery error: {e}")
        return None

def create_satellite_map(location, date_range, imagery_type="optical", off_nadir=False, hurricane_name=None):
    """Create a map with satellite imagery with improved error handling"""
    try:
        # Get satellite imagery
        sat_data = get_satellite_imagery(location, date_range, imagery_type, off_nadir)
        if not sat_data:
            print("❌ No satellite data available, creating basic map instead")
            return create_basic_map(location, hurricane_name)
        
        # Get location for map center
        location_data = get_coordinates(location)
        if not location_data:
            return None
        
        lat, lon = location_data['lat'], location_data['lon']
        
        # Create geemap Map (inherits from folium)
        m = geemap.Map(center=[lat, lon], zoom=8)
        
        # Add satellite imagery layer
        try:
            if imagery_type == "optical":
                # Check if it's Sentinel-2 or Landsat based on band names
                image_bands = sat_data["image"].bandNames().getInfo()
                
                if 'B4' in image_bands:  # Sentinel-2
                    if off_nadir:
                        vis_params = {
                            'bands': ['B4', 'B3', 'B2'],
                            'min': 0,
                            'max': 3000,
                            'gamma': 1.4
                        }
                        layer_name = f"Sentinel-2 Off-Nadir {hurricane_name if hurricane_name else 'Imagery'}"
                    else:
                        vis_params = {
                            'bands': ['B4', 'B3', 'B2'],
                            'min': 0,
                            'max': 3000,
                            'gamma': 1.4
                        }
                        layer_name = f"Sentinel-2 {hurricane_name if hurricane_name else 'Imagery'}"
                
                elif 'SR_B4' in image_bands:  # Landsat
                    vis_params = {
                        'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
                        'min': 0.0,
                        'max': 0.3,
                        'gamma': 1.4
                    }
                    layer_name = f"Landsat {hurricane_name if hurricane_name else 'Imagery'}"
                
                else:
                    print(f"⚠️ Unknown band structure: {image_bands}")
                    vis_params = {
                        'bands': image_bands[:3],  # Use first 3 bands
                        'min': 0,
                        'max': 3000
                    }
                    layer_name = f"Satellite {hurricane_name if hurricane_name else 'Imagery'}"
            
            elif imagery_type == "radar":
                # Sentinel-1 SAR visualization
                vis_params = {
                    'bands': ['VV'],
                    'min': -25,
                    'max': 5
                }
                layer_name = f"Sentinel-1 SAR {hurricane_name if hurricane_name else 'Imagery'}"
            
            # Add the image to the map
            m.addLayer(sat_data["image"], vis_params, layer_name)
            print(f"✅ Added {layer_name} layer to map")
            
        except Exception as e:
            print(f"❌ Error adding satellite layer: {e}")
            # Add a basic marker instead
            folium.Marker(
                [lat, lon],
                popup=f"Satellite imagery unavailable for {location}",
                icon=folium.Icon(color="red")
            ).add_to(m)
        
        # Add location marker
        folium.Marker(
            [lat, lon],
            popup=f"{location} - {hurricane_name if hurricane_name else 'Satellite View'}",
            icon=folium.DivIcon(html=f"<div style='font-size: 20px;'>{EMOJI_ICONS['satellite']}</div>"),
            tooltip=f"Satellite imagery for {location}"
        ).add_to(m)
        
        # Add hurricane marker if applicable
        if hurricane_name:
            folium.Marker(
                [lat, lon],
                popup=f"Hurricane {hurricane_name.title()} - {location}",
                icon=folium.DivIcon(html=f"<div style='font-size: 20px;'>{EMOJI_ICONS['hurricane']}</div>"),
                tooltip=f"Hurricane {hurricane_name.title()}"
            ).add_to(m)
        
        # Add legend
        legend_items = ["satellite"]
        if hurricane_name:
            legend_items.append("hurricane")
        add_emoji_legend(m, legend_items)
        
        # Save map
        output_file = f"satellite_map_{location.replace(' ', '_')}_{hurricane_name if hurricane_name else 'imagery'}.html"
        m.save(output_file)
        print(f"✅ Satellite map saved to {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Satellite map creation error: {e}")
        return create_basic_map(location, hurricane_name)

def create_basic_map(location, hurricane_name=None):
    """Create a basic map when satellite imagery is not available"""
    try:
        location_data = get_coordinates(location)
        if not location_data:
            return None
        
        lat, lon = location_data['lat'], location_data['lon']
        
        # Create basic folium map
        m = folium.Map(location=[lat, lon], zoom_start=10)
        
        # Add location marker
        folium.Marker(
            [lat, lon],
            popup=f"{location} - Basic View",
            icon=folium.Icon(color="red"),
            tooltip=f"Basic view for {location}"
        ).add_to(m)
        
        # Add hurricane marker if applicable
        if hurricane_name:
            folium.Marker(
                [lat, lon],
                popup=f"Hurricane {hurricane_name.title()} - {location}",
                icon=folium.DivIcon(html=f"<div style='font-size: 20px;'>{EMOJI_ICONS['hurricane']}</div>"),
                tooltip=f"Hurricane {hurricane_name.title()}"
            ).add_to(m)
        
        # Save map
        output_file = f"basic_map_{location.replace(' ', '_')}_{hurricane_name if hurricane_name else 'view'}.html"
        m.save(output_file)
        print(f"✅ Basic map saved to {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Basic map creation error: {e}")
        return None

def analyze_change_detection(location, before_date, after_date, analysis_type="ndvi"):
    """Perform change detection analysis with improved error handling"""
    try:
        location_data = get_coordinates(location)
        if not location_data:
            return None
        
        lat, lon = location_data['lat'], location_data['lon']
        
        # Create area of interest
        if location.lower() == "florida":
            aoi = ee.Geometry.Rectangle([-87.6, 24.5, -80.0, 31.0])
        else:
            point = ee.Geometry.Point([lon, lat])
            aoi = point.buffer(25000)  # 25km buffer
        
        print(f"🔍 Analyzing change from {before_date} to {after_date}")
        
        # Expand date ranges to increase chance of finding imagery
        before_start = (datetime.datetime.strptime(before_date, "%Y-%m-%d") - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        before_end = (datetime.datetime.strptime(before_date, "%Y-%m-%d") + datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        
        after_start = (datetime.datetime.strptime(after_date, "%Y-%m-%d") - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        after_end = (datetime.datetime.strptime(after_date, "%Y-%m-%d") + datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Get before and after imagery
        before_collection = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate(before_start, before_end) \
            .filterBounds(aoi) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
        
        after_collection = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate(after_start, after_end) \
            .filterBounds(aoi) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
        
        # Check if collections have images
        before_size = before_collection.size().getInfo()
        after_size = after_collection.size().getInfo()
        
        print(f"📊 Before collection: {before_size} images")
        print(f"📊 After collection: {after_size} images")
        
        if before_size == 0 or after_size == 0:
            print(f"❌ Insufficient imagery for change detection")
            
            # Try with Landsat instead
            print("🔄 Trying Landsat imagery...")
            
            before_collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterDate(before_start, before_end) \
                .filterBounds(aoi) \
                .filter(ee.Filter.lt('CLOUD_COVER', 50))
            
            after_collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterDate(after_start, after_end) \
                .filterBounds(aoi) \
                .filter(ee.Filter.lt('CLOUD_COVER', 50))
            
            before_size = before_collection.size().getInfo()
            after_size = after_collection.size().getInfo()
            
            if before_size == 0 or after_size == 0:
                print(f"❌ No Landsat imagery available either")
                return create_basic_change_map(location, before_date, after_date)
        
        before_image = before_collection.median()
        after_image = after_collection.median()
        
        if analysis_type == "ndvi":
            # Calculate NDVI - check if Sentinel-2 or Landsat
            try:
                # Try Sentinel-2 bands first
                before_ndvi = before_image.normalizedDifference(['B8', 'B4']).rename('NDVI_before')
                after_ndvi = after_image.normalizedDifference(['B8', 'B4']).rename('NDVI_after')
                sensor_type = "Sentinel-2"
            except:
                # Try Landsat bands
                before_ndvi = before_image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI_before')
                after_ndvi = after_image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI_after')
                sensor_type = "Landsat"
            
            # Calculate difference
            ndvi_diff = after_ndvi.subtract(before_ndvi).rename('NDVI_change')
            
            # Create change map
            m = geemap.Map(center=[lat, lon], zoom=9)
            
            # Add layers
            try:
                m.addLayer(before_ndvi, {'min': -1, 'max': 1, 'palette': ['red', 'yellow', 'green']}, f'NDVI Before ({sensor_type})')
                m.addLayer(after_ndvi, {'min': -1, 'max': 1, 'palette': ['red', 'yellow', 'green']}, f'NDVI After ({sensor_type})')
                m.addLayer(ndvi_diff, {'min': -0.5, 'max': 0.5, 'palette': ['red', 'white', 'green']}, 'NDVI Change')
                
                print(f"✅ Added NDVI layers using {sensor_type} data")
                
            except Exception as e:
                print(f"❌ Error adding NDVI layers: {e}")
                return create_basic_change_map(location, before_date, after_date)
            
            # Add marker
            folium.Marker(
                [lat, lon],
                popup=f"NDVI Change Analysis - {location}",
                icon=folium.DivIcon(html=f"<div style='font-size: 20px;'>{EMOJI_ICONS['ndvi']}</div>"),
                tooltip="NDVI Change Detection"
            ).add_to(m)
            
            add_emoji_legend(m, ["ndvi"])
            
            output_file = f"ndvi_change_{location.replace(' ', '_')}.html"
            m.save(output_file)
            print(f"✅ NDVI change analysis saved to {output_file}")
            
            return output_file
            
    except Exception as e:
        print(f"❌ Change detection error: {e}")
        return create_basic_change_map(location, before_date, after_date)

def create_basic_change_map(location, before_date, after_date):
    """Create a basic change map when satellite analysis fails"""
    try:
        location_data = get_coordinates(location)
        if not location_data:
            return None
        
        lat, lon = location_data['lat'], location_data['lon']
        
        # Create basic folium map
        m = folium.Map(location=[lat, lon], zoom_start=10)
        
        # Add location marker
        folium.Marker(
            [lat, lon],
            popup=f"Change Analysis - {location}<br>Before: {before_date}<br>After: {after_date}<br>(Satellite data unavailable)",
            icon=folium.Icon(color="orange"),
            tooltip=f"Change analysis for {location}"
        ).add_to(m)
        
        # Save map
        output_file = f"basic_change_map_{location.replace(' ', '_')}.html"
        m.save(output_file)
        print(f"✅ Basic change map saved to {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Basic change map creation error: {e}")
        return None

# ------------------------ Realtime Overlay --------------------------------------
def add_real_time_overlay(fmap, lat, lon):
    try:
        for dlat in np.linspace(-0.15, 0.15, 5):
            for dlon in np.linspace(-0.15, 0.15, 5):
                res = requests.get("https://api.open-meteo.com/v1/forecast", params={
                    "latitude": lat + dlat,
                    "longitude": lon + dlon,
                    "hourly": "wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,precipitation",
                    "timezone": "auto"
                })
                js = res.json().get("hourly", {})

                wspd = js.get("wind_speed_10m", [None])[0]
                wdir = js.get("wind_direction_10m", [None])[0]
                temp = js.get("temperature_2m", [None])[0]
                hum = js.get("relative_humidity_2m", [None])[0]
                rain = js.get("precipitation", [None])[0]

                popup = []
                color = "#000000"

                if wspd is not None:
                    color = f"#{int(min(255, wspd * 10)):02x}6499"
                    popup.append(f"Wind: {wspd} m/s")
                if wdir is not None:
                    dx = 0.02 * np.sin(np.radians(wdir))
                    dy = 0.02 * np.cos(np.radians(wdir))
                    folium.PolyLine([(lat + dlat, lon + dlon), (lat + dlat + dy, lon + dlon + dx)], color="blue", weight=2).add_to(fmap)
                    popup.append(f"Dir: {wdir}°")

                # Use emoji marker
                emoji = ""
                if temp is not None:
                    emoji = EMOJI_ICONS.get("temperature", "")
                    popup.append(f"Temp: {temp} °C")
                elif hum is not None:
                    emoji = EMOJI_ICONS.get("humidity", "")
                    popup.append(f"Humidity: {hum}%")
                elif rain is not None:
                    emoji = EMOJI_ICONS.get("precipitation", "")
                    popup.append(f"Rain: {rain} mm")

                if emoji:
                    folium.Marker(
                        location=[lat + dlat, lon + dlon],
                        icon=folium.DivIcon(html=f"<div style='font-size: 20px;'>{emoji}</div>"),
                        popup="<br>".join(popup),
                        tooltip="Hover for details"
                    ).add_to(fmap)

        print("🌪️ Realtime weather overlay added")
    except Exception as e:
        print(f"❌ Overlay error: {e}")

# ------------------------ Emoji Legend --------------------------------
def add_emoji_legend(fmap, features_used):
    legend_html = """
     <div style='position: fixed; top: 10px; left: 10px; z-index: 9999;
                 background-color: white; padding: 10px; border: 2px solid black;
                 font-size: 14px; border-radius: 6px;'>
        <b>🧾 Legend</b><br>
    """
    for feature in features_used:
        emoji = EMOJI_ICONS.get(feature, "")
        if emoji:
            legend_html += f"{emoji} : {feature.capitalize()}<br>"
    legend_html += "</div>"
    fmap.get_root().html.add_child(folium.Element(legend_html))

# ------------------------ Enhanced LLM Reasoning --------------------------------
def parse_query_to_steps(query: str):
    try:
        query_lower = query.lower()
        steps = []

        # Location detection
        known_places = ["florida", "chennai", "mumbai", "texas", "california", "new york"]
        for place in known_places:
            if place in query_lower:
                steps.append({"action": "geocode", "place": place})
                break

        # Satellite imagery type
        if "radar" in query_lower:
            steps.append({"action": "satellite_imagery", "imagery_type": "radar", "off_nadir": False})
        elif "off-nadir" in query_lower:
            steps.append({"action": "satellite_imagery", "imagery_type": "optical", "off_nadir": True})
        elif "satellite" in query_lower or "imagery" in query_lower:
            steps.append({"action": "satellite_imagery", "imagery_type": "optical", "off_nadir": False})

        # Hurricane detection
        for name in ["milton", "ian", "michael", "irma", "matthew"]:
            if name in query_lower:
                steps.append({"action": "hurricane_analysis", "hurricane": name})
                break

        # Layers
        for keyword, layer in {
            "road": "road",
            "hospital": "hospital",
            "flood": "flood",
            "wind": "windspeed",
            "temperature": "temperature",
            "humidity": "humidity",
            "rain": "precipitation",
            "precipitation": "precipitation"
        }.items():
            if keyword in query_lower:
                steps.append({"action": "add_layer", "type": layer})

        # Change detection
        if "change" in query_lower or ("before" in query_lower and "after" in query_lower):
            steps.append({"action": "change_detection", "analysis_type": "ndvi"})

        return steps

    except Exception as e:
        print(f"❌ Reasoning parse error: {e}")
        return []


def handle_query(query):
    steps = parse_query_to_steps(query)
    query_lower = query.lower()

    place = "Chennai"
    radius = 10
    layer_type = "rainfall"
    layers_requested = []
    satellite_requested = False
    hurricane_analysis = False

    hurricane_found = None

    for step in steps:
        if step.get("action") == "geocode":
            place = step.get("place", place)
        elif step.get("action") == "buffer":
            radius = step.get("distance_km", radius)
        elif step.get("action") == "add_layer":
            layers_requested.append(step.get("type"))
            layer_type = step.get("type", layer_type)
        elif step.get("action") == "satellite_imagery":
            satellite_requested = True
            imagery_type = step.get("imagery_type", "optical")
            off_nadir = step.get("off_nadir", False)
        elif step.get("action") == "hurricane_analysis":
            hurricane_analysis = True
            hurricane_found = step.get("hurricane", hurricane_found)
        elif step.get("action") == "change_detection":
            analysis_type = step.get("analysis_type", "ndvi")

    print(f"🔍 Processing query: {query}")
    print(f"📍 Location: {place}")
    print(f"🎯 Satellite requested: {satellite_requested}")
    print(f"🌀 Hurricane analysis: {hurricane_analysis}")

    # Handle satellite imagery requests
    if satellite_requested or hurricane_analysis:
        if hurricane_found:
            hurricane_dates = get_hurricane_dates(hurricane_found, 2024)
            if hurricane_dates:
                date_range = {
                    "start": hurricane_dates["start"],
                    "end": hurricane_dates["end"]
                }
                create_satellite_map(place, date_range, "optical", "off-nadir" in query_lower, hurricane_found)
            else:
                print(f"⚠️ No date information found for Hurricane {hurricane_found}")
        else:
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=7)
            date_range = {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            }
            create_satellite_map(place, date_range, "optical", "off-nadir" in query_lower)

    # Handle traditional mapping
    if layers_requested and not satellite_requested:
        print(f"🗺️ Generating traditional map for {place} with buffer {radius}km and layers {layers_requested}")
        generate_map(place, radius, layers_requested)
        generate_query_graph(layer_type, place)

    # Handle change detection
    if "change" in query_lower:
        if hurricane_found:
            hurricane_dates = get_hurricane_dates(hurricane_found, 2024)
            if hurricane_dates:
                before_date = hurricane_dates["start"]
                after_date = hurricane_dates["end"]
                analyze_change_detection(place, before_date, after_date, "ndvi")


# ------------------------ Run ------------------------
if __name__ == "__main__":
    print("🚀 Enhanced GIS Assistant with Google Earth Engine")
    print("📡 Supports satellite imagery, hurricane analysis, and change detection")
    print("🌍 Examples:")
    print("  - 'Show me the off-nadir imagery for Florida just after Hurricane Milton made landfall'")
    print("  - 'Analyze NDVI changes in Chennai after flooding'")
    print("  - 'Show hospitals and roads around Mumbai with weather data'")
    print("  - 'Display Sentinel-1 radar imagery for coastal erosion analysis'")
    print()

    user_query = input("🧭 Enter your spatial query: ")
    handle_query(user_query)
