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
import matplotlib
import shutil


# ------------------------ CONFIG --------------------------------------
BEARER_TOKEN = "<YOUR_NASA_TOKEN_HERE>"
NASA_DOWNLOAD_DIR = "nasa_downloads"
os.makedirs(NASA_DOWNLOAD_DIR, exist_ok=True)

# ------------------------ EMOJI LEGEND --------------------------------
EMOJI_ICONS = {
    "temperature": "🌡️",
    "humidity": "💧",
    "precipitation": "🌧️",
    "rain": "🌧️",
    "hospital": "🏥",
    "road": "🛣️",
    "fire": "🔥",
    "buildings": "🏢",
    "population": "👥",
    "flood": "🌧️"

}


# ------------------------ GEE AUTH --------------------------------------
def authenticate_gee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

# ------------------------ GEE IMAGERY FETCH ----------------------------
def get_gee_off_nadir_imagery(place, start_date, end_date):
    coords = get_coordinates(place)
    if not coords:
        return None

    point = Point(coords['lon'], coords['lat'])
    gdf = gpd.GeoDataFrame(geometry=[point], crs='EPSG:4326')
    gdf_proj = gdf.to_crs(epsg=32643)
    buffered = gdf_proj.buffer(30000).to_crs(epsg=4326)  # 30km radius
    region = buffered.geometry.iloc[0].__geo_interface__

    collection = (ee.ImageCollection("COPERNICUS/S2_SR")
                  .filterDate(start_date, end_date)
                  .filterBounds(ee.Geometry.Polygon(region['coordinates']))
                  .sort("CLOUDY_PIXEL_PERCENTAGE"))

    img = collection.first()
    url = img.getThumbURL({
        'region': region,
        'dimensions': 1024,
        'format': 'png',
        'min': 0,
        'max': 3000,
        'bands': ['B4', 'B3', 'B2']
    })
    output = f"gee_{place.replace(' ', '_')}.png"
    with open(output, 'wb') as f:
        f.write(requests.get(url).content)
    print(f"🚁️ GEE image saved as {output}")
    return output


def get_coordinates(place_name):
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": place_name, "format": "json", "limit": 1},
            headers={"User-Agent": "llm-gis-assistant"}
        )
        data = response.json()
        if not data:
            print(f"⚠️ Geocoding failed for: {place_name}")
            return None
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        print(f"📍 Geocoded: {place_name} -> ({lat}, {lon})")
        return {"lat": lat, "lon": lon, "name": place_name}
    except Exception as e:
        print(f"❌ Geocoding error: {e}")
        return None

# ------------------------ UTILITIES --------------------------------------
def fix_quotes_and_json(text):
    text = text.replace("\u201c", "\"").replace("\u201d", "\"").replace("\u2018", "'").replace("\u2019", "'")
    json_match = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
    return json_match.group(0) if json_match else "[]"

def get_real_data_from_api(data_type, lat, lon):
    try:
        today = datetime.date.today()
        start = today - datetime.timedelta(days=180)
        end = today - datetime.timedelta(days=1)


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
        if "daily" not in data:
            print("❌ Real data missing 'daily'")
            print("🔍 Full response:", data)
            return None
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

def generate_query_graph(data_type="rainfall", location="Chennai",output_path = "None"):
    matplotlib.use("Agg")  # Use a non-GUI backend

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
        print("📉 No data for graph. Creating empty placeholder.")
        plt.figure(figsize=(10, 5))
        plt.title(f"No {data_type.capitalize()} Data for {location}", fontsize=14)
        plt.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=12)
        plt.axis("off")
        fallback_path = f"static/query_graph_{location.lower().replace(' ', '_')}.png"
        plt.savefig(fallback_path)
        plt.close()
        return fallback_path


    plt.figure(figsize=(10, 5))
    plt.title(f"{data_type.capitalize()} Trend in {location}", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel(f"{data_type.capitalize()} Level")

    plt.plot(list(values.keys()), list(values.values()), marker='o', linestyle='-', color='blue')
    plt.grid(True)
    plt.tight_layout()

    output_path = output_path or f"query_graph_{location.lower().replace(' ', '_')}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Graph saved to {output_path}")
    return output_path

def get_fire_hotspots(lat, lon, radius_km=10):
    try:
        # Define bounding box for ~50km radius
        delta = 0.1  # ~10km
        bbox = f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"
        
        # NASA FIRMS WFS endpoint (no API key needed for public WFS)
        fire_url = f"https://firms.modaps.eosdis.nasa.gov/mapserver/wfs?service=WFS&version=1.1.0&request=GetFeature&typeName=viirs:VNP14IMGTDL_NRT&bbox={bbox}&outputFormat=application/json"
        
        res = requests.get(fire_url)
        fire_data = res.json()

        fire_points = []
        for feature in fire_data.get("features", []):
            coords = feature["geometry"]["coordinates"]
            fire_points.append((coords[1], coords[0]))

        return fire_points
    except Exception as e:
        print(f"🔥 FIRMS error: {e}")
        return []


# ------------------------ Realtime Overlay --------------------------------------
def add_real_time_overlay(fmap, lat, lon, layers_requested=None):
    if layers_requested is None:
        layers_requested = []

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
                emoji = ""

                # Add wind direction as arrow
                if wdir is not None:
                    dx = 0.02 * np.sin(np.radians(wdir))
                    dy = 0.02 * np.cos(np.radians(wdir))
                    folium.PolyLine([(lat + dlat, lon + dlon), (lat + dlat + dy, lon + dlon + dx)],
                                    color="blue", weight=2).add_to(fmap)
                    popup.append(f"Wind Dir: {wdir}°")

                if temp is not None and "temperature" in layers_requested:
                    emoji = EMOJI_ICONS["temperature"]
                    popup.append(f"Temp: {temp} °C")
                elif hum is not None and "humidity" in layers_requested:
                    emoji = EMOJI_ICONS["humidity"]
                    popup.append(f"Humidity: {hum}%")
                elif rain is not None and "precipitation" in layers_requested:
                    emoji = EMOJI_ICONS["precipitation"]
                    popup.append(f"Rain: {rain} mm")

                if emoji:
                    folium.Marker(
                        location=[lat + dlat, lon + dlon],
                        icon=folium.DivIcon(html=f"<div style='font-size: 18px;'>{emoji}</div>"),
                        popup="<br>".join(popup),
                        tooltip="Details"
                    ).add_to(fmap)

        print("🌪️ Realtime emoji overlay added")

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

# ------------------------ LLM Reasoning --------------------------------
def parse_query_to_steps(query: str):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:latest",
                "prompt": f"""
You are a professional geospatial and climate analyst assistant.

When a user gives you a query, do the following:

1. Return a **detailed, scientific, formatted explanation** of the requested topic.
2. Use structured layout with:
   - Sections
   - Emojis
   - Bullet points
   - Scientific notations
3. Keep the output visually readable, concise, and numbered where appropriate.
4. Then, output a JSON list of GIS steps for further spatial processing.

💡 Example output for explanation:

🌍 **City:** Chennai  
🌀 **Climate Type:** Tropical Wet and Dry (Aw)  
📅 **Seasons & Rainfall:**  
1. **Dry Season (Jan–Mar):** <10 mm/month  
2. **Pre-Monsoon (Apr–May):** ~50 mm  
3. **SW Monsoon (Jun–Aug):** ~100 mm  
4. **NE Monsoon (Oct–Nov):** 150–200 mm — Peak Season 🌧️  
5. **Post-Monsoon (Dec):** Rainfall drops back to ~40 mm  

💧 **Humidity:** 60–80%  
💨 **Wind Speed:** ~6–10 m/s avg during pre-monsoon  

📈 **Analysis Note:**  
- Use Open-Meteo for monthly data  
- Add precipitation + windspeed layers  
- Compare annual vs seasonal variance  
- Move to next line after each point

---

🎯 Now respond to this query with:
1. A well-formatted explanation as described above.
2. JSON steps under a `Steps:` section for GIS use.

Query: \"{query}\"
"""
,
                "stream": False
            }
        )
        full_output = response.json().get("response", "").strip()

        # 🧠 Extract explanation and JSON
        explanation = ""
        steps_json = "[]"

        if "Steps:" in full_output:
            explanation = full_output.split("Steps:")[0].replace("Explanation:", "").strip()
            steps_json = fix_quotes_and_json(full_output.split("Steps:")[1])
        else:
            explanation = full_output

        steps = json.loads(steps_json)
        return explanation, steps

    except Exception as e:
        print(f"❌ Reasoning error: {e}")
        return "Reasoning failed due to an error.", [{"action": "geocode", "place": "Chennai"}]


# ------------------------ Map Generator ------------------------
def generate_sample_map(place_name="Chennai", buffer_km=10, layers_requested=None, output_name="output_map.html"):
    if layers_requested is None:
        layers_requested = []

    location = get_coordinates(place_name)
    if not location:
        print("❌ Unable to geocode location.")
        return None

    lat, lon, city = location["lat"], location["lon"], location["name"]
    point = Point(lon, lat)
    gdf_point = gpd.GeoDataFrame([{"city": city}], geometry=[point], crs="EPSG:4326")

    # Buffer creation
    gdf_proj = gdf_point.to_crs(epsg=32643)
    gdf_buffered = gdf_proj.buffer(buffer_km * 1000)
    buffer_wgs = gdf_buffered.to_crs(epsg=4326)
    buffer_geo = gpd.GeoDataFrame(geometry=buffer_wgs, crs="EPSG:4326")
    buffer_poly = union_all(buffer_geo.geometry)

    # Create map
    fmap = folium.Map(location=[lat, lon], zoom_start=11)
    folium.Marker([lat, lon], popup=city, icon=folium.Icon(color="red")).add_to(fmap)

    # ✅ Add buffer to map
    folium.GeoJson(
        data=buffer_geo.__geo_interface__,
        name="Buffer",
        style_function=lambda x: {
            "fillColor": "orange",
            "color": "orange",
            "weight": 2,
            "fillOpacity": 0.3
        }
    ).add_to(fmap)

    # Add emoji overlays (temp/humidity/wind/rain)
    if any(x in layers_requested for x in ["windspeed", "temperature", "humidity", "precipitation"]):
        add_real_time_overlay(fmap, lat, lon, layers_requested)

    # Flood marker
    if "flood" in layers_requested:
        folium.Marker(
            location=[lat, lon],
            popup="Potential Flood Area",
            icon=folium.DivIcon(html=f"<div style='font-size:24px;'>{EMOJI_ICONS['flood']}</div>"),
            tooltip="Flood zone"
        ).add_to(fmap)
        print("🌧️ Flood emoji added")

    # Building footprints
    if "buildings" in layers_requested:
        try:
            buildings = ox.geometries_from_place(place_name, tags={"building": True})
            buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
            folium.GeoJson(
                data=buildings.__geo_interface__,
                name="Buildings",
                style_function=lambda x: {
                    "fillColor": "#800080",
                    "color": "#800080",
                    "weight": 0.5,
                    "fillOpacity": 0.4
                }
            ).add_to(fmap)
        except Exception as e:
            print(f"❌ Building load error: {e}")

    # Population
    if "population" in layers_requested:
        folium.TileLayer(
            tiles="https://tile.worldpop.org/wopr/Population_Density_2020/{z}/{x}/{y}.png",
            name="Population",
            attr="WorldPop",
            overlay=True,
            opacity=0.6
        ).add_to(fmap)

    # OSM features: road/hospital
    if "road" in layers_requested or "hospital" in layers_requested:
        try:
            tags = {}
            if "road" in layers_requested:
                tags["highway"] = True
            if "hospital" in layers_requested:
                tags["amenity"] = "hospital"

            features = ox.features_from_polygon(buffer_poly, tags).to_crs(epsg=4326)

            if "highway" in features.columns:
                folium.GeoJson(
                    features[features['highway'].notnull()],
                    name="Roads",
                    style_function=lambda x: {"color": "gray", "weight": 1}
                ).add_to(fmap)

            if "amenity" in features.columns:
                hospitals = features[features["amenity"] == "hospital"]
                for _, row in hospitals.iterrows():
                    if row.geometry.geom_type == "Point":
                        folium.Marker(
                            location=[row.geometry.y, row.geometry.x],
                            popup="Hospital",
                            icon=folium.DivIcon(html=f"<div style='font-size:20px'>{EMOJI_ICONS['hospital']}</div>")
                        ).add_to(fmap)

        except Exception as e:
            print(f"⚠️ OSM error: {e}")

    # Final touches
    add_emoji_legend(fmap, layers_requested)
    folium.LayerControl().add_to(fmap)
    os.makedirs("static", exist_ok=True)
    fmap.save(os.path.join("static", output_name))



# ------------------------ Handler ------------------------
def handle_query(query):
    explanation, steps = parse_query_to_steps(query)
    query_lower = query.lower()

    radius = 10
    climate_layers_priority = ["temperature", "humidity", "precipitation", "windspeed"]
    layers_requested = []

    # Detect all requested layers from query
    for word, layer in {
        "road": "road",
        "hospital": "hospital",
        "fire": "fire",
        "wind": "windspeed",
        "temperature": "temperature",
        "humidity": "humidity",
        "rain": "precipitation",
        "precipitation": "precipitation",
        "building": "buildings",
        "urban": "buildings",
        "population": "population",
        "density": "population"
    }.items():
        if word in query_lower and layer not in layers_requested:
            layers_requested.append(layer)

    # Select the first climate-related layer to use for graph
    layer_type = next((l for l in climate_layers_priority if l in layers_requested), "rainfall")


    # Heuristic layer detection
    for word, layer in {
        "road": "road",
        "hospital": "hospital",
        "fire": "fire",
        "wind": "windspeed",
        "temperature": "temperature",
        "humidity": "humidity",
        "rain": "precipitation",
        "precipitation": "precipitation",
        "building": "buildings",
        "urban": "buildings",
        "population": "population",
        "density": "population"
    }.items():
        if word in query_lower and layer not in layers_requested:
            layers_requested.append(layer)

    # Add flood if temporal/flood action found
    for step in steps:
        if step.get("action") in ["temporal_filter", "flood_extent_analysis"] and "flood" not in layers_requested:
            layers_requested.append("flood")

    # Get cities
    # Extract cities from steps or explanation fallback
    cities = [step.get("place") for step in steps if step.get("action") == "geocode" and step.get("place")]

# Fallback if LLM skipped geocode steps
    if not cities:
    # Try extracting from explanation
        found_in_text = re.findall(r"\*\*City\s\d?:\*\* ([A-Za-z\s]+)", explanation)
        if found_in_text:
            cities = [city.strip() for city in found_in_text]
        else:
            # Try from query string
            cities = re.findall(r"\b(?:in|near|around)\s+([A-Z][a-z]+)", query)
            if not cities:
                cities = ["Chennai"]  # Final fallback


    for step in steps:
        if step.get("action") == "buffer":
            radius = step.get("distance_km", radius)
        elif step.get("action") == "add_layer":
            layer_type = step.get("type", layer_type)
            if layer_type not in layers_requested:
                layers_requested.append(layer_type)

    # Map and graph generation
    map_outputs = []
    for place in cities:
        try:
            filename = f"map_{place.lower().replace(',', '').replace(' ', '_')}.html"
            generate_sample_map(place, radius, layers_requested, output_name=filename)
            graph_file = f"static/query_graph_{place.lower().replace(' ', '_')}.png"
            generate_query_graph(layer_type, place, output_path=graph_file)

            loc = get_coordinates(place)
            map_outputs.append({
                "city": place,
                "file": filename,
                'graph': graph_file,  # ✅ new
                "center": [loc["lat"], loc["lon"]],
                "layers": layers_requested
            })
        except Exception as e:
            print(f"❌ Failed for {place}: {e}")

    try:
        shutil.copy("query_graph.png", os.path.join("static", "query_graph.png"))
    except Exception as e:
        print(f"❌ Error copying graph to static: {e}")

    return {
        "message": explanation,
        "mapData": map_outputs,
        "graphData": {
            "type": f"{layer_type}_trend",
            "title": f"{layer_type.title()} Trends",
            "data": []
        }
    }


# ------------------------ Run ------------------------
if __name__ == "__main__":
    user_query = input("🧭 Enter your spatial query: ")
    handle_query(user_query)
