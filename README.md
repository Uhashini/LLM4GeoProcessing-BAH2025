# LLM4GeoProcessing-BAH2025

# 🌍 LLM-Powered Geospatial Assistant

An interactive, AI-driven assistant that transforms natural language queries into dynamic geospatial workflows, maps, and visualizations. Designed for planners, researchers, and emergency responders, it integrates LLM reasoning with real-time spatial data sources.

---

## 📑 Table of Contents

- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Demo Video Link](#demo)

---

## 🧠 About

The *LLM-Powered GIS Assistant* enables users to submit queries like "Show rainfall and population density in Chennai" and receive real-time maps, climate graphs, and spatial analysis. It combines natural language understanding from a local LLM with APIs like Open-Meteo, NASA, WorldPop, and OSM.

It is ideal for applications in:
- Urban and environmental planning
- Disaster risk analysis
- Climate monitoring and adaptation

---

## 🚀 Features

✅ Accepts natural language spatial queries  
✅ Parses and interprets queries via local LLM (Ollama)  
✅ Generates maps with:
- Real-time overlays (rainfall, humidity, wind, etc.)
- Building footprints, roads, hospitals from OpenStreetMap
- Population density via WorldPop  
✅ Produces climate trend graphs  
✅ Supports flood detection using Sentinel-1 (via GEE)  
✅ Emoji-based map annotations for intuitive visuals  
✅ REST API for integration  
✅ Web frontend with interactive UI

---

## 🧰 Tech Stack

| Layer         | Technology                                   |
|---------------|----------------------------------------------|
| Frontend      | React (via create-react-app)                 |
| Backend       | Python 3.9+, Flask, Folium, GeoPandas        |
| LLM Reasoning | Ollama + Gemma3 (can be swapped with others) |
| GIS Tools     | Shapely, OSMnx, Earth Engine (optional)      |
| Data Sources  | Open-Meteo, WorldPop, NASA FIRMS, OSM        |
| Visualization | Folium, Matplotlib                          |
| Security      | CORS-enabled API, local execution            |

---

## ⚙ Getting Started

### ✅ Prerequisites

- Python 3.9+
- Node.js & npm (for frontend)
- Git
- Ollama with gemma3 or llama3 installed
- (Optional) Earth Engine CLI authenticated for flood detection

---

### 🔧 Backend Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/llm-gis-assistant.git
cd llm-gis-assistant/backend

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install backend dependencies
pip install -r requirements.txt
# Or manually:
pip install flask folium geopandas osmnx shapely requests matplotlib numpy

# 4. Run backend
python app.py


Demo Link : https://drive.google.com/file/d/1TdwBWu28bOCepypg1AmwgyBUpH8qseqw/view?usp=sharing
