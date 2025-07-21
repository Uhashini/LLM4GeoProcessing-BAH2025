# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import app1  # Your existing code
from app1 import handle_query


app = Flask(__name__)
CORS(app)  # Enable CORS for React

@app.route('/api/spatial-query', methods=['POST'])
def spatial_query():
    data = request.json
    query = data.get('query', '')
    
    # Use your existing handle_query function
    result = handle_query(query)

    print("🪵 Final result to return:", result)  # ✅ log here
    return jsonify({
        'message': result.get('message'),
        'mapData': result.get('mapData'),
        'graphData': result.get('graphData')
    })
    
    return jsonify({
        'message': f'Analysis complete for: {query}',
        'mapData': result.get('mapData'),
        'graphData': result.get('graphData')
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.getcwd(), filename)
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)