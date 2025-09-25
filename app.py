"""
app.py - TribalGIS OCR+NER + WebGIS demo with UI/UX and persistence (SQLite).
Run: python app.py
Open: http://127.0.0.1:5000
"""

import os
import sqlite3
import time
from flask import Flask, request, jsonify, render_template_string, render_template, g, redirect, url_for, session
from PIL import Image
import pytesseract
import spacy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from flask_cors import CORS
from functools import wraps

# ----------------- CONFIG -----------------
DB_PATH = "claims.db"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# optionally set tesseract path on Windows:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------- APP -----------------
app = Flask(__name__, template_folder='templates')
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["JSON_SORT_KEYS"] = False
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Demo users (replace with database in production)
USERS = {
    "admin": "admin123",
    "user": "user123"
}

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ----------------- DB -----------------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        try:
            db = g._database = sqlite3.connect(DB_PATH)
            db.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            # If database is corrupted, delete it and try again
            if "disk image is malformed" in str(e):
                try:
                    os.remove(DB_PATH)
                    db = g._database = sqlite3.connect(DB_PATH)
                    db.row_factory = sqlite3.Row
                    init_db()  # Reinitialize tables
                except Exception as ex:
                    print(f"Failed to recreate database: {ex}")
                    raise
    return db

def init_db():
    db = get_db()
    try:
        db.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            text TEXT,
            entities TEXT,
            saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        db.execute("""
        CREATE TABLE IF NOT EXISTS points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id INTEGER,
            label TEXT,
            name TEXT,
            lat REAL,
            lon REAL,
            seq INTEGER,
            FOREIGN KEY (claim_id) REFERENCES claims(id)
        )""")
        db.commit()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization error: {e}")
        raise

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

# ----------------- NLP + GEO -----------------
# Load spaCy model (NER)
nlp = spacy.load("en_core_web_sm")

# Geopy Nominatim with rate limiter (avoid hammering the server)
geolocator = Nominatim(user_agent="tribalgis_demo_app")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2, error_wait_seconds=2.0)

# ----------------- HTML (embedded template) -----------------
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TribalGIS — OCR + NER + WebGIS</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
  <style>
    body { background: linear-gradient(135deg,#f0f6f6 0%,#eef7f7 50%); }
    .topbar { background: linear-gradient(90deg,#0d6efd,#0b7285); color:white; padding:12px 20px; display:flex; align-items:center; justify-content:space-between; }
    .brand { font-weight:700; font-size:1.2rem; letter-spacing:0.6px; }
    .container-main { margin-top:20px; }
    .panel { background: #ffffff; border-radius:12px; padding:18px; box-shadow: 0 6px 20px rgba(12, 75, 75, 0.06); }
    #map { height:520px; border-radius:10px; }
    pre { background:#f7fcfc; padding:10px; border-radius:8px; border:1px solid #eef6f6; max-height:220px; overflow:auto; }
    .chips { display:flex; flex-wrap:wrap; gap:8px; }
    .chip { background:#e6f7f7; padding:6px 10px; border-radius:999px; color:#0b5a5a; font-weight:600; border:1px solid #d3f0f0; }
    .small-note { color:#4d6b6b; font-size:0.9rem; }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="brand">TribalGIS — OCR & NER + WebGIS</div>
    <div class="small-note">Demo: OCR → NER → Geocode → Map & Save</div>
  </div>

  <div class="container container-main">
    <div class="row g-3">
      <div class="col-lg-4">
        <div class="panel">
          <h5>Upload claim form</h5>
          <form id="uploadForm">
            <input id="fileInput" type="file" accept="image/*,.pdf" class="form-control my-2" />
            <div class="d-grid gap-2">
              <button id="btnProcess" class="btn btn-primary" type="button">Process & Preview</button>
              <button id="btnSave" class="btn btn-success" type="button" disabled>Save to DB</button>
            </div>
          </form>

          <hr/>

          <div>
            <h6 class="mb-1">Extracted Text</h6>
            <pre id="extractedText">No text yet. Upload an image to run OCR.</pre>
          </div>

          <div class="mt-3">
            <h6 class="mb-1">Detected Entities</h6>
            <div id="entities" class="chips"></div>
          </div>

          <div class="mt-3 small-note">
            Tip: Use clear scanned images for best OCR results. Geocoding uses OpenStreetMap Nominatim (rate limited).
          </div>
        </div>
      </div>

      <div class="col-lg-8">
        <div class="panel">
          <div class="d-flex justify-content-between mb-2">
            <h5 class="m-0">Map — Saved Claims & Current Extraction</h5>
            <div>
              <button id="btnRefresh" class="btn btn-outline-secondary btn-sm">Refresh Saved</button>
              <button id="btnClearCurrent" class="btn btn-outline-danger btn-sm">Clear Current</button>
            </div>
          </div>
          <div id="map"></div>
          <div class="mt-2 small-note">Markers in <span style="color:#0b7285;font-weight:700">teal</span> = saved claims, <span style="color:#d9534f;font-weight:700">red</span> = current extraction. Polyline traces order of detected places. No distance is shown.</div>
        </div>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
  <script>
    // Leaflet map setup
    const map = L.map('map').setView([20.5937,78.9629],5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{ attribution:'© OSM contributors' }).addTo(map);

    let savedLayer = L.layerGroup().addTo(map);
    let currentLayer = L.layerGroup().addTo(map);
    let currentPolyline = null;

    // helper functions
    function toJsonInputFile(files){
      if(!files || files.length===0) return null;
      return files[0];
    }

    async function fetchSaved() {
      savedLayer.clearLayers();
      const res = await fetch('/markers');
      const data = await res.json();
      data.forEach(c => {
        // group by claim id
        if(c.lat && c.lon){
          const m = L.circleMarker([c.lat,c.lon], {radius:6, color:'#0b8b88', fillColor:'#8ee0dd', fillOpacity:0.9}).addTo(savedLayer)
            .bindPopup(`<b>Claim:</b> ${c.claim_id}<br><b>Name:</b> ${c.name || '-'}<br><b>Label:</b> ${c.label}`);
        }
      });
    }

    document.getElementById('btnRefresh').addEventListener('click', () => fetchSaved());
    fetchSaved(); // load on start

    document.getElementById('btnProcess').addEventListener('click', async () => {
      const f = document.getElementById('fileInput').files;
      if(!f || f.length===0) { alert('Select an image or pdf first'); return; }
      const form = new FormData();
      form.append('file', f[0]);
      // show loading
      document.getElementById('extractedText').textContent = 'Processing...';
      const resp = await fetch('/extract', { method:'POST', body: form });
      const j = await resp.json();
      if(j.error){ alert('Error: ' + j.error); document.getElementById('extractedText').textContent='No text'; return; }

      // fill extracted text
      document.getElementById('extractedText').textContent = j.text || '(no text)';
      // show entities as chips
      const entDiv = document.getElementById('entities');
      entDiv.innerHTML = '';
      j.entities.forEach(e => {
        const el = document.createElement('div');
        el.className = 'chip';
        el.textContent = `${e.label}: ${e.text}`;
        entDiv.appendChild(el);
      });

      // plot current markers (only points)
      currentLayer.clearLayers();
      if(currentPolyline){ map.removeLayer(currentPolyline); currentPolyline = null;}
      let hasPoint = false;
      j.entities.forEach(e => {
        if(e.coordinates){
          hasPoint = true;
          L.marker([e.coordinates.lat, e.coordinates.lon], {icon: L.divIcon({className:'', html:'<div style="width:12px;height:12px;border-radius:50%;background:#d9534f;border:2px solid #fff"></div>'})})
            .addTo(currentLayer).bindPopup(`<b>${e.text}</b><br>${e.label}`);
        }
      });
      if(hasPoint){
        // fit to first point
        const first = j.entities.find(e => e.coordinates);
        if(first){
          map.setView([first.coordinates.lat, first.coordinates.lon], 10);
        }
      } else if(j.entities.length>0){
        // center to India if none geocoded
        map.setView([20.5937,78.9629],5);
      }

      // allow save
      const btnSave = document.getElementById('btnSave');
      btnSave.disabled = false;
      btnSave._lastResult = j; // stash last result
    });

    document.getElementById('btnSave').addEventListener('click', async () => {
      const btn = document.getElementById('btnSave');
      if(!btn._lastResult){ alert('No processed result to save'); return; }
      const payload = btn._lastResult;
      const res = await fetch('/save', {
        method:'POST',
        headers:{ 'Content-Type':'application/json' },
        body: JSON.stringify(payload)
      });
      const js = await res.json();
      if(js.success){
        alert('Saved claim and points to DB');
        document.getElementById('btnSave').disabled = true;
        fetchSaved();
      } else {
        alert('Save failed: ' + (js.error || 'unknown'));
      }
    });

    document.getElementById('btnClearCurrent').addEventListener('click', () => {
      currentLayer.clearLayers();
      if(currentPolyline){ map.removeLayer(currentPolyline); currentPolyline = null; }
      document.getElementById('extractedText').textContent = 'No text yet. Upload an image to run OCR.';
      document.getElementById('entities').innerHTML = '';
      document.getElementById('btnSave').disabled = true;
      document.getElementById('btnSave')._lastResult = null;
    });

  </script>
</body>
</html>
"""

# ----------------- API: extract OCR + NER + geocode -----------------
@app.route("/extract", methods=["POST"])
def extract():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        f = request.files["file"]
        filename = f.filename or f"upload_{int(time.time())}.png"
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(saved_path)

        # OCR
        try:
            text = pytesseract.image_to_string(Image.open(saved_path))
        except Exception as e:
            return jsonify({"error": f"OCR failed: {e}"}), 500

        # NER via spaCy
        doc = nlp(text)
        entities = []
        # track order
        seq = 0
        for ent in doc.ents:
            seq += 1
            entd = {"label": ent.label_, "text": ent.text, "seq": seq}
            # only attempt geocode for GPE/LOC/PLACE-like labels and if text length reasonable
            if ent.label_ in ("GPE", "LOC", "FAC", "NORP", "ORG") and len(ent.text) < 120:
                try:
                    place = geocode(ent.text)
                    if place:
                        entd["coordinates"] = {"lat": place.latitude, "lon": place.longitude}
                except Exception as ge:
                    # ignore geocode failure (rate limited or not found)
                    entd["geo_error"] = str(ge)
            entities.append(entd)

        return jsonify({"text": text, "entities": entities})
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

# ----------------- API: save result into DB -----------------
@app.route("/save", methods=["POST"])
def save():
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error":"No JSON body"}), 400
        text = payload.get("text","")
        entities = payload.get("entities",[])
        filename = payload.get("filename","uploaded")

        db = get_db()
        cur = db.cursor()
        cur.execute("INSERT INTO claims (filename, text, entities) VALUES (?,?,?)", (filename, text, str(entities)))
        claim_id = cur.lastrowid

        seq = 0
        for e in entities:
            if "coordinates" in e:
                seq += 1
                lat = e["coordinates"]["lat"]
                lon = e["coordinates"]["lon"]
                label = e.get("label","")
                name = e.get("text","")
                cur.execute("INSERT INTO points (claim_id, label, name, lat, lon, seq) VALUES (?,?,?,?,?,?)",
                            (claim_id, label, name, lat, lon, seq))
        db.commit()
        return jsonify({"success": True, "claim_id": claim_id})
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

# ----------------- API: get saved markers -----------------
@app.route("/markers", methods=["GET"])
def markers():
    try:
        db = get_db()
        cur = db.cursor()
        rows = cur.execute("SELECT p.id, p.claim_id, p.label, p.name, p.lat, p.lon, p.seq FROM points p ORDER BY p.id DESC").fetchall();
        out = []
        for r in rows:
            out.append({"id": r["id"], "claim_id": r["claim_id"], "label": r["label"], "name": r["name"], "lat": r["lat"], "lon": r["lon"], "seq": r["seq"]})
        return jsonify(out)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500
    


# ----------------- Auth Routes -----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        
        if username in USERS and USERS[username] == password:
            session['username'] = username
            return jsonify({"success": True}), 200
        return jsonify({"error": "Invalid credentials"}), 401
    
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# ----------------- UI Routes -----------------
@app.route("/")
def index():
    return redirect(url_for('login'))

@app.route("/app")
@login_required
def main_app():
    return render_template_string(HTML)

# ----------------- UI: Database Viewer -----------------
DB_VIEW_HTML = """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Database Viewer</title>
  <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css' rel='stylesheet'>
</head>
<body>
  <div class='container mt-4'>
    <h3>Claims Database Viewer</h3>
    <a href='/' class='btn btn-secondary btn-sm mb-3'>Back to App</a>
    <div id='db-content'></div>
  </div>
  <script>
    async function loadDB(){
      const res = await fetch('/db_data');
      const data = await res.json();
      let html = '<h5>Claims</h5><table class="table table-bordered"><thead><tr><th>ID</th><th>Filename</th><th>Text</th><th>Entities</th><th>Saved At</th></tr></thead><tbody>';
      data.claims.forEach(c => {
        html += `<tr><td>${c.id}</td><td>${c.filename}</td><td><pre>${c.text}</pre></td><td><pre>${c.entities}</pre></td><td>${c.saved_at}</td></tr>`;
      });
      html += '</tbody></table>';
      html += '<h5>Points</h5><table class="table table-bordered"><thead><tr><th>ID</th><th>Claim ID</th><th>Label</th><th>Name</th><th>Lat</th><th>Lon</th><th>Seq</th></tr></thead><tbody>';
      data.points.forEach(p => {
        html += `<tr><td>${p.id}</td><td>${p.claim_id}</td><td>${p.label}</td><td>${p.name}</td><td>${p.lat}</td><td>${p.lon}</td><td>${p.seq}</td></tr>`;
      });
      html += '</tbody></table>';
      document.getElementById('db-content').innerHTML = html;
    }
    loadDB();
  </script>
</body>
</html>
"""

@app.route("/db")
def db_view():
    return render_template_string(DB_VIEW_HTML)

@app.route("/db_data")
def db_data():
    try:
        db = get_db()
        claims = db.execute("SELECT * FROM claims ORDER BY id DESC").fetchall()
        points = db.execute("SELECT * FROM points ORDER BY id DESC").fetchall()
        claims_out = [dict(row) for row in claims]
        points_out = [dict(row) for row in points]
        return jsonify({"claims": claims_out, "points": points_out})
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

# ----------------- MAIN -----------------
if __name__ == "__main__":
    # initialize DB
    with app.app_context():
        init_db()
    print("Starting TribalGIS demo app on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
