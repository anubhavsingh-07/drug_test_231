import os
import io
import time
import json
import base64
import hashlib
import sqlite3
import numpy as np
import cv2
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# --- 1. SQLITE DATABASE INITIALIZATION ---
DB_FILE = "spectro_ndps_records.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS test_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id TEXT UNIQUE,
            timestamp_ist TEXT,
            officer_id TEXT,
            location_name TEXT,
            coords TEXT,
            reagent_name TEXT,
            analyte TEXT,
            result_status TEXT,
            delta_e REAL,
            rate REAL,
            sha256_seal TEXT,
            frame_t0 TEXT,
            frame_final TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- 2. REAGENT & ARUCO CONFIGURATION ---
REAGENT_DATABASE = {
    "scott_cocaine": {
        "name": "Modified Scott (Cobalt Thiocyanate)",
        "analyte": "Cocaine HCl",
        "duration_sec": 5,
        "target_lab": [82.0, 152.0, 69.0],
        "tolerance_de": 42.0,
        "ndps_schedule": "Schedule I (Commercial / Small)"
    },
    "marquis_opiates": {
        "name": "Marquis Reagent (Formaldehyde/H2SO4)",
        "analyte": "Opiates / Heroin",
        "duration_sec": 15,
        "target_lab": [75.0, 148.0, 115.0],
        "tolerance_de": 38.0,
        "ndps_schedule": "Schedule I (Narcotic Drug)"
    }
}

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
if hasattr(cv2.aruco, 'DetectorParameters'):
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()
else:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()

ARUCO_PARAMS.adaptiveThreshWinSizeMin = 3
ARUCO_PARAMS.adaptiveThreshWinSizeMax = 45
ARUCO_PARAMS.adaptiveThreshWinSizeStep = 4
ARUCO_PARAMS.adaptiveThreshConstant = 7
ARUCO_PARAMS.minMarkerPerimeterRate = 0.03
ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

try:
    DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    USE_NEW_API = True
except AttributeError:
    USE_NEW_API = False

def extract_well_data(frame_bytes):
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_boosted = clahe.apply(gray)

    if USE_NEW_API:
        corners, ids, _ = DETECTOR.detectMarkers(gray_boosted)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray_boosted, ARUCO_DICT, parameters=ARUCO_PARAMS)

    if ids is None or len(ids) < 4:
        # Fallback orthogonal crop if markers occluded
        h, w = frame.shape[:2]
        size = min(h, w)
        sy, sx = (h - size) // 2, (w - size) // 2
        calibrated = cv2.resize(frame[sy:sy+size, sx:sx+size], (600, 600))
    else:
        pts_src = []
        id_order = [0, 1, 2, 3]
        flat_ids = ids.flatten().tolist()
        for marker_id in id_order:
            if marker_id in flat_ids:
                idx = flat_ids.index(marker_id)
                pts_src.append(corners[idx][0][0])
            else:
                pts_src.append([0, 0])
        pts_dst = np.array([[0, 0], [600, 0], [600, 600], [0, 600]], dtype="float32")
        h_matrix = cv2.getPerspectiveTransform(np.array(pts_src, dtype="float32"), pts_dst)
        calibrated = cv2.warpPerspective(frame, h_matrix, (600, 600))

    # 18% Neutral Gray Calibration Patch (Top Center: y=80..130, x=270..330)
    gray_roi = calibrated[80:130, 270:330]
    mean_bgr = np.mean(gray_roi, axis=(0, 1))
    mean_b, mean_g, mean_r = max(mean_bgr[0], 1), max(mean_bgr[1], 1), max(mean_bgr[2], 1)

    scale_r = 119.0 / mean_r
    scale_g = 119.0 / mean_g
    scale_b = 119.0 / mean_b

    cal_b = np.clip(calibrated[:, :, 0] * scale_b, 0, 255).astype(np.uint8)
    cal_g = np.clip(calibrated[:, :, 1] * scale_g, 0, 255).astype(np.uint8)
    cal_r = np.clip(calibrated[:, :, 2] * scale_r, 0, 255).astype(np.uint8)
    photometric_frame = cv2.merge([cal_b, cal_g, cal_r])

    # Center Reaction Well (30x30 px ROI: y=285..315, x=285..315)
    well = photometric_frame[285:315, 285:315]
    glare_mask = cv2.inRange(well, np.array([220, 220, 220]), np.array([255, 255, 255]))
    non_glare = well[glare_mask == 0]

    if len(non_glare) > 30:
        sample_bgr = non_glare.reshape(-1, 1, 3)
        sample_lab = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2LAB)
        avg_lab = np.median(sample_lab, axis=0)[0]
        avg_rgb = [int(np.median(non_glare[:, 2])), int(np.median(non_glare[:, 1])), int(np.median(non_glare[:, 0]))]
    else:
        lab_well = cv2.cvtColor(well, cv2.COLOR_BGR2LAB)
        avg_lab = np.median(lab_well, axis=(0, 1))
        avg_rgb = [int(np.median(well[:, :, 2])), int(np.median(well[:, :, 1])), int(np.median(well[:, :, 0]))]

    _, buffer = cv2.imencode('.jpg', photometric_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
    calibrated_b64 = base64.b64encode(buffer).decode('utf-8')

    return avg_lab, avg_rgb, calibrated_b64

# --- 3. API ENDPOINTS ---

@app.route('/api/kinetic_assay', methods=['POST'])
def kinetic_assay():
    data = request.get_json()
    if not data or 'frames' not in data:
        return jsonify({"error": "Payload missing frames"}), 400

    frames_b64 = data['frames']
    reagent_key = data.get('reagent', 'scott_cocaine')
    officer_id = data.get('officer_id', 'POLICE-UK-8842')
    location_name = data.get('location_name', 'Dehradun, Uttarakhand')
    coords = data.get('coords', '30.3165° N, 78.0322° E')
    cfg = REAGENT_DATABASE.get(reagent_key, REAGENT_DATABASE['scott_cocaine'])

    lab_series = []
    rgb_series = []
    saved_frames = []

    for idx, f_b64 in enumerate(frames_b64):
        raw_bytes = base64.b64decode(f_b64.split(',')[1] if ',' in f_b64 else f_b64)
        avg_lab, avg_rgb, clean_frame_b64 = extract_well_data(raw_bytes)
        if avg_lab is not None:
            lab_series.append(avg_lab)
            rgb_series.append(avg_rgb)
            if idx == 0 or idx == len(frames_b64) - 1:
                saved_frames.append(clean_frame_b64)

    if len(lab_series) < 2:
        return jsonify({"error": "Insufficient valid frames"}), 400

    frame_t0_b64 = saved_frames[0]
    frame_final_b64 = saved_frames[-1]

    # Kinetic shift (T_final vs T_0)
    delta_es = []
    l0, a0, b0 = lab_series[0]
    for (l, a, b) in lab_series:
        de = float(np.sqrt((l - l0)**2 + (a - a0)**2 + (b - b0)**2))
        delta_es.append(round(de, 2))

    max_delta_e = max(delta_es)
    time_delta = cfg['duration_sec']
    kinetic_rate = round(max_delta_e / max(time_delta, 1), 2)

    # Reference match against UNODC chromophore
    cur_l, cur_a, cur_b = lab_series[-1]
    tgt_l, tgt_a, tgt_b = cfg['target_lab']
    delta_e_ref = round(float(np.sqrt((cur_l - tgt_l)**2 + (cur_a - tgt_a)**2 + (cur_b - tgt_b)**2)), 1)

    # Verdict classification
    if kinetic_rate < 3.0 and delta_e_ref < cfg['tolerance_de']:
        verdict = "ADULTERANT FLAGGED: STATIC DYE"
        badge_class = "badge-danger"
    elif delta_e_ref <= cfg['tolerance_de']:
        verdict = f"PRESUMPTIVE POSITIVE: {cfg['analyte'].upper()}"
        badge_class = "badge-success"
    else:
        verdict = "INCONCLUSIVE / ANOMALOUS SPECTRUM"
        badge_class = "badge-warning"

    now = datetime.now()
    timestamp_ist = now.strftime("%Y-%m-%d %H:%M:%S IST")
    test_id = f"S52A-{now.strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"

    # SHA-256 seal of the peak calibrated frame
    raw_final_bytes = base64.b64decode(frame_final_b64)
    sha256_seal = hashlib.sha256(raw_final_bytes).hexdigest().upper()

    # Save to SQLite
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            INSERT INTO test_records (
                test_id, timestamp_ist, officer_id, location_name, coords,
                reagent_name, analyte, result_status, delta_e, rate,
                sha256_seal, frame_t0, frame_final
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            test_id, timestamp_ist, officer_id, location_name, coords,
            cfg['name'], cfg['analyte'], verdict, delta_e_ref, kinetic_rate,
            sha256_seal, frame_t0_b64, frame_final_b64
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print("DB Insert Error:", e)

    return jsonify({
        "test_id": test_id,
        "timestamp_ist": timestamp_ist,
        "officer_id": officer_id,
        "location_name": location_name,
        "coords": coords,
        "analyte": cfg['analyte'],
        "reagent": cfg['name'],
        "delta_e_ref": delta_e_ref,
        "tolerance": cfg['tolerance_de'],
        "rate": kinetic_rate,
        "verdict": verdict,
        "badge_class": badge_class,
        "delta_es": delta_es,
        "sha256_seal": sha256_seal,
        "frame_t0": frame_t0_b64,
        "frame_final": frame_final_b64
    })

@app.route('/api/records', methods=['GET'])
def get_records():
    search_q = request.args.get('q', '').strip().lower()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if search_q:
        c.execute('''
            SELECT test_id, timestamp_ist, officer_id, location_name, analyte, result_status, delta_e, sha256_seal 
            FROM test_records 
            WHERE lower(test_id) LIKE ? OR lower(analyte) LIKE ? OR lower(officer_id) LIKE ? OR lower(location_name) LIKE ? OR lower(sha256_seal) LIKE ?
            ORDER BY id DESC
        ''', (f"%{search_q}%", f"%{search_q}%", f"%{search_q}%", f"%{search_q}%", f"%{search_q}%"))
    else:
        c.execute('''
            SELECT test_id, timestamp_ist, officer_id, location_name, analyte, result_status, delta_e, sha256_seal 
            FROM test_records ORDER BY id DESC LIMIT 20
        ''')
    rows = c.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "test_id": r[0],
            "timestamp": r[1],
            "officer": r[2],
            "location": r[3],
            "analyte": r[4],
            "verdict": r[5],
            "delta_e": r[6],
            "sha256": r[7][:16] + "..."
        })
    return jsonify(results)

# --- 4. FRONTEND HTML TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SpectroNDPS | MHA Field Companion</title>
<style>
  :root { --bg: #0d1117; --panel: #161b22; --border: #30363d; --accent: #238636; --text: #e6edf3; --muted: #8b949e; --warn: #d29922; --danger: #da3633; }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
  body { background: var(--bg); color: var(--text); padding: 12px; }
  .container { max-width: 680px; margin: 0 auto; }
  .header { display: flex; align-items: center; justify-content: space-between; padding-bottom: 12px; border-bottom: 1px solid var(--border); margin-bottom: 14px; }
  .header h1 { font-size: 1.1rem; color: #58a6ff; letter-spacing: 0.5px; }
  .header span { font-size: 0.75rem; background: #21262d; padding: 4px 8px; border-radius: 4px; border: 1px solid var(--border); }
  
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 12px; margin-bottom: 14px; }
  label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; font-weight: 600; display: block; margin-bottom: 4px; }
  select, input { width: 100%; background: #0d1117; border: 1px solid var(--border); color: var(--text); padding: 8px; border-radius: 4px; margin-bottom: 10px; font-size: 0.85rem; }
  
  .video-box { position: relative; width: 100%; height: 280px; background: #000; border-radius: 4px; overflow: hidden; border: 1px solid var(--border); margin-bottom: 10px; }
  video { width: 100%; height: 100%; object-fit: cover; }
  .overlay-ring { position: absolute; top: 50%; left: 50%; width: 70px; height: 70px; border: 2px dashed #58a6ff; border-radius: 50%; transform: translate(-50%, -50%); pointer-events: none; }
  
  .btn { width: 100%; background: var(--accent); color: #fff; font-weight: 700; border: none; padding: 12px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  
  .memo-box { background: #0d1117; border: 1px solid #58a6ff; border-radius: 6px; padding: 14px; margin-top: 14px; }
  .memo-header { font-size: 0.95rem; font-weight: bold; color: #58a6ff; border-bottom: 1px dashed var(--border); padding-bottom: 6px; margin-bottom: 10px; display: flex; justify-content: space-between; }
  .memo-meta { font-size: 0.75rem; line-height: 1.5; margin-bottom: 10px; }
  .memo-meta b { color: #58a6ff; }
  
  .snapshots-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0; }
  .snap-card { text-align: center; border: 1px solid var(--border); border-radius: 4px; padding: 6px; background: #161b22; }
  .snap-card img { width: 100%; height: 110px; object-fit: cover; border-radius: 3px; }
  .snap-card p { font-size: 0.65rem; color: var(--muted); margin-top: 4px; }
  
  .badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; margin-bottom: 8px; }
  .badge-success { background: rgba(35, 134, 54, 0.2); color: #3fb950; border: 1px solid #238636; }
  .badge-danger { background: rgba(218, 54, 51, 0.2); color: #f85149; border: 1px solid #da3633; }
  .badge-warning { background: rgba(210, 153, 34, 0.2); color: #d29922; border: 1px solid #9e6a03; }
  
  .disclaimer-box { font-size: 0.68rem; color: #8b949e; background: #21262d; border-left: 3px solid var(--warn); padding: 8px; border-radius: 0 4px 4px 0; margin-top: 10px; line-height: 1.35; }
  
  .tabs { display: flex; gap: 6px; margin-bottom: 12px; }
  .tab-btn { flex: 1; padding: 8px; background: #21262d; color: var(--text); border: 1px solid var(--border); border-radius: 4px; cursor: pointer; font-size: 0.8rem; font-weight: 600; }
  .tab-btn.active { background: #30363d; border-color: #58a6ff; }
  
  .table-box { overflow-x: auto; max-height: 260px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.72rem; }
  th, td { border: 1px solid var(--border); padding: 6px; text-align: left; }
  th { background: #21262d; color: #58a6ff; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>SPECTRO-NDPS // MHA SIH26231</h1>
    <span>SEC 52A CERTIFIED</span>
  </div>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchView('assay')">Live Assay & Terminal</button>
    <button class="tab-btn" onclick="switchView('logs')">Audit Records Log</button>
  </div>

  <!-- VIEW 1: ASSAY & LIVE MEMO -->
  <div id="assayView">
    <div class="card">
      <label>Target Reagent Protocol</label>
      <select id="reagentSelect">
        <option value="scott_cocaine">Modified Scott Reagent (Cocaine HCl) — 5s Scan</option>
        <option value="marquis_opiates">Marquis Reagent (Opiates/Heroin) — 15s Scan</option>
      </select>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
        <div>
          <label>Officer ID / Badge</label>
          <input type="text" id="officerId" value="IO-DEH-4091">
        </div>
        <div>
          <label>Location Landmark</label>
          <input type="text" id="locName" value="Herbertpur, Dehradun">
        </div>
      </div>

      <div class="video-box">
        <video id="webcam" autoplay playsinline muted></video>
        <div class="overlay-ring"></div>
      </div>

      <button class="btn" id="runBtn" onclick="triggerKineticAssay()">Execute Kinetic Assay (5s)</button>
    </div>

    <div id="memoArea" style="display:none;"></div>
  </div>

  <!-- VIEW 2: SEARCHABLE LOGS -->
  <div id="logsView" style="display:none;">
    <div class="card">
      <label>Search Judicial Seizure Database</label>
      <input type="text" id="logSearch" placeholder="Filter by Test ID, Drug, Officer, City, or Hash..." onkeyup="fetchLogs()">
      <div class="table-box">
        <table>
          <thead>
            <tr>
              <th>Test ID</th>
              <th>Date & Location</th>
              <th>Analyte</th>
              <th>Result</th>
              <th>ΔE</th>
              <th>SHA-256 Seal</th>
            </tr>
          </thead>
          <tbody id="logsTbody">
            <tr><td colspan="6" style="text-align:center;">Loading audit database...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<canvas id="compressCanvas" width="480" height="480" style="display:none;"></canvas>

<script>
let video = document.getElementById('webcam');
let currentLat = "30.3165 N", currentLng = "78.0322 E";

// Sequentially chain permissions to prevent bubble block
navigator.mediaDevices.getUserMedia({
  video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 1280 } }
}).then(stream => {
  video.srcObject = stream;
  setTimeout(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(pos => {
        currentLat = pos.coords.latitude.toFixed(4) + " N";
        currentLng = pos.coords.longitude.toFixed(4) + " E";
      }, () => {});
    }
  }, 1000);
}).catch(err => {
  console.log("Webcam error:", err);
});

function switchView(view) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  if (view === 'assay') {
    document.querySelectorAll('.tab-btn')[0].classList.add('active');
    document.getElementById('assayView').style.display = 'block';
    document.getElementById('logsView').style.display = 'none';
  } else {
    document.querySelectorAll('.tab-btn')[1].classList.add('active');
    document.getElementById('assayView').style.display = 'none';
    document.getElementById('logsView').style.display = 'block';
    fetchLogs();
  }
}

function triggerKineticAssay() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.innerText = "Assaying Reaction Kinetics (Sampling 5 Frames)...";

  let frames = [];
  let canvas = document.getElementById('compressCanvas');
  let ctx = canvas.getContext('2d');
  let count = 0;

  let interval = setInterval(() => {
    ctx.drawImage(video, 0, 0, 480, 480);
    frames.push(canvas.toDataURL('image/jpeg', 0.65));
    count++;
    if (count >= 5) {
      clearInterval(interval);
      btn.innerText = "Computing Spectrophotometric Model...";
      sendAssay(frames);
    }
  }, 1000);
}

function sendAssay(frames) {
  const payload = {
    frames: frames,
    reagent: document.getElementById('reagentSelect').value,
    officer_id: document.getElementById('officerId').value,
    location_name: document.getElementById('locName').value,
    coords: currentLat + ", " + currentLng
  };

  fetch('/api/kinetic_assay', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById('runBtn').disabled = false;
    document.getElementById('runBtn').innerText = "Execute Kinetic Assay (5s)";
    renderMemo(data);
  })
  .catch(err => {
    alert("Assay error: " + err);
    document.getElementById('runBtn').disabled = false;
  });
}

function renderMemo(d) {
  const area = document.getElementById('memoArea');
  area.style.display = 'block';
  area.innerHTML = `
    <div class="memo-box">
      <div class="memo-header">
        <span>SECTION 52A SEIZURE CERTIFICATE</span>
        <span>${d.test_id}</span>
      </div>

      <div class="badge ${d.badge_class}">${d.verdict}</div>

      <div class="memo-meta">
        <div><b>Jurisdiction:</b> ${d.location_name} (${d.coords})</div>
        <div><b>Timestamp:</b> ${d.timestamp_ist}</div>
        <div><b>Officer In-Charge:</b> ${d.officer_id}</div>
        <div><b>Protocol:</b> ${d.reagent} (Tolerance: ΔE ≤ ${d.tolerance})</div>
        <div><b>Observed Metrics:</b> ΔE(Ref) = ${d.delta_e_ref} | Kinetic Velocity = ${d.rate}/s</div>
      </div>

      <label>FORENSIC CHAIN-OF-CUSTODY SNAPSHOTS</label>
      <div class="snapshots-grid">
        <div class="snap-card">
          <img src="data:image/jpeg;base64,${d.frame_t0}">
          <p>T=0s Baseline (Unreacted)</p>
        </div>
        <div class="snap-card">
          <img src="data:image/jpeg;base64,${d.frame_final}">
          <p>T=Final Chromophore Peak</p>
        </div>
      </div>

      <div class="memo-meta" style="word-break: break-all; margin-top:8px;">
        <b>SHA-256 DIGITAL EVIDENCE SEAL (BSA Sec 63):</b><br>
        <code>${d.sha256_seal}</code>
      </div>

      <div class="disclaimer-box">
        <b>STATUTORY FORENSIC NOTICE (MHA / NDPS Sec 52A):</b><br>
        This output represents a presumptive, objective field-test screening result establishing reasonable belief for seizure under Sections 42/43 NDPS Act. Confirmatory qualitative and quantitative analysis is performed by CFSL/SFSL via GC-MS / HPLC.
      </div>

      <button class="btn" style="margin-top:10px; background:#21262d; border:1px solid var(--border);" onclick="window.print()">
        Export / Print Signed Memo
      </button>
    </div>
  `;
}

function fetchLogs() {
  let q = document.getElementById('logSearch').value;
  fetch('/api/records?q=' + encodeURIComponent(q))
  .then(res => res.json())
  .then(rows => {
    let tbody = document.getElementById('logsTbody');
    if (rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;">No matching seizure records.</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map(r => `
      <tr>
        <td><b>${r.test_id}</b></td>
        <td>${r.timestamp}<br><span style="color:#8b949e">${r.location}</span></td>
        <td>${r.analyte}</td>
        <td>${r.verdict.includes('POSITIVE') ? '<span style="color:#3fb950">POSITIVE</span>' : r.verdict}</td>
        <td>${r.delta_e}</td>
        <td><code>${r.sha256}</code></td>
      </tr>
    `).join('');
  });
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
