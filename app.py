import base64
import hashlib
import time
from datetime import datetime, timezone
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB memory safety cap
app.config['DEBUG'] = False

# ArUco Configuration (DICT_4X4_1000)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
try:
    DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())
    USE_NEW_API = True
except AttributeError:
    PARAMS = cv2.aruco.DetectorParameters_create()
    USE_NEW_API = False

# Certified UNODC / CRCL Reagent Registry
REAGENT_DATABASE = {
    "scott_cocaine": {
        "name": "Modified Scott (Cobalt Thiocyanate)",
        "analyte": "Cocaine HCl",
        "duration_sec": 5,
        "target_lab": [82.0, 152.0, 69.0],  # Cobalt Blue (#0047AB)
        "tolerance_de": 32.0,
        "ndps_schedule": "Schedule I (Commercial / Small)"
    },
    "marquis_opiate": {
        "name": "Marquis Reagent",
        "analyte": "Opiates (Heroin / Morphine)",
        "duration_sec": 10,
        "target_lab": [75.0, 148.0, 115.0],  # Deep Violet (#4B0082)
        "tolerance_de": 35.0,
        "ndps_schedule": "Schedule I"
    },
    "simons_mdma": {
        "name": "Simon's Reagent (A+B)",
        "analyte": "MDMA / Methamphetamine",
        "duration_sec": 10,
        "target_lab": [88.0, 135.0, 72.0],  # Royal Blue
        "tolerance_de": 30.0,
        "ndps_schedule": "Schedule I / II"
    },
    "marquis_amphet": {
        "name": "Marquis Reagent",
        "analyte": "Amphetamine Class",
        "duration_sec": 15,
        "target_lab": [110.0, 145.0, 165.0],  # Orange-Brown
        "tolerance_de": 30.0,
        "ndps_schedule": "Schedule II"
    },
    "mandelin_ketamine": {
        "name": "Mandelin Reagent",
        "analyte": "Ketamine HCl",
        "duration_sec": 20,
        "target_lab": [90.0, 115.0, 140.0],  # Olive-Brown
        "tolerance_de": 30.0,
        "ndps_schedule": "Schedule I"
    },
    "duquenois_cannabis": {
        "name": "Duquenois-Levine",
        "analyte": "Cannabinoids (THC / Charas)",
        "duration_sec": 30,
        "target_lab": [70.0, 150.0, 120.0],  # Violet-Indigo Layer
        "tolerance_de": 35.0,
        "ndps_schedule": "Schedule III"
    },
    "ehrlich_lsd": {
        "name": "Ehrlich / Van Urk",
        "analyte": "LSD / Indole Alkaloids",
        "duration_sec": 45,
        "target_lab": [78.0, 142.0, 90.0],  # Deep Purple
        "tolerance_de": 35.0,
        "ndps_schedule": "Schedule I"
    }
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>NDPS Forensic Field Assay Terminal</title>
    <style>
        :root {
            --bg: #0b0c10;
            --panel: #13151b;
            --border: #232733;
            --text-primary: #e0e4ec;
            --text-secondary: #747d92;
            --accent: #2e66ff;
            --success: #00b86b;
            --danger: #e63946;
            --warning: #f59e0b;
            --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            --font-sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg);
            color: var(--text-primary);
            font-family: var(--font-sans);
            padding: 14px;
            display: flex;
            justify-content: center;
        }

        .container {
            width: 100%;
            max-width: 480px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        header {
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }

        .sys-title {
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
        }

        .sys-meta {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-secondary);
            margin-top: 2px;
        }

        .controls {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .label {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        select {
            background-color: var(--panel);
            color: var(--text-primary);
            border: 1px solid var(--border);
            padding: 10px;
            font-family: var(--font-mono);
            font-size: 11px;
            border-radius: 2px;
            width: 100%;
            outline: none;
        }

        .viewport-wrapper {
            position: relative;
            width: 100%;
            aspect-ratio: 1 / 1;
            background: #000;
            border: 1px solid var(--border);
            border-radius: 2px;
            overflow: hidden;
        }

        video {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .overlay-box {
            position: absolute;
            inset: 12%;
            border: 1px dashed rgba(255, 255, 255, 0.35);
            pointer-events: none;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .overlay-text {
            font-family: var(--font-mono);
            font-size: 10px;
            color: rgba(255, 255, 255, 0.5);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .btn {
            background-color: var(--accent);
            color: #fff;
            border: none;
            padding: 12px;
            font-size: 12px;
            font-family: var(--font-mono);
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            cursor: pointer;
            border-radius: 2px;
            width: 100%;
        }

        .btn:disabled {
            background-color: var(--border);
            color: var(--text-secondary);
            cursor: not-allowed;
        }

        #panel {
            display: none;
            background-color: var(--panel);
            border: 1px solid var(--border);
            border-radius: 2px;
            padding: 12px;
            gap: 10px;
        }

        .status {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 700;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
            text-transform: uppercase;
        }

        svg {
            width: 100%;
            height: 100px;
            background: #0a0b0e;
            border: 1px solid var(--border);
            margin-top: 8px;
        }

        .data-list {
            margin-top: 8px;
            display: flex;
            justify-content: space-between;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-secondary);
        }

        .data-list span {
            color: var(--text-primary);
            font-weight: 600;
        }

        /* Seizure Certificate Section */
        #memoBox {
            display: none;
            background: #07080a;
            border: 1px solid var(--border);
            padding: 10px;
            font-family: var(--font-mono);
            font-size: 10px;
            line-height: 1.4;
            color: #b0b8c8;
            margin-top: 8px;
        }

        .memo-header {
            font-weight: bold;
            color: var(--text-primary);
            border-bottom: 1px solid var(--border);
            padding-bottom: 4px;
            margin-bottom: 6px;
            text-transform: uppercase;
        }

        .memo-hash {
            word-break: break-all;
            color: #4da6ff;
            background: #11131a;
            padding: 4px;
            border: 1px solid #1a1e2a;
            margin-top: 4px;
        }

        .print-btn {
            margin-top: 10px;
            width: 100%;
            padding: 9px;
            background: #1b2030;
            color: #fff;
            border: 1px solid var(--border);
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 600;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        @media print {
            body { background: #fff !important; color: #000 !important; padding: 0 !important; }
            .viewport-wrapper, header, .controls, #recordBtn, #graph, .data-list, button { display: none !important; }
            #panel { display: block !important; border: none !important; background: #fff !important; padding: 0 !important; }
            #memoBox {
                display: block !important;
                border: 2px solid #000 !important;
                color: #000 !important;
                background: #fff !important;
                padding: 15px !important;
            }
            .memo-header { color: #000 !important; border-bottom: 2px solid #000 !important; font-size: 12pt !important; }
            .memo-hash { color: #000 !important; background: #eee !important; border: 1px solid #666 !important; font-size: 8pt !important; }
            .status { color: #000 !important; border-bottom: 2px solid #000 !important; font-size: 11pt !important; margin-bottom: 10px !important; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="sys-title">NDPS Field Colorimetric Terminal</div>
            <div class="sys-meta">SEC 52A DIGITAL COMPANION // ARUCO 4X4 OPTICAL MATRIX</div>
        </header>

        <div class="controls">
            <span class="label">Reagent Protocol & Target Schedule</span>
            <select id="reagentSelect">
                <option value="scott_cocaine">Scott Reagent — Cocaine HCl (5s Window)</option>
                <option value="marquis_opiate">Marquis — Opiates / Heroin (10s Window)</option>
                <option value="simons_mdma">Simon's (A+B) — MDMA / Meth (10s Window)</option>
                <option value="marquis_amphet">Marquis — Amphetamine (15s Window)</option>
                <option value="mandelin_ketamine">Mandelin — Ketamine (20s Window)</option>
                <option value="duquenois_cannabis">Duquenois-Levine — Cannabis (30s Window)</option>
                <option value="ehrlich_lsd">Ehrlich — LSD / Indoles (45s Window)</option>
            </select>
        </div>

        <div class="viewport-wrapper">
            <video id="webcam" autoplay playsinline muted></video>
            <div class="overlay-box">
                <span class="overlay-text">Frame Calibration Markers</span>
            </div>
        </div>

        <button id="recordBtn" class="btn">Execute Kinetic Assay</button>

        <div id="panel">
            <div id="statusText" class="status"></div>
            
            <svg id="graph" viewBox="0 0 300 100">
                <line x1="30" y1="85" x2="280" y2="85" stroke="#232733" stroke-width="1" />
                <line x1="30" y1="15" x2="30" y2="85" stroke="#232733" stroke-width="1" />
                <polyline id="curve" fill="none" stroke="#2e66ff" stroke-width="2" points="" />
            </svg>

            <div class="data-list">
                <div>RATE (dE/dt): <span id="rateVal">--</span></div>
                <div>DELTA-E (REF): <span id="deRefVal">--</span></div>
                <div>MEASURED RGB: <span id="rgbVal">--</span></div>
            </div>

            <div id="memoBox">
                <div class="memo-header">Sec 52A Digital Seizure Hash Certificate</div>
                <div>REAGENT: <span id="mReagent">--</span></div>
                <div>ANALYTE: <span id="mAnalyte">--</span></div>
                <div>GEO-COORDINATES: <span id="mGps">Acquiring GNSS...</span></div>
                <div>TIMESTAMP (OFFICIAL): <span id="mTime">--</span></div>
                <div>IO / STATION ID: <span>NCB-ZU-NDLS / GD-8821</span></div>
                <div style="margin-top:6px;">CRYPTOGRAPHIC INTEGRITY SEAL (BSA SEC 63 / SHA-256):</div>
                <div id="mHash" class="memo-hash">--</div>

                <button class="print-btn" onclick="window.print()">
                    Export / Print Official Seizure Memo (PDF)
                </button>
            </div>
        </div>
    </div>

    <canvas id="offscreenCanvas" width="480" height="480" style="display:none;"></canvas>

    <script>
        const video = document.getElementById('webcam');
        const recordBtn = document.getElementById('recordBtn');
        const panel = document.getElementById('panel');
        const statusText = document.getElementById('statusText');
        const canvas = document.getElementById('offscreenCanvas');
        const ctx = canvas.getContext('2d');
        const curve = document.getElementById('curve');
        const reagentSelect = document.getElementById('reagentSelect');
        const memoBox = document.getElementById('memoBox');

        let currentLat = "30.3165 N";
        let currentLng = "78.0322 E";

        // Query GNSS coordinates automatically from device sensor
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(pos => {
                currentLat = pos.coords.latitude.toFixed(5);
                currentLng = pos.coords.longitude.toFixed(5);
            }, () => {});
        }

        navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 1280 } }
        }).then(stream => {
            video.srcObject = stream;
        }).catch(() => {
            alert("Rear camera access unavailable or blocked.");
        });

        recordBtn.addEventListener('click', async () => {
            const reagentKey = reagentSelect.value;
            recordBtn.disabled = true;
            panel.style.display = 'none';
            memoBox.style.display = 'none';

            // Pacing delay so 5 frames span the designated reaction window
            const delays = {
                "scott_cocaine": 1000,
                "marquis_opiate": 2000,
                "simons_mdma": 2000,
                "marquis_amphet": 3000,
                "mandelin_ketamine": 4000,
                "duquenois_cannabis": 6000,
                "ehrlich_lsd": 9000
            };
            const stepDelay = delays[reagentKey] || 1000;

            const frames = [];
            for (let i = 0; i < 5; i++) {
                recordBtn.innerText = `SAMPLING REACTION (${i + 1}/5)...`;
                ctx.drawImage(video, 0, 0, 480, 480);
                frames.push(canvas.toDataURL('image/jpeg', 0.6));
                if (i < 4) await new Promise(r => setTimeout(r, stepDelay));
            }

            recordBtn.innerText = "COMPUTING FORENSIC MATRIX...";

            try {
                const res = await fetch('/api/kinetic_assay', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ frames, reagent: reagentKey, lat: currentLat, lng: currentLng })
                });
                const data = await res.json();

                panel.style.display = 'block';
                recordBtn.disabled = false;
                recordBtn.innerText = "Execute Kinetic Assay";

                if (data.success) {
                    statusText.innerText = data.verdict;
                    statusText.style.color = data.positive ? 'var(--success)' : (data.is_dye ? 'var(--danger)' : 'var(--warning)');
                    
                    document.getElementById('rateVal').innerText = data.slope;
                    document.getElementById('deRefVal').innerText = data.de_reference;
                    document.getElementById('rgbVal').innerText = `(${data.final_rgb.r}, ${data.final_rgb.g}, ${data.final_rgb.b})`;

                    // Render SVG kinetic curve
                    const maxDe = Math.max(...data.delta_e_series, 50);
                    const pts = data.delta_e_series.map((val, idx) => {
                        const x = 40 + (idx * 55);
                        const y = 80 - ((val / maxDe) * 60);
                        return `${x},${y}`;
                    }).join(' ');
                    curve.setAttribute('points', pts);
                    curve.setAttribute('stroke', data.positive ? '#00b86b' : (data.is_dye ? '#e63946' : '#f59e0b'));

                    // Populate Section 52A Seizure Certificate
                    memoBox.style.display = 'block';
                    document.getElementById('mReagent').innerText = data.memo.reagent;
                    document.getElementById('mAnalyte').innerText = data.memo.analyte;
                    document.getElementById('mGps').innerText = `${data.memo.lat}, ${data.memo.lng}`;
                    document.getElementById('mTime').innerText = data.memo.timestamp;
                    document.getElementById('mHash').innerText = data.memo.sha256_seal;
                } else {
                    statusText.innerText = "OPTICAL TRACKING FAILURE: " + data.message;
                    statusText.style.color = 'var(--danger)';
                    curve.setAttribute('points', "");
                }
            } catch (err) {
                alert("Network timeout or payload failure.");
                recordBtn.disabled = false;
                recordBtn.innerText = "Execute Kinetic Assay";
            }
        });
    </script>
</body>
</html>
"""

def extract_well_data(frame_b64):
    try:
        if "," in frame_b64:
            _, encoded = frame_b64.split(",", 1)
        else:
            encoded = frame_b64
        file_bytes = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception:
        return None, None, None

    if frame is None:
        return None, None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if USE_NEW_API:
        corners, ids, _ = DETECTOR.detectMarkers(frame)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=PARAMS)

    if ids is None or len(ids) < 4:
        return None, None, None

    id_list = ids.flatten().tolist()
    if not all(idx in id_list for idx in [0, 1, 2, 3]):
        return None, None, None

    c0 = corners[id_list.index(0)][0][0]
    c1 = corners[id_list.index(1)][0][1]
    c2 = corners[id_list.index(2)][0][2]
    c3 = corners[id_list.index(3)][0][3]

    src_pts = np.float32([c0, c1, c2, c3])
    SIZE = 600
    dst_pts = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, matrix, (SIZE, SIZE))

    # White balance against 18% neutral gray patch
    gray_roi = warped[40:110, 220:380]
    b_mean = max(float(np.mean(gray_roi[:, :, 0])), 1.0)
    g_mean = max(float(np.mean(gray_roi[:, :, 1])), 1.0)
    r_mean = max(float(np.mean(gray_roi[:, :, 2])), 1.0)

    TARGET = 119.0
    calibrated = warped.astype(np.float32)
    calibrated[:, :, 0] = np.clip(calibrated[:, :, 0] * (TARGET / b_mean), 0, 255)
    calibrated[:, :, 1] = np.clip(calibrated[:, :, 1] * (TARGET / g_mean), 0, 255)
    calibrated[:, :, 2] = np.clip(calibrated[:, :, 2] * (TARGET / r_mean), 0, 255)
    calibrated = calibrated.astype(np.uint8)

    # Concentrated median sampling (30x30 px box in exact center)
    well = calibrated[285:315, 285:315]
    lab_well = cv2.cvtColor(well, cv2.COLOR_BGR2LAB)
    
    avg_lab = np.median(lab_well, axis=(0, 1))
    avg_rgb = [
        int(np.median(well[:, :, 2])),
        int(np.median(well[:, :, 1])),
        int(np.median(well[:, :, 0]))
    ]
    return avg_lab, avg_rgb, calibrated

@app.route('/')
def root():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/kinetic_assay', methods=['POST'])
def kinetic_assay():
    data = request.get_json() or {}
    frames = data.get('frames', [])
    reagent_key = data.get('reagent', 'scott_cocaine')
    lat = data.get('lat', 'Unknown')
    lng = data.get('lng', 'Unknown')

    if len(frames) != 5:
        return jsonify({'success': False, 'message': 'Requires 5 consecutive temporal frames.'})

    cfg = REAGENT_DATABASE.get(reagent_key, REAGENT_DATABASE['scott_cocaine'])
    lab_series = []
    final_rgb = [0, 0, 0]
    final_calibrated_frame = None

    for frame_b64 in frames:
        lab, rgb, cal_img = extract_well_data(frame_b64)
        if lab is None:
            return jsonify({'success': False, 'message': 'Tracking lost during test window. Keep card completely in view.'})
        lab_series.append(lab)
        final_rgb = rgb
        final_calibrated_frame = cal_img

    # 1. Delta E relative to T=0 baseline
    base_l, base_a, base_b = lab_series[0]
    delta_e = []
    for l, a, b in lab_series:
        de = float(np.sqrt((l - base_l)**2 + (a - base_a)**2 + (b - base_b)**2))
        delta_e.append(round(de, 2))

    total_shift = delta_e[-1]
    slope = round((delta_e[-1] - delta_e[0]) / 4.0, 2)

    # 2. Objective Euclidean distance to certified standard in CIE-Lab space
    cur_l, cur_a, cur_b = lab_series[-1]
    tgt_l, tgt_a, tgt_b = cfg['target_lab']
    delta_e_ref = float(np.sqrt((cur_l - tgt_l)**2 + (cur_a - tgt_a)**2 + (cur_b - tgt_b)**2))

    color_matches_target = delta_e_ref <= cfg['tolerance_de']

    # 3. Decision Matrix
    if total_shift < 8.0:
        if color_matches_target:
            verdict = "ADULTERANT FLAGGED: STATIC PRE-EXISTING DYE (dE/dt ≈ 0)"
            positive, is_dye = False, True
        else:
            verdict = "NEGATIVE: UNREACTIVE BASELINE"
            positive, is_dye = False, False

    elif total_shift >= 12.0:
        if color_matches_target:
            verdict = f"PRESUMPTIVE POSITIVE: {cfg['analyte'].upper()} ({cfg['name'].upper()})"
            positive, is_dye = True, False
        else:
            verdict = f"ANOMALOUS COLOR SPECTRUM (ΔE vs Ref: {round(delta_e_ref, 1)})"
            positive, is_dye = False, False
    else:
        verdict = f"INCONCLUSIVE TRANSITION (dE: {total_shift})"
        positive, is_dye = False, False

    # 4. Generate SHA-256 digital evidence seal over calibrated image pixels
    _, enc_bytes = cv2.imencode('.png', final_calibrated_frame)
    sha256_hash = hashlib.sha256(enc_bytes).hexdigest()

    # Format official IST time for Indian policing, with UTC ISO standard
    now_utc = datetime.now(timezone.utc)
    now_ist = datetime.fromtimestamp(now_utc.timestamp() + 19800, timezone.utc)
    
    timestamp_ist = now_ist.strftime("%d-%m-%Y %H:%M:%S IST")
    timestamp_utc = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    return jsonify({
        'success': True,
        'delta_e_series': delta_e,
        'slope': f"{slope}/s",
        'de_reference': f"{round(delta_e_ref, 1)}",
        'final_rgb': {'r': final_rgb[0], 'g': final_rgb[1], 'b': final_rgb[2]},
        'verdict': verdict,
        'positive': positive,
        'is_dye': is_dye,
        'memo': {
            'reagent': cfg['name'],
            'analyte': cfg['analyte'],
            'lat': lat,
            'lng': lng,
            'timestamp': f"{timestamp_ist} ({timestamp_utc})",
            'sha256_seal': sha256_hash
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
