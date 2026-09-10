import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
try:
    DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())
    USE_NEW_API = True
except AttributeError:
    PARAMS = cv2.aruco.DetectorParameters_create()
    USE_NEW_API = False

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>NDPS Field Assay System</title>
    <style>
        :root {
            --bg: #090a0f;
            --panel: #11131a;
            --border: #222736;
            --text-primary: #dce1eb;
            --text-secondary: #70788d;
            --accent: #2e66ff;
            --success: #00b86b;
            --danger: #e63946;
            --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            --font-sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg);
            color: var(--text-primary);
            font-family: var(--font-sans);
            padding: 16px;
            display: flex;
            justify-content: center;
        }

        .container {
            width: 100%;
            max-width: 460px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        header {
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }

        .sys-title {
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .sys-meta {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-secondary);
            margin-top: 2px;
        }

        .viewport-wrapper {
            position: relative;
            width: 100%;
            aspect-ratio: 1 / 1;
            background: #000;
            border: 1px solid var(--border);
            border-radius: 4px;
            overflow: hidden;
        }

        video {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .overlay-box {
            position: absolute;
            inset: 15%;
            border: 1px dashed rgba(255, 255, 255, 0.3);
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
            border-radius: 4px;
            padding: 12px;
        }

        .status {
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 700;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
            text-transform: uppercase;
        }

        .chart-box {
            margin-top: 10px;
        }

        .chart-title {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-secondary);
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        svg {
            width: 100%;
            height: 120px;
            background: #0d0f17;
            border: 1px solid var(--border);
        }

        .data-list {
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-secondary);
        }

        .data-list span {
            color: var(--text-primary);
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="sys-title">NDPS Reagent Kinetic Analyzer</div>
            <div class="sys-meta">SEC 52A // 5-POINT TEMPORAL ASSAY // ARUCO 4X4</div>
        </header>

        <div class="viewport-wrapper">
            <video id="webcam" autoplay playsinline muted></video>
            <div class="overlay-box">
                <span class="overlay-text">Align Card Corners Here</span>
            </div>
        </div>

        <button id="recordBtn" class="btn">Record 5s Kinetic Assay</button>

        <div id="panel">
            <div id="statusText" class="status"></div>
            
            <div class="chart-box">
                <div class="chart-title">Reaction Kinetics (Delta-E vs Time)</div>
                <svg id="graph" viewBox="0 0 300 100">
                    <line x1="30" y1="85" x2="280" y2="85" stroke="#222736" stroke-width="1" />
                    <line x1="30" y1="15" x2="30" y2="85" stroke="#222736" stroke-width="1" />
                    <polyline id="curve" fill="none" stroke="#2e66ff" stroke-width="2" points="" />
                </svg>
            </div>

            <div class="data-list">
                <div>RATE (dE/dt): <span id="rateVal">--</span></div>
                <div>FINAL RGB: <span id="rgbVal">--</span></div>
            </div>
        </div>
    </div>

    <canvas id="offscreenCanvas" width="640" height="640" style="display:none;"></canvas>

    <script>
        const video = document.getElementById('webcam');
        const recordBtn = document.getElementById('recordBtn');
        const panel = document.getElementById('panel');
        const statusText = document.getElementById('statusText');
        const canvas = document.getElementById('offscreenCanvas');
        const ctx = canvas.getContext('2d');
        const curve = document.getElementById('curve');

        navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 1280 } }
        }).then(stream => {
            video.srcObject = stream;
        }).catch(err => {
            alert("Camera access denied or unavailable.");
        });

        recordBtn.addEventListener('click', async () => {
            recordBtn.disabled = true;
            panel.style.display = 'none';

            const frames = [];
            for (let i = 0; i < 5; i++) {
                recordBtn.innerText = `SAMPLING REACTION (${i + 1}/5)...`;
                ctx.drawImage(video, 0, 0, 640, 640);
                frames.push(canvas.toDataURL('image/jpeg', 0.7));
                if (i < 4) await new Promise(r => setTimeout(r, 1000));
            }

            recordBtn.innerText = "PROCESSING ASSAY...";

            try {
                const res = await fetch('/api/kinetic_assay', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ frames })
                });
                const data = await res.json();

                panel.style.display = 'block';
                recordBtn.disabled = false;
                recordBtn.innerText = "Record 5s Kinetic Assay";

                if (data.success) {
                    statusText.innerText = data.verdict;
                    statusText.style.color = data.positive ? 'var(--success)' : (data.is_dye ? 'var(--danger)' : 'var(--text-secondary)');
                    document.getElementById('rateVal').innerText = data.slope;
                    document.getElementById('rgbVal').innerText = `(${data.final_rgb.r}, ${data.final_rgb.g}, ${data.final_rgb.b})`;

                    // Render SVG kinetic points: scale x [0..4] to [40..270], y [0..max_de] to [80..20]
                    const maxDe = Math.max(...data.delta_e_series, 50);
                    const pts = data.delta_e_series.map((val, idx) => {
                        const x = 40 + (idx * 55);
                        const y = 80 - ((val / maxDe) * 60);
                        return `${x},${y}`;
                    }).join(' ');
                    curve.setAttribute('points', pts);
                    curve.setAttribute('stroke', data.positive ? '#00b86b' : (data.is_dye ? '#e63946' : '#2e66ff'));
                } else {
                    statusText.innerText = "LOCK FAILED: " + data.message;
                    statusText.style.color = 'var(--danger)';
                    curve.setAttribute('points', "");
                }
            } catch (err) {
                alert("Network error processing assay.");
                recordBtn.disabled = false;
                recordBtn.innerText = "Record 5s Kinetic Assay";
            }
        });
    </script>
</body>
</html>
"""

def extract_well_lab(frame_b64):
    header, encoded = frame_b64.split(",", 1)
    file_bytes = np.frombuffer(base64.b64decode(encoded), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if USE_NEW_API:
        corners, ids, _ = DETECTOR.detectMarkers(frame)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=PARAMS)

    if ids is None or len(ids) < 4:
        return None, None

    id_list = ids.flatten().tolist()
    if not all(idx in id_list for idx in [0, 1, 2, 3]):
        return None, None

    c0 = corners[id_list.index(0)][0][0]
    c1 = corners[id_list.index(1)][0][1]
    c2 = corners[id_list.index(2)][0][2]
    c3 = corners[id_list.index(3)][0][3]

    src_pts = np.float32([c0, c1, c2, c3])
    SIZE = 600
    dst_pts = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, matrix, (SIZE, SIZE))

    # Optical gray patch calibration
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

    # Reaction well sampling: center is at x:300, y:300, sample 100x100 box
    well = calibrated[250:350, 250:350]
    lab_well = cv2.cvtColor(well, cv2.COLOR_BGR2LAB)
    
    avg_lab = np.mean(lab_well, axis=(0, 1))
    avg_rgb = [int(np.mean(well[:, :, 2])), int(np.mean(well[:, :, 1])), int(np.mean(well[:, :, 0]))]
    return avg_lab, avg_rgb

@app.route('/')
def root():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/kinetic_assay', methods=['POST'])
def kinetic_assay():
    data = request.get_json() or {}
    frames = data.get('frames', [])

    if len(frames) != 5:
        return jsonify({'success': False, 'message': 'Requires exactly 5 temporal frames.'})

    lab_series = []
    final_rgb = [0, 0, 0]

    for frame_b64 in frames:
        lab, rgb = extract_well_lab(frame_b64)
        if lab is None:
            return jsonify({'success': False, 'message': 'Card tracking lost during 5s burst. Keep frame stable.'})
        lab_series.append(lab)
        final_rgb = rgb

    # Calculate Delta E from baseline (T=0)
    base_l, base_a, base_b = lab_series[0]
    delta_e = []
    for l, a, b in lab_series:
        de = float(np.sqrt((l - base_l)**2 + (a - base_a)**2 + (b - base_b)**2))
        delta_e.append(round(de, 2))

    total_shift = delta_e[-1]
    slope = round((delta_e[-1] - delta_e[0]) / 4.0, 2)

    # Chemical Evaluation
    r, g, b = final_rgb
    is_blue = (b > r + 20) and (b > g)

    if total_shift < 8.0 and not is_blue:
        verdict = "NEGATIVE: UNREACTIVE BASELINE"
        positive, is_dye = False, False
    elif total_shift < 8.0 and is_blue:
        # Liquid was blue from T=0 with near-zero transition
        verdict = "ADULTERANT FLAGGED: STATIC DYE (dE/dt ≈ 0)"
        positive, is_dye = False, True
    elif total_shift >= 12.0 and is_blue:
        verdict = "POSITIVE: COCAINE HCl (KINETIC TRANSITION CONFIRMED)"
        positive, is_dye = True, False
    else:
        verdict = f"ANOMALOUS REACTION (dE: {total_shift})"
        positive, is_dye = False, False

    return jsonify({
        'success': True,
        'delta_e_series': delta_e,
        'slope': f"{slope}/s",
        'final_rgb': {'r': r, 'g': g, 'b': b},
        'verdict': verdict,
        'positive': positive,
        'is_dye': is_dye
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
