import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ArUco Configuration: DICT_4X4_1000
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
    <title>NDPS Field Assay Calibration</title>
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

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            -webkit-tap-highlight-color: transparent;
        }

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
            gap: 16px;
        }

        header {
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }

        .sys-title {
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: var(--text-primary);
        }

        .sys-meta {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        .capture-card {
            background-color: var(--panel);
            border: 1px dashed var(--border);
            padding: 24px 16px;
            text-align: center;
            border-radius: 4px;
        }

        .file-label {
            display: block;
            background-color: var(--accent);
            color: #fff;
            padding: 12px 16px;
            font-size: 13px;
            font-family: var(--font-mono);
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            cursor: pointer;
            border-radius: 2px;
        }

        .file-input {
            display: none;
        }

        .hint {
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 10px;
            font-family: var(--font-mono);
        }

        #processing {
            display: none;
            background: var(--panel);
            border: 1px solid var(--border);
            padding: 14px;
            font-family: var(--font-mono);
            font-size: 12px;
            color: var(--text-secondary);
            text-align: center;
        }

        #results {
            display: none;
            background-color: var(--panel);
            border: 1px solid var(--border);
            border-radius: 4px;
            overflow: hidden;
        }

        .status-bar {
            padding: 10px 14px;
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border);
            text-transform: uppercase;
        }

        .status-success {
            background-color: rgba(0, 184, 107, 0.1);
            color: var(--success);
            border-color: rgba(0, 184, 107, 0.2);
        }

        .status-failed {
            background-color: rgba(230, 57, 70, 0.1);
            color: var(--danger);
            border-color: rgba(230, 57, 70, 0.2);
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-family: var(--font-mono);
            font-size: 11px;
        }

        .data-table tr {
            border-bottom: 1px solid var(--border);
        }

        .data-table td {
            padding: 10px 14px;
        }

        .data-table td:first-child {
            color: var(--text-secondary);
            width: 40%;
        }

        .data-table td:last-child {
            color: var(--text-primary);
            text-align: right;
            font-weight: 600;
        }

        .preview-container {
            padding: 14px;
            border-top: 1px solid var(--border);
        }

        .preview-title {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-secondary);
            margin-bottom: 8px;
            text-transform: uppercase;
        }

        .preview-img {
            width: 100%;
            display: block;
            border: 1px solid var(--border);
            border-radius: 2px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="sys-title">Optical Assay Calibration Unit</div>
            <div class="sys-meta">NDPS SEC 52A // REF: ARUCO 4X4 // D65 NORMALIZED</div>
        </header>

        <div class="capture-card">
            <label class="file-label" for="cameraInput">Acquire Target Scan</label>
            <input type="file" id="cameraInput" class="file-input" accept="image/*" capture="environment">
            <div class="hint">Center all 4 corner markers in camera frame</div>
        </div>

        <div id="processing">CALIBRATING PERSPECTIVE & ILLUMINATION...</div>

        <div id="results">
            <div id="statusBar" class="status-bar"></div>
            <table class="data-table">
                <tbody>
                    <tr>
                        <td>GEOMETRIC LOCK</td>
                        <td id="lockData">--</td>
                    </tr>
                    <tr>
                        <td>WHITE BALANCE GAIN</td>
                        <td id="gainData">--</td>
                    </tr>
                    <tr>
                        <td>NORMALIZED RGB</td>
                        <td id="rgbData">--</td>
                    </tr>
                    <tr>
                        <td>PRESUMPTIVE ASSAY</td>
                        <td id="assayData">--</td>
                    </tr>
                </tbody>
            </table>
            <div class="preview-container">
                <div class="preview-title">Rectified Output Matrix</div>
                <img id="calibratedPreview" class="preview-img" alt="Orthorectified View">
            </div>
        </div>
    </div>

    <script>
        const input = document.getElementById('cameraInput');
        const processing = document.getElementById('processing');
        const results = document.getElementById('results');
        const statusBar = document.getElementById('statusBar');

        input.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            processing.style.display = 'block';
            results.style.display = 'none';

            const formData = new FormData();
            formData.append('target_image', file);

            try {
                const response = await fetch('/api/calibrate', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                processing.style.display = 'none';
                results.style.display = 'block';

                if (data.success) {
                    statusBar.className = 'status-bar status-success';
                    statusBar.innerText = 'LOCK ACQUIRED: OPTICAL MATRIX STABLE';
                    document.getElementById('lockData').innerText = `4/4 (IDs ${data.ids.join(',')})`;
                    document.getElementById('gainData').innerText = `R:${data.gains.r} G:${data.gains.g} B:${data.gains.b}`;
                    document.getElementById('rgbData').innerText = `(${data.rgb.r}, ${data.rgb.g}, ${data.rgb.b})`;
                    document.getElementById('assayData').innerText = data.assay_result;
                    document.getElementById('calibratedPreview').src = data.rectified_image;
                    document.querySelector('.preview-container').style.display = 'block';
                } else {
                    statusBar.className = 'status-bar status-failed';
                    statusBar.innerText = 'LOCK FAILED: ' + data.error_code;
                    document.getElementById('lockData').innerText = data.detected_markers || 'None';
                    document.getElementById('gainData').innerText = '--';
                    document.getElementById('rgbData').innerText = '--';
                    document.getElementById('assayData').innerText = data.message;
                    document.querySelector('.preview-container').style.display = 'none';
                }
            } catch (err) {
                processing.style.display = 'none';
                results.style.display = 'block';
                statusBar.className = 'status-bar status-failed';
                statusBar.innerText = 'NETWORK / TRANSMISSION ERROR';
                document.getElementById('assayData').innerText = err.message;
                document.querySelector('.preview-container').style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def root():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    file = request.files.get('target_image')
    if not file:
        return jsonify({'success': False, 'error_code': 'NO_DATA', 'message': 'No image stream uploaded.'})

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'success': False, 'error_code': 'DECODE_ERROR', 'message': 'Corrupt image payload.'})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if USE_NEW_API:
        corners, ids, _ = DETECTOR.detectMarkers(frame)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=PARAMS)

    if ids is None or len(ids) < 4:
        found = ids.flatten().tolist() if ids is not None else []
        return jsonify({
            'success': False,
            'error_code': 'INCOMPLETE_CONSTELLATION',
            'detected_markers': f"IDs {found}",
            'message': 'All 4 corner markers (0, 1, 2, 3) must be visible.'
        })

    id_list = ids.flatten().tolist()
    if not all(idx in id_list for idx in [0, 1, 2, 3]):
        return jsonify({
            'success': False,
            'error_code': 'INVALID_ID_SET',
            'detected_markers': f"IDs {id_list}",
            'message': 'Missing one or more required markers (0, 1, 2, 3).'
        })

    # Order outer boundary points
    c0 = corners[id_list.index(0)][0][0]
    c1 = corners[id_list.index(1)][0][1]
    c2 = corners[id_list.index(2)][0][2]
    c3 = corners[id_list.index(3)][0][3]

    src_pts = np.float32([c0, c1, c2, c3])
    CANVAS_SIZE = 600
    dst_pts = np.float32([
        [0, 0],
        [CANVAS_SIZE, 0],
        [CANVAS_SIZE, CANVAS_SIZE],
        [0, CANVAS_SIZE]
    ])

    # Perspective rectification
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, matrix, (CANVAS_SIZE, CANVAS_SIZE))

    # Color normalization against 18% neutral gray patch
    # Target coordinates on 600x600 plane: y: 50..100, x: 225..375
    gray_roi = warped[50:100, 225:375]
    b_mean = float(np.mean(gray_roi[:, :, 0]))
    g_mean = float(np.mean(gray_roi[:, :, 1]))
    r_mean = float(np.mean(gray_roi[:, :, 2]))

    TARGET_GRAY = 119.0
    gain_b = TARGET_GRAY / max(b_mean, 1.0)
    gain_g = TARGET_GRAY / max(g_mean, 1.0)
    gain_r = TARGET_GRAY / max(r_mean, 1.0)

    calibrated = warped.astype(np.float32)
    calibrated[:, :, 0] = np.clip(calibrated[:, :, 0] * gain_b, 0, 255)
    calibrated[:, :, 1] = np.clip(calibrated[:, :, 1] * gain_g, 0, 255)
    calibrated[:, :, 2] = np.clip(calibrated[:, :, 2] * gain_r, 0, 255)
    calibrated = calibrated.astype(np.uint8)

    # Reaction well analysis: y: 220..380, x: 220..380
    well_roi = calibrated[220:380, 220:380]
    well_b = int(np.mean(well_roi[:, :, 0]))
    well_g = int(np.mean(well_roi[:, :, 1]))
    well_r = int(np.mean(well_roi[:, :, 2]))

    # Quantitative presumptive assessment
    if well_b > (well_r + 25) and well_b > well_g:
        assay_status = "COBALT BLUE CONFIRMED (SCOTT POSITIVE)"
    else:
        assay_status = "UNREACTIVE / BASELINE"

    # Draw diagnostic overlays on the output image
    cv2.rectangle(calibrated, (225, 50), (375, 100), (0, 255, 0), 2)
    cv2.circle(calibrated, (300, 300), 75, (255, 0, 0), 2)

    _, enc = cv2.imencode('.jpg', calibrated)
    b64_img = "data:image/jpeg;base64," + base64.b64encode(enc).decode('utf-8')

    return jsonify({
        'success': True,
        'ids': [0, 1, 2, 3],
        'gains': {
            'r': f"{gain_r:.2f}",
            'g': f"{gain_g:.2f}",
            'b': f"{gain_b:.2f}"
        },
        'rgb': {'r': well_r, 'g': well_g, 'b': well_b},
        'assay_result': assay_status,
        'rectified_image': b64_img
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
