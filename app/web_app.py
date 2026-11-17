import os, sys, uuid, json, base64, time
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from ultralytics import YOLO
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE, '..', 'best.pt'))

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found at", MODEL_PATH)
    sys.exit(1)

print("Loading model...")
model = YOLO(MODEL_PATH)
CLASS_NAMES = list(model.names.values())
print(f"Model loaded! Detects {len(CLASS_NAMES)} classes")

if torch.cuda.is_available():
    device = 'cuda'
else:
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    device = 'cpu'
print(f"Device: {device}")

DEFAULT_IMGSZ = 320
JPEG_QUALITY = 70
MAX_PREVIEW_W = 960
DEFAULT_SKIP = 5
DEFAULT_MAX_FRAMES = 1200

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def encode_jpeg(img, max_w=MAX_PREVIEW_W, quality=JPEG_QUALITY):
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ''
    return base64.b64encode(buf).decode('utf-8')


def parse_detections(results):
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({'class': model.names.get(cls_id, '?'), 'confidence': round(conf, 4)})
    return detections


def run_inference(img, imgsz):
    results = model(img, imgsz=imgsz, device=device, verbose=False)
    annotated = results[0].plot()
    return annotated, parse_detections(results)


def cap_pos_int(value, default, lo=1, hi=10**9):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


@app.route('/')
def index():
    return render_template('index.html', class_names=CLASS_NAMES, device=device)


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file'}), 400

    imgsz = cap_pos_int(request.form.get('imgsz', DEFAULT_IMGSZ), DEFAULT_IMGSZ, 128, 640)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.webm', '.mkv']:
        return jsonify({'error': 'Videos must be uploaded via the streaming mode'}), 400

    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img_array = np.frombuffer(open(filepath, 'rb').read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 400
        annotated, detections = run_inference(img, imgsz)
        img_b64 = encode_jpeg(annotated)
        return jsonify({'type': 'image', 'image': img_b64, 'detections': detections})
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass


@app.route('/detect_stream', methods=['POST'])
def detect_stream():
    def generate():
        if 'file' not in request.files:
            yield f"data: {json.dumps({'type': 'error', 'error': 'No file'})}\n\n"
            return
        file = request.files['file']
        if file.filename == '':
            yield f"data: {json.dumps({'type': 'error', 'error': 'No file'})}\n\n"
            return

        imgsz = cap_pos_int(request.form.get('imgsz', DEFAULT_IMGSZ), DEFAULT_IMGSZ, 128, 640)
        skip = cap_pos_int(request.form.get('skip', DEFAULT_SKIP), DEFAULT_SKIP)
        max_frames = cap_pos_int(request.form.get('max_frames', DEFAULT_MAX_FRAMES), DEFAULT_MAX_FRAMES, 1, 5000)

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.mp4', '.avi', '.mov', '.webm', '.mkv']:
            yield f"data: {json.dumps({'type': 'error', 'error': 'File is not a video'})}\n\n"
            return

        filename = f"{uuid.uuid4()}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        total_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        estimated_total = max(1, total_count // max(1, skip))
        if estimated_total > max_frames:
            estimated_total = max_frames

        processed = 0
        frame_idx = 0
        t_start = time.time()
        t_window = time.time()
        fps = 0.0
        truncated = False

        try:
            if not cap.isOpened():
                yield f"data: {json.dumps({'type': 'error', 'error': 'Could not open video'})}\n\n"
                return

            while True:
                if frame_idx % skip == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    annotated, detections = run_inference(frame, imgsz)
                    img_b64 = encode_jpeg(annotated)
                    processed += 1

                    now = time.time()
                    dt = now - t_window
                    if dt >= 1.0:
                        fps = 1.0 / dt
                        t_window = now

                    progress = min(100, int(processed / max(1, estimated_total) * 100))
                    payload = {
                        'type': 'frame',
                        'frame': processed,
                        'image': img_b64,
                        'detections': detections,
                        'current_fps': round(fps, 1),
                        'progress': progress,
                        'estimated_total': estimated_total,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                    if processed >= max_frames:
                        truncated = True
                        break
                else:
                    if not cap.grab():
                        break
                frame_idx += 1
        except GeneratorExit:
            pass
        finally:
            cap.release()
            try:
                os.remove(filepath)
            except OSError:
                pass

        elapsed = round(time.time() - t_start, 1)
        done = {
            'type': 'complete',
            'total_processed': processed,
            'elapsed_sec': elapsed,
            'estimated_total': estimated_total,
            'truncated': truncated,
        }
        yield f"data: {json.dumps(done)}\n\n"

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'X-Accel-Buffering': 'no',
                    })


if __name__ == '__main__':
    import socket
    port = int(os.environ.get('PORT', 5000))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('127.0.0.1', port)) == 0:
            port = 5001
    print(f"\n========================================")
    print(f" Object Detection Web App Running!")
    print(f" Access URL: http://127.0.0.1:{port}")
    print(f"========================================\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)