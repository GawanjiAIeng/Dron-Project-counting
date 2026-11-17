# Drone Object Detection and Counting System

Detect people and vehicles in images and recorded videos using a custom Ultralytics YOLO model. The project includes a Flask web interface and a Tkinter desktop application for viewing annotated detections, confidence scores, and object counts.

## Features

- Image detection with bounding boxes, class labels, and confidence scores.
- Video processing with streamed annotated frames in the browser.
- Browser controls for stopping analysis, reviewing buffered frames, playback, and scrubbing.
- Class filters for the browser detection list and displayed count. Filters do not remove boxes from the annotated image.
- Adjustable inference resolution and video frame sampling.
- Desktop video reports with cumulative detections, peak detections, and average detections per processed frame.
- Automatic CUDA selection when available, with CPU fallback.

Counts represent detections in processed frames. The same object can be counted again in later frames; unique object tracking and line-crossing counts are not implemented. Inputs are uploaded or selected files; a live drone or camera connection is not included.

## Detected classes

The bundled model metadata lists these 10 VisDrone classes:

| ID | Class |
| --- | --- |
| 0 | pedestrian |
| 1 | people |
| 2 | bicycle |
| 3 | car |
| 4 | van |
| 5 | truck |
| 6 | tricycle |
| 7 | awning-tricycle |
| 8 | bus |
| 9 | motor |

The applications read class names directly from `best.pt` when starting.

## Setup

The application was run on Windows with Python 3.10. A CPU is sufficient; GPU use requires a CUDA-capable PyTorch installation. The desktop application also requires Tkinter, included with typical Windows Python installations.

Clone the repository and create a virtual environment in PowerShell:

```powershell
git clone https://github.com/GawanjiAIeng/Dron-Project-counting.git
cd Dron-Project-counting
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r app/requirements.txt
```

Dependencies include Ultralytics, PyTorch (installed through Ultralytics), OpenCV, NumPy, Pillow, Flask, and Flask-CORS. Versions in `requirements.txt` are minimum versions rather than a pinned environment.

Keep `best.pt` in the repository root. Both entry points load this file on startup. An OpenVINO export is also included in `best_openvino_model/`, but the current applications use the PyTorch model.

## Run the web application

From the repository root:

```powershell
.\.venv\Scripts\python.exe app/web_app.py
```

Open **http://127.0.0.1:5000** in your browser. Wait for the model-loaded message before opening the page. If port 5000 is occupied, the application attempts port 5001; check the terminal for the actual URL.

To request a different port in PowerShell:

```powershell
$env:PORT = '8000'
.\.venv\Scripts\python.exe app/web_app.py
```

Press `Ctrl+C` in the terminal to stop the server. The included Flask development server listens on all network interfaces and is intended for local development.

### Use the web interface

1. Choose the resolution and video mode before selecting a file.
2. Click **Browse Files** or drop an image or video into the upload area.
3. Review the annotated preview, confidence scores, and active detection count.
4. Toggle class chips to filter the results list.
5. For videos, use **Stop** to cancel analysis and the playback controls to review buffered frames.

The file picker accepts JPG, JPEG, PNG, BMP, WEBP, MP4, AVI, and MOV. The streaming backend also recognizes WEBM and MKV; actual decoding depends on the installed OpenCV codecs. The upload limit is 500 MB per request.

| Setting | Options | Browser default |
| --- | --- | --- |
| Inference resolution | 320, 480, 640 | 480 |
| Video sampling | Fast Preview: every 10th frame; Balanced: every 5th; Detailed: every 2nd | Balanced |

Lower resolution and wider sampling reduce processing work. The backend defaults to 320 when no resolution is supplied and limits video analysis to 1,200 processed frames by default. API requests can set `max_frames` up to 5,000; the browser does not expose this setting. Video processing can therefore end before the source video finishes.

## Run the desktop application

From the repository root:

```powershell
.\.venv\Scripts\python.exe app/app.py
```

Click **Select Image/Video** to choose a local file. Resolution options are 320, 480, and 640, and video frame sampling ranges from every frame to every fifth frame. The desktop defaults are resolution 640 and sampling every third frame.

On Windows, `run.bat` also launches the desktop application. It uses the system `py` launcher or `python`; use the explicit virtual-environment command above if dependencies are installed only in `.venv`.

## Project structure

```text
Dron-Project-counting/
|-- app/
|   |-- app.py                 # Tkinter desktop application
|   |-- web_app.py             # Flask server and inference routes
|   |-- requirements.txt       # Python dependencies
|   |-- static/
|   |   |-- style.css          # Browser styling
|   |   `-- script.js          # Upload, streaming, filters, and playback
|   |-- templates/
|   |   `-- index.html         # Web interface
|   `-- uploads/               # Temporary uploads; ignored by Git
|-- best.pt                    # Model used by both applications
|-- best_openvino_model/       # Included OpenVINO export and metadata
|-- Vehicle Dataset Sample 3.mp4
|-- run.bat                    # Windows desktop launcher
|-- .gitignore
`-- README.md
```

Additional JPG files in the root can be used as sample inputs. Temporary uploads, logs, and Python caches are excluded from Git.

## Web API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/` | Render the web interface |
| POST | `/detect` | Detect objects in an uploaded image |
| POST | `/detect_stream` | Stream video detection events |

Both POST routes accept multipart form data with `file` and optional `imgsz` (128-640). The video route also accepts `skip` (positive integer) and `max_frames` (1-5,000).

The image response contains `type`, a base64 JPEG in `image`, and a `detections` array with `class` and `confidence` values. The video response uses `text/event-stream` with JSON events of type `frame`, `complete`, or `error`. Because uploads use POST, clients should read the response stream with `fetch` or an equivalent streaming HTTP client.

Example image request from PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:5000/detect -F "file=@zebra_crossing.jpg" -F "imgsz=480"
```

## Troubleshooting

- **Model not found:** verify that `best.pt` is in the root, one level above `app/`.
- **Missing Python module:** install `app/requirements.txt` using the same interpreter used to launch the application.
- **Slow video analysis:** select resolution 320 and Fast Preview, or test with a shorter video. Buffered browser frames can consume substantial memory on long runs.
- **Video cannot be opened:** try an MP4 file with a codec supported by your OpenCV installation.
- **Port already in use:** inspect the printed startup URL or set `PORT` to an available port. The fallback to 5001 is a single attempt.
- **GPU is not selected:** confirm that your PyTorch installation reports CUDA availability. Runtime device selection is printed by the web server; the browser engine label is static and may not reflect the actual device.

## License information

No project-level license file is currently included. The bundled OpenVINO metadata identifies its model export as AGPL-3.0. Refer to the relevant dependency and model license files when reusing or distributing them.
