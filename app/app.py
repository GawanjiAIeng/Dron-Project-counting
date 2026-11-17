import os, sys, tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
import torch

# Optimize PyTorch CPU execution
torch.set_num_threads(8)
torch.set_grad_enabled(False)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE, '..', 'best.pt'))

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found at", MODEL_PATH)
    sys.exit(1)

print("Loading model...")
model = YOLO(MODEL_PATH)
CLASS_NAMES = list(model.names.values())
print(f"Model loaded! Detects {len(CLASS_NAMES)} classes")


class App:
    def __init__(self, root):
        self.root = root
        root.title("Object Detection")
        root.geometry("900x700")
        root.configure(bg="#1a1a2e")

        title = tk.Label(root, text="Object Detection", font=("Arial", 22, "bold"),
                         fg="#00d4ff", bg="#1a1a2e")
        title.pack(pady=(16, 4))

        tags = "  ".join(CLASS_NAMES)
        tk.Label(root, text=tags, font=("Arial", 10), fg="#b8b8ff", bg="#1a1a2e",
                 wraplength=850).pack(pady=(0, 12))

        btn_frame = tk.Frame(root, bg="#1a1a2e")
        btn_frame.pack(pady=6)
        tk.Button(btn_frame, text="Select Image/Video", command=self.select_image,
                  font=("Arial", 12), bg="#7b2ff7", fg="white", padx=20, pady=6,
                  cursor="hand2").pack(side=tk.LEFT, padx=6)
        self.status = tk.Label(btn_frame, text="No file selected", font=("Arial", 10),
                               fg="#888", bg="#1a1a2e")
        self.status.pack(side=tk.LEFT, padx=12)

        # Optimization settings frame
        settings_frame = tk.LabelFrame(root, text="Optimization Settings", font=("Arial", 10, "bold"),
                                       fg="#00d4ff", bg="#1a1a2e", bd=1, relief=tk.GROOVE, padx=10, pady=4)
        settings_frame.pack(fill=tk.X, padx=16, pady=4)

        # 1. Inference resolution (imgsz)
        tk.Label(settings_frame, text="Resolution:", font=("Arial", 9), fg="#b8b8ff", bg="#1a1a2e").pack(side=tk.LEFT, padx=(5, 2))
        self.imgsz_var = tk.IntVar(value=640)
        imgsz_menu = tk.OptionMenu(settings_frame, self.imgsz_var, 320, 480, 640)
        imgsz_menu.config(font=("Arial", 9), bg="#242440", fg="white", bd=0, highlightthickness=0, activebackground="#32325c", activeforeground="white")
        imgsz_menu["menu"].config(bg="#242440", fg="white")
        imgsz_menu.pack(side=tk.LEFT, padx=(0, 15))

        # 2. Frame skip
        tk.Label(settings_frame, text="Frame Skip:", font=("Arial", 9), fg="#b8b8ff", bg="#1a1a2e").pack(side=tk.LEFT, padx=(5, 2))
        self.skip_var = tk.IntVar(value=3)
        skip_menu = tk.OptionMenu(settings_frame, self.skip_var, 1, 2, 3, 4, 5)
        skip_menu.config(font=("Arial", 9), bg="#242440", fg="white", bd=0, highlightthickness=0, activebackground="#32325c", activeforeground="white")
        skip_menu["menu"].config(bg="#242440", fg="white")
        skip_menu.pack(side=tk.LEFT, padx=(0, 15))

        # 3. Running Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device_label_text = f"Device: GPU (CUDA)" if self.device == 'cuda' else f"Device: CPU"
        device_color = "#00ffcc" if self.device == 'cuda' else "#ffcc00"
        tk.Label(settings_frame, text=device_label_text, font=("Arial", 9, "bold"), fg=device_color, bg="#1a1a2e").pack(side=tk.RIGHT, padx=5)

        img_frame = tk.Frame(root, bg="#0f0f1a", bd=1, relief=tk.SUNKEN)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=6)
        img_frame.pack_propagate(False)
        self.image_label = tk.Label(img_frame, bg="#0f0f1a")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.img_frame = img_frame

        self.result_text = tk.Text(root, height=5, font=("Consolas", 10),
                                    bg="#242440", fg="#e0e0e0", bd=0, padx=10, pady=6)
        self.result_text.pack(fill=tk.X, padx=16, pady=(0, 12))
        self.result_text.insert(tk.END, "Results will appear here")
        self.result_text.config(state=tk.DISABLED)

        self.photo = None
        self.current_img = None

        # Thread-safe settings cache
        self._skip = 3
        self._imgsz = 640
        self._container_w = 820
        self._container_h = 520

        # Video state variables
        self.queue = queue.Queue()
        self.video_running = False
        self.video_thread = None
        self.video_cap = None

        # Register window close protocol to stop thread cleanly
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _resize_to_fit(self, rgb):
        h, w = rgb.shape[:2]
        fw = self.img_frame.winfo_width()
        fh = self.img_frame.winfo_height()
        if fw > 10 and fh > 10:
            scale = min(fw / w, fh / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            if (new_w, new_h) != (w, h):
                return cv2.resize(rgb, (new_w, new_h))
        return rgb

    def select_image(self):
        # Stop any existing video processing
        self.stop_video()

        path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=[
                ("All Supported Files", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov"),
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Videos", "*.mp4 *.avi *.mov"),
                ("All Files", "*.*")
            ]
        )
        if not path:
            return

        # Check if the selected file is a video
        ext = os.path.splitext(path)[1].lower()
        is_video = ext in ['.mp4', '.avi', '.mov']

        if is_video:
            self.status.config(text="Starting video...")
            self.root.update()

            # Cache current settings for thread safety
            self._skip = self.skip_var.get()
            self._imgsz = self.imgsz_var.get()

            # Clear queue
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            
            self.video_running = True
            self.video_thread = threading.Thread(target=self.video_worker, args=(path,), daemon=True)
            self.video_thread.start()
            self.root.after(20, self.poll_queue)
        else:
            self.status.config(text="Processing...")
            self.root.update()

            img = cv2.imread(path)
            if img is None:
                messagebox.showerror("Error", "Failed to read image")
                self.status.config(text="Error loading image")
                return

            # Run YOLO inference on the original image for maximum accuracy
            imgsz = self.imgsz_var.get()
            with torch.inference_mode():
                results = model(img, imgsz=imgsz, device=self.device, verbose=False)
            annotated = results[0].plot()

            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    detections.append({'cls': model.names.get(cls_id, '?'), 'conf': conf})

            self.root.update_idletasks()
            rgb = self._resize_to_fit(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            self.current_img = Image.fromarray(rgb)
            self.photo = ImageTk.PhotoImage(self.current_img)
            self.image_label.config(image=self.photo)

            text = f"Detected {len(detections)} object(s):\n"
            for i, d in enumerate(detections, 1):
                text += f"  {i}. {d['cls']} — {d['conf']*100:.1f}%\n"
            if not detections:
                text += "  (no objects detected)\n"

            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, text)
            self.result_text.config(state=tk.DISABLED)

            self.status.config(text=f"Done — {len(detections)} object(s)")

    def stop_video(self):
        self.video_running = False
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.video_thread = None

    def video_worker(self, path):
        try:
            cap = cv2.VideoCapture(path)
            self.video_cap = cap
            frame_idx = 0

            stats = {
                'total_frames': 0,
                'peak_counts': {},
                'sum_counts': {}
            }

            while self.video_running:
                skip_rate = self._skip
                imgsz = self._imgsz

                if frame_idx % skip_rate == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    with torch.inference_mode():
                        results = model(frame, imgsz=imgsz, device=self.device, verbose=False)
                    annotated = results[0].plot()

                    detections = []
                    frame_counts = {}
                    for r in results:
                        if r.boxes is None:
                            continue
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            cls_name = model.names.get(cls_id, '?')
                            conf = float(box.conf[0])
                            detections.append({'cls': cls_name, 'conf': conf})
                            frame_counts[cls_name] = frame_counts.get(cls_name, 0) + 1

                    stats['total_frames'] += 1
                    for cls_name, count in frame_counts.items():
                        stats['sum_counts'][cls_name] = stats['sum_counts'].get(cls_name, 0) + count
                        stats['peak_counts'][cls_name] = max(stats['peak_counts'].get(cls_name, 0), count)

                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    cw, ch = self._container_w, self._container_h
                    h, w = rgb.shape[:2]
                    scale = min(cw / w, ch / h, 1.0)
                    if scale < 1.0:
                        new_w, new_h = int(w * scale), int(h * scale)
                        rgb = cv2.resize(rgb, (new_w, new_h))
                    pil_img = Image.fromarray(rgb)

                    if self.video_running:
                        self.queue.put((pil_img, detections, dict(stats)))
                else:
                    ret = cap.grab()
                    if not ret:
                        break

                frame_idx += 1

            cap.release()
            if self.video_cap == cap:
                self.video_cap = None
            self.queue.put((None, stats))
        except Exception as e:
            self.queue.put((None, {'error': str(e)}))

    def poll_queue(self):
        try:
            item = self.queue.get_nowait()
            if item[0] is None:
                stats = item[1]
                if 'error' in stats:
                    self.status.config(text="Video error")
                    self.result_text.config(state=tk.NORMAL)
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, f"ERROR during video processing:\n{stats['error']}")
                    self.result_text.config(state=tk.DISABLED)
                    self.stop_video()
                    return

                self.status.config(text="Video playback completed")
                report = "=== VIDEO PROCESSING REPORT ===\n"
                report += f"Total processed frames: {stats['total_frames']}\n\n"
                report += "Peak Detections (Max in a single frame):\n"
                if stats['peak_counts']:
                    for cls, peak in sorted(stats['peak_counts'].items()):
                        report += f"  - {cls}: {peak}\n"
                else:
                    report += "  - No objects detected\n"

                report += "\nAverage Detections per Frame:\n"
                if stats['sum_counts']:
                    for cls, total in sorted(stats['sum_counts'].items()):
                        avg = total / stats['total_frames']
                        report += f"  - {cls}: {avg:.1f}\n"
                else:
                    report += "  - No objects detected\n"

                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, report)
                self.result_text.config(state=tk.DISABLED)

                self.stop_video()
                return

            pil_img, detections, stats = item
            self.photo = ImageTk.PhotoImage(pil_img)
            self.image_label.config(image=self.photo)

            # Refresh cached values from main thread (thread-safe)
            self._skip = self.skip_var.get()
            self._imgsz = self.imgsz_var.get()
            self._container_w = self.img_frame.winfo_width()
            self._container_h = self.img_frame.winfo_height()
            if self._container_w < 10:
                self._container_w = 820
            if self._container_h < 10:
                self._container_h = 520

            report = f"=== Running Report (frame {stats['total_frames']}) ===\n"
            report += f"Per-frame: {len(detections)} object(s)\n"
            if detections:
                for d in detections:
                    report += f"  {d['cls']} — {d['conf']*100:.1f}%\n"
            else:
                report += "  (no objects detected)\n"

            report += "\nCumulative Detections:\n"
            if stats['sum_counts']:
                for cls, total in sorted(stats['sum_counts'].items()):
                    report += f"  {cls}: {total}\n"
            else:
                report += "  (none yet)\n"

            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, report)
            self.result_text.config(state=tk.DISABLED)

            self.status.config(text=f"Frame {stats['total_frames']} — {len(detections)} object(s)")
        except queue.Empty:
            pass

        # Reschedule polling as long as the video is running OR the queue is not empty
        if self.video_running or not self.queue.empty():
            self.root.after(20, self.poll_queue)

    def on_close(self):
        self.stop_video()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
