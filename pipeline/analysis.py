# In pipeline/analysis.py

import os
import cv2
import numpy as np
from ultralytics import YOLO
from . import schema
# NEW: Import your zero-shot detector
from backend.zero_shot_detection import get_zero_shot_detector, Detection

# --- UTILITIES (get_dominant_color, iou, etc. are unchanged) ---
def get_dominant_color(image):
    # ... (code is unchanged)

# --- FINAL ANALYSIS FUNCTION ---
def analyze_video_content(keyframes_dir, video_metadata):
    print("-> Step 2c: Building Knowledge Base with YOLO and Zero-Shot Analysis...")
    yolo_model = YOLO("yolov8l.pt")
    zero_shot_detector = get_zero_shot_detector()

    frame_files = sorted([f for f in os.listdir(keyframes_dir) if f.endswith(".jpg")])
    if not frame_files: return

    for frame_file in frame_files:
        timestamp = float(frame_file.replace("frame_", "").replace(".jpg", ""))
        frame_path = os.path.join(keyframes_dir, frame_file)
        img = cv2.imread(frame_path)
        if img is None: continue

        # --- Run YOLO for common objects (Fast Path) ---
        yolo_results = yolo_model(img, verbose=False)
        percepts = []
        for res in yolo_results:
            for box in res.boxes:
                if float(box.conf.item()) > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = yolo_model.names[int(box.cls)]
                    color = get_dominant_color(img[y1:y2, x1:x2])
                    percepts.append(f"a {color} {class_name} is at position ({x1},{y1})")

        # --- Run Zero-Shot for specific, complex queries (Slower, more powerful path) ---
        # In a real system, you might run this based on the user's prompt
        # For the hackathon, we can run it on a few key classes to demonstrate capability
        zero_shot_queries = ["safety vest", "traffic cone", "warning sign"]
        if HAS_ZERO_SHOT:
            zero_shot_detections = zero_shot_detector.detect_objects_zero_shot(img, query_classes=zero_shot_queries)
            for det in zero_shot_detections:
                percepts.append(f"a zero-shot detected {det.description} is at position {det.bbox[:2]}")

        if percepts:
            video_metadata.events.append({
                "time": timestamp,
                "type": "visual_percepts",
                "content": ", ".join(percepts)
            })
