import os
import sys
import json
import time
import cv2
import shutil
import numpy as np
from ultralytics import YOLO
from PIL import Image

# =========================================================
# Configuration
# =========================================================
PROCESSED_FOLDER = "data/processed"
TEMP_FOLDER = "data/temp"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)


# =========================================================
# Utility Function: Dominant Color Detection
# =========================================================
def get_dominant_color(image):
    """
    Detect the dominant color in an image crop using K-means clustering.

    Parameters:
        image (numpy.ndarray) : BGR image crop.

    Returns:
        str : Color name ('white', 'black', 'red', 'blue', 'unknown color').
    """
    try:
        # Flatten the pixels for clustering
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)

        # K-means clustering to find the most dominant color
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
        _, _, centers = cv2.kmeans(
            pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        b, g, r = map(int, centers[0])

        # Simple color thresholds
        if r > 180 and g > 180 and b > 180:
            return "white"
        if r < 70 and g < 70 and b < 70:
            return "black"
        if r > 150 and g < 100 and b < 100:
            return "red"
        if b > 150 and r < 100 and g < 100:
            return "blue"
        return "unknown color"

    except Exception:
        return "unknown color"


# =========================================================
# Main Video Processing Pipeline
# =========================================================
def run_pipeline(video_path, video_hash):
    """
    Full pipeline for:
    1. Sampling keyframes.
    2. Detecting objects + dominant colors with YOLOv8.
    3. Saving structured percept logs.

    Parameters:
        video_path (str) : Path to the input video file.
        video_hash (str) : Unique hash/ID for the video.

    Returns:
        str : Path to generated metadata JSON.
    """
    start_time = time.time()
    print(f"--- Starting Pipeline for {os.path.basename(video_path)} ---")

    events = []

    try:
        # -------------------------------------------------
        # 1. Keyframe Sampling
        # -------------------------------------------------
        keyframe_dir = os.path.join(TEMP_FOLDER, video_hash)
        os.makedirs(keyframe_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Sample 15 frames evenly across the video
        num_frames_to_sample = 15
        frame_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_time = i / fps
                cv2.imwrite(
                    os.path.join(keyframe_dir, f"frame_{frame_time:.2f}.jpg"),
                    frame
                )
        cap.release()

        # -------------------------------------------------
        # 2. Perception Extraction (Objects + Colors)
        # -------------------------------------------------
        print("-> Extracting percepts from frames...")
        model = YOLO("yolov8l.pt")  # Large YOLOv8 model for accuracy

        frame_files = sorted(
            [f for f in os.listdir(keyframe_dir) if f.endswith(".jpg")]
        )

        for frame_file in frame_files:
            timestamp = float(frame_file.replace("frame_", "").replace(".jpg", ""))
            img = cv2.imread(os.path.join(keyframe_dir, frame_file))
            if img is None:
                continue

            results = model(img, verbose=False)
            percepts = []

            for res in results:
                for box in res.boxes:
                    if float(box.conf.item()) > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_name = model.names[int(box.cls)]
                        color = get_dominant_color(img[y1:y2, x1:x2])
                        percepts.append(
                            f"a {color} {class_name} is at position ({x1},{y1})"
                        )

            if percepts:
                events.append({
                    "time": timestamp,
                    "type": "visual_percepts",
                    "content": ", ".join(percepts)
                })

    except Exception as e:
        print(f"Error during pipeline execution: {e}")

    finally:
        # -------------------------------------------------
        # 3. Save Metadata and Cleanup
        # -------------------------------------------------
        if os.path.exists(keyframe_dir):
            shutil.rmtree(keyframe_dir)

    output_path = os.path.join(PROCESSED_FOLDER, f"{video_hash}_metadata.json")
    metadata = {
        "video_file": os.path.basename(video_path),
        "events": events
    }

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[pipeline] Done. Wrote percept log in {time.time() - start_time:.2f}s.")
    return output_path


# =========================================================
# Command-Line Execution
# =========================================================
if __name__ == "__main__":
    if len(sys.argv) > 2:
        run_pipeline(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python -m pipeline.run <video_path> <video_hash>")
