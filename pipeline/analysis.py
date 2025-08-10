import os
import cv2
import numpy as np
from ultralytics import YOLO
from . import schema

# -----------------------------------------
# Utility: Detect the dominant color of an image region
# -----------------------------------------
def get_dominant_color(image):
    """
    Extracts the dominant color in the central region of the image.
    Returns a basic color label (white, black, red, blue, or unknown).
    """
    try:
        # Focus on the central region to avoid noisy edges
        pixels = image[
            image.shape[0] // 4 : 3 * image.shape[0] // 4,
            image.shape[1] // 4 : 3 * image.shape[1] // 4
        ].reshape(-1, 3)

        if pixels.size == 0:
            return "unknown"

        # Convert to float for KMeans
        pixels = np.float32(pixels)

        # Apply KMeans to find the dominant color cluster
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
        _, _, centers = cv2.kmeans(
            pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Extract RGB values
        b, g, r = map(int, centers[0])

        # Basic heuristic-based color naming
        if r > 180 and g > 180 and b > 180:
            return "white"
        if r < 70 and g < 70 and b < 70:
            return "black"
        if r > 150 and g < 100 and b < 100:
            return "red"
        if b > 150 and r < 100 and g < 100:
            return "blue"
        return "unknown"

    except Exception:
        return "unknown"


# -----------------------------------------
# Main Analysis: Object detection + attributes from video keyframes
# -----------------------------------------
def analyze_video_content(keyframes_dir, video_metadata):
    """
    Analyzes extracted keyframes from a video to detect:
    - Objects
    - Basic color attributes
    - Positions

    Appends these perceptual events into `video_metadata`.
    """
    print("-> Step 2: Analyzing frames for objects and attributes...")

    # Load YOLO model (Large version for higher accuracy)
    model = YOLO("yolov8l.pt")

    # Get sorted list of all keyframe images
    frame_files = sorted(
        [f for f in os.listdir(keyframes_dir) if f.endswith(".jpg")]
    )

    for frame_file in frame_files:
        # Extract timestamp from filename
        timestamp = float(frame_file.replace("frame_", "").replace(".jpg", ""))

        # Load image
        img_path = os.path.join(keyframes_dir, frame_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Run YOLO inference
        results = model(img, verbose=False)

        percepts = []
        for res in results:
            for box in res.boxes:
                # Apply confidence threshold
                if float(box.conf.item()) > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = model.names[int(box.cls)]

                    # Detect object color
                    color = get_dominant_color(img[y1:y2, x1:x2])

                    # Create a perceptual description
                    percepts.append(
                        f"a {color} {class_name} is at position ({x1},{y1})"
                    )

        # Append perceptual data to metadata if detected
        if percepts:
            video_metadata.events.append({
                "time": timestamp,
                "type": "visual_percepts",
                "content": ", ".join(percepts)
            })
