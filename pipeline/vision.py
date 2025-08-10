import os
import cv2
from transformers import pipeline


# =========================================================
# Video Processing: Keyframe Sampling & Action Recognition
# =========================================================

def sample_keyframes(video_path: str, video_hash: str, interval_sec: float) -> str:
    """
    Samples keyframes from a video at a fixed interval.

    Args:
        video_path (str)    : Path to the input video file.
        video_hash (str)    : Unique identifier for the video (used for storage path).
        interval_sec (float): Time interval (in seconds) between keyframes.

    Returns:
        str: Path to the directory containing sampled keyframes.
    """
    print(f"-> Sampling keyframes every {interval_sec}s...")

    keyframe_dir = f"data/temp/keyframes/{video_hash}"
    os.makedirs(keyframe_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(max(1, round(fps * interval_sec)))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            ts = frame_idx / fps
            frame_filename = os.path.join(keyframe_dir, f"frame_{ts:.2f}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame_idx += 1

    cap.release()
    return keyframe_dir


def recognize_actions(video_path: str) -> list:
    """
    Recognizes high-level actions in a video using a pre-trained VideoMAE model.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        list: Top 3 predicted action labels.
    """
    print("-> Recognizing high-level actions...")

    try:
        clf = pipeline(
            "video-classification",
            model="MCG-NJU/videomae-base-finetuned-kinetics",
            device=0  # Use GPU if available
        )
        results = clf(video_path)
        return [r['label'] for r in results[:3]]

    except Exception as e:
        print(f"[Error] Action recognition failed: {e}")
        return []
