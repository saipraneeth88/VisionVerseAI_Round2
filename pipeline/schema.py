from typing import List, Dict
from pydantic import BaseModel, Field

# =========================================================
# Data Models for Video Analysis
# =========================================================

class ObjectState(BaseModel):
    """
    Represents the state of a detected object at a specific timestamp.
    
    Attributes:
        time (float)       : Timestamp in seconds.
        bbox (List[int])   : Bounding box [x1, y1, x2, y2].
        color (str)        : Dominant color of the object.
    """
    time: float
    bbox: List[int]
    color: str


class TrackedObject(BaseModel):
    """
    Represents a tracked object across multiple frames.
    
    Attributes:
        id (int)                   : Unique object tracking ID.
        class_name (str)           : YOLO-detected class name.
        states (List[ObjectState]) : List of object states over time.
    """
    id: int
    class_name: str
    states: List[ObjectState] = Field(default_factory=list)


class VideoMetadata(BaseModel):
    """
    Represents metadata and analysis events for an entire video.
    
    Attributes:
        video_file (str)       : Name of the processed video file.
        duration_sec (float)   : Duration of the video in seconds.
        events (List[Dict])    : List of event dictionaries (percepts, dialogues, etc.).
    """
    video_file: str
    duration_sec: float
    events: List[Dict] = Field(default_factory=list)
