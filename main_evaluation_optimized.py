"""
Evaluation-Optimized Backend for Visual Understanding
Designed specifically for external evaluation systems
"""

import os
import shutil
import uuid
import time
import cv2
import torch
import numpy as np
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Optional imports with fallbacks
try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# --- App Setup ---
app = FastAPI(title="Evaluation-Optimized Visual Assistant")
app.mount("/static", StaticFiles(directory="frontend/static", html=True), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Configuration
TEMP_FOLDER = "temp_files"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Global models
llava_model = None
llava_processor = None
yolo_model = None

def load_models():
    """Load models optimized for evaluation"""
    global llava_model, llava_processor, yolo_model
    
    if HAS_TRANSFORMERS:
        try:
            MODEL_ID = "llava-hf/llava-1.5-7b-hf"
            llava_model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            if torch.cuda.is_available():
                llava_model = llava_model.to("cuda")
            llava_processor = AutoProcessor.from_pretrained(MODEL_ID)
            print("✅ LLaVA loaded")
        except Exception as e:
            print(f"❌ LLaVA failed: {e}")
    
    if HAS_YOLO:
        try:
            yolo_model = YOLO("yolov8m.pt")  # Medium model for balance
            print("✅ YOLO loaded")
        except Exception as e:
            print(f"❌ YOLO failed: {e}")

def extract_key_frames(video_path: str, max_frames: int = 8) -> List[Image.Image]:
    """Extract key frames with scene change detection"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Adaptive sampling
        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        prev_gray = None
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Scene change detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_small = cv2.resize(gray, (64, 64))
                
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray_small)
                    change_ratio = np.count_nonzero(diff > 30) / (64 * 64)
                    if change_ratio < 0.05 and len(frames) > 0:  # Skip similar frames
                        continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                prev_gray = gray_small
        
        cap.release()
    except Exception as e:
        print(f"Frame extraction error: {e}")
    
    return frames

def analyze_color_precise(frame_np: np.ndarray, bbox=None) -> str:
    """Precise color analysis optimized for evaluation"""
    try:
        if bbox:
            x1, y1, x2, y2 = bbox
            roi = frame_np[y1:y2, x1:x2]
        else:
            roi = frame_np
        
        if roi.size == 0:
            return "unknown"
        
        # Convert to HSV for robust color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        
        # Precise color classification
        if v_mean < 50:
            return "black"
        elif s_mean < 30:
            return "white" if v_mean > 180 else "gray"
        elif h_mean < 10 or h_mean > 170:
            return "red"
        elif 10 <= h_mean < 25:
            return "orange"
        elif 25 <= h_mean < 35:
            return "yellow"
        elif 35 <= h_mean < 80:
            return "green"
        elif 80 <= h_mean < 130:
            return "blue"
        elif 130 <= h_mean < 160:
            return "purple"
        else:
            return "pink"
    except:
        return "unknown"

def detect_and_count_objects(frames: List[Image.Image], target_class: str = None) -> Dict:
    """Object detection and counting across frames"""
    if not yolo_model:
        return {"count": 0, "objects": [], "colors": []}
    
    all_detections = []
    object_counts = {}
    colors_found = []
    
    for frame in frames:
        frame_np = np.array(frame)
        try:
            results = yolo_model(frame_np, verbose=False)
            for result in results:
                for box in result.boxes:
                    if float(box.conf.item()) > 0.3:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_name = yolo_model.names[int(box.cls)]
                        
                        # Count objects
                        object_counts[class_name] = object_counts.get(class_name, 0) + 1
                        
                        # Analyze color
                        color = analyze_color_precise(frame_np, [x1, y1, x2, y2])
                        colors_found.append(f"{color} {class_name}")
                        
                        all_detections.append({
                            'class': class_name,
                            'color': color,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(box.conf.item())
                        })
        except Exception as e:
            print(f"Detection error: {e}")
    
    # Get max count for target class
    if target_class:
        count = max([1 for det in all_detections if target_class.lower() in det['class'].lower()], default=0)
        count = len([det for det in all_detections if target_class.lower() in det['class'].lower()])
    else:
        count = len(set(det['class'] for det in all_detections))
    
    return {
        "count": count,
        "objects": list(object_counts.keys()),
        "colors": list(set(colors_found)),
        "detections": all_detections
    }

def create_evaluation_prompt(question: str, frames_info: Dict) -> str:
    """Create evaluation-optimized prompts"""
    question_lower = question.lower()
    
    # Detect question type and create focused prompt
    if any(word in question_lower for word in ['color', 'what color']):
        return f"""Analyze the images and identify colors precisely. Focus on the dominant colors of objects.

Objects detected: {', '.join(frames_info.get('objects', []))}
Colors found: {', '.join(frames_info.get('colors', []))}

Question: {question}

Provide a direct, specific answer about the color. Use basic color names: red, blue, green, yellow, orange, purple, pink, black, white, gray, brown."""

    elif any(word in question_lower for word in ['how many', 'count', 'number']):
        return f"""Count objects systematically across all frames. Avoid double-counting the same object.

Objects detected: {', '.join(frames_info.get('objects', []))}
Total unique objects: {frames_info.get('count', 0)}

Question: {question}

Provide the exact number. If counting specific objects, focus only on that type."""

    elif any(word in question_lower for word in ['move', 'motion', 'moving', 'speed']):
        return f"""Analyze movement and motion across the video frames. Look for objects changing position.

Objects detected: {', '.join(frames_info.get('objects', []))}

Question: {question}

Describe what is moving, how it moves, and any motion patterns you observe."""

    elif any(word in question_lower for word in ['change', 'different', 'before', 'after']):
        return f"""Compare the frames to identify changes in object states, positions, or properties.

Objects detected: {', '.join(frames_info.get('objects', []))}

Question: {question}

Describe specific changes you observe between frames."""

    else:
        return f"""Analyze the video frames systematically and provide a comprehensive answer.

Objects detected: {', '.join(frames_info.get('objects', []))}
Colors found: {', '.join(frames_info.get('colors', []))}

Question: {question}

Provide a clear, direct answer based on what you observe in the images."""

@app.post("/infer", response_class=PlainTextResponse)
async def infer_evaluation(video: UploadFile = File(...), prompt: str = Form(...)):
    """Evaluation-optimized inference endpoint"""
    temp_video_path = os.path.join(TEMP_FOLDER, f"eval_{uuid.uuid4()}.mp4")
    
    try:
        # Save video
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Extract frames
        frames = extract_key_frames(temp_video_path, max_frames=6)
        if not frames:
            return PlainTextResponse("Unable to process video")
        
        # Analyze frames
        frames_info = detect_and_count_objects(frames)
        
        # Create optimized prompt
        evaluation_prompt = create_evaluation_prompt(prompt, frames_info)
        
        # Generate response
        if llava_model and llava_processor:
            try:
                with torch.inference_mode():
                    inputs = llava_processor(
                        text=evaluation_prompt,
                        images=frames[:3],  # Use up to 3 frames for stability
                        return_tensors="pt"
                    )
                    
                    if torch.cuda.is_available():
                        inputs = inputs.to("cuda")
                    
                    output = llava_model.generate(
                        **inputs,
                        max_new_tokens=150,  # Concise answers for evaluation
                        do_sample=False,     # Deterministic for consistency
                        temperature=0.0,
                        pad_token_id=llava_processor.tokenizer.eos_token_id
                    )
                    
                    response = llava_processor.decode(output[0], skip_special_tokens=True)
                    answer = response.split("Question:")[-1].split("Provide")[0].strip()
                    
                    # Clean up answer
                    if not answer or len(answer) < 5:
                        answer = response.split("\n")[-1].strip()
                    
                    return PlainTextResponse(answer)
                    
            except Exception as e:
                print(f"Generation error: {e}")
                return PlainTextResponse("Processing error occurred")
        else:
            return PlainTextResponse("Model not available")
            
    except Exception as e:
        print(f"Request error: {e}")
        return PlainTextResponse("Request processing failed")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "llava_loaded": llava_model is not None,
        "yolo_loaded": yolo_model is not None,
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/")
async def root():
    return FileResponse('frontend/index.html')

# Load models on startup
print("🚀 Loading Evaluation-Optimized Backend")
load_models()
print("✅ Ready for evaluation!")
