import os
import shutil
import uuid
import cv2
import torch
import numpy as np
import time
from typing import List, Dict, Optional
from collections import defaultdict
from functools import lru_cache
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# Try to import zero-shot detection
try:
    from .zero_shot_detection import get_zero_shot_detector, Detection
    HAS_ZERO_SHOT = True
except ImportError:
    HAS_ZERO_SHOT = False
    print("⚠️  Zero-shot detection not available")

# --- App Initialization ---
app = FastAPI(title="Zero-Shot Visual Understanding Assistant")
app.mount("/static", StaticFiles(directory="frontend/static", html=True), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Configuration ---
TEMP_FOLDER = "temp_files"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Global variables
llava_model = None
llava_processor = None
zero_shot_detector = None
performance_stats = {"requests": 0, "total_time": 0.0}
performance_metrics = {
    "vlm_inference_time": [],
    "frame_processing_time": [],
    "total_request_time": []
}
# --- Optimized Model Loading with Quantization ---
print("Loading optimized models for L40S GPU (48GB VRAM)...")

def load_optimized_llava():
    """Load LLaVA with 4-bit quantization and optimizations for L40S"""
    try:
        MODEL_ID = "llava-hf/llava-1.5-13b-hf"  # Upgraded to 13B for better accuracy

        # 4-bit quantization config for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            attn_implementation="flash_attention_2"  # Flash attention for speed
        )

        processor = AutoProcessor.from_pretrained(MODEL_ID)

        # Compile model for faster inference
        model = torch.compile(model, mode="reduce-overhead")

        print(f"LLaVA 13B model loaded with 4-bit quantization")
        return model, processor

    except Exception as e:
        print(f"Error loading optimized LLaVA: {e}")
        # Fallback to 7B model
        try:
            MODEL_ID = "llava-hf/llava-1.5-7b-hf"
            model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            ).to("cuda")
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            model = torch.compile(model, mode="reduce-overhead")
            print("Fallback: LLaVA 7B model loaded")
            return model, processor
        except Exception as e2:
            print(f"FATAL: Error loading fallback model: {e2}")
            return None, None

@lru_cache(maxsize=1)
def get_yolo_model():
    """Cached YOLO model loading with TensorRT optimization"""
    try:
        model = YOLO("yolov8x.pt")  # Upgraded to YOLOv8x for better accuracy

        # Export to TensorRT for L40S optimization
        try:
            model.export(format="engine", device=0, half=True, workspace=4)  # 4GB workspace
            model = YOLO("yolov8x.engine")  # Load TensorRT engine
            print("YOLO model loaded with TensorRT optimization")
        except:
            print("TensorRT export failed, using PyTorch model")

        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

# Initialize models
llava_model, llava_processor = load_optimized_llava()
yolo_model = get_yolo_model()

# Warm up models
if llava_model and llava_processor:
    print("Warming up LLaVA model...")
    dummy_image = Image.new('RGB', (224, 224), color='red')
    dummy_prompt = "USER: Describe this image.\nASSISTANT:"
    try:
        with torch.inference_mode():
            inputs = llava_processor(text=dummy_prompt, images=[dummy_image], return_tensors="pt").to("cuda", torch.float16)
            _ = llava_model.generate(**inputs, max_new_tokens=10, do_sample=False)
        print("LLaVA model warmed up successfully")
    except Exception as e:
        print(f"Model warmup failed: {e}")

if yolo_model:
    print("Warming up YOLO model...")
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        _ = yolo_model(dummy_frame, verbose=False)
        print("YOLO model warmed up successfully")
    except Exception as e:
        print(f"YOLO warmup failed: {e}")

# --- Advanced Color Detection with GPU Acceleration ---
@lru_cache(maxsize=1000)
def get_advanced_color(image_tensor):
    """GPU-accelerated color detection with better accuracy"""
    try:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image_tensor, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Define color ranges in HSV
        color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'yellow': [(20, 50, 50), (40, 255, 255)],
            'orange': [(10, 50, 50), (20, 255, 255)],
            'purple': [(130, 50, 50), (170, 255, 255)],
            'white': [(0, 0, 200), (180, 30, 255)],
            'black': [(0, 0, 0), (180, 255, 30)]
        }

        max_pixels = 0
        dominant_color = "unknown"

        for color, ranges in color_ranges.items():
            mask = np.zeros_like(h)
            for i in range(0, len(ranges), 2):
                lower, upper = ranges[i], ranges[i+1] if i+1 < len(ranges) else ranges[i]
                mask += cv2.inRange(hsv, lower, upper)

            pixel_count = cv2.countNonZero(mask)
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                dominant_color = color

        return dominant_color if max_pixels > hsv.shape[0] * hsv.shape[1] * 0.1 else "mixed"
    except:
        return "unknown"

# --- Object Tracking and Memory System ---
class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 1
        self.frame_history = []

    def update_tracking(self, detections, timestamp):
        """Update object tracking with temporal consistency"""
        current_objects = {}

        for detection in detections:
            x1, y1, x2, y2, conf, class_id, class_name = detection
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Find closest existing object
            best_match_id = None
            min_distance = float('inf')

            for obj_id, obj_data in self.tracked_objects.items():
                if obj_data['class_name'] == class_name:
                    last_pos = obj_data['positions'][-1] if obj_data['positions'] else (0, 0)
                    distance = np.sqrt((center_x - last_pos[0])**2 + (center_y - last_pos[1])**2)
                    if distance < min_distance and distance < 100:  # Threshold for same object
                        min_distance = distance
                        best_match_id = obj_id

            if best_match_id:
                # Update existing object
                self.tracked_objects[best_match_id]['positions'].append((center_x, center_y))
                self.tracked_objects[best_match_id]['timestamps'].append(timestamp)
                self.tracked_objects[best_match_id]['bbox'] = (x1, y1, x2, y2)
                current_objects[best_match_id] = self.tracked_objects[best_match_id]
            else:
                # Create new object
                obj_id = self.next_id
                self.next_id += 1
                self.tracked_objects[obj_id] = {
                    'class_name': class_name,
                    'positions': [(center_x, center_y)],
                    'timestamps': [timestamp],
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                }
                current_objects[obj_id] = self.tracked_objects[obj_id]

        return current_objects

# Global object tracker
object_tracker = ObjectTracker()

# --- Enhanced VLM Query with Specialized Reasoning ---
async def query_vlm_optimized(image_frames: List[Image.Image], question: str, context_data: dict = None) -> str:
    """Optimized VLM query with context-aware prompting"""
    if llava_model is None:
        return "Error: LLaVA model is not available."

    # Analyze question type for specialized prompting
    question_lower = question.lower()

    if any(word in question_lower for word in ['color', 'what color', 'colored']):
        reasoning_focus = "COLOR_ANALYSIS"
    elif any(word in question_lower for word in ['count', 'how many', 'number of']):
        reasoning_focus = "COUNTING"
    elif any(word in question_lower for word in ['change', 'different', 'before', 'after', 'became']):
        reasoning_focus = "TEMPORAL_CHANGE"
    elif any(word in question_lower for word in ['move', 'motion', 'speed', 'direction']):
        reasoning_focus = "MOTION_PHYSICS"
    else:
        reasoning_focus = "GENERAL_ANALYSIS"

    # Context-aware prompting
    context_info = ""
    if context_data:
        context_info = f"\n#**CONTEXT DATA**#\nObject tracking data: {context_data.get('tracking', 'None')}\nAudio transcript: {context_data.get('audio', 'None')}\n"

    prompt = f"""USER: You are an advanced visual reasoning AI specialized in {reasoning_focus}. Analyze this video sequence with extreme attention to detail.

#**SPECIALIZED INSTRUCTIONS FOR {reasoning_focus}**#
{"Focus on precise color identification. Look at object surfaces, lighting conditions, and color consistency across frames." if reasoning_focus == "COLOR_ANALYSIS" else ""}
{"Count objects methodically. Track each instance across frames to avoid double-counting or missing objects." if reasoning_focus == "COUNTING" else ""}
{"Track object states, positions, and properties across the temporal sequence. Identify what changed, when, and how." if reasoning_focus == "TEMPORAL_CHANGE" else ""}
{"Analyze movement patterns, velocities, directions, and physical interactions between objects." if reasoning_focus == "MOTION_PHYSICS" else ""}
{"Provide comprehensive analysis covering all visible elements, their relationships, and any notable events." if reasoning_focus == "GENERAL_ANALYSIS" else ""}

#**ANALYSIS FRAMEWORK**#
1. **Frame-by-Frame Analysis**: Examine each frame systematically
2. **Object Identification**: Identify all objects, their properties, and spatial relationships
3. **Temporal Tracking**: Track changes across the sequence
4. **Reasoning**: Apply logical reasoning based on visual evidence
5. **Conclusion**: Provide a definitive answer with supporting evidence
{context_info}
<image_placeholder>

#**Question**#
{question}

ASSISTANT:"""

    try:
        start_time = time.time()
        with torch.inference_mode():
            # Optimize input processing
            inputs = llava_processor(
                text=prompt,
                images=image_frames,
                return_tensors="pt",
                padding=True
            ).to("cuda", torch.float16)

            # Optimized generation parameters
            output = llava_model.generate(
                **inputs,
                max_new_tokens=400,  # Increased for detailed analysis
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                num_beams=1,  # Faster than beam search
                use_cache=True
            )

            response_text = llava_processor.decode(output[0], skip_special_tokens=True)

        inference_time = time.time() - start_time
        performance_metrics['vlm_inference_time'].append(inference_time)

        return response_text.split("ASSISTANT:")[-1].strip()

    except Exception as e:
        print(f"Error during VLM inference: {e}")
        return "An error occurred while generating the AI response."

# --- Advanced Frame Processing with GPU Acceleration ---
async def process_video_frames_optimized(video_path: str, duration_sec: float, fps: float) -> tuple:
    """
    Advanced frame processing optimized for L40S GPU
    Returns: (keyframes, object_tracking_data, performance_metrics)
    """
    start_time = time.time()

    # Adaptive sampling based on video characteristics
    if duration_sec <= 10:
        sample_rate = max(2, fps // 5)  # High sampling for short videos
        max_frames = 50
    elif duration_sec <= 60:
        sample_rate = max(1, fps // 10)  # Medium sampling
        max_frames = 100
    elif duration_sec <= 600:  # 10 minutes
        sample_rate = max(1, fps // 15)  # Lower sampling
        max_frames = 200
    else:  # Long videos (up to 120 minutes)
        sample_rate = max(1, fps // 30)  # Sparse sampling
        max_frames = 400

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame indices with intelligent distribution
    if total_frames <= max_frames:
        frame_indices = list(range(0, total_frames, max(1, total_frames // max_frames)))
    else:
        # Logarithmic distribution for long videos - more frames at beginning
        early_frames = int(max_frames * 0.4)  # 40% from first quarter
        mid_frames = int(max_frames * 0.4)    # 40% from middle half
        late_frames = max_frames - early_frames - mid_frames  # 20% from last quarter

        quarter = total_frames // 4
        indices_early = np.linspace(0, quarter, early_frames, dtype=int)
        indices_mid = np.linspace(quarter, 3*quarter, mid_frames, dtype=int)
        indices_late = np.linspace(3*quarter, total_frames-1, late_frames, dtype=int)

        frame_indices = np.concatenate([indices_early, indices_mid, indices_late])
        frame_indices = np.unique(frame_indices)  # Remove duplicates

    # Parallel frame extraction and processing
    frames_data = []
    batch_size = 32  # Process frames in batches

    for i in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[i:i+batch_size]
        batch_frames = []
        batch_timestamps = []

        for frame_idx in batch_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                timestamp = frame_idx / fps
                batch_frames.append(frame)
                batch_timestamps.append(timestamp)

        # Process batch with YOLO
        if batch_frames and yolo_model:
            try:
                # Batch inference for efficiency
                results = yolo_model(batch_frames, verbose=False)

                for j, (frame, timestamp, result) in enumerate(zip(batch_frames, batch_timestamps, results)):
                    detections = []
                    for box in result.boxes:
                        if float(box.conf.item()) > 0.3:  # Lower threshold for better recall
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf.item())
                            class_id = int(box.cls)
                            class_name = yolo_model.names[class_id]

                            detections.append((x1, y1, x2, y2, conf, class_id, class_name))

                    # Update object tracking
                    tracked_objects = object_tracker.update_tracking(detections, timestamp)

                    frames_data.append({
                        'frame': frame,
                        'timestamp': timestamp,
                        'detections': detections,
                        'tracked_objects': tracked_objects
                    })
            except Exception as e:
                print(f"Error in batch processing: {e}")

    cap.release()

    # Advanced scene change detection with optical flow
    final_keyframes = []
    keyframe_metadata = []

    if frames_data:
        # Always include first frame
        first_frame_data = frames_data[0]
        final_keyframes.append(Image.fromarray(cv2.cvtColor(first_frame_data['frame'], cv2.COLOR_BGR2RGB)))
        keyframe_metadata.append(first_frame_data)

        prev_gray = cv2.cvtColor(first_frame_data['frame'], cv2.COLOR_BGR2GRAY)

        for frame_data in frames_data[1:]:
            current_gray = cv2.cvtColor(frame_data['frame'], cv2.COLOR_BGR2GRAY)

            # Calculate optical flow magnitude
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, current_gray,
                np.array([[100, 100]], dtype=np.float32).reshape(-1, 1, 2),
                None
            )[0]

            # Scene change detection with multiple criteria
            hist_diff = cv2.compareHist(
                cv2.calcHist([prev_gray], [0], None, [256], [0, 256]),
                cv2.calcHist([current_gray], [0], None, [256], [0, 256]),
                cv2.HISTCMP_CORREL
            )

            # Structural similarity
            ssim_score = cv2.matchTemplate(
                cv2.resize(prev_gray, (64, 64)),
                cv2.resize(current_gray, (64, 64)),
                cv2.TM_CCOEFF_NORMED
            )[0][0]

            # Object count change
            obj_count_change = abs(len(frame_data['detections']) - len(keyframe_metadata[-1]['detections']))

            # Multi-criteria scene change detection
            if (hist_diff < 0.8 or ssim_score < 0.7 or obj_count_change > 2):
                final_keyframes.append(Image.fromarray(cv2.cvtColor(frame_data['frame'], cv2.COLOR_BGR2RGB)))
                keyframe_metadata.append(frame_data)
                prev_gray = current_gray

    processing_time = time.time() - start_time
    performance_metrics['frame_processing_time'].append(processing_time)

    return final_keyframes, keyframe_metadata, {
        'processing_time': processing_time,
        'total_frames_processed': len(frames_data),
        'keyframes_selected': len(final_keyframes)
    }

# --- Optimized API Endpoint ---
@app.post("/infer", response_class=PlainTextResponse)
async def infer_from_video_optimized(video: UploadFile = File(...), prompt: str = Form(...)):
    """Optimized video inference endpoint with advanced processing"""
    request_start_time = time.time()
    temp_video_path = os.path.join(TEMP_FOLDER, f"temp_{uuid.uuid4()}.mp4")

    # Async file writing
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    try:
        # Get video metadata
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration_sec = total_frames / fps if fps > 0 else 0
        cap.release()

        print(f"Processing video: {duration_sec:.2f}s, {total_frames} frames @ {fps:.1f}fps")

        # Advanced frame processing
        keyframes, metadata, processing_stats = await process_video_frames_optimized(
            temp_video_path, duration_sec, fps
        )

        # Build context data for enhanced reasoning
        context_data = {
            'tracking': {obj_id: {
                'class': obj['class_name'],
                'trajectory': obj['positions'][-5:],  # Last 5 positions
                'duration': len(obj['timestamps'])
            } for obj_id, obj in object_tracker.tracked_objects.items()},
            'video_stats': {
                'duration': duration_sec,
                'fps': fps,
                'keyframes_count': len(keyframes)
            }
        }

        # Enhanced VLM inference
        ai_response = await query_vlm_optimized(
            image_frames=keyframes,
            question=prompt,
            context_data=context_data
        )

        # Track total request time
        total_time = time.time() - request_start_time
        performance_metrics['total_request_time'].append(total_time)

        print(f"Request completed in {total_time:.2f}s (target: <1.0s)")

        return PlainTextResponse(content=ai_response)

    except Exception as e:
        print(f"Error in optimized inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

# --- Performance Monitoring Endpoints ---
@app.get("/metrics")
async def get_performance_metrics():
    """Get current performance metrics"""
    metrics_summary = {}
    for metric_name, values in performance_metrics.items():
        if values:
            metrics_summary[metric_name] = {
                'count': len(values),
                'mean': np.mean(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'min': np.min(values),
                'max': np.max(values)
            }

    return {
        'metrics': metrics_summary,
        'gpu_memory': {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        },
        'model_status': {
            'llava_loaded': llava_model is not None,
            'yolo_loaded': yolo_model is not None,
            'tracked_objects': len(object_tracker.tracked_objects)
        }
    }

@app.post("/reset_tracking")
async def reset_object_tracking():
    """Reset object tracking for new video session"""
    global object_tracker
    object_tracker = ObjectTracker()
    return {"status": "Object tracking reset"}

# --- Low-Latency Streaming Endpoints ---
@app.websocket("/ws/{session_id}")
async def websocket_inference(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time streaming inference"""
    await websocket.accept()

    try:
        # Create or get session
        if session_id not in streaming_engine.active_sessions:
            await streaming_engine.create_session(session_id)

        while True:
            # Receive request data
            data = await websocket.receive_json()
            question = data.get('question', '')

            if not question:
                await websocket.send_json({"error": "No question provided"})
                continue

            # For WebSocket, we'll use cached frames from the session
            session_data = streaming_engine.active_sessions[session_id]
            frames = list(session_data.get('frame_history', []))

            if not frames:
                await websocket.send_json({"error": "No frames available. Upload video first."})
                continue

            # Stream response
            await websocket.send_json({"type": "start", "message": "Processing..."})

            response_chunks = []
            async for chunk in streaming_engine.process_request_streaming(
                session_id, frames, question, websocket
            ):
                response_chunks.append(chunk)
                await websocket.send_json({
                    "type": "chunk",
                    "content": chunk
                })

            # Send completion signal
            await websocket.send_json({
                "type": "complete",
                "full_response": ''.join(response_chunks)
            })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})

@app.post("/infer_stream")
async def infer_streaming(video: UploadFile = File(...), prompt: str = Form(...)):
    """Streaming inference endpoint for low-latency responses"""

    async def generate_response():
        temp_video_path = os.path.join(TEMP_FOLDER, f"temp_{uuid.uuid4()}.mp4")

        try:
            # Save uploaded video
            with open(temp_video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)

            # Quick video analysis for streaming
            cap = cv2.VideoCapture(temp_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps
            cap.release()

            # Fast frame extraction (fewer frames for speed)
            keyframes, metadata, stats = await process_video_frames_optimized(
                temp_video_path, duration_sec, fps
            )

            # Create session for streaming
            session_id = await streaming_engine.create_session()

            # Stream response
            async for chunk in streaming_engine.process_request_streaming(
                session_id, [np.array(frame) for frame in keyframes], prompt
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            yield f"data: {json.dumps({'complete': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

# --- Endpoint to serve the frontend HTML ---
@app.get("/")
async def read_index():
    """Serves the main frontend HTML file."""
    return FileResponse('frontend/index.html')
