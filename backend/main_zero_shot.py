"""
Zero-Shot Visual Understanding Backend
Optimized for open-vocabulary object detection and reasoning
"""

import os
import shutil
import uuid
import cv2
import torch
import numpy as np
import time
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Optional imports with fallbacks
try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    HAS_LLAVA = True
except ImportError:
    HAS_LLAVA = False
    print("⚠️  LLaVA not available")

try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("⚠️  CLIP not available")

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  Transformers pipeline not available")

# --- App Setup ---
app = FastAPI(title="Zero-Shot Visual Understanding Assistant")
app.mount("/static", StaticFiles(directory="frontend/static", html=True), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Configuration
TEMP_FOLDER = "temp_files"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Global models
llava_model = None
llava_processor = None
clip_model = None
clip_preprocess = None
segmentation_model = None
performance_stats = {"requests": 0, "total_time": 0.0}

def load_models():
    """Load zero-shot capable models"""
    global llava_model, llava_processor, clip_model, clip_preprocess, segmentation_model
    
    print("🚀 Loading Zero-Shot Models...")
    
    # Load LLaVA for visual reasoning
    if HAS_LLAVA:
        try:
            MODEL_ID = "llava-hf/llava-1.5-7b-hf"
            llava_model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            if torch.cuda.is_available():
                llava_model = llava_model.to("cuda")
            llava_processor = AutoProcessor.from_pretrained(MODEL_ID)
            print("✅ LLaVA loaded for visual reasoning")
        except Exception as e:
            print(f"❌ LLaVA failed: {e}")
    
    # Load CLIP for zero-shot classification
    if HAS_CLIP:
        try:
            clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
            print("✅ CLIP loaded for zero-shot classification")
        except Exception as e:
            print(f"❌ CLIP failed: {e}")
    
    # Load segmentation model
    if HAS_TRANSFORMERS:
        try:
            segmentation_model = pipeline(
                "image-segmentation",
                model="facebook/detr-resnet-50-panoptic",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Segmentation model loaded")
        except Exception as e:
            print(f"❌ Segmentation failed: {e}")

def extract_frames_smart(video_path: str, max_frames: int = 8) -> List[Image.Image]:
    """Smart frame extraction with scene change detection"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            # Distribute frames across video
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

def zero_shot_detect(image: np.ndarray, query_classes: List[str]) -> List[Dict]:
    """Zero-shot object detection using CLIP + segmentation"""
    if not clip_model or not segmentation_model:
        return []
    
    detections = []
    try:
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get segments
        segments = segmentation_model(pil_image)
        
        for segment in segments:
            mask = np.array(segment['mask'])
            coords = np.where(mask > 0)
            
            if len(coords[0]) == 0:
                continue
            
            # Get bounding box
            y1, y2 = coords[0].min(), coords[0].max()
            x1, x2 = coords[1].min(), coords[1].max()
            
            # Extract ROI
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            # CLIP classification
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            roi_preprocessed = clip_preprocess(roi_pil).unsqueeze(0).to(clip_model.device)
            
            text_queries = [f"a photo of a {cls}" for cls in query_classes]
            text_tokens = clip.tokenize(text_queries).to(clip_model.device)
            
            with torch.no_grad():
                image_features = clip_model.encode_image(roi_preprocessed)
                text_features = clip_model.encode_text(text_tokens)
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                best_idx = similarities.argmax().item()
                confidence = similarities[0, best_idx].item()
            
            if confidence > 0.15:  # Threshold
                detections.append({
                    'class': query_classes[best_idx],
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'color': analyze_color(roi)
                })
    
    except Exception as e:
        print(f"Zero-shot detection error: {e}")
    
    return detections

def analyze_color(roi: np.ndarray) -> str:
    """Analyze dominant color"""
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        
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

def create_zero_shot_prompt(question: str, detections: List[Dict]) -> str:
    """Create prompt optimized for zero-shot understanding"""
    question_lower = question.lower()
    
    # Extract detected objects info
    objects_info = []
    colors_found = set()
    
    for det in detections:
        objects_info.append(f"{det['color']} {det['class']} (confidence: {det['confidence']:.2f})")
        colors_found.add(det['color'])
    
    objects_text = ", ".join(objects_info) if objects_info else "No objects detected"
    colors_text = ", ".join(colors_found) if colors_found else "No specific colors identified"
    
    # Question-type specific prompting
    if any(word in question_lower for word in ['color', 'what color']):
        focus = "Focus on identifying colors precisely. Use the detected color information."
    elif any(word in question_lower for word in ['count', 'how many']):
        focus = f"Count systematically. Detected {len(detections)} objects total."
    elif any(word in question_lower for word in ['what', 'describe', 'see']):
        focus = "Provide a comprehensive description of all visible elements."
    else:
        focus = "Analyze the visual content systematically."
    
    return f"""USER: You are an expert visual AI with zero-shot understanding capabilities.

DETECTED OBJECTS: {objects_text}
COLORS IDENTIFIED: {colors_text}

INSTRUCTION: {focus}

Question: {question}

Provide a direct, accurate answer based on the visual evidence.

ASSISTANT:"""

async def query_llava_zero_shot(frames: List[Image.Image], question: str) -> str:
    """Query LLaVA with zero-shot detection context"""
    if not llava_model or not llava_processor:
        return "Visual reasoning model not available"
    
    try:
        # Perform zero-shot detection on first frame
        frame_np = np.array(frames[0])
        
        # Extract query classes from question
        question_words = question.lower().split()
        common_objects = [
            "person", "car", "bicycle", "dog", "cat", "bird", "bottle", "chair", 
            "table", "book", "phone", "laptop", "ball", "cup", "plate"
        ]
        
        # Add color-based queries
        colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white"]
        color_objects = [f"{color} object" for color in colors]
        
        query_classes = common_objects + color_objects
        
        # Perform zero-shot detection
        detections = zero_shot_detect(frame_np, query_classes)
        
        # Create enhanced prompt
        prompt = create_zero_shot_prompt(question, detections)
        
        # Generate response
        with torch.inference_mode():
            inputs = llava_processor(
                text=prompt,
                images=frames[:3],  # Use up to 3 frames
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            output = llava_model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.0,
                pad_token_id=llava_processor.tokenizer.eos_token_id
            )
            
            response = llava_processor.decode(output[0], skip_special_tokens=True)
            answer = response.split("ASSISTANT:")[-1].strip()
            
            return answer if answer else "Unable to analyze the video"
    
    except Exception as e:
        print(f"Zero-shot query error: {e}")
        return "Error processing the visual content"

@app.post("/infer", response_class=PlainTextResponse)
async def infer_zero_shot(video: UploadFile = File(...), prompt: str = Form(...)):
    """Zero-shot inference endpoint"""
    temp_video_path = os.path.join(TEMP_FOLDER, f"zero_shot_{uuid.uuid4()}.mp4")
    
    try:
        performance_stats["requests"] += 1
        start_time = time.time()
        
        # Save video
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Extract frames
        frames = extract_frames_smart(temp_video_path, max_frames=6)
        if not frames:
            return PlainTextResponse("Unable to extract frames from video")
        
        # Zero-shot analysis
        response = await query_llava_zero_shot(frames, prompt)
        
        # Update stats
        processing_time = time.time() - start_time
        performance_stats["total_time"] += processing_time
        
        print(f"✅ Zero-shot request completed in {processing_time:.2f}s")
        return PlainTextResponse(response)
        
    except Exception as e:
        print(f"❌ Zero-shot inference error: {e}")
        return PlainTextResponse("Error processing the request")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "llava_loaded": llava_model is not None,
            "clip_loaded": clip_model is not None,
            "segmentation_loaded": segmentation_model is not None
        },
        "capabilities": ["zero_shot_detection", "visual_reasoning", "color_analysis"],
        "stats": performance_stats
    }

@app.get("/")
async def root():
    """Serve frontend"""
    return FileResponse('frontend/index.html')

# Load models on startup
print("🚀 Starting Zero-Shot Visual Understanding Backend")
load_models()
print("✅ Zero-shot backend ready!")
"""
Zero-Shot Visual Understanding Backend
Optimized for open-vocabulary object detection and reasoning
"""

import os
import shutil
import uuid
import cv2
import torch
import numpy as np
import time
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Optional imports with fallbacks
try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    HAS_LLAVA = True
except ImportError:
    HAS_LLAVA = False
    print("⚠️  LLaVA not available")

try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("⚠️  CLIP not available")

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  Transformers pipeline not available")

# --- App Setup ---
app = FastAPI(title="Zero-Shot Visual Understanding Assistant")
app.mount("/static", StaticFiles(directory="frontend/static", html=True), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Configuration
TEMP_FOLDER = "temp_files"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Global models
llava_model = None
llava_processor = None
clip_model = None
clip_preprocess = None
segmentation_model = None
performance_stats = {"requests": 0, "total_time": 0.0}

def load_models():
    """Load zero-shot capable models"""
    global llava_model, llava_processor, clip_model, clip_preprocess, segmentation_model
    
    print("🚀 Loading Zero-Shot Models...")
    
    # Load LLaVA for visual reasoning
    if HAS_LLAVA:
        try:
            MODEL_ID = "llava-hf/llava-1.5-7b-hf"
            llava_model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            if torch.cuda.is_available():
                llava_model = llava_model.to("cuda")
            llava_processor = AutoProcessor.from_pretrained(MODEL_ID)
            print("✅ LLaVA loaded for visual reasoning")
        except Exception as e:
            print(f"❌ LLaVA failed: {e}")
    
    # Load CLIP for zero-shot classification
    if HAS_CLIP:
        try:
            clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
            print("✅ CLIP loaded for zero-shot classification")
        except Exception as e:
            print(f"❌ CLIP failed: {e}")
    
    # Load segmentation model
    if HAS_TRANSFORMERS:
        try:
            segmentation_model = pipeline(
                "image-segmentation",
                model="facebook/detr-resnet-50-panoptic",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Segmentation model loaded")
        except Exception as e:
            print(f"❌ Segmentation failed: {e}")

def extract_frames_smart(video_path: str, max_frames: int = 8) -> List[Image.Image]:
    """Smart frame extraction with scene change detection"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            # Distribute frames across video
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

def zero_shot_detect(image: np.ndarray, query_classes: List[str]) -> List[Dict]:
    """Zero-shot object detection using CLIP + segmentation"""
    if not clip_model or not segmentation_model:
        return []
    
    detections = []
    try:
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get segments
        segments = segmentation_model(pil_image)
        
        for segment in segments:
            mask = np.array(segment['mask'])
            coords = np.where(mask > 0)
            
            if len(coords[0]) == 0:
                continue
            
            # Get bounding box
            y1, y2 = coords[0].min(), coords[0].max()
            x1, x2 = coords[1].min(), coords[1].max()
            
            # Extract ROI
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            # CLIP classification
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            roi_preprocessed = clip_preprocess(roi_pil).unsqueeze(0).to(clip_model.device)
            
            text_queries = [f"a photo of a {cls}" for cls in query_classes]
            text_tokens = clip.tokenize(text_queries).to(clip_model.device)
            
            with torch.no_grad():
                image_features = clip_model.encode_image(roi_preprocessed)
                text_features = clip_model.encode_text(text_tokens)
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                best_idx = similarities.argmax().item()
                confidence = similarities[0, best_idx].item()
            
            if confidence > 0.15:  # Threshold
                detections.append({
                    'class': query_classes[best_idx],
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'color': analyze_color(roi)
                })
    
    except Exception as e:
        print(f"Zero-shot detection error: {e}")
    
    return detections

def analyze_color(roi: np.ndarray) -> str:
    """Analyze dominant color"""
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        
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

def create_zero_shot_prompt(question: str, detections: List[Dict]) -> str:
    """Create prompt optimized for zero-shot understanding"""
    question_lower = question.lower()
    
    # Extract detected objects info
    objects_info = []
    colors_found = set()
    
    for det in detections:
        objects_info.append(f"{det['color']} {det['class']} (confidence: {det['confidence']:.2f})")
        colors_found.add(det['color'])
    
    objects_text = ", ".join(objects_info) if objects_info else "No objects detected"
    colors_text = ", ".join(colors_found) if colors_found else "No specific colors identified"
    
    # Question-type specific prompting
    if any(word in question_lower for word in ['color', 'what color']):
        focus = "Focus on identifying colors precisely. Use the detected color information."
    elif any(word in question_lower for word in ['count', 'how many']):
        focus = f"Count systematically. Detected {len(detections)} objects total."
    elif any(word in question_lower for word in ['what', 'describe', 'see']):
        focus = "Provide a comprehensive description of all visible elements."
    else:
        focus = "Analyze the visual content systematically."
    
    return f"""USER: You are an expert visual AI with zero-shot understanding capabilities.

DETECTED OBJECTS: {objects_text}
COLORS IDENTIFIED: {colors_text}

INSTRUCTION: {focus}

Question: {question}

Provide a direct, accurate answer based on the visual evidence.

ASSISTANT:"""

async def query_llava_zero_shot(frames: List[Image.Image], question: str) -> str:
    """Query LLaVA with zero-shot detection context"""
    if not llava_model or not llava_processor:
        return "Visual reasoning model not available"
    
    try:
        # Perform zero-shot detection on first frame
        frame_np = np.array(frames[0])
        
        # Extract query classes from question
        question_words = question.lower().split()
        common_objects = [
            "person", "car", "bicycle", "dog", "cat", "bird", "bottle", "chair", 
            "table", "book", "phone", "laptop", "ball", "cup", "plate"
        ]
        
        # Add color-based queries
        colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white"]
        color_objects = [f"{color} object" for color in colors]
        
        query_classes = common_objects + color_objects
        
        # Perform zero-shot detection
        detections = zero_shot_detect(frame_np, query_classes)
        
        # Create enhanced prompt
        prompt = create_zero_shot_prompt(question, detections)
        
        # Generate response
        with torch.inference_mode():
            inputs = llava_processor(
                text=prompt,
                images=frames[:3],  # Use up to 3 frames
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            output = llava_model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.0,
                pad_token_id=llava_processor.tokenizer.eos_token_id
            )
            
            response = llava_processor.decode(output[0], skip_special_tokens=True)
            answer = response.split("ASSISTANT:")[-1].strip()
            
            return answer if answer else "Unable to analyze the video"
    
    except Exception as e:
        print(f"Zero-shot query error: {e}")
        return "Error processing the visual content"

@app.post("/infer", response_class=PlainTextResponse)
async def infer_zero_shot(video: UploadFile = File(...), prompt: str = Form(...)):
    """Zero-shot inference endpoint"""
    temp_video_path = os.path.join(TEMP_FOLDER, f"zero_shot_{uuid.uuid4()}.mp4")
    
    try:
        performance_stats["requests"] += 1
        start_time = time.time()
        
        # Save video
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Extract frames
        frames = extract_frames_smart(temp_video_path, max_frames=6)
        if not frames:
            return PlainTextResponse("Unable to extract frames from video")
        
        # Zero-shot analysis
        response = await query_llava_zero_shot(frames, prompt)
        
        # Update stats
        processing_time = time.time() - start_time
        performance_stats["total_time"] += processing_time
        
        print(f"✅ Zero-shot request completed in {processing_time:.2f}s")
        return PlainTextResponse(response)
        
    except Exception as e:
        print(f"❌ Zero-shot inference error: {e}")
        return PlainTextResponse("Error processing the request")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "llava_loaded": llava_model is not None,
            "clip_loaded": clip_model is not None,
            "segmentation_loaded": segmentation_model is not None
        },
        "capabilities": ["zero_shot_detection", "visual_reasoning", "color_analysis"],
        "stats": performance_stats
    }

@app.get("/")
async def root():
    """Serve frontend"""
    return FileResponse('frontend/index.html')

# Load models on startup
print("🚀 Starting Zero-Shot Visual Understanding Backend")
load_models()
print("✅ Zero-shot backend ready!")
