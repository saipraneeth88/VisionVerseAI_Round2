"""
Evaluation-Focused Visual Understanding Backend
Optimized specifically for evaluation metrics
"""

import os
import shutil
import uuid
import cv2
import torch
import numpy as np
import time
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Optional imports
try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    HAS_LLAVA = True
except ImportError:
    HAS_LLAVA = False

try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

# App setup
app = FastAPI(title="Evaluation-Focused Visual Assistant")
app.mount("/static", StaticFiles(directory="frontend/static", html=True), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

TEMP_FOLDER = "temp_files"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Global models
llava_model = None
llava_processor = None
clip_model = None
clip_preprocess = None
stats = {"requests": 0, "total_time": 0.0}

def load_models():
    """Load models optimized for evaluation"""
    global llava_model, llava_processor, clip_model, clip_preprocess
    
    print("🚀 Loading Evaluation-Optimized Models...")
    
    if HAS_LLAVA:
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
    
    if HAS_CLIP:
        try:
            clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
            print("✅ CLIP loaded")
        except Exception as e:
            print(f"❌ CLIP failed: {e}")

def extract_frames_optimized(video_path: str, max_frames: int = 6) -> List[Image.Image]:
    """Optimized frame extraction for evaluation"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
    except Exception as e:
        print(f"Frame extraction error: {e}")
    
    return frames

def analyze_color_precise(roi: np.ndarray) -> str:
    """Ultra-precise color analysis for evaluation"""
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_val = np.median(hsv[:, :, 0])
        s_val = np.median(hsv[:, :, 1])
        v_val = np.median(hsv[:, :, 2])
        
        # Very precise color classification
        if v_val < 35:
            return "black"
        elif s_val < 20:
            return "white" if v_val > 200 else "gray"
        elif h_val < 8 or h_val > 172:
            return "red"
        elif 8 <= h_val < 22:
            return "orange"
        elif 22 <= h_val < 38:
            return "yellow"
        elif 38 <= h_val < 75:
            return "green"
        elif 75 <= h_val < 130:
            return "blue"
        elif 130 <= h_val < 160:
            return "purple"
        else:
            return "pink"
    except:
        return "unknown"

def create_evaluation_prompt(question: str, frame_analysis: Dict) -> str:
    """Create prompt optimized for specific evaluation categories"""
    question_lower = question.lower()
    
    # Extract analysis info
    colors = frame_analysis.get('colors', [])
    objects = frame_analysis.get('objects', [])
    count = frame_analysis.get('count', 0)
    
    colors_text = ", ".join(colors) if colors else "No colors detected"
    objects_text = ", ".join(objects) if objects else "No objects detected"
    
    # Category-specific optimization
    if any(word in question_lower for word in ['color', 'what color']):
        return f"""You are a color recognition expert. Answer with EXACT color names only.

DETECTED COLORS: {colors_text}

RULES:
- Use only these colors: red, blue, green, yellow, orange, purple, pink, black, white, gray
- Give direct answers (e.g., "red", "blue")
- Ignore lighting effects

Question: {question}
Answer:"""

    elif any(word in question_lower for word in ['count', 'how many']):
        return f"""You are a counting expert. Give EXACT numbers only.

DETECTED OBJECTS: {count} total
OBJECTS: {objects_text}

RULES:
- Count each unique object once
- Give exact numbers (e.g., "3", "5")
- Don't use ranges or approximations

Question: {question}
Answer:"""

    elif any(word in question_lower for word in ['change', 'different', 'move', 'motion']):
        return f"""You are a motion and change detection expert.

OBJECTS: {objects_text}
COLORS: {colors_text}

RULES:
- Describe specific changes or movements
- Be precise about what changed
- Use clear motion descriptions

Question: {question}
Answer:"""

    else:
        return f"""You are a visual analysis expert.

OBJECTS: {objects_text}
COLORS: {colors_text}
COUNT: {count}

RULES:
- Be specific and accurate
- Use clear, direct language
- Focus on observable facts

Question: {question}
Answer:"""

async def analyze_video_for_evaluation(frames: List[Image.Image], question: str) -> str:
    """Analyze video with evaluation-specific optimizations"""
    if not llava_model or not llava_processor:
        return "Model not available"
    
    try:
        # Quick analysis of first frame
        frame_np = np.array(frames[0])
        
        # Simple object detection using color analysis
        colors_found = []
        objects_found = []
        
        # Analyze different regions for colors
        h, w = frame_np.shape[:2]
        regions = [
            frame_np[h//4:3*h//4, w//4:3*w//4],  # Center
            frame_np[:h//2, :w//2],              # Top-left
            frame_np[:h//2, w//2:],              # Top-right
            frame_np[h//2:, :w//2],              # Bottom-left
            frame_np[h//2:, w//2:]               # Bottom-right
        ]
        
        for region in regions:
            if region.size > 0:
                color = analyze_color_precise(region)
                if color != "unknown":
                    colors_found.append(color)
        
        # Remove duplicates and count
        unique_colors = list(set(colors_found))
        
        frame_analysis = {
            'colors': unique_colors,
            'objects': unique_colors,  # Use colors as object proxies
            'count': len(unique_colors)
        }
        
        # Create optimized prompt
        prompt = create_evaluation_prompt(question, frame_analysis)
        
        # Generate response with optimized parameters
        with torch.inference_mode():
            inputs = llava_processor(
                text=prompt,
                images=frames[:2],  # Use fewer frames for speed
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            output = llava_model.generate(
                **inputs,
                max_new_tokens=50,   # Short answers for evaluation
                do_sample=False,     # Deterministic
                temperature=0.0,
                pad_token_id=llava_processor.tokenizer.eos_token_id
            )
            
            response = llava_processor.decode(output[0], skip_special_tokens=True)
            answer = response.split("Answer:")[-1].strip()
            
            # Clean up answer
            if not answer:
                answer = response.split("\n")[-1].strip()
            
            return answer if answer else "Unable to analyze"
    
    except Exception as e:
        print(f"Analysis error: {e}")
        return "Error in analysis"

@app.post("/infer", response_class=PlainTextResponse)
async def infer_evaluation(video: UploadFile = File(...), prompt: str = Form(...)):
    """Evaluation-optimized inference endpoint"""
    temp_video_path = os.path.join(TEMP_FOLDER, f"eval_{uuid.uuid4()}.mp4")
    
    try:
        stats["requests"] += 1
        start_time = time.time()
        
        # Save video
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Extract frames
        frames = extract_frames_optimized(temp_video_path, max_frames=4)
        if not frames:
            return PlainTextResponse("Unable to process video")
        
        # Analyze with evaluation focus
        response = await analyze_video_for_evaluation(frames, prompt)
        
        # Update stats
        processing_time = time.time() - start_time
        stats["total_time"] += processing_time
        
        print(f"✅ Evaluation request completed in {processing_time:.2f}s")
        return PlainTextResponse(response)
        
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return PlainTextResponse("Processing error")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models": {
            "llava_loaded": llava_model is not None,
            "clip_loaded": clip_model is not None
        },
        "optimization": "evaluation_focused",
        "stats": stats
    }

@app.get("/")
async def root():
    return FileResponse('frontend/index.html')

# Load models
print("🚀 Starting Evaluation-Focused Backend")
load_models()
print("✅ Evaluation backend ready!")
