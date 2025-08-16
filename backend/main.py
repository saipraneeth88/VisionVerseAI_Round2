import os
import shutil
import uuid
import cv2
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# --- App Initialization & Frontend Serving ---
app = FastAPI(title="Definitive Vuencode Assistant")
# This serves your CSS, JS, and image files from the 'frontend' folder
app.mount("/static", StaticFiles(directory="frontend/static", html=True), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Configuration ---
TEMP_FOLDER = "temp_files"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# --- AI Model Loading (Stable 7B Model) ---
print("Loading LLaVA 7B model into GPU memory...")
try:
    MODEL_ID = "llava-hf/llava-1.5-7b-hf"
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    llava_processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("LLaVA 7B model loaded successfully.")
except Exception as e:
    print(f"FATAL: Error loading LLaVA model: {e}")
    llava_model = None

# --- VLM Query Function with Chain-of-Thought Reasoning Prompt ---
def query_vlm(image_frames: list, question: str) -> str:
    if llava_model is None:
        return "Error: LLaVA model is not available."

    # This prompt is specifically designed to force the AI to reason
    prompt = f"""USER: You are a world-class AI reasoning engine. Your task is to answer a question about a video by analyzing a sequence of image frames.

#**Chain of Thought Instructions**#
1.  **Analyze the User's Question:** Identify the core reasoning task (e.g., color identification, memory/state change, motion/physics, counting).
2.  **Analyze the Frames Sequentially:** Look at the image frames to understand the objects, their attributes, and their positions.
3.  **Detect Changes (Memory):** Compare the frames to identify what has changed. Note objects that appear, disappear, or move.
4.  **Formulate a Conclusion:** Based on the changes and states you observed in the images, form a logical conclusion that directly answers the user's question.
5.  **Provide the Final Answer:** State your conclusion concisely and directly. If the frames do not contain enough information to answer, state that the data is insufficient.

<image_placeholder>

#**User's Question**#
{question}

ASSISTANT:"""

    try:
        # Use torch.inference_mode() for a significant speed boost
        with torch.inference_mode():
            inputs = llava_processor(text=prompt, images=image_frames, return_tensors="pt").to("cuda", torch.float16)
            output = llava_model.generate(**inputs, max_new_tokens=250)
            response_text = llava_processor.decode(output[0], skip_special_tokens=True)
        return response_text.split("ASSISTANT:")[-1].strip()
    except Exception as e:
        print(f"Error during VLM inference: {e}")
        return "An error occurred while generating the AI response."

# --- API Endpoint ---
@app.post("/infer", response_class=PlainTextResponse)
async def infer_from_video(video: UploadFile = File(...), prompt: str = Form(...)):
    temp_video_path = os.path.join(TEMP_FOLDER, f"temp_{uuid.uuid4()}.mp4")
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    frames = []
    try:
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Sample exactly 10 frames for a good balance of speed and context
        frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        
        if not frames:
            raise HTTPException(status_code=500, detail="Could not extract any frames from the video.")

        # This entire process is now very fast, no caching needed for evaluation
        ai_response = query_vlm(image_frames=frames, question=prompt)
        return PlainTextResponse(content=ai_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

# --- Endpoint to serve the frontend HTML ---
@app.get("/")
async def read_index():
    """Serves the main frontend HTML file."""
    return FileResponse('frontend/index.html')
