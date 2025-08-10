import os
import shutil
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import google.generativeai as genai

# ==============================================================
# 🚀 Hackathon Project: Definitive Vuencode Assistant
# Purpose: Accept a video from the user, extract key frames,
# send them to Google's Gemini API, and return a reasoning-based answer.
# ==============================================================

# =========================
# 1️⃣ Google API Setup
# =========================
try:
    # Read Google API key from environment variables (secure way to handle secrets)
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    # Configure the Google Gemini API client
    genai.configure(api_key=api_key)
    print("✅ Google API Key configured successfully from environment variable.")
except Exception as e:
    print(f"❌ FATAL: Could not configure Google API Key. Error: {e}")

# =========================
# 2️⃣ FastAPI Initialization
# =========================
app = FastAPI(title="Definitive Vuencode Assistant")

# Serve static frontend files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="frontend/static", html=True), name="static")

# Enable CORS for cross-origin requests (important for frontend-backend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins for demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary folder to store uploaded videos
TEMP_FOLDER = "temp_files"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# =========================
# 3️⃣ Gemini API Query Function
# =========================
def query_gemini_api(image_frames: list, question: str) -> str:
    """
    Sends extracted video frames + user question to Gemini API for analysis.
    The prompt instructs Gemini to perform sequential reasoning across frames.
    """
    try:
        print("🧠 Initializing Gemini 1.5 Flash model...")
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Reasoning-focused prompt for video question answering
        prompt = f"""You are a world-class AI reasoning engine. Your task is to answer a question about a video by analyzing a sequence of image frames.
        
Instructions:
1. Analyze the User's Question.
2. Look at frames sequentially to understand objects & positions.
3. Compare frames to detect changes over time (memory).
4. Draw logical conclusions.
5. Provide a concise final answer.
If information is insufficient, say so.

User Question: {question}"""

        # The API expects prompt text + frame images
        content = [prompt] + image_frames
        
        print("📡 Sending request to Google Gemini API...")
        response = model.generate_content(content)
        
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Error during Gemini API call: {e}")
        return f"An error occurred while communicating with the AI model: {e}"

# =========================
# 4️⃣ Video Inference Endpoint
# =========================
@app.post("/infer", response_class=PlainTextResponse)
async def infer_from_video(video: UploadFile = File(...), prompt: str = Form(...)):
    """
    API endpoint to:
    1. Receive a video + user question.
    2. Extract 12 evenly spaced frames.
    3. Send them to Gemini API for reasoning.
    4. Return AI-generated answer as plain text.
    """
    # Save uploaded video temporarily
    temp_video_path = os.path.join(TEMP_FOLDER, f"temp_{uuid.uuid4()}.mp4")
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    frames = []
    try:
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Select 12 evenly spaced frames for better scene coverage
        frame_indices = np.linspace(0, total_frames - 1, 12, dtype=int)

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert OpenCV BGR format to PIL RGB format
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        
        if not frames:
            raise HTTPException(status_code=500, detail="No frames extracted from video.")

        # Query Gemini API with frames and user question
        ai_response = query_gemini_api(image_frames=frames, question=prompt)
        return PlainTextResponse(content=ai_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temporary video file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

# =========================
# 5️⃣ Serve Frontend
# =========================
@app.get("/")
async def read_index():
    """Serves the main frontend HTML file."""
    return FileResponse('frontend/index.html')
