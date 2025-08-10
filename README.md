# VisionVerseAI – Round 2

# 🚦 VisionVerseAI - High-Performance, Scalable Visual Understanding Chat Assistant

**VisionVerseAI** is an advanced visual understanding chat assistant capable of processing **long-duration video streams** with **low latency**, maintaining **multi-turn conversational context**, and delivering **scalable performance** across deployment environments.

> 📣 **Submitted for Mantra Hackathon 2025 – Round 2: High-Performance & Scalable Visual Understanding Chat Assistant**  
> 🔥 **Team Name**: Phoenix  
>
> 👥 **Team Members**:
> - Gorla Sai Praneeth Reddy  
> - Polisetty Surya Teja  
> - Nabajyoti Chandra Deb
>
> ✅ Round 2 Enhancements:
> - 🚀 Optimized low-latency video processing pipeline  
> - 🌐 Scalable architecture for longer video streams  
> - 💬 Context-aware multi-turn conversations  
> - ⚡ Robustness improvements for real-time/near-real-time responses

---

## 📌 Project Overview

In **Round 2**, our goal is to **extend** VisionVerseAI’s capabilities from short clips to **continuous or long-duration videos**, while ensuring:

- **Low-latency** stream processing  
- **High accuracy** in event recognition & summarization  
- **Multi-turn conversational capabilities** with memory  
- **Scalability** for real-time deployments

Our system is capable of:
- Accepting extended-duration video inputs or streams
- Extracting keyframes & detecting important objects/events efficiently
- Providing **guideline adherence and violation summaries** with timestamps
- Maintaining chat memory for **follow-up queries**
- Deploying on high-performance infrastructure for **scalable throughput**

---

## 🧱 Architecture Diagram

Below is the updated **Round 2** architecture diagram, optimized for high-performance and scalability.

![Architecture Diagram](architecture_diagram.png)

**Key Components:**

1. **Frontend (HTML/CSS/JS)** – Video upload & chat interface  
2. **Backend (FastAPI)** – Handles stream/video processing, chat state, and model inference  
3. **Video Pre-processing Module** – Keyframe sampling & audio extraction with **low-latency optimizations**  
4. **Object & Event Detection** – YOLOv8-based frame analysis with dominant color detection  
5. **Audio Transcription** – Faster-Whisper for quick speech-to-text conversion  
6. **Summarization Engine** – Contextual summaries with guideline adherence/violations  
7. **Multi-turn Chat Handler** – Maintains conversation history  
8. **Scalability Layer** – Parallelized processing & resource-optimized inference

---

## 🧠 Tech Stack Justification

| Component         | Technology             | Justification |
|------------------|-------------------------|----------------|
| **Backend**       | FastAPI + Python        | High-performance async API handling |
| **Frontend**      | HTML, CSS, JavaScript   | Lightweight, responsive user interface |
| **Video Processing** | OpenCV              | Efficient frame sampling & manipulation |
| **Object Detection** | YOLOv8              | High accuracy with GPU acceleration |
| **Audio Processing** | Faster-Whisper      | Low-latency, accurate speech recognition |
| **Chat Handling** | Python logic + Context Memory | Maintains conversational flow |
| **Deployment**    | Uvicorn + Scalable Model Serving | Supports multiple concurrent sessions |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/saipraneeth88/VisionVerseAI-R2.git
cd VisionVerseAI-R2
```
### 2️⃣ Create virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```
### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```
### 4️⃣ Start the server
```
uvicorn backend.main:app --host 0.0.0.0 --port 8080
```
### 5️⃣ Open the web interface
Open index.html from the frontend directory in your browser
OR

If integrated, visit http://localhost:8080 for the full interface.

