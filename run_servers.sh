#!/bin/bash

# --- Step 1: Start vLLM server for LLaVA model ---
echo "Starting vLLM server for LLaVA..."
python -m vllm.entrypoints.openai.api_server \
    --model llava-hf/llava-1.5-13b-hf \   # Model to serve
    --host 0.0.0.0 \                      # Listen on all network interfaces
    --port 8000 \                         # vLLM API port
    &
VLLM_PID=$!                               # Store vLLM server PID for later termination
echo "vLLM server started with PID $VLLM_PID."

# --- Step 2: Wait for vLLM server to fully initialize ---
sleep 30

# --- Step 3: Start FastAPI backend ---
echo "Starting FastAPI backend server..."
uvicorn backend.main:app --host 0.0.0.0 --port 8080

# --- Step 4: Stop vLLM server when FastAPI stops ---
kill $VLLM_PID
echo "Servers stopped."
