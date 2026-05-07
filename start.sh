#!/bin/bash
# Start FastAPI in the background
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Wait for API to be ready
sleep 5

# Start Streamlit on the port HF Spaces expects (7860)
streamlit run app.py \
  --server.port=7860 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false