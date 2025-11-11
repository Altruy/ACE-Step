#!/bin/bash
# Run script for Streamlit + Uvicorn with venv inside tmux (RunPod compatible)

set -e  # stop on errors

# Path to your virtual environment
VENV_PATH="venv"

# Kill old tmux sessions if they exist
tmux kill-session -t streamlit 2>/dev/null || true
tmux kill-session -t api 2>/dev/null || true

# Start Streamlit session (with venv activation)
tmux new-session -d -s streamlit "source venv/bin/activate && \
streamlit run streamlit_app.py \
  --server.port 5000 \
  --server.address 0.0.0.0 \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.enableWebsocketCompression=false"

echo "✅ Streamlit running in tmux session: streamlit (port 5000)"

# Start Uvicorn API session (with venv activation)
tmux new-session -d -s api "source venv/bin/activate && \
uvicorn infer-api:app --host 0.0.0.0 --port 8000"

echo "✅ Uvicorn API running in tmux session: api (port 8000)"

# Display helpful info
echo ""
echo "🚀 Both services started successfully!"
echo "📺 Streamlit: https://<your-instance-id>-5000.proxy.runpod.net"
echo "📡 API Docs:  https://<your-instance-id>-8000.proxy.runpod.net/docs"
echo ""
echo "Use these commands to view logs:"
echo "  tmux attach -t streamlit"
echo "  tmux attach -t api"
echo ""
echo "To detach: press Ctrl+B, then D"