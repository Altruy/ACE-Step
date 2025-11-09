#!/bin/bash
set -e

echo "🚀 Setting up ACE-Step API on RunPod..."

# === 1️⃣ System Setup ===
apt-get update -y
apt-get install -y git tmux ffmpeg libsndfile1 wget

# === 2️⃣ Clone Repo ===
cd /workspace
if [ ! -d "ace-step-api" ]; then
  echo "📦 Cloning ACE-Step API repository..."
  git clone https://github.com/yourusername/ace-step-api.git
fi
cd ace-step-api

# === 3️⃣ Virtual Environment ===
if [ ! -d "venv" ]; then
  echo "🐍 Creating Python virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip

# === 4️⃣ Dependencies ===
echo "📦 Installing dependencies..."
pip install torch torchvision torchaudio
pip install fastapi uvicorn pydantic soundfile torchcodec huggingface_hub
pip install git+https://github.com/pytorch/torchcodec.git || true

# === 5️⃣ Download Model Checkpoint ===
echo "⬇️ Downloading ACE-Step model checkpoint..."
mkdir -p /workspace/ace-step-api/checkpoints

python3 << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    "ACE-Step/ACE-Step-v1-3.5B",
    cache_dir="/workspace/ace-step-api/checkpoints",
    local_dir="/workspace/ace-step-api/checkpoints/ACE-Step-v1-3.5B",
    local_dir_use_symlinks=False
)
EOF

# === 6️⃣ Create output directory ===
mkdir -p /workspace/ace-step-api/generated

# === 7️⃣ Launch FastAPI Server in tmux ===
echo "🧠 Launching FastAPI server in tmux session..."
tmux kill-session -t ace-step || true
tmux new -d -s ace-step "cd /workspace/ace-step-api && source venv/bin/activate && python3 ACE-Step/infer-api.py --port 8888"

echo ""
echo "✅ ACE-Step API setup complete!"
echo "🌐 Access it at: https://<your-pod-id>-8888.proxy.runpod.io/docs"
echo "📁 Generated audio files will be saved in: /workspace/ace-step-api/generated/"
echo ""
echo "Use 'tmux attach -t ace-step' to view logs or stop the server."