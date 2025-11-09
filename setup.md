# 🚀 ACE-Step Setup Guide

Complete setup instructions for ACE-Step Inference API Server on various platforms including local machines, cloud services, and RunPod.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Cloud Setup (RunPod)](#cloud-setup-runpod)
- [Docker Setup](#docker-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Python**: 3.10 or later
- **GPU**: NVIDIA GPU with CUDA support (recommended)
  - Minimum: 8GB VRAM (with CPU offload)
  - Recommended: 16GB+ VRAM
  - Supported: RTX 3090, RTX 4090, A100, A5000, etc.
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space (for model checkpoints)
- **OS**: Linux, macOS, or Windows

### Software Dependencies

- Git
- FFmpeg (for audio processing)
- CUDA Toolkit (for GPU acceleration)
- Python package manager (pip/conda)

---

## Local Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/ace-step/ACE-Step.git
cd ACE-Step
```

### Step 2: Create Virtual Environment

#### Option A: Using Conda (Recommended)

```bash
# Create environment
conda create -n ace_step python=3.10 -y

# Activate environment
conda activate ace_step
```

#### Option B: Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Linux/macOS:
source venv/bin/activate

# On Windows (cmd.exe):
venv\Scripts\activate.bat

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```

### Step 3: Install PyTorch (Windows Only)

If you're on Windows and using an NVIDIA GPU, install PyTorch with CUDA support first:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> **Note**: Adjust `cu126` based on your CUDA version. Check [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your specific setup.

### Step 4: Install ACE-Step Dependencies

```bash
# Install ACE-Step package and dependencies
pip install -e .

# Install additional API dependencies
pip install fastapi uvicorn
```

### Step 5: Download Model Checkpoints

The model will automatically download on first use, or you can specify a custom path:

```bash
# Set custom checkpoint directory (optional)
export CHECKPOINT_PATH="/path/to/checkpoints"

# Default location: ~/.cache/ace-step/checkpoints
```

### Step 6: Start the API Server

```bash
# Basic startup
python infer-api.py

# Or with custom configuration
python -c "
import uvicorn
from infer_api import app
uvicorn.run(app, host='0.0.0.0', port=8000)
"
```

The server will start on `http://0.0.0.0:8000`

### Step 7: Verify Installation

```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy"}

# Access API documentation
# Open browser: http://localhost:8000/docs
```

---

## Cloud Setup (RunPod)

Complete guide for deploying ACE-Step API Server on RunPod with RTX A5000 (24GB VRAM).

### Step 1: Configure RunPod Pod

1. Go to [RunPod Console](https://runpod.io/console/gpu)
2. Click **"Deploy Pod"**
3. Configure the following:

   **Template Image:**
   ```
   runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
   ```

   **Hardware:**
   - GPU: ✅ **RTX A5000** (24 GB VRAM)
   - CPU: Intel Xeon Gold 6342 @ 2.8 GHz (9 vCPUs)
   - RAM: 50 GB
   - Volume Disk: 50 GB (minimum)

   **Storage:**
   - Mount Path: `/workspace`

   **Networking:**
   - ✅ Enable **Global Networking**
   - Expose Ports:
     - HTTP: `8000`
     - TCP (SSH): `22`

   **Container:**
   - Start Command: `/bin/bash`

4. Click **"Deploy"**

### Step 2: Connect to Pod

**Option A: Web Terminal**
- Click **"Web Terminal"** in RunPod console

**Option B: SSH (Optional)**
```bash
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### Step 3: Run Automated Setup Script

```bash
# Navigate to workspace
cd /workspace

# Download setup script
wget https://sandbox.openai.com/mnt/data/runpod_ace_step_setup.sh -O runpod_ace_step_setup.sh

# Make executable
chmod +x runpod_ace_step_setup.sh

# Run setup
./runpod_ace_step_setup.sh
```

**What the script does:**
- ✅ Installs system dependencies (git, ffmpeg, sndfile, tmux)
- ✅ Clones ACE-Step repository
- ✅ Creates Python virtual environment
- ✅ Installs required Python packages
- ✅ Downloads ACE-Step-v1-3.5B model checkpoint
- ✅ Launches FastAPI server in tmux on port 8000

### Step 4: Verify Installation

```bash
# Check model checkpoint
ls -lh /workspace/ace-step-api/checkpoints/ACE-Step-v1-3.5B

# Expected structure:
# - config.json
# - pytorch_model.bin
# - tokenizer.json
# - ... (other model files)

# Check server status
tmux attach -t ace-step

# Look for:
# Uvicorn running on http://0.0.0.0:8000

# Detach from tmux: Ctrl+B, then D
```

### Step 5: Access API Dashboard

**Swagger UI:**
```
https://<your-pod-id>-8000.proxy.runpod.io/docs
```

**Health Check:**
```
https://<your-pod-id>-8000.proxy.runpod.io/health
```

**With Global Networking:**
```
http://<your-public-ip>:8000/docs
```

### Step 6: Manual Server Start (If Needed)

If you need to manually start the server:

```bash
cd /workspace/ace-step-api
source venv/bin/activate

# Start in tmux
tmux new -d -s ace-step "python3 ACE-Step/infer-api.py"

# Or directly
python3 ACE-Step/infer-api.py
```

### Step 7: Auto-Start on Pod Boot

Create startup script:

```bash
# Create startup script
cat > /workspace/ace-step-api/start.sh << 'EOF'
#!/bin/bash
cd /workspace/ace-step-api
source venv/bin/activate
tmux new -d -s ace-step "python3 ACE-Step/infer-api.py"
EOF

# Make executable
chmod +x /workspace/ace-step-api/start.sh
```

**Configure in RunPod UI:**
- Go to Pod Settings
- Set **Container Start Command** to:
  ```bash
  bash /workspace/ace-step-api/start.sh
  ```

### RunPod Performance Optimization

For RTX A5000 (24GB VRAM), use these recommended settings:

| Setting | Recommendation | Reason |
|---------|---------------|--------|
| `audio_duration` | ≤ 90 seconds | Avoid VRAM overflow |
| `bf16` | `true` | Faster and lighter on GPU |
| `infer_step` | 50-60 | Good quality/speed balance |
| `guidance_scale` | 12-16 | Natural sound tone |
| Concurrent requests | 1 at a time | A5000 has 24GB VRAM |

---

## Docker Setup

### Step 1: Build Docker Image

```bash
# Build from Dockerfile
docker build -t ace-step-api:latest .

# Or use docker-compose
docker-compose build
```

### Step 2: Run Container

```bash
# Basic run
docker run -d \
  --name ace-step-api \
  -p 8000:8000 \
  --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/outputs:/workspace/outputs \
  ace-step-api:latest

# Or with docker-compose
docker-compose up -d
```

### Step 3: Verify Container

```bash
# Check logs
docker logs ace-step-api

# Check status
docker ps

# Access API
curl http://localhost:8000/health
```

---

## Verification

### Test Health Endpoint

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{"status": "healthy"}
```

### Test Generation Endpoint

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_path": "/path/to/checkpoints/ACE-Step-v1-3.5B",
    "audio_duration": 30.0,
    "prompt": "pop, upbeat, energetic",
    "lyrics": "[verse]\nTest lyrics\n\n[chorus]\nChorus here",
    "infer_step": 60,
    "guidance_scale": 15.0,
    "scheduler_type": "euler",
    "cfg_type": "apg",
    "omega_scale": 10.0,
    "actual_seeds": [12345],
    "guidance_interval": 0.5,
    "guidance_interval_decay": 0.0,
    "min_guidance_scale": 3.0,
    "use_erg_tag": true,
    "use_erg_lyric": true,
    "use_erg_diffusion": true,
    "oss_steps": []
  }' \
  --output test_output.wav
```

### Check API Documentation

Open in browser:
```
http://localhost:8000/docs
```

This provides interactive Swagger UI for testing all endpoints.

---

## Troubleshooting

### Common Issues and Solutions

#### 1. TorchCodec Missing Error

**Error:**
```
TorchCodec is required for save_with_torchcodec
```

**Solution:**
```bash
pip install torchcodec
```

Or ensure you're using WAV format (default).

#### 2. CUDA Out of Memory

**Error:**
```
CUDA out of memory
```

**Solutions:**
- Reduce `audio_duration` (try 30-60 seconds)
- Reduce `infer_step` (try 27-50)
- Enable CPU offload (when available in API)
- Use `bf16: true` for lower memory usage
- Close other GPU applications

#### 3. Model Checkpoint Not Found

**Error:**
```
Checkpoint not loaded, loading checkpoint...
```

**Solutions:**
```bash
# Check checkpoint path
ls -lh /path/to/checkpoints/ACE-Step-v1-3.5B

# Re-download if needed
# Model will auto-download on first use, or:
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ACE-Step/ACE-Step-v1-3.5B', cache_dir='/path/to/checkpoints')
"
```

#### 4. Port Already in Use

**Error:**
```
Address already in use
```

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
# In infer-api.py, change:
uvicorn.run(app, host="0.0.0.0", port=8001)
```

#### 5. File Permission Errors

**Error:**
```
Permission denied: /path/to/outputs
```

**Solution:**
```bash
# Create and set permissions
mkdir -p ./outputs
chmod 755 ./outputs
```

#### 6. Python Version Issues

**Error:**
```
Python version must be 3.10 or higher
```

**Solution:**
```bash
# Check Python version
python --version

# If < 3.10, install correct version
conda create -n ace_step python=3.10 -y
conda activate ace_step
```

#### 7. Missing Dependencies

**Error:**
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:**
```bash
# Reinstall dependencies
pip install -e .
pip install fastapi uvicorn
```

#### 8. RunPod Server Not Responding

**Issues:**
- Port 8000 not exposed
- Global networking not enabled
- Server crashed

**Solutions:**
```bash
# Check server status
tmux attach -t ace-step

# Restart server
tmux kill-session -t ace-step
cd /workspace/ace-step-api
source venv/bin/activate
tmux new -d -s ace-step "python3 ACE-Step/infer-api.py"

# Verify port exposure in RunPod UI
# Ensure Global Networking is enabled
```

#### 9. Slow Generation

**Possible Causes:**
- GPU not being used
- Too many inference steps
- Slow scheduler

**Solutions:**
```bash
# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU utilization
nvidia-smi

# Use faster settings:
# - infer_step: 27-50
# - scheduler_type: "euler"
# - bf16: true
```

#### 10. Invalid Parameter Errors

**Error:**
```
object of type 'int' has no len()
```

**Solution:**
- Ensure `actual_seeds` is an array: `[12345]` not `12345`
- Ensure `oss_steps` is an array: `[]` not `""`
- Check all required parameters are provided

---

## Quick Reference

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/generate` | POST | Generate music from prompt + lyrics |
| `/download/{filename}` | GET | Download generated audio file |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `./outputs/` | Generated audio files (default) |
| `~/.cache/ace-step/checkpoints/` | Model checkpoints (default) |
| `/workspace/ace-step-api/` | RunPod workspace directory |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU device ID | `0` |
| `CHECKPOINT_PATH` | Model checkpoint directory | `~/.cache/ace-step/checkpoints` |
| `HF_HOME` | HuggingFace cache directory | `~/.cache/huggingface` |

---

## Next Steps

After successful setup:

1. ✅ Test the API with a simple request
2. ✅ Explore the Swagger UI at `/docs`
3. ✅ Read the [API Documentation](./README.md#api-endpoints) for detailed parameter explanations
4. ✅ Check [Examples](./README.md#examples) for sample payloads
5. ✅ Review [Performance Tips](./README.md#performance--hardware) for optimization

---

## Additional Resources

- **Main README**: [README.md](./README.md)
- **Training Guide**: [TRAIN_INSTRUCTION.md](./TRAIN_INSTRUCTION.md)
- **LoRA Guide**: [ZH_RAP_LORA.md](./ZH_RAP_LORA.md)
- **Technical Report**: [arXiv:2506.00045](https://arxiv.org/abs/2506.00045)
- **GitHub Repository**: [ace-step/ACE-Step](https://github.com/ace-step/ACE-Step)
- **HuggingFace Model**: [ACE-Step-v1-3.5B](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)

---

**Last Updated:** 2025-01-09  
**Setup Guide Version:** 1.0
