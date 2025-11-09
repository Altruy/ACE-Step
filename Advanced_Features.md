Reviewing the current implementation to draft the README:


[3 tools called]


# ACE-Step Inference API Server

REST API server for the ACE-Step music generation model. Generate music from text prompts and lyrics via HTTP endpoints.

## Table of Contents

- [Overview](#overview)
- [Model Information](#model-information)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Request Parameters](#request-parameters)
- [Response Format](#response-format)
- [Examples](#examples)
- [Performance & Hardware](#performance--hardware)
- [Error Handling](#error-handling)
- [Troubleshooting](#troubleshooting)
- [Advanced Features](#advanced-features)
- [Contributing](#contributing)

---

## Overview

The ACE-Step Inference API Server provides a RESTful interface to the ACE-Step music generation foundation model. It enables:

- Text-to-music generation from prompts and lyrics
- Fast inference (up to 4 minutes of music in ~20 seconds on A100)
- Support for 19 languages
- Multiple music styles and genres
- Lyric alignment with structure tags
- Reproducible generation via seeds

### Key Features

- FastAPI-based REST API
- Direct audio file download
- Detailed timing metrics
- Automatic output file management
- Health check endpoint
- Comprehensive error handling

---

## Model Information

### ACE-Step v1-3.5B

ACE-Step is a music generation foundation model that combines:
- Diffusion-based generation with flow matching
- Deep Compression AutoEncoder (DCAE) for audio compression
- Lightweight linear transformer architecture
- MERT and m-hubert for semantic alignment

### Model Architecture

- Base Model: 3.5B parameters
- Input: Text prompts + lyrics (with structure tags)
- Output: High-quality audio (48kHz sample rate, WAV format)
- Generation Speed: 15-34× real-time on modern GPUs

### Model Checkpoints

- Default: `ACE-Step/ACE-Step-v1-3.5B` (HuggingFace)
- Alternative: ModelScope (China)
- Fine-tuned LoRAs: Available on HuggingFace (e.g., RapMachine)

### Supported Languages

Top 10 well-performing languages:
- English, Chinese, Russian, Spanish, Japanese, German, French, Portuguese, Italian, Korean

Additional supported languages:
- Polish, Turkish, Czech, Dutch, Arabic, Hungarian, Hindi

---

## Installation

### Prerequisites

- Python 3.10 or later
- CUDA-capable GPU (recommended) or CPU/MPS
- 8GB+ VRAM (with CPU offload) or 16GB+ VRAM (without)
- FFmpeg (for some audio formats)

### Step 1: Clone Repository

```bash
git clone https://github.com/ace-step/ACE-Step.git
cd ACE-Step
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n ace_step python=3.10 -y
conda activate ace_step

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install ACE-Step package
pip install -e .

# Install additional API dependencies
pip install fastapi uvicorn
```

### Step 4: Download Model Checkpoints

The model will auto-download on first use, or manually:

```bash
# Set checkpoint path
export CHECKPOINT_PATH="/path/to/checkpoints"

# Model will download to ~/.cache/ace-step/checkpoints by default
```

### Step 5: Verify Installation

```bash
python infer-api.py
```

Server should start on `http://0.0.0.0:8000`

---

## Quick Start

### 1. Start the Server

```bash
python infer-api.py
```

Or with custom host/port:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Test Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy"}
```

### 3. Generate Your First Audio

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_path": "/path/to/checkpoints/ACE-Step-v1-3.5B",
    "audio_duration": 30.0,
    "prompt": "pop, upbeat, energetic",
    "lyrics": "[verse]\nHello world\nThis is a test\n\n[chorus]\nMusic generation\nIs amazing",
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
  --output generated_audio.wav
```

---

## API Endpoints

### POST `/generate`

Generates music from text prompts and lyrics.

**Request Body:** JSON (see [Request Parameters](#request-parameters))

**Response:** Audio file (WAV format) with timing headers

**Response Headers:**
- `X-Output-Path`: Full path to generated file
- `X-Generation-Time`: Total generation time (seconds)
- `X-Init-Time`: Pipeline initialization time
- `X-Prep-Time`: Parameter preparation time
- `X-Generation-Time-Detailed`: Audio generation time

**Example:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d @request.json \
  --output output.wav
```

### GET `/download/{filename}`

Downloads a previously generated audio file from the `./outputs` directory.

**Path Parameters:**
- `filename`: Name of the file to download (e.g., `output_abc123.wav`)

**Response:** Audio file

**Response Headers:**
- `X-Download-Time`: Download processing time
- `X-File-Path`: Full path to the file

**Example:**
```bash
curl "http://localhost:8000/download/output_abc123.wav" \
  --output downloaded_audio.wav
```

**Security:** Prevents directory traversal attacks

### GET `/health`

Health check endpoint.

**Response:**
```json
{"status": "healthy"}
```

---

## Request Parameters

### Required Parameters

| Parameter | Type | Description |
|----------|------|-------------|
| `checkpoint_path` | string | Path to model checkpoint directory |
| `audio_duration` | float | Duration in seconds (use `-1` for random 30-240s) |
| `prompt` | string | Music style/genre description (comma-separated tags) |
| `lyrics` | string | Lyrics with structure tags (`[verse]`, `[chorus]`, etc.) |
| `infer_step` | integer | Number of diffusion steps (27-100, typical: 60) |
| `guidance_scale` | float | How strongly to follow prompt/lyrics (10-20, default: 15.0) |
| `scheduler_type` | string | `"euler"`, `"heun"`, or `"pingpong"` |
| `cfg_type` | string | `"apg"` (recommended), `"cfg"`, or `"cfg_star"` |
| `omega_scale` | float | Flow matching parameter (default: 10.0) |
| `actual_seeds` | array[int] | Random seeds for reproducibility |
| `guidance_interval` | float | Fraction of steps with active guidance (0.0-1.0) |
| `guidance_interval_decay` | float | Guidance decay factor (0.0 = constant) |
| `min_guidance_scale` | float | Minimum guidance when using decay |
| `use_erg_tag` | boolean | Enable ERG for text encoder |
| `use_erg_lyric` | boolean | Enable ERG for lyric encoder |
| `use_erg_diffusion` | boolean | Enable ERG for diffusion model |
| `oss_steps` | array[int] | Custom inference steps (empty = use all) |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bf16` | boolean | `true` | Use bfloat16 precision |
| `torch_compile` | boolean | `false` | Enable torch.compile optimization |
| `device_id` | integer | `0` | GPU device ID |
| `output_path` | string | `null` | Custom output path (auto-generated if null) |
| `guidance_scale_text` | float | `0.0` | Separate text guidance (0.0 = disabled) |
| `guidance_scale_lyric` | float | `0.0` | Separate lyric guidance (0.0 = disabled) |

### Parameter Details

#### `scheduler_type`
- `"euler"`: Fast, recommended
- `"heun"`: Slower, potentially better quality
- `"pingpong"`: Uses SDE for better consistency

#### `cfg_type`
- `"apg"`: Adaptive guidance with momentum (recommended)
- `"cfg"`: Standard classifier-free guidance
- `"cfg_star"`: Variant of CFG

#### ERG (Entropy Rectifying Guidance)
- `use_erg_tag`: Weaker tag condition → more diversity
- `use_erg_lyric`: Weaker lyric condition → more variation
- `use_erg_diffusion`: Increases generation diversity

#### `guidance_interval`
- `0.5`: Guidance active for middle 50% of steps
- `1.0`: Guidance active for all steps
- `0.0`: No guidance (not recommended)

#### `oss_steps` (Optimized Step Sampling)
- `[]`: Use all inference steps
- `[1, 10, 20, 30, 40, 50, 60]`: Use only specified steps

---

## Response Format

### Success Response

**Status Code:** `200 OK`

**Content-Type:** `audio/wav`

**Body:** Binary audio file

**Headers:**
```
X-Output-Path: /path/to/output.wav
X-Generation-Time: 25.43
X-Init-Time: 2.15
X-Prep-Time: 0.05
X-Generation-Time-Detailed: 23.20
```

### Error Response

**Status Code:** `400 Bad Request` or `500 Internal Server Error`

**Content-Type:** `application/json`

**Body:**
```json
{
  "detail": "Error message description"
}
```

---

## Examples

### Example 1: Basic Generation

```json
{
  "checkpoint_path": "/workspace/ace-step-api/checkpoints/ACE-Step-v1-3.5B",
  "audio_duration": 60.0,
  "prompt": "pop, rock, electronic, upbeat, energetic",
  "lyrics": "[verse]\nWalking through the city streets at night\nLights are shining bright\n\n[chorus]\nThis is our time to shine\nTogether we'll make it fine",
  "infer_step": 60,
  "guidance_scale": 15.0,
  "scheduler_type": "euler",
  "cfg_type": "apg",
  "omega_scale": 10.0,
  "actual_seeds": [1234567890],
  "guidance_interval": 0.5,
  "guidance_interval_decay": 0.0,
  "min_guidance_scale": 3.0,
  "use_erg_tag": true,
  "use_erg_lyric": true,
  "use_erg_diffusion": true,
  "oss_steps": []
}
```

### Example 2: Rap Generation

```json
{
  "checkpoint_path": "/workspace/ace-step-api/checkpoints/ACE-Step-v1-3.5B",
  "audio_duration": 70.0,
  "prompt": "rap, hip hop, trap-pop, fast-paced, emotional, melodic, passionate, high energy",
  "lyrics": "[intro]\nYeah, this one's for Meena...\nLet's go.\n\n[verse 1]\nLate nights, city lights, I remember that glow,\nEvery dream got your name in the flow,\n\n[chorus]\nMeena, I miss you, no disguise,\nEvery bar I spit got your vibe inside",
  "infer_step": 60,
  "guidance_scale": 15.0,
  "scheduler_type": "euler",
  "cfg_type": "apg",
  "omega_scale": 10.0,
  "actual_seeds": [135792468],
  "guidance_interval": 0.5,
  "guidance_interval_decay": 0.0,
  "min_guidance_scale": 3.0,
  "use_erg_tag": true,
  "use_erg_lyric": true,
  "use_erg_diffusion": true,
  "oss_steps": []
}
```

### Example 3: High Quality (More Steps)

```json
{
  "checkpoint_path": "/workspace/ace-step-api/checkpoints/ACE-Step-v1-3.5B",
  "audio_duration": 120.0,
  "prompt": "jazz, smooth, relaxing, instrumental",
  "lyrics": "[verse]\nIn the quiet of the evening\nWhen the world slows down",
  "infer_step": 100,
  "guidance_scale": 18.0,
  "scheduler_type": "heun",
  "cfg_type": "apg",
  "omega_scale": 10.0,
  "actual_seeds": [987654321],
  "guidance_interval": 0.6,
  "guidance_interval_decay": 0.2,
  "min_guidance_scale": 5.0,
  "use_erg_tag": true,
  "use_erg_lyric": false,
  "use_erg_diffusion": true,
  "oss_steps": []
}
```

### Example 4: Fast Generation (Fewer Steps)

```json
{
  "checkpoint_path": "/workspace/ace-step-api/checkpoints/ACE-Step-v1-3.5B",
  "audio_duration": 30.0,
  "prompt": "electronic, dance, upbeat",
  "lyrics": "[verse]\nQuick test\nFast generation",
  "infer_step": 27,
  "guidance_scale": 15.0,
  "scheduler_type": "euler",
  "cfg_type": "apg",
  "omega_scale": 10.0,
  "actual_seeds": [11111],
  "guidance_interval": 0.5,
  "guidance_interval_decay": 0.0,
  "min_guidance_scale": 3.0,
  "use_erg_tag": true,
  "use_erg_lyric": true,
  "use_erg_diffusion": true,
  "oss_steps": []
}
```

### Python Client Example

```python
import requests
import json

url = "http://localhost:8000/generate"

payload = {
    "checkpoint_path": "/workspace/ace-step-api/checkpoints/ACE-Step-v1-3.5B",
    "audio_duration": 60.0,
    "prompt": "pop, rock, electronic",
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
    "use_erg_tag": True,
    "use_erg_lyric": True,
    "use_erg_diffusion": True,
    "oss_steps": []
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    # Save audio file
    with open("output.wav", "wb") as f:
        f.write(response.content)
    
    # Get timing information
    print(f"Generation time: {response.headers.get('X-Generation-Time')}s")
    print(f"Output path: {response.headers.get('X-Output-Path')}")
else:
    print(f"Error: {response.json()}")
```

---

## Performance & Hardware

### Hardware Requirements

| Device | RTF (27 steps) | Time for 1 min (27 steps) | RTF (60 steps) | Time for 1 min (60 steps) |
|--------|----------------|---------------------------|----------------|----------------------------|
| NVIDIA RTX 4090 | 34.48× | 1.74s | 15.63× | 3.84s |
| NVIDIA A100 | 27.27× | 2.20s | 12.27× | 4.89s |
| NVIDIA RTX 3090 | 12.76× | 4.70s | 6.48× | 9.26s |
| MacBook M2 Max | 2.27× | 26.43s | 1.03× | 58.25s |

RTF = Real-Time Factor (higher is faster)

### Memory Requirements

- Minimum: 8GB VRAM (with CPU offload)
- Recommended: 16GB+ VRAM (without CPU offload)
- Model Size: ~3.5B parameters (~14GB in float32, ~7GB in bfloat16)

### Performance Tips

1. Use `bf16: true` for faster inference
2. Use `torch_compile: true` for additional speedup (requires triton on Windows)
3. Lower `infer_step` for faster generation (27 vs 60)
4. Use `scheduler_type: "euler"` for fastest generation
5. Enable CPU offload if VRAM is limited

---

## Error Handling

### Common Errors

#### 1. Model Not Found
```
Error: Checkpoint not loaded, loading checkpoint...
```
Solution: Ensure `checkpoint_path` is correct and model files exist

#### 2. CUDA Out of Memory
```
Error: CUDA out of memory
```
Solution:
- Enable CPU offload (when available in API)
- Reduce `audio_duration`
- Use lower precision (`bf16: true`)

#### 3. Invalid Parameters
```
Error: object of type 'int' has no len()
```
Solution: Ensure all array parameters are properly formatted

#### 4. File Not Found
```
Error: [Errno 2] No such file or directory
```
Solution: Ensure output directory exists and is writable

#### 5. TorchCodec Error
```
Error: TorchCodec is required for save_with_torchcodec
```
Solution: Install torchcodec or ensure format is "wav"

### Error Response Format

All errors return JSON:
```json
{
  "detail": "Error description"
}
```

Status codes:
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (file not found)
- `500`: Internal Server Error (generation failed)

---

## Troubleshooting

### Server Won't Start

1. Check Python version: `python --version` (need 3.10+)
2. Verify dependencies: `pip list | grep fastapi`
3. Check port availability: `netstat -an | grep 8000`

### Slow Generation

1. Check GPU utilization: `nvidia-smi`
2. Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
3. Try fewer inference steps: `infer_step: 27`
4. Use faster scheduler: `scheduler_type: "euler"`

### Poor Audio Quality

1. Increase inference steps: `infer_step: 100`
2. Use better scheduler: `scheduler_type: "heun"`
3. Adjust guidance scale: `guidance_scale: 18.0`
4. Check lyrics format (structure tags)

### Memory Issues

1. Enable CPU offload (when available)
2. Reduce audio duration
3. Use bfloat16: `bf16: true`
4. Close other GPU applications

### Model Download Issues

1. Check internet connection
2. Verify HuggingFace access
3. Set `HF_HOME` environment variable
4. Manually download to checkpoint path

---

## Advanced Features

### Future Enhancements

The following features are planned but not yet implemented in the API:

1. LoRA Support: Fine-tuned model adapters
2. Audio2Audio: Style transfer from reference audio
3. Lyric Editing: Modify lyrics while preserving melody
4. Repainting: Regenerate specific time segments
5. Extending: Add audio to beginning/end
6. Batch Generation: Generate multiple files in parallel
7. Format Options: MP3, OGG output formats
8. CPU Offload: Memory optimization
9. Overlapped Decode: Faster long audio generation

See the [Feature Exploration Plan](#feature-exploration-plan) for implementation details.

---

## Feature Exploration Plan

### Phase 1: LoRA/Fine-tuned Models

**Status:** Not yet implemented

**Purpose:** Use fine-tuned model adapters for specialized styles

**Implementation:**
- Add `lora_name_or_path` parameter (HuggingFace ID or local path)
- Add `lora_weight` parameter (0.0-1.0)
- Example: `"ACE-Step/ACE-Step-v1-chinese-rap-LoRA"`

**Use Cases:**
- RapMachine for Chinese rap
- Style-specific generation
- Language improvements

### Phase 2: Memory Optimization

**Status:** Not yet implemented

**Features:**
- CPU Offload: Reduce VRAM to ~8GB
- Overlapped Decode: Faster long audio generation

### Phase 3: Audio2Audio

**Status:** Not yet implemented

**Purpose:** Transform reference audio to match new prompt/lyrics

**Parameters:**
- `audio2audio_enable`: Enable feature
- `ref_audio_input`: Path to reference audio
- `ref_audio_strength`: Preservation strength (0.0-1.0)

### Phase 4: Editing Features

**Status:** Not yet implemented

**Features:**
- Lyric Editing: Modify lyrics preserving melody
- Repainting: Regenerate time segments
- Extending: Add audio to beginning/end

---

## Contributing

Contributions welcome. Areas:

1. Additional API endpoints
2. LoRA support implementation
3. Audio2Audio feature
4. Performance optimizations
5. Documentation improvements
6. Example scripts and tools

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](./LICENSE) for details.

---

## References

- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step)
- [ACE-Step HuggingFace](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)
- [Technical Report](https://arxiv.org/abs/2506.00045)
- [Project Website](https://ace-step.github.io/)

---

## Support

- GitHub Issues: [Report Issues](https://github.com/ace-step/ACE-Step/issues)
- Discord: [Join Community](https://discord.gg/PeWDxrkdj7)
- Documentation: See main [README.md](./README.md)

---

**Last Updated:** 2025-01-09  
**API Version:** 1.0  
**Model Version:** ACE-Step-v1-3.5B

---

This README covers the inference API server. For model training, fine-tuning, and advanced features, see the main [README.md](./README.md) and [TRAIN_INSTRUCTION.md](./TRAIN_INSTRUCTION.md).