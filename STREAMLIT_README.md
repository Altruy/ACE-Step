# Streamlit Frontend for ACE-Step Music Generator

A user-friendly Streamlit interface for generating music using ACE-Step with AI-powered lyrics and prompt generation.

## Features

- 🎵 **AI-Powered Generation**: Uses GPT to generate lyrics and prompts based on your description
- 🎛️ **Advanced Controls**: Full access to all ACE-Step parameters via expandable advanced settings
- 🎧 **Audio Playback**: Automatic audio playback after generation
- 📥 **Download Support**: Download generated audio files
- ⚙️ **Customizable**: Edit generated lyrics and prompts before generation

## Setup

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API Key**

   You need an OpenAI API key for GPT-powered lyrics and prompt generation. You can set it in one of two ways:

   **Option A: Environment Variable**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

   **Option B: Streamlit Secrets** (Recommended for deployment)
   
   Create a `.streamlit/secrets.toml` file:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

3. **Configure API Endpoint**

   The default API endpoint is set to `https://h4c6smqnf4l7ft-8000.proxy.runpod.net/generate`. You can change this in the sidebar when running the app, or modify the default in `streamlit_app.py`.

## Running the App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. **Describe Your Music**: Enter a description of the music you want (e.g., "A fast-paced rap song about missing someone, with emotional and melodic elements")

2. **Generate Lyrics & Prompt**: Click "✨ Generate Lyrics & Prompt" to use GPT to create structured lyrics and a prompt

3. **Review & Edit**: Review the generated lyrics and prompt. You can edit them directly in the text areas if needed

4. **Adjust Settings** (Optional): Click "⚙️ Advanced Settings" to customize:
   - Core parameters (duration, inference steps, guidance scale, etc.)
   - ERG settings (Entropy Rectifying Guidance)
   - Performance settings (BF16, torch compile)
   - Advanced guidance controls
   - OSS steps

5. **Generate Music**: Click "🎵 Generate Music" to send the request to the API and generate your audio

6. **Play & Download**: The audio will automatically play when ready, and you can download it using the download button

## API Endpoint

The app makes POST requests to the `/generate` endpoint with the following payload structure:

```json
{
  "checkpoint_path": "./checkpoints/ACE-Step-v1-3.5B",
  "bf16": true,
  "torch_compile": false,
  "device_id": 0,
  "output_path": null,
  "audio_duration": 70.0,
  "prompt": "rap, hip hop, trap-pop, fast-paced, emotional, melodic, passionate, high energy",
  "lyrics": "[intro]\n...",
  "infer_step": 60,
  "guidance_scale": 15.0,
  "scheduler_type": "euler",
  "cfg_type": "apg",
  "omega_scale": 10.0,
  "actual_seeds": [135767468],
  "guidance_interval": 0.5,
  "guidance_interval_decay": 0.0,
  "min_guidance_scale": 3.0,
  "use_erg_tag": true,
  "use_erg_lyric": true,
  "use_erg_diffusion": true,
  "oss_steps": [],
  "guidance_scale_text": 0.0,
  "guidance_scale_lyric": 0.0
}
```

## Default Parameters

The app uses the following default parameters (matching the sample payload):

- **Audio Duration**: 70 seconds
- **Inference Steps**: 60
- **Guidance Scale**: 15.0
- **Scheduler Type**: euler
- **CFG Type**: apg
- **Omega Scale**: 10.0
- **Guidance Interval**: 0.5
- **All ERG settings**: Enabled

See `PARAMETER_CONTROL_GUIDE.md` for detailed information about each parameter.

## Troubleshooting

### OpenAI API Key Not Found

If you see an error about the OpenAI API key:
- Make sure you've set `OPENAI_API_KEY` as an environment variable or in Streamlit secrets
- Restart the Streamlit app after setting the key

### API Connection Errors

- Check that the API endpoint URL is correct
- Ensure the API server is running and accessible
- Check network connectivity and firewall settings

### Generation Timeout

- Music generation can take several minutes depending on duration and inference steps
- The default timeout is 600 seconds (10 minutes)
- For longer tracks, consider reducing `audio_duration` or `infer_step`

## Notes

- The app uses GPT-4o-mini for lyrics generation (configurable in code)
- Generated audio files are temporarily stored and can be downloaded
- All parameters can be customized in the Advanced Settings section
- The seed can be changed to generate different variations with the same parameters

