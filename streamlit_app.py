import streamlit as st
import requests
import os
import tempfile
import random
import time
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()   

# Page configuration
st.set_page_config(
    page_title="Sonic Suppliment Generator",
    page_icon="🎵",
    layout="wide"
)

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# API endpoint
API_URL = "https://h4c6smqnf4l7ft-8000.proxy.runpod.net/generate"

# Default parameters based on the sample payload
DEFAULT_PARAMS = {
    "checkpoint_path": "./checkpoints/ACE-Step-v1-3.5B",
    "bf16": True,
    "torch_compile": False,
    "device_id": 0,
    "audio_duration": -1.0,
    "infer_step": 60,
    "guidance_scale": 15.0,
    "scheduler_type": "euler",
    "cfg_type": "apg",
    "omega_scale": 10.0,
    "actual_seeds": [random.randint(0, 2**32-1)],
    "guidance_interval": 0.5,
    "guidance_interval_decay": 0.0,
    "min_guidance_scale": 3.0,
    "use_erg_tag": True,
    "use_erg_lyric": True,
    "use_erg_diffusion": True,
    "oss_steps": [],
    "guidance_scale_text": 0.0,
    "guidance_scale_lyric": 0.0
}

def generate_with_gpt(user_prompt: str) -> tuple[str, str]:
    """Generate lyrics and prompt using GPT based on user description."""
    client = get_openai_client()
    if not client:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in secrets or environment variables.")
        return None, None
    
    system_prompt = """You are a music generation assistant. Based on the user's description of the music they want, generate:
1. A PROMPT: Comma-separated tags describing the style, mood, genre, tempo, and instrumentation (e.g., "rap, hip hop, trap-pop, fast-paced, emotional, melodic, passionate, high energy")
2. LYRICS: Structured song lyrics with sections like [intro], [verse 1], [chorus], [verse 2], [outro]. Each line should be 8-12 syllables for rhythmic coherence.

Format your response as JSON with two keys: "prompt" and "lyrics".
Example format:
{
  "prompt": "rap, hip hop, trap-pop, fast-paced, emotional, melodic, passionate, high energy",
  "lyrics": "[intro]\\nYeah, this one's for Me...\\nLet's go.\\n\\n[verse 1]\\nLate nights, city lights, I remember that glow,\\nEvery dream got your name in the flow,\\nHeartbeat drum when I think of your face,\\nTryna move on but you can't be replaced.\\n\\n[chorus]\\nMeena, I miss you, no disguise,\\nEvery bar I spit got your vibe inside,\\nWe were fire, lightning, rhythm, rhyme,\\nStill chasing that spark, one more time.\\n\\n[verse 2]\\nSkyline dreams and the sound of your laugh,\\nWhole world froze when I looked at the past,\\nEvery lyric I write, got a piece of you,\\nEvery line that I drop, yeah it's real and true.\\n\\n[chorus]\\nI miss you, no disguise,\\nEvery bar I spit got your vibe inside,\\nWe were fire, lightning, rhythm, rhyme,\\nStill chasing that spark, one more time.\\n\\n[outro]\\nYeah, this one's for Me,\\nForever in the beat, forever in the dream-a,\\nTill we meet again, I'ma keep that flame,\\nLove and music — same name, same game."
}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate music prompt and lyrics for: {user_prompt}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.8
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("prompt", ""), result.get("lyrics", "")
    except Exception as e:
        st.error(f"Error generating with GPT: {str(e)}")
        return None, None

def submit_generation_task(payload: dict) -> str:
    """Submit generation task and return task_id."""
    try:
        print(f"[DEBUG] Submitting generation task to API: {API_URL}")
        print(f"[DEBUG] Payload keys: {list(payload.keys())}")
        print(f"[DEBUG] Audio duration: {payload.get('audio_duration')}")
        print(f"[DEBUG] Prompt length: {len(payload.get('prompt', ''))}")
        print(f"[DEBUG] Lyrics length: {len(payload.get('lyrics', ''))}")
        
        response = requests.post(API_URL, json=payload, timeout=30)
        print(f"[DEBUG] Response status code: {response.status_code}")
        response.raise_for_status()
        
        result = response.json()
        print(f"[DEBUG] Task submission response: {result}")
        
        task_id = result.get("task_id")
        if not task_id:
            print("[ERROR] No task_id in response")
            st.error("Failed to get task_id from API response")
            return None
        
        print(f"[DEBUG] Task ID: {task_id}")
        return task_id
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request exception: {str(e)}")
        st.error(f"Error submitting generation task: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[ERROR] Response status: {e.response.status_code}")
            print(f"[ERROR] Response text: {e.response.text[:500]}")
            st.error(f"Response: {e.response.text}")
        return None

def poll_task_status(task_id: str, status_placeholder=None) -> dict:
    """Poll task status until completed."""
    status_url = f"{API_URL.replace('/generate', '')}/status/{task_id}"
    print(f"[DEBUG] Polling status from: {status_url}")
    
    max_polls = 300  # 5 minutes max (1 second intervals)
    poll_count = 0
    
    while poll_count < max_polls:
        try:
            response = requests.get(status_url, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            status = result.get("status")
            print(f"[DEBUG] Poll {poll_count + 1}: Status = {status}")
            
            if status_placeholder:
                status_placeholder.write(f"Status: {status}... (Poll {poll_count + 1})")
            
            if status == "completed":
                print(f"[DEBUG] Task completed!")
                print(f"[DEBUG] Result: {result}")
                return result
            elif status == "failed" or status == "error":
                error_msg = result.get("message", "Unknown error")
                print(f"[ERROR] Task failed: {error_msg}")
                st.error(f"Generation failed: {error_msg}")
                return None
            
            # Wait before next poll
            time.sleep(15)
            poll_count += 1
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Error polling status: {str(e)}")
            if poll_count == 0:  # Only show error on first poll
                st.error(f"Error checking task status: {str(e)}")
            time.sleep(2)  # Wait longer on error
            poll_count += 1
    
    print(f"[ERROR] Timeout waiting for task completion after {max_polls} polls")
    st.error("Timeout waiting for generation to complete")
    return None

def download_audio_file(filename: str) -> bytes:
    """Download audio file by filename."""
    download_url = f"{API_URL.replace('/generate', '')}/download/{filename}"
    print(f"[DEBUG] Downloading audio from: {download_url}")
    
    try:
        response = requests.get(download_url, timeout=60)
        print(f"[DEBUG] Download response status: {response.status_code}")
        print(f"[DEBUG] Download response headers: {dict(response.headers)}")
        response.raise_for_status()
        
        audio_data = response.content
        print(f"[DEBUG] Downloaded audio: {len(audio_data)} bytes")
        
        if not audio_data:
            print("[ERROR] Downloaded content is empty!")
            return None
        
        if len(audio_data) < 1000:
            print(f"[WARNING] Downloaded content is very small ({len(audio_data)} bytes)")
            print(f"[DEBUG] Content preview: {audio_data[:500]}")
        
        return audio_data
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Error downloading audio: {str(e)}")
        st.error(f"Error downloading audio file: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[ERROR] Response status: {e.response.status_code}")
            print(f"[ERROR] Response text: {e.response.text[:500]}")
        return None

# Main UI
st.title("🎵 Sonic Suppliment Generator")
st.markdown("Generate music from your description using AI-powered lyrics and prompt generation")

# User input section
st.header("📝 Describe Your Music")
user_description = st.text_area(
    "What kind of music do you want?",
    placeholder="e.g., A fast-paced rap song about missing someone, with emotional and melodic elements",
    height=100
)

# Default prompt and lyrics
DEFAULT_PROMPT = "rap, hip hop, trap-pop, fast-paced, emotional, melodic, passionate, high energy"
DEFAULT_LYRICS = """[intro]
Yeah, let's go.
This is a new track.

[verse 1]
Writing lyrics for the beat,
Making music that's complete,
Every line flows with the rhythm,
Creating sounds that make you listen.

[chorus]
Music generation, that's the name,
Creating tracks, that's the game,
With AI power, we create,
Songs that resonate.

[verse 2]
From the prompt to the final mix,
Every element we fix,
Lyrics and beats in harmony,
Creating music for you and me.

[chorus]
Music generation, that's the name,
Creating tracks, that's the game,
With AI power, we create,
Songs that resonate.

[outro]
That's the track, hope you like it,
Music made with AI, we did it."""

# Button row for generating or using defaults
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("✨ Generate Lyrics & Prompt", type="primary", use_container_width=True):
        if not user_description:
            st.warning("Please enter a description of the music you want.")
        else:
            with st.spinner("Generating lyrics and prompt with GPT..."):
                prompt, lyrics = generate_with_gpt(user_description)
                if prompt and lyrics:
                    st.session_state['generated_prompt'] = prompt
                    st.session_state['generated_lyrics'] = lyrics
                    st.success("Lyrics and prompt generated successfully!")

with col_btn2:
    if st.button("📝 Use Custom Prompt & Lyrics", use_container_width=True):
        st.session_state['generated_prompt'] = DEFAULT_PROMPT
        st.session_state['generated_lyrics'] = DEFAULT_LYRICS


# Display generated content
if 'generated_prompt' in st.session_state:
    st.header("📋 Generated Content")
    
    # Instructions expander
    with st.expander("ℹ️ How to Format Prompt & Lyrics", expanded=False):
        col_inst1, col_inst2 = st.columns(2)
        
        with col_inst1:
            st.markdown("### 📝 Prompt Formatting")
            st.markdown("""
            **Best Practices:**
            - Use **comma-separated tags** (not full sentences)
            - Order matters: place most important first
            - Use 5-10 descriptors for best results
            - Mix genre, mood, and instrumentation
            
            **Examples:**
            - ✅ `"rap, hip hop, trap-pop, fast-paced, emotional, melodic"`
            - ✅ `"jazz, smooth, relaxing, saxophone, piano, late night"`
            - ❌ `"I want a pop rock electronic song that is upbeat"`
            
            **Categories to include:**
            - **Genre**: pop, rock, jazz, electronic, hip hop
            - **Mood**: upbeat, melancholic, energetic, relaxing
            - **Instruments**: piano, guitar, drums, strings, brass
            - **Tempo**: fast-paced, slow, moderate, 130 bpm
            - **Style**: cinematic, lo-fi, orchestral, acoustic
            """)
        
        with col_inst2:
            st.markdown("### 🎤 Lyrics Formatting")
            st.markdown("""
            **Structure Tags:**
            - `[intro]` - Opening section
            - `[verse 1]`, `[verse 2]` - Main verses
            - `[chorus]` - Repeating chorus
            - `[bridge]` - Transitional section
            - `[outro]` - Closing section
            
            **Rhythm Guidelines:**
            - **8-12 syllables per line** for best rhythm
            - Use line breaks to indicate phrasing
            - Empty lines create pauses/breaths
            - Keep lines consistent in length
            
            **Multilingual Support:**
            - Supports 19 languages (auto-detected)
            - English, Chinese, Spanish, Japanese, Korean, etc.
            - Language detected per line automatically
            """)
    
    col2, col1 = st.columns(2)
    
    with col1:
        st.subheader("Prompt")
        with st.expander("💡 Prompt Tips", expanded=False):
            st.markdown("""
            **Format:** Comma-separated tags describing style, mood, genre, tempo, and instrumentation.
            
            **Example:**
            ```
            rap, hip hop, trap-pop, fast-paced, emotional, melodic, passionate, high energy
            ```
            
            **Avoid:** Full sentences or descriptions. Use short, descriptive tags instead.
            """)
        prompt = st.text_area("Prompt", value=st.session_state['generated_prompt'], height=100, help="Comma-separated tags describing the music style, mood, genre, tempo, and instrumentation")
    
    with col2:
        st.subheader("Lyrics")
        with st.expander("💡 Lyrics Tips", expanded=False):
            st.markdown("""
            **Format:** Structured lyrics with section tags and 8-12 syllables per line.
            
            **Example:**
            ```
            [verse 1]
            Late nights, city lights, I remember that glow,
            Every dream got your name in the flow,
            
            [chorus]
            Music generation, that's the name,
            Creating tracks, that's the game,
            ```
            
            **Tips:**
            - Use structure tags: [intro], [verse], [chorus], [outro]
            - Keep lines 8-12 syllables for rhythm
            - Empty lines create natural pauses
            """)
        lyrics = st.text_area("Lyrics", value=st.session_state['generated_lyrics'], height=300, help="Structured lyrics with section tags. Use 8-12 syllables per line for best rhythm.")
    
    # Initialize parameters with defaults
    checkpoint_path = DEFAULT_PARAMS["checkpoint_path"]
    audio_duration = float(DEFAULT_PARAMS["audio_duration"])
    infer_step = DEFAULT_PARAMS["infer_step"]
    guidance_scale = DEFAULT_PARAMS["guidance_scale"]
    scheduler_type = DEFAULT_PARAMS["scheduler_type"]
    cfg_type = DEFAULT_PARAMS["cfg_type"]
    omega_scale = DEFAULT_PARAMS["omega_scale"]
    guidance_interval = DEFAULT_PARAMS["guidance_interval"]
    guidance_interval_decay = DEFAULT_PARAMS["guidance_interval_decay"]
    min_guidance_scale = DEFAULT_PARAMS["min_guidance_scale"]
    device_id = DEFAULT_PARAMS["device_id"]
    seed_input = st.session_state.get('seed_input', DEFAULT_PARAMS["actual_seeds"][0])
    use_erg_tag = DEFAULT_PARAMS["use_erg_tag"]
    use_erg_lyric = DEFAULT_PARAMS["use_erg_lyric"]
    use_erg_diffusion = DEFAULT_PARAMS["use_erg_diffusion"]
    bf16 = DEFAULT_PARAMS["bf16"]
    torch_compile = DEFAULT_PARAMS["torch_compile"]
    guidance_scale_text = DEFAULT_PARAMS["guidance_scale_text"]
    guidance_scale_lyric = DEFAULT_PARAMS["guidance_scale_lyric"]
    oss_steps = DEFAULT_PARAMS["oss_steps"]
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.subheader("Core Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            checkpoint_path = st.text_input("Checkpoint Path", value=checkpoint_path)
            # Allow -1.0 for random duration (30-240s)
            audio_duration_value = float(audio_duration) if audio_duration is not None else -1.0
            audio_duration = st.number_input(
                "Audio Duration (seconds)", 
                min_value=-1.0, 
                max_value=240.0, 
                value=audio_duration_value, 
                step=5.0, 
                help="-1 means random duration (30-240s)"
            )
            infer_step = st.number_input("Inference Steps", min_value=20, max_value=120, value=infer_step, step=1)
            guidance_scale = st.number_input("Guidance Scale", min_value=1.0, max_value=25.0, value=guidance_scale, step=0.5)
        
        with col2:
            scheduler_options = ["euler", "heun", "pingpong"]
            scheduler_index = scheduler_options.index(scheduler_type) if scheduler_type in scheduler_options else 0
            scheduler_type = st.selectbox("Scheduler Type", scheduler_options, index=scheduler_index)
            cfg_options = ["apg", "cfg", "cfg_star"]
            cfg_index = cfg_options.index(cfg_type) if cfg_type in cfg_options else 0
            cfg_type = st.selectbox("CFG Type", cfg_options, index=cfg_index)
            omega_scale = st.number_input("Omega Scale", min_value=8.0, max_value=12.0, value=omega_scale, step=0.1)
            guidance_interval = st.number_input("Guidance Interval", min_value=0.0, max_value=1.0, value=guidance_interval, step=0.1)
        
        with col3:
            guidance_interval_decay = st.number_input("Guidance Interval Decay", min_value=0.0, max_value=1.0, value=guidance_interval_decay, step=0.1)
            min_guidance_scale = st.number_input("Min Guidance Scale", min_value=0.0, max_value=10.0, value=min_guidance_scale, step=0.5)
            device_id = st.number_input("Device ID", min_value=0, max_value=7, value=device_id, step=1)
            seed_col1, seed_col2 = st.columns([3, 1])
            with seed_col1:
                seed_input = st.number_input("Seed", min_value=0, max_value=2**32-1, value=seed_input, step=1, key="seed_input")
            with seed_col2:
                if st.button("🎲", help="Randomize seed", key="randomize_seed"):
                    new_seed = random.randint(0, 2**32-1)
                    st.session_state['seed_input'] = new_seed
                    st.rerun()
        
        st.subheader("ERG (Entropy Rectifying Guidance)")
        col1, col2, col3 = st.columns(3)
        with col1:
            use_erg_tag = st.checkbox("Use ERG Tag", value=use_erg_tag)
        with col2:
            use_erg_lyric = st.checkbox("Use ERG Lyric", value=use_erg_lyric)
        with col3:
            use_erg_diffusion = st.checkbox("Use ERG Diffusion", value=use_erg_diffusion)
        
        st.subheader("Performance Settings")
        col1, col2 = st.columns(2)
        with col1:
            bf16 = st.checkbox("BF16", value=bf16, help="Use bfloat16 for faster inference")
        with col2:
            torch_compile = st.checkbox("Torch Compile", value=torch_compile, help="Compile model for additional speedup")
        
        st.subheader("Advanced Guidance")
        col1, col2 = st.columns(2)
        with col1:
            guidance_scale_text = st.number_input("Guidance Scale Text", min_value=0.0, max_value=25.0, value=guidance_scale_text, step=0.5)
        with col2:
            guidance_scale_lyric = st.number_input("Guidance Scale Lyric", min_value=0.0, max_value=25.0, value=guidance_scale_lyric, step=0.5)
        
        st.subheader("OSS Steps (Optional)")
        oss_steps_str = ",".join(map(str, oss_steps)) if oss_steps else ""
        oss_steps_input = st.text_input("OSS Steps (comma-separated)", value=oss_steps_str, help="Leave empty to use all steps. Example: 1,5,10,20,30,40,50,60")
        if oss_steps_input.strip():
            try:
                oss_steps = [int(x.strip()) for x in oss_steps_input.split(",") if x.strip()]
            except:
                oss_steps = []
        else:
            oss_steps = []
    
    # Generate audio button
    if st.button("🎵 Generate Music", type="primary", use_container_width=True):
        if not prompt or not lyrics:
            st.warning("Please generate lyrics and prompt first.")
        else:
            # Prepare payload
            payload = {
                "checkpoint_path": checkpoint_path,
                "bf16": bf16,
                "torch_compile": torch_compile,
                "device_id": device_id,
                "output_path": None,  # Let API generate the path
                "audio_duration": audio_duration,
                "prompt": prompt,
                "lyrics": lyrics,
                "infer_step": infer_step,
                "guidance_scale": guidance_scale,
                "scheduler_type": scheduler_type,
                "cfg_type": cfg_type,
                "omega_scale": omega_scale,
                "actual_seeds": [seed_input],
                "guidance_interval": guidance_interval,
                "guidance_interval_decay": guidance_interval_decay,
                "min_guidance_scale": min_guidance_scale,
                "use_erg_tag": use_erg_tag,
                "use_erg_lyric": use_erg_lyric,
                "use_erg_diffusion": use_erg_diffusion,
                "oss_steps": oss_steps,
                "guidance_scale_text": guidance_scale_text,
                "guidance_scale_lyric": guidance_scale_lyric
            }
            
            print(f"[DEBUG] Starting music generation...")
            print(f"[DEBUG] Payload audio_duration: {audio_duration}")
            
            # Submit generation task
            task_id = submit_generation_task(payload)
            
            if task_id:
                # Create status placeholder
                status_placeholder = st.empty()
                status_placeholder.info("Task submitted, waiting for generation to start...")
                
                # Poll for completion
                with st.spinner("Generating music... This may take a few minutes."):
                    result = poll_task_status(task_id, status_placeholder)
                
                if result:
                    output_path = result.get("output_path")
                    generation_time = result.get("message", "Unknown")
                    
                    print(f"[DEBUG] Generation completed!")
                    print(f"[DEBUG] Output path: {output_path}")
                    print(f"[DEBUG] Generation time: {generation_time}")
                    
                    # Extract filename from output_path
                    if output_path:
                        filename = output_path.split("/")[-1]
                        print(f"[DEBUG] Extracted filename: {filename}")
                        
                        # Download the audio file
                        status_placeholder.info("Downloading audio file...")
                        audio_data = download_audio_file(filename)
                        
                        if audio_data:
                            print(f"[DEBUG] Received audio data: {len(audio_data)} bytes")
                            
                            # Save to temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                tmp_file.write(audio_data)
                                tmp_file_path = tmp_file.name
                                print(f"[DEBUG] Saved audio to: {tmp_file_path}")
                                print(f"[DEBUG] File size: {os.path.getsize(tmp_file_path)} bytes")
                            
                            # Verify file was written correctly
                            if os.path.exists(tmp_file_path) and os.path.getsize(tmp_file_path) > 0:
                                st.session_state['audio_file'] = tmp_file_path
                                st.session_state['generation_time'] = generation_time
                                st.session_state['filename'] = filename
                                
                                status_placeholder.empty()
                                st.success(f"Music generated successfully!")
                                
                                # Show generation time
                                st.info(f"⏱️ {generation_time}")
                                
                                # Show debug info
                                with st.expander("🔍 Debug Info", expanded=False):
                                    st.write(f"**Task ID:** {task_id}")
                                    st.write(f"**Output path:** {output_path}")
                                    st.write(f"**Filename:** {filename}")
                                    st.write(f"**File path:** {tmp_file_path}")
                                    st.write(f"**File size:** {os.path.getsize(tmp_file_path):,} bytes")
                                    st.write(f"**Content type:** WAV audio")
                                    if len(audio_data) < 1000:
                                        st.warning("⚠️ Audio file is very small, might be corrupted or empty")
                                
                                # Auto-play audio
                                try:
                                    st.audio(tmp_file_path, format="audio/wav", autoplay=True)
                                    print(f"[DEBUG] Audio player displayed")
                                except Exception as e:
                                    print(f"[ERROR] Failed to display audio: {str(e)}")
                                    st.error(f"Failed to play audio: {str(e)}")
                                
                                # Download button
                                try:
                                    with open(tmp_file_path, "rb") as f:
                                        audio_bytes = f.read()
                                        st.download_button(
                                            label="📥 Download Audio",
                                            data=audio_bytes,
                                            file_name=filename,
                                            mime="audio/wav"
                                        )
                                    print(f"[DEBUG] Download button created")
                                except Exception as e:
                                    print(f"[ERROR] Failed to create download button: {str(e)}")
                                    st.error(f"Failed to create download button: {str(e)}")
                            else:
                                print(f"[ERROR] File was not created or is empty")
                                st.error("Failed to save audio file or file is empty")
                        else:
                            print(f"[ERROR] Failed to download audio file")
                            st.error("Failed to download audio file from server")
                    else:
                        print(f"[ERROR] No output_path in result")
                        st.error("No output path in completion result")
                else:
                    status_placeholder.empty()
                    st.error("Generation task did not complete successfully")
            else:
                st.error("Failed to submit generation task")

# Display previous audio if available
if 'audio_file' in st.session_state and os.path.exists(st.session_state['audio_file']):
    st.header("🎧 Generated Audio")
    if 'generation_time' in st.session_state:
        st.info(f"⏱️ {st.session_state['generation_time']}")
    st.audio(st.session_state['audio_file'], format="audio/wav")
    if 'filename' in st.session_state:
        with open(st.session_state['audio_file'], "rb") as f:
            st.download_button(
                label="📥 Download Previously Generated Audio",
                data=f.read(),
                file_name=st.session_state['filename'],
                mime="audio/wav"
            )




