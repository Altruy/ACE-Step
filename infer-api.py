from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.data_sampler import DataSampler
import uuid
import time

app = FastAPI(title="ACEStep Pipeline API")

class ACEStepInput(BaseModel):
    checkpoint_path: str
    bf16: bool = True
    torch_compile: bool = False
    device_id: int = 0
    output_path: Optional[str] = None
    audio_duration: float
    prompt: str
    lyrics: str
    infer_step: int
    guidance_scale: float
    scheduler_type: str
    cfg_type: str
    omega_scale: float
    actual_seeds: List[int]
    guidance_interval: float
    guidance_interval_decay: float
    min_guidance_scale: float
    use_erg_tag: bool
    use_erg_lyric: bool
    use_erg_diffusion: bool
    oss_steps: List[int]
    guidance_scale_text: float = 0.0
    guidance_scale_lyric: float = 0.0

class ACEStepOutput(BaseModel):
    status: str
    output_path: Optional[str]
    message: str

def initialize_pipeline(checkpoint_path: str, bf16: bool, torch_compile: bool, device_id: int) -> ACEStepPipeline:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    return ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
    )

@app.post("/generate")
async def generate_audio(input_data: ACEStepInput):
    start_time = time.time()
    print(f"[GENERATE] Request received at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize pipeline
        init_start = time.time()
        print(f"[GENERATE] Initializing pipeline...")
        model_demo = initialize_pipeline(
            input_data.checkpoint_path,
            input_data.bf16,
            input_data.torch_compile,
            input_data.device_id
        )
        init_time = time.time() - init_start
        print(f"[GENERATE] Pipeline initialized in {init_time:.2f} seconds")

        # Prepare parameters
        prep_start = time.time()
        # Pass manual_seeds as list (pipeline accepts list, string, or int)
        manual_seeds = input_data.actual_seeds if input_data.actual_seeds else None
        
        # Handle oss_steps - convert to string (empty string for empty list)
        # The pipeline will convert empty string to empty list
        if input_data.oss_steps and len(input_data.oss_steps) > 0:
            oss_steps_str = ", ".join(map(str, input_data.oss_steps))
        else:
            oss_steps_str = ""  # Empty string, not None

        # Generate output path if not provided
        # Ensure we always have a directory in the path
        if input_data.output_path:
            output_path = input_data.output_path
            # If it's just a filename without directory, add ./outputs/ directory
            if os.path.dirname(output_path) == "":
                output_dir = "./outputs"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, output_path)
            else:
                # Ensure the directory exists
                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)
        else:
            # Generate default output path in ./outputs/ directory
            output_dir = "./outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"output_{uuid.uuid4().hex}.wav")
        prep_time = time.time() - prep_start
        print(f"[GENERATE] Parameters prepared in {prep_time:.2f} seconds")
        print(f"[GENERATE] Output path: {output_path}")

        # Run pipeline with explicit keyword arguments to avoid parameter order issues
        generation_start = time.time()
        print(f"[GENERATE] Starting audio generation...")
        model_demo(
            audio_duration=input_data.audio_duration,
            prompt=input_data.prompt,
            lyrics=input_data.lyrics,
            infer_step=input_data.infer_step,
            guidance_scale=input_data.guidance_scale,
            scheduler_type=input_data.scheduler_type,
            cfg_type=input_data.cfg_type,
            omega_scale=input_data.omega_scale,
            manual_seeds=manual_seeds,
            guidance_interval=input_data.guidance_interval,
            guidance_interval_decay=input_data.guidance_interval_decay,
            min_guidance_scale=input_data.min_guidance_scale,
            use_erg_tag=input_data.use_erg_tag,
            use_erg_lyric=input_data.use_erg_lyric,
            use_erg_diffusion=input_data.use_erg_diffusion,
            oss_steps=oss_steps_str,
            guidance_scale_text=input_data.guidance_scale_text,
            guidance_scale_lyric=input_data.guidance_scale_lyric,
            save_path=output_path
        )
        generation_time = time.time() - generation_start
        print(f"[GENERATE] Audio generation completed in {generation_time:.2f} seconds")
        
        # Verify file exists before returning
        verify_start = time.time()
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail=f"Generated file not found: {output_path}")
        verify_time = time.time() - verify_start
        
        total_time = time.time() - start_time
        print(f"[GENERATE] Total endpoint execution time: {total_time:.2f} seconds")
        print(f"[GENERATE] Breakdown - Init: {init_time:.2f}s, Prep: {prep_time:.2f}s, Generation: {generation_time:.2f}s, Verify: {verify_time:.2f}s")
        
        # Return the audio file
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=os.path.basename(output_path),
            headers={
                "X-Output-Path": output_path,
                "X-Generation-Time": f"{total_time:.2f}",
                "X-Init-Time": f"{init_time:.2f}",
                "X-Prep-Time": f"{prep_time:.2f}",
                "X-Generation-Time-Detailed": f"{generation_time:.2f}"
            }
        )

    except Exception as e:
        total_time = time.time() - start_time
        print(f"[GENERATE] Error occurred after {total_time:.2f} seconds: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

@app.get("/download/{filename}")
async def download_audio(filename: str):
    """
    Download an audio file from the ./outputs directory by filename.
    
    Args:
        filename: The name of the file to download (e.g., "output_abc123.wav")
    
    Returns:
        The audio file if found, otherwise a 404 error.
    """
    start_time = time.time()
    print(f"[DOWNLOAD] Request received at {time.strftime('%Y-%m-%d %H:%M:%S')} for filename: {filename}")
    
    try:
        # Security: prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Construct the full path
        path_start = time.time()
        output_dir = "./outputs"
        file_path = os.path.join(output_dir, filename)
        
        # Normalize and convert to absolute paths to prevent directory traversal
        file_path = os.path.abspath(os.path.normpath(file_path))
        output_dir = os.path.abspath(os.path.normpath(output_dir))
        path_time = time.time() - path_start
        
        # Ensure the file is within the outputs directory
        if not file_path.startswith(output_dir):
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Check if file exists
        check_start = time.time()
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        # Check if it's a file (not a directory)
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=400, detail=f"Path is not a file: {filename}")
        check_time = time.time() - check_start
        
        # Determine media type based on extension
        media_type = "audio/wav"
        if filename.lower().endswith(".mp3"):
            media_type = "audio/mpeg"
        elif filename.lower().endswith(".ogg"):
            media_type = "audio/ogg"
        
        total_time = time.time() - start_time
        print(f"[DOWNLOAD] File found: {file_path}")
        print(f"[DOWNLOAD] Total endpoint execution time: {total_time:.2f} seconds")
        print(f"[DOWNLOAD] Breakdown - Path resolution: {path_time:.2f}s, File check: {check_time:.2f}s")
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename,
            headers={
                "X-Download-Time": f"{total_time:.2f}",
                "X-File-Path": file_path
            }
        )
    except HTTPException:
        total_time = time.time() - start_time
        print(f"[DOWNLOAD] Request failed after {total_time:.2f} seconds")
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
