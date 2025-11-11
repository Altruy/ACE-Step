from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import time
from acestep.pipeline_ace_step import ACEStepPipeline

app = FastAPI(title="ACEStep Pipeline API (Async Version)")

# ---- In-memory task tracking ----
tasks = {}

# ---- Request/Response Models ----
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


# ---- Helper: Initialize Pipeline ----
def initialize_pipeline(checkpoint_path: str, bf16: bool, torch_compile: bool, device_id: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    return ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
    )


# ---- Background Generation Job ----
def run_generation(task_id: str, input_data: ACEStepInput):
    start_time = time.time()
    try:
        tasks[task_id] = {"status": "running", "message": "Initializing pipeline..."}
        model = initialize_pipeline(
            input_data.checkpoint_path,
            input_data.bf16,
            input_data.torch_compile,
            input_data.device_id
        )

        # Prepare OSS steps
        oss_steps_str = ", ".join(map(str, input_data.oss_steps)) if input_data.oss_steps else ""

        # Prepare output path
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)
        if input_data.output_path:
            output_path = os.path.join(output_dir, os.path.basename(input_data.output_path))
        else:
            output_path = os.path.join(output_dir, f"output_{task_id}.wav")

        tasks[task_id] = {"status": "running", "message": "Generating audio..."}

        # Run generation
        model(
            audio_duration=input_data.audio_duration,
            prompt=input_data.prompt,
            lyrics=input_data.lyrics,
            infer_step=input_data.infer_step,
            guidance_scale=input_data.guidance_scale,
            scheduler_type=input_data.scheduler_type,
            cfg_type=input_data.cfg_type,
            omega_scale=input_data.omega_scale,
            manual_seeds=input_data.actual_seeds,
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

        if not os.path.exists(output_path):
            raise Exception("Output file not found after generation")

        tasks[task_id] = {
            "status": "completed",
            "output_path": output_path,
            "message": f"Generation finished in {time.time() - start_time:.2f}s",
        }

    except Exception as e:
        tasks[task_id] = {"status": "error", "message": str(e)}


# ---- API Endpoints ----
@app.post("/generate")
async def generate_audio(input_data: ACEStepInput, background_tasks: BackgroundTasks):
    """
    Start a long-running generation task asynchronously.
    Returns a task ID immediately to prevent timeouts.
    """
    task_id = uuid.uuid4().hex
    tasks[task_id] = {"status": "pending", "message": "Queued for processing"}
    background_tasks.add_task(run_generation, task_id, input_data)
    return {"task_id": task_id, "status": "started"}


@app.get("/status/{task_id}")
async def check_status(task_id: str):
    """
    Check the current status of a generation task.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


@app.get("/download/{filename}")
async def download_audio(filename: str):
    """
    Download a generated audio file from the outputs directory.
    """
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.abspath(os.path.join("./outputs", filename))
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    media_type = "audio/wav"
    if filename.lower().endswith(".mp3"):
        media_type = "audio/mpeg"
    elif filename.lower().endswith(".ogg"):
        media_type = "audio/ogg"

    return FileResponse(path=file_path, media_type=media_type, filename=filename)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ---- Run Server ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

