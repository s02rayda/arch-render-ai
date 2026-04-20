# ============================================================
# Arch-Render AI - Phase 3: FastAPI Backend (Canny Precision Mode)
# ============================================================
import os
os.environ["DIFFUSERS_DISABLE_XFORMERS"] = "1"

import torch
import numpy as np
from PIL import Image
import cv2
import uuid
import time
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)

# ============================================================
# STEP 1: INITIALIZE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Arch-Render AI",
    description="Local AI architectural rendering pipeline",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# STEP 2: SETUP PATHS & DEVICE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# STEP 3: LOAD ALL AI MODELS AT STARTUP
# ============================================================

print("\n📦 Loading AI models — please wait...")

# --- Load ControlNet (CANNY) ---
print("Loading ControlNet (Canny)...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16
)
print("✅ ControlNet ready!")

# --- Load Realistic Vision ---
print("Loading Realistic Vision...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(device)
pipe.enable_model_cpu_offload()
print("✅ Realistic Vision ready!")

print("\n🚀 All models loaded! Server is ready.\n")

# ============================================================
# STEP 4: HELPER FUNCTIONS
# ============================================================

def extract_canny_edges(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """Extract edges from image using OpenCV Canny Edge Detection."""
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def run_pipeline(
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_scale: float,
    low_threshold: int,
    high_threshold: int,
    seed: int
) -> tuple:
    """Run the full AI pipeline"""
    image_resized = image.resize((512, 512))
    edge_map = extract_canny_edges(image_resized, low_threshold, high_threshold)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=edge_map,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_scale,
        generator=generator,
    )
    
    return output.images[0], edge_map

# ============================================================
# STEP 5: API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Arch-Render AI is running!",
        "gpu": torch.cuda.get_device_name(0),
        "vram_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    }

@app.get("/health")
def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "device": device,
        "gpu": torch.cuda.get_device_name(0),
        "vram_used": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
        "vram_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
        "models_loaded": True
    }

@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(default="cartoon, blurry, low quality, deformed, unrealistic"),
    num_inference_steps: int = Form(default=30),
    guidance_scale: float = Form(default=7.5),
    controlnet_scale: float = Form(default=1.0),
    low_threshold: int = Form(default=100),
    high_threshold: int = Form(default=200),
    seed: int = Form(default=42)
):
    """Main generation endpoint."""
    print(f"\n📨 New request received!")
    print(f"   Prompt: {prompt[:50]}...")
    
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    try:
        contents = await file.read()
        input_image = Image.open(BytesIO(contents)).convert("RGB")
        
        start_time = time.time()
        print("🎨 Running AI pipeline (Canny Mode)...")
        
        rendered_image, edge_map = run_pipeline(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_scale=controlnet_scale,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            seed=seed
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Generation complete in {elapsed:.1f} seconds!")
        
        job_id = str(uuid.uuid4())[:8]
        
        render_path = OUTPUTS_DIR / f"render_{job_id}.png"
        edge_path = OUTPUTS_DIR / f"edge_{job_id}.png"
        input_path = OUTPUTS_DIR / f"input_{job_id}.png"
        
        rendered_image.save(render_path)
        edge_map.save(edge_path)
        input_image.save(input_path)
        
        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "render_url": f"/outputs/{job_id}/render",
            "edge_url": f"/outputs/{job_id}/edge",
            "input_url": f"/outputs/{job_id}/input",
            "generation_time": f"{elapsed:.1f}s",
            "vram_used": f"{torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB"
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outputs/{job_id}/render")
def get_render(job_id: str):
    """Serve the rendered image"""
    path = OUTPUTS_DIR / f"render_{job_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Render not found")
    return FileResponse(path, media_type="image/png")

@app.get("/outputs/{job_id}/edge")
def get_edge(job_id: str):
    """Serve the edge map"""
    path = OUTPUTS_DIR / f"edge_{job_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Edge map not found")
    return FileResponse(path, media_type="image/png")

@app.get("/outputs/{job_id}/input")
def get_input(job_id: str):
    """Serve the original input image"""
    path = OUTPUTS_DIR / f"input_{job_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Input not found")
    return FileResponse(path, media_type="image/png")

#=============================================================
@app.post("/generate-text-to-image")
async def generate_text_to_image(
    prompt: str = Form(...),
    negative_prompt: str = Form(default="cartoon, blurry, low quality, deformed, unrealistic"),
    num_inference_steps: int = Form(default=30),
    guidance_scale: float = Form(default=7.5),
    seed: int = Form(default=42)
):
    """
    Creative Mode: Pure Text-to-Image.
    We pass a dummy image but set the conditioning scale to 0.0.
    """
    try:
        # Create a dummy 512x512 image (required by the pipeline structure)
        dummy_image = Image.new("RGB", (512, 512), color=(0, 0, 0))
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # 💡 The magic happens here: controlnet_conditioning_scale=0.0
        # This tells the pipeline to ignore the dummy_image completely.
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=dummy_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=0.0, 
            generator=generator,
        )
        
        rendered_image = output.images[0]
        job_id = str(uuid.uuid4())[:8]
        render_path = OUTPUTS_DIR / f"render_{job_id}.png"
        rendered_image.save(render_path)
        
        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "render_url": f"/outputs/{job_id}/render"
        })
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# STEP 6: RUN THE SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )