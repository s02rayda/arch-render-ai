# ============================================================
# Arch-Render AI - Phase 3: FastAPI Backend
# ============================================================
from fastapi.responses import Response, JSONResponse # Add Response here
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
    version="1.0.0"
)

# Allow React frontend to talk to this API
# This is called CORS - Cross Origin Resource Sharing
# Without this, the browser blocks frontend → backend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# STEP 2: SETUP PATHS & DEVICE
# ============================================================

# All generated images will be saved here
# All generated images will be saved here
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# STEP 3: LOAD ALL AI MODELS AT STARTUP
# We load models ONCE when the server starts, not on every
# request. This is critical — loading takes 30-60 seconds
# but generating takes only 30-60 seconds per image.
# ============================================================

print("\n📦 Loading AI models — please wait...")

# --- Load MiDaS ---
print("Loading MiDaS...")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
print("✅ MiDaS ready!")

# --- Load ControlNet ---
print("Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
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

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)
pipe.to(device)
pipe.enable_model_cpu_offload()
print("✅ Realistic Vision ready!")

print("\n🚀 All models loaded! Server is ready.\n")

# ============================================================
# STEP 4: HELPER FUNCTIONS
# ============================================================

def extract_depth_map(image: Image.Image) -> Image.Image:
    """Extract depth map from image using MiDaS"""
    img_np = np.array(image.convert("RGB"))
    input_batch = transform(img_np).to(device)
    
    with torch.no_grad():
        depth_prediction = midas(input_batch)
        depth_prediction = torch.nn.functional.interpolate(
            depth_prediction.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_np = depth_prediction.cpu().numpy()
    depth_np = cv2.normalize(
        depth_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )
    depth_rgb = cv2.cvtColor(depth_np, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(depth_rgb)


def run_pipeline(
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_scale: float,
    seed: int
) -> tuple:
    """Run the full AI pipeline"""
    
    # Resize to SD native resolution
    image_resized = image.resize((512, 512))
    
    # Extract depth
    depth_map = extract_depth_map(image_resized)
    
    # Set seed
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Run diffusion
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=depth_map,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_scale,
        generator=generator,
    )
    
    return output.images[0], depth_map

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
    num_inference_steps: int = Form(default=30),
    guidance_scale: float = Form(default=7.5),
    controlnet_scale: float = Form(default=1.0),
    seed: int = Form(default=42)
):
    # Read image
    contents = await file.read()
    input_image = Image.open(BytesIO(contents)).convert("RGB")
    
    # Run pipeline
    rendered_image, _ = run_pipeline(
        input_image, prompt, "cartoon, blurry, low quality", 
        num_inference_steps, guidance_scale, controlnet_scale, seed
    )
    
    # Return image bytes directly
    img_byte_arr = BytesIO()
    rendered_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")
    """
    Main generation endpoint.
    Accepts an image + parameters, returns rendered image.
    
    This is what the React frontend will call when the
    user clicks the Generate button.
    """
    
    print(f"\n📨 New request received!")
    print(f"   Prompt: {prompt[:50]}...")
    print(f"   Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    
    # --- Validate file type ---
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a PNG, JPG, or WebP image."
        )
    
    try:
        # --- Read uploaded image ---
        contents = await file.read()
        input_image = Image.open(BytesIO(contents)).convert("RGB")
        print(f"   Image size: {input_image.size}")
        
        # --- Run the AI pipeline ---
        start_time = time.time()
        print("🎨 Running AI pipeline...")
        
        rendered_image, depth_map = run_pipeline(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_scale=controlnet_scale,
            seed=seed
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Generation complete in {elapsed:.1f} seconds!")
        
        # --- Save outputs with unique ID ---
        # We use UUID so multiple users don't overwrite each other's files
        job_id = str(uuid.uuid4())[:8]
        
        render_path = OUTPUTS_DIR / f"render_{job_id}.png"
        depth_path = OUTPUTS_DIR / f"depth_{job_id}.png"
        input_path = OUTPUTS_DIR / f"input_{job_id}.png"
        
        rendered_image.save(render_path)
        depth_map.save(depth_path)
        input_image.save(input_path)
        
        print(f"💾 Saved outputs with job ID: {job_id}")
        
        # --- Return response ---
        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "render_url": f"/outputs/{job_id}/render",
            "depth_url": f"/outputs/{job_id}/depth",
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


@app.get("/outputs/{job_id}/depth")
def get_depth(job_id: str):
    """Serve the depth map"""
    path = OUTPUTS_DIR / f"depth_{job_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Depth map not found")
    return FileResponse(path, media_type="image/png")


@app.get("/outputs/{job_id}/input")
def get_input(job_id: str):
    """Serve the original input image"""
    path = OUTPUTS_DIR / f"input_{job_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Input not found")
    return FileResponse(path, media_type="image/png")


# ============================================================
# STEP 6: RUN THE SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Must be False — we can't reload GPU models
    )