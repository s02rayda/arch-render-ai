import os
os.environ["DIFFUSERS_DISABLE_XFORMERS"]="1"
import torch
import numpy as np
from PIL import Image
import cv2
import requests
from io import BytesIO

# Diffusers imports
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)

# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# --- Your prompt goes here ---
PROMPT = "a photorealistic modern living room, interior design, \
natural lighting, hardwood floors, luxury furniture, \
architectural visualization, 8k, highly detailed"

NEGATIVE_PROMPT = "cartoon, illustration, painting, blurry, \
dark, ugly, deformed, low quality, unrealistic"

# ============================================================
# STEP 2: LOAD MIDAS (Depth Extractor)
# 💡 NVIDIA CERT CONNECTION: MiDaS is a vision transformer
# that understands spatial relationships in a flat image.
# It outputs a depth map — a grayscale image where white
# pixels = close to camera, black pixels = far away.
# This depth map becomes our "conditioning signal".
# ============================================================

print("\n📦 Loading MiDaS depth model...")

midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

print("✅ MiDaS loaded!")

# ============================================================
# STEP 3: LOAD CONTROLNET
# 💡 NVIDIA CERT CONNECTION: ControlNet adds extra
# "conditioning layers" to the U-Net. It takes our depth
# map and injects it into the U-Net's cross-attention layers
# at every denoising step, forcing the output image to
# respect our room's geometry. This is CONDITIONING in action.
# ============================================================

print("\n📦 Loading ControlNet Depth model...")
print("⏳ First run will download ~1.5GB - please wait...")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16  # float16 saves VRAM
)

print("✅ ControlNet loaded!")

# ============================================================
# STEP 4: LOAD REALISTIC VISION (The Main SD Model)
# 💡 NVIDIA CERT CONNECTION: This IS the U-Net architecture
# you studied. It's a Stable Diffusion 1.5 based model
# fine-tuned on photorealistic images. The U-Net runs
# num_inference_steps times, each step removing a little
# noise from the latent space until a clean image emerges.
# ============================================================

print("\n📦 Loading Realistic Vision model...")
print("⏳ First run will download ~4GB - please wait...")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None  # Disabled for architectural use
)

# ============================================================
# STEP 5: MEMORY OPTIMIZATIONS FOR 8GB VRAM
# These are critical for your RTX 5060!
# ============================================================

# UniPC scheduler - faster than default, same quality
# 💡 NVIDIA CERT CONNECTION: This is the NOISE SCHEDULER.
# It defines the sequence of noise levels (timesteps) the
# U-Net works through. UniPC can reach good quality in
# just 20 steps instead of 50, saving us time and memory.
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)

# Move to GPU
pipe.to(device)

# Enable memory optimizations
pipe.enable_model_cpu_offload()  # Moves unused models to CPU RAM

print("✅ Realistic Vision loaded!")

# ============================================================
# STEP 6: DEPTH EXTRACTION FUNCTION
# ============================================================

def extract_depth_map(image: Image.Image) -> Image.Image:
    """
    Takes a PIL Image, runs MiDaS on it, returns a depth map.
    
    💡 NVIDIA CERT CONNECTION: This is where we create our
    CONDITIONING SIGNAL. The depth map captures the latent
    geometric structure of the scene — essentially encoding
    spatial information that will guide the diffusion process.
    """
    print("\n🔍 Extracting depth map with MiDaS...")
    
    # Convert PIL to numpy for MiDaS
    img_np = np.array(image.convert("RGB"))
    
    # Apply MiDaS transforms
    input_batch = transform(img_np).to(device)
    
    # Run MiDaS inference
    with torch.no_grad():
        depth_prediction = midas(input_batch)
        depth_prediction = torch.nn.functional.interpolate(
            depth_prediction.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert to numpy and normalize to 0-255
    depth_np = depth_prediction.cpu().numpy()
    depth_np = cv2.normalize(
        depth_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )
    
    # Convert to RGB (ControlNet expects 3-channel input)
    depth_rgb = cv2.cvtColor(depth_np, cv2.COLOR_GRAY2RGB)
    depth_image = Image.fromarray(depth_rgb)
    
    print("✅ Depth map extracted!")
    return depth_image

# ============================================================
# STEP 7: GENERATION FUNCTION
# ============================================================

def generate_render(
    input_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    seed: int = 42
) -> tuple:
    """
    Full pipeline: image → depth map → photorealistic render
    
    💡 NVIDIA CERT CONNECTION:
    - num_inference_steps: How many denoising steps the U-Net
      takes. Think of it as how many times it refines the image.
      More steps = better quality but slower. 25 is our sweet spot.
    
    - guidance_scale: Controls how strongly the U-Net follows
      your text prompt. Mathematically it's the weight on the
      "conditional" vs "unconditional" score in classifier-free
      guidance. 7.5 = balanced. Higher = more prompt-faithful
      but less natural looking.
    
    - controlnet_conditioning_scale: How strongly the depth map
      influences the generation. 0.8 = strong geometry control
      while still allowing creative freedom.
    """
    
    # Resize input to 512x512 (SD 1.5 native resolution)
    input_resized = input_image.resize((512, 512))
    
    # Extract depth map
    depth_map = extract_depth_map(input_resized)
    
    # Set seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"\n🎨 Generating render...")
    print(f"   Steps: {num_inference_steps}")
    print(f"   Guidance Scale: {guidance_scale}")
    print(f"   ControlNet Scale: {controlnet_conditioning_scale}")
    print(f"   Seed: {seed}")
    
    # 💡 THIS IS THE DIFFUSION PROCESS RUNNING ON YOUR GPU!
    # The U-Net denoises a latent tensor num_inference_steps
    # times, guided by your text prompt AND the depth map.
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=depth_map,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
    )
    
    generated_image = output.images[0]
    print("✅ Render complete!")
    
    return generated_image, depth_map

# ============================================================
# STEP 8: RUN THE PIPELINE
# ============================================================

if __name__ == "__main__":
    # 1. Put the path to your image here
    image_path = "input1.png" 
    
    print(f"\n📸 Loading your sketch: {image_path}...")
    
    # 2. Open the image
    input_image = Image.open(image_path).convert("RGB")
    print("✅ Sketch loaded!")
    
    # 3. Run the pipeline
    generated_image, depth_map = generate_render(
        input_image=input_image,
        prompt="photorealistic interior, gray brick wall on left, \
        large L-shaped white leather sofa with brown pillows, \
        two white armchairs with brown pillows, round dark coffee table, \
        three framed artworks on wall, pendant lamp, dark hardwood floor, \
        warm lighting, architectural visualization, 8k, highly detailed",
        negative_prompt="different layout, cartoon, blurry, \
        low quality, extra rooms, outdoor, people",
        num_inference_steps=30,
        guidance_scale=8.5,        # ← increased to follow prompt more strictly
        controlnet_conditioning_scale=1.0,  # ← keep at max
        seed=123                   # ← change seed for variation
    )
    
    # 4. Save results
    input_image.save("output_original.png")
    depth_map.save("output_depth.png")
    generated_image.save("output_render.png")
    
    print("\n🎉 SUCCESS! Check your folder for the new images.")
    # Show VRAM usage
    print(f"\n💾 Peak VRAM used: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")