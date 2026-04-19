import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# Load Canny ControlNet
device = "cuda"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE", 
    controlnet=controlnet, 
    torch_dtype=torch.float16, 
    safety_checker=None
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(device)
pipe.enable_model_cpu_offload()

def get_canny_image(image):
    # Convert to numpy
    image = np.array(image)
    # Detect edges
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)

def generate_canny_render(input_image, prompt):
    input_resized = input_image.resize((512, 512))
    canny_image = get_canny_image(input_resized)
    
    output = pipe(
        prompt=prompt, 
        image=canny_image, 
        num_inference_steps=30, 
        guidance_scale=7.5
    ).images[0]
    return output