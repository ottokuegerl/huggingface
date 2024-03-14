"""
################################################
## text-to-image
## create an image on behalf engine="stabilityai/stable-diffusion-2-1"
#
# pip install diffusers transformers accelerate scipy safetensors
# pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
################################################
"""

import platform
import os
import time
from dotenv import load_dotenv
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


if __name__ == "__main__":
    start_time = time.time()  # Start time measurement
    clear_screen()
    # Load environment variables from .env file
    # STABILITYAI_API_KEY
    load_dotenv()

    # Paste your API Key below.
    # os.environ['STABILITY_KEY'] = 'key-goes-here'

    model_id = "stabilityai/stable-diffusion-2-1"

    # use float16 on GPU
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # use float32 on CPU only
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # CUDA (Compute Unified Device Architecture) is a parallel computing platform and API model created by NVIDIA
    # pipe = pipe.to("cuda")
    pipe = pipe.to("cpu")

    # If you have low GPU RAM available, make sure to add a pipe.enable_attention_slicing()
    # after sending it to cuda for less VRAM usage (to the cost of speed)
    pipe.enable_attention_slicing()

    # Settings for enhanced quality
    guidance_scale = 7.5  # Increase the guidance scale for stronger adherence to the prompt (can affect quality)
    num_inference_steps = (
        50  # Increase the number of inference steps for better quality
    )

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]

    image.save("astronaut_rides_horse2.png")

    end_time = time.time()  # End time measurement
    execution_time = end_time - start_time  # Calculate execution time
    print(f"Execution time: ---> {execution_time:.2f} <--- seconds")
