import torch
from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.models import AutoencoderKL
from computing import Computing


computing = Computing()

def get_text_to_image_pipleline(model_id:str="stabilityai/stable-diffusion-2-1-base",vae_id:str="stabilityai/sd-vae-ft-mse",):
    vae = AutoencoderKL.from_pretrained(vae_id)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model_id, vae=vae, subfolder="scheduler"
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=computing.datatype, scheduler=scheduler
    )
    if computing._device=="cuda":
        pipeline = pipeline.to("cuda")

    return pipeline
