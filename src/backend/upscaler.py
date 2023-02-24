import torch
from diffusers import StableDiffusionLatentUpscalePipeline

# load model and scheduler
# https://github.com/huggingface/diffusers/pull/2059
upscale_model_id = "stabilityai/sd-x2-latent-upscaler"
upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
    upscale_model_id, torch_dtype=torch.float16
)
upscaler.to("cuda")
