from typing import Optional, Union

from pydantic import BaseModel, Field, validator
from stablediffusion.samplers import Sampler


class StableDiffusionSetting(BaseModel):
    prompt: str
    negative_prompt: Optional[str]
    image_height: Optional[int] = 512
    image_width: Optional[int] = 512
    inference_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5
    number_of_images: Optional[int] = 1
    scheduler: Optional[str] = Sampler.DPMSolverMultistepScheduler.value
    seed: Optional[int] = -1
    attention_slicing: Optional[bool] = True
    vae_slicing: Optional[bool] = True
