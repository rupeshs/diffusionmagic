from pydantic import BaseModel
from typing import Optional


class WurstchenSetting(BaseModel):
    prompt: str
    negative_prompt: Optional[str]
    image_height: Optional[int] = 512
    image_width: Optional[int] = 512
    prior_guidance_scale: Optional[float] = 4.0
    number_of_images: Optional[int] = 1
    seed: Optional[int] = -1
