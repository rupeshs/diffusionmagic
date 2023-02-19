from enum import Enum


class StableDiffusionType(str, Enum):
    """Stable diffusion types"""

    base = "Base"
    inpainting = "InPainting"
    depth2img = "DepthToImage"
