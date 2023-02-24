from enum import Enum


class StableDiffusionType(str, Enum):
    """Stable diffusion types"""

    base = "Base"
    inpainting = "InPainting"
    depth2img = "DepthToImage"
    instruct_pix2pix = "InstructPixToPix"


def get_diffusion_type(
    model_id: str,
) -> StableDiffusionType:
    stable_diffusion_type = StableDiffusionType.base
    if "inpainting" in model_id:
        stable_diffusion_type = StableDiffusionType.inpainting
    elif "depth" in model_id:
        stable_diffusion_type = StableDiffusionType.depth2img
    elif "instruct-pix2pix" in model_id:
        stable_diffusion_type = StableDiffusionType.instruct_pix2pix

    return stable_diffusion_type
