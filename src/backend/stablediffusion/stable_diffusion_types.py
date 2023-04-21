from enum import Enum


class StableDiffusionType(str, Enum):
    """Stable diffusion types"""

    base = "Base"
    inpainting = "InPainting"
    depth2img = "DepthToImage"
    instruct_pix2pix = "InstructPixToPix"
    controlnet_canny = "controlnet_canny"
    controlnet_line = "controlnet_line"
    controlnet_normal = "controlnet_normal"
    controlnet_hed = "controlnet_hed"


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
    elif "controlnet-canny" in model_id:
        stable_diffusion_type = StableDiffusionType.controlnet_canny
    elif "controlnet-mlsd" in model_id:
        stable_diffusion_type = StableDiffusionType.controlnet_line
    elif "controlnet-normal" in model_id:
        stable_diffusion_type = StableDiffusionType.controlnet_normal
    elif "controlnet-hed" in model_id:
        stable_diffusion_type = StableDiffusionType.controlnet_hed

    return stable_diffusion_type
