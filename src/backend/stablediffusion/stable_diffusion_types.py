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
    controlnet_pose = "controlnet_pose"
    controlnet_depth = "controlnet_depth"
    controlnet_scribble = "controlnet_scribble"
    controlnet_seg = "controlnet_seg"


def get_diffusion_type(
    model_id: str,
) -> StableDiffusionType:
    stable_diffusion_type = StableDiffusionType.base
    if "inpainting" in model_id:
        stable_diffusion_type = StableDiffusionType.inpainting
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
    elif "controlnet-openpose" in model_id:
        stable_diffusion_type = StableDiffusionType.controlnet_pose
    elif "sd-controlnet-depth" in model_id:
        stable_diffusion_type = StableDiffusionType.controlnet_depth
    elif "depth" in model_id:
        stable_diffusion_type = StableDiffusionType.depth2img
    elif "controlnet-scribble" in model_id:
        stable_diffusion_type = StableDiffusionType.controlnet_scribble
    elif "controlnet-seg" in model_id:
        stable_diffusion_type = StableDiffusionType.controlnet_seg
    return stable_diffusion_type
