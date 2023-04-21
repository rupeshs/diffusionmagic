from backend.stablediffusion.stable_diffusion_types import (
    StableDiffusionType,
)
from backend.controlnet.controls.canny_control import CannyControl
from backend.controlnet.controls.line_control import LineControl


class ImageControlFactory:
    def create_control(self, controlnet_type: StableDiffusionType):
        if controlnet_type == StableDiffusionType.controlnet_canny:
            return CannyControl()
        elif controlnet_type == StableDiffusionType.controlnet_line:
            return LineControl()
        else:
            print("Error: Control type not implemented!")
            raise Exception("Error: Control type not implemented!")
