from backend.controlnet.controls.canny_control import CannyControl
from backend.controlnet.controls.depth_control import DepthControl
from backend.controlnet.controls.hed_control import HedControl
from backend.controlnet.controls.line_control import LineControl
from backend.controlnet.controls.normal_control import NormalControl
from backend.controlnet.controls.pose_control import PoseControl
from backend.controlnet.controls.scribble_control import ScribbleControl
from backend.controlnet.controls.seg_control import SegControl
from backend.stablediffusion.stable_diffusion_types import StableDiffusionType


class ImageControlFactory:
    def create_control(self, controlnet_type: StableDiffusionType):
        if controlnet_type == StableDiffusionType.controlnet_canny:
            return CannyControl()
        elif controlnet_type == StableDiffusionType.controlnet_line:
            return LineControl()
        elif controlnet_type == StableDiffusionType.controlnet_normal:
            return NormalControl()
        elif controlnet_type == StableDiffusionType.controlnet_hed:
            return HedControl()
        elif controlnet_type == StableDiffusionType.controlnet_pose:
            return PoseControl()
        elif controlnet_type == StableDiffusionType.controlnet_depth:
            return DepthControl()
        elif controlnet_type == StableDiffusionType.controlnet_scribble:
            return ScribbleControl()
        elif controlnet_type == StableDiffusionType.controlnet_seg:
            return SegControl()
        else:
            print("Error: Control type not implemented!")
            raise Exception("Error: Control type not implemented!")
