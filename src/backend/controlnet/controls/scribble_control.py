from PIL import Image

from backend.controlnet.controls.control_interface import ControlInterface
from controlnet_aux import HEDdetector


class ScribbleControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")
        image = hed(image, scribble=True)
        return image
