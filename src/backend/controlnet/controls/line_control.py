from PIL import Image

from backend.controlnet.controls.control_interface import ControlInterface
from controlnet_aux import MLSDdetector


class LineControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
        image = mlsd(image)
        return image
