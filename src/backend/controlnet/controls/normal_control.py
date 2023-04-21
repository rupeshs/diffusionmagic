import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

from backend.controlnet.controls.control_interface import ControlInterface


class NormalControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        image = depth_estimator(image)["predicted_depth"][0]
        image = image.numpy()

        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)

        bg_threhold = 0.4

        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < bg_threhold] = 0

        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < bg_threhold] = 0

        z = np.ones_like(x) * np.pi * 2.0

        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image**2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        return image
