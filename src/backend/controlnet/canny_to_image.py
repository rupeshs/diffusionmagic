from time import time
from typing import Any
import numpy as np
from cv2 import Canny, bitwise_not
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image

from backend.computing import Computing
from backend.stablediffusion.models.scheduler_types import SchedulerType
from backend.stablediffusion.models.setting import (
    StableDiffusionControlnetSetting,
)
from backend.stablediffusion.scheduler_mixin import SamplerMixin


class StableDiffusionCannyToImage(SamplerMixin):
    def __init__(self, compute: Computing):
        self.compute = compute
        self.device = self.compute.name
        super().__init__()

    def get_canny_to_image_pipleline(
        self,
        model_id: str = "lllyasviel/sd-controlnet-canny",
        stable_diffusion_model="runwayml/stable-diffusion-v1-5",
        low_vram_mode: bool = False,
        sampler: str = SchedulerType.UniPCMultistepScheduler.value,
    ):
        self.low_vram_mode = low_vram_mode
        print(f"StableDiffusion - {self.compute.name},{self.compute.datatype}")
        print(f"Using ControlNet Model  {model_id}")
        print(f"Using Stable diffusion Model  {stable_diffusion_model}")
        self.control_net_model_id = model_id
        self.model_id = stable_diffusion_model
        self.default_sampler = self.find_sampler(
            sampler,
            self.model_id,
        )

        tic = time()
        self._load_model()
        delta = time() - tic
        print(f"Model loaded in {delta:.2f}s ")
        self._pipeline_to_device()

    def canny_to_image(
        self,
        setting: StableDiffusionControlnetSetting,
    ):
        if setting.scheduler is None:
            raise Exception("Scheduler cannot be  empty")
        print("Running canny to image pipeline")
        self.canny_pipeline.scheduler = self.find_sampler(
            setting.scheduler,
            self.model_id,
        )
        generator = None
        if setting.seed != -1 and setting.seed:
            print(f"Using seed {setting.seed}")
            generator = torch.Generator(self.device).manual_seed(setting.seed)

        if setting.attention_slicing:
            self.canny_pipeline.enable_attention_slicing()
        else:
            self.canny_pipeline.disable_attention_slicing()

        if setting.vae_slicing:
            self.canny_pipeline.enable_vae_slicing()
        else:
            self.canny_pipeline.disable_vae_slicing()

        base_image = setting.image.convert("RGB").resize(
            (
                setting.image_width,
                setting.image_height,
            ),
            Image.Resampling.LANCZOS,
        )

        canny_image, canny_image_inv = self.get_canny_image(base_image)
        images = self.canny_pipeline(
            prompt=setting.prompt,
            image=canny_image,
            guidance_scale=setting.guidance_scale,
            num_inference_steps=setting.inference_steps,
            negative_prompt=setting.negative_prompt,
            num_images_per_prompt=setting.number_of_images,
            generator=generator,
        ).images
        images.append(canny_image_inv)
        return images

    def _pipeline_to_device(self):
        if self.low_vram_mode:
            print("Running in low VRAM mode,slower to generate images")
            self.canny_pipeline.enable_sequential_cpu_offload()
        else:
            self.canny_pipeline.enable_model_cpu_offload()

    def _load_full_precision_model(self):
        self.controlnet = ControlNetModel.from_pretrained(self.control_net_model_id)
        self.canny_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id, controlnet=self.controlnet
        )

    def _load_model(self):
        print("Loading model...")
        if self.compute.name == "cuda":
            try:
                self.controlnet = ControlNetModel.from_pretrained(
                    self.control_net_model_id,
                    torch_dtype=torch.float16,
                )
                self.canny_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    self.model_id,
                    controlnet=self.controlnet,
                    torch_dtype=torch.float16,
                    revision="fp16",
                )
            except Exception as ex:
                print(
                    f" The fp16 of the model not found using full precision model,  {ex}"
                )
                self._load_full_precision_model()
        else:
            self._load_full_precision_model()

    def get_canny_image(self, image: Image) -> Any:
        low_threshold = 100
        high_threshold = 200
        image = np.array(image)
        image = Canny(image, low_threshold, high_threshold)
        image_inv = bitwise_not(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image), Image.fromarray(image_inv)
