from time import time

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import ImageOps

from backend.computing import Computing
from backend.controlnet.controls.image_control_factory import ImageControlFactory
from backend.image_ops import resize_pil_image
from backend.stablediffusion.models.scheduler_types import SchedulerType
from backend.stablediffusion.models.setting import StableDiffusionControlnetSetting
from backend.stablediffusion.scheduler_mixin import SamplerMixin
from backend.stablediffusion.stable_diffusion_types import (
    StableDiffusionType,
    get_diffusion_type,
)


class ControlnetContext(SamplerMixin):
    def __init__(self, compute: Computing):
        self.compute = compute
        self.device = self.compute.name
        super().__init__()

    def init_control_to_image_pipleline(
        self,
        model_id: str = "lllyasviel/sd-controlnet-canny",
        stable_diffusion_model="runwayml/stable-diffusion-v1-5",
        low_vram_mode: bool = False,
        sampler: str = SchedulerType.UniPCMultistepScheduler.value,
    ):
        self.low_vram_mode = low_vram_mode
        self.controlnet_type = get_diffusion_type(model_id)
        image_control_factory = ImageControlFactory()
        self.image_control = image_control_factory.create_control(self.controlnet_type)
        print(f"StableDiffusion - {self.compute.name},{self.compute.datatype}")
        print(f"Controlnet - { self.controlnet_type }")
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

    def control_to_image(
        self,
        setting: StableDiffusionControlnetSetting,
    ):
        if setting.scheduler is None:
            raise Exception("Scheduler cannot be  empty")
        print("Running controlnet image pipeline")
        self.controlnet_pipeline.scheduler = self.find_sampler(
            setting.scheduler,
            self.model_id,
        )
        generator = None
        if setting.seed != -1 and setting.seed:
            print(f"Using seed {setting.seed}")
            generator = torch.Generator(self.device).manual_seed(setting.seed)

        self._enable_slicing(setting)
        base_image = resize_pil_image(
            setting.image, setting.image_width, setting.image_height
        )
        control_img = self.image_control.get_control_image(base_image)

        images = self.controlnet_pipeline(
            prompt=setting.prompt,
            image=control_img,
            guidance_scale=setting.guidance_scale,
            num_inference_steps=setting.inference_steps,
            negative_prompt=setting.negative_prompt,
            num_images_per_prompt=setting.number_of_images,
            generator=generator,
        ).images
        if (
            self.controlnet_type == StableDiffusionType.controlnet_canny
            or self.controlnet_type == StableDiffusionType.controlnet_line
        ):
            inverted_image = ImageOps.invert(control_img)
            images.append(inverted_image)
        else:
            images.append(control_img)

        return images

    def _pipeline_to_device(self):
        if self.low_vram_mode:
            print("Running in low VRAM mode,slower to generate images")
            self.controlnet_pipeline.enable_sequential_cpu_offload()
        else:
            self.controlnet_pipeline.enable_model_cpu_offload()

    def _load_full_precision_model(self):
        self.controlnet = ControlNetModel.from_pretrained(self.control_net_model_id)
        self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=self.controlnet,
        )

    def _load_model(self):
        print("Loading model...")
        if self.compute.name == "cuda":
            try:
                self.controlnet = ControlNetModel.from_pretrained(
                    self.control_net_model_id,
                    torch_dtype=torch.float16,
                )
                self.controlnet_pipeline = (
                    StableDiffusionControlNetPipeline.from_pretrained(
                        self.model_id,
                        controlnet=self.controlnet,
                        torch_dtype=torch.float16,
                        revision="fp16",
                    )
                )
            except Exception as ex:
                print(
                    f" The fp16 of the model not found using full precision model,  {ex}"
                )
                self._load_full_precision_model()
        else:
            self._load_full_precision_model()

    def _enable_slicing(self, setting: StableDiffusionControlnetSetting):
        if setting.attention_slicing:
            self.controlnet_pipeline.enable_attention_slicing()
        else:
            self.controlnet_pipeline.disable_attention_slicing()

        if setting.vae_slicing:
            self.controlnet_pipeline.enable_vae_slicing()
        else:
            self.controlnet_pipeline.disable_vae_slicing()
