# Based on https://huggingface.co/spaces/AP123/IllusionDiffusion/
from time import time

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
)

from backend.computing import Computing
from backend.stablediffusion.models.scheduler_types import SchedulerType
from backend.stablediffusion.models.setting import IllusionDiffusionSetting
from backend.stablediffusion.scheduler_mixin import SamplerMixin
from backend.stablediffusion.stable_diffusion_types import (
    get_diffusion_type,
)
from backend.image_ops import get_black_and_white_image


class IllusionDiffusion(SamplerMixin):
    def __init__(self, compute: Computing):
        self.compute = compute
        self.device = self.compute.name
        super().__init__()

    def init_control_to_image_pipleline(
        self,
        model_id: str = "monster-labs/control_v1p_sd15_qrcode_monster",
        stable_diffusion_model="runwayml/stable-diffusion-v1-5",
        low_vram_mode: bool = False,
        sampler: str = SchedulerType.EulerDiscreteScheduler.value,
    ):
        self.low_vram_mode = low_vram_mode
        self.controlnet_type = get_diffusion_type(model_id)
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
        setting: IllusionDiffusionSetting,
    ):
        if setting.scheduler is None:
            raise Exception("Scheduler cannot be  empty")
        print("Running illusion diffusion image pipeline")
        self.controlnet_pipeline.scheduler = self.find_sampler(
            setting.scheduler,
            self.model_id,
        )
        generator = None
        if setting.seed != -1 and setting.seed:
            print(f"Using seed {setting.seed}")
            generator = torch.Generator(self.device).manual_seed(setting.seed)

        control_image_bw = get_black_and_white_image(setting.control_image)

        control_image_small = self._center_crop_resize(control_image_bw, (512, 512))
        inf_steps = (
            setting.inference_steps - 5
            if setting.inference_steps
            else setting.inference_steps
        )

        latents = self.controlnet_pipeline(
            prompt=setting.prompt,
            negative_prompt=setting.negative_prompt,
            image=control_image_small,
            guidance_scale=setting.guidance_scale,
            controlnet_conditioning_scale=setting.controlnet_conditioning_scale,
            generator=generator,
            control_guidance_start=setting.control_guidance_start,
            control_guidance_end=setting.control_guidance_end,
            num_inference_steps=inf_steps,
            output_type="latent",
        )
        control_image_large = self._center_crop_resize(control_image_bw, (1024, 1024))
        upscaled_latents = self._upscale(latents, "nearest-exact", 2)
        out_images = self.image_pipeline(
            prompt=setting.prompt,
            negative_prompt=setting.negative_prompt,
            control_image=control_image_large,
            image=upscaled_latents,
            guidance_scale=setting.guidance_scale,
            generator=generator,
            num_inference_steps=setting.inference_steps,
            strength=setting.upscaler_strength,
            control_guidance_start=setting.control_guidance_start,
            control_guidance_end=setting.control_guidance_end,
            controlnet_conditioning_scale=setting.controlnet_conditioning_scale,
        )
        return out_images["images"]

    def _pipeline_to_device(self):
        if self.compute.name == "cuda":
            self.controlnet_pipeline = self.controlnet_pipeline.to("cuda")
            self.image_pipeline = self.image_pipeline.to("cuda")
        elif self.compute.name == "mps":
            self.controlnet_pipeline = self.controlnet_pipeline.to("mps")
            self.image_pipeline = self.image_pipeline.to("mps")

    def _load_full_precision_model(self):
        self.controlnet = ControlNetModel.from_pretrained(self.control_net_model_id)
        self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=self.controlnet,
        )
        self.image_pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.model_id,
            unet=self.controlnet_pipeline.unet,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
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
                        safety_checker=None,
                        torch_dtype=torch.float16,
                    )
                )

                self.image_pipeline = (
                    StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                        self.model_id,
                        unet=self.controlnet_pipeline.unet,
                        controlnet=self.controlnet,
                        safety_checker=None,
                        torch_dtype=torch.float16,
                    )
                )

            except Exception as ex:
                print(
                    f" The fp16 of the model not found using full precision model,  {ex}"
                )
                self._load_full_precision_model()
        else:
            self._load_full_precision_model()

    def _common_upscale(
        self,
        samples,
        width,
        height,
        upscale_method,
        crop=False,
    ):
        if crop == "center":
            old_width = samples.shape[3]
            old_height = samples.shape[2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = samples[:, :, y : old_height - y, x : old_width - x]
        else:
            s = samples
        return torch.nn.functional.interpolate(
            s, size=(height, width), mode=upscale_method
        )

    def _upscale(
        self,
        samples,
        upscale_method,
        scale_by,
    ):
        # s = samples.copy()
        width = round(samples["images"].shape[3] * scale_by)
        height = round(samples["images"].shape[2] * scale_by)
        s = self._common_upscale(
            samples["images"], width, height, upscale_method, "disabled"
        )
        return s

    def _center_crop_resize(self, img, output_size=(512, 512)):
        width, height = img.size

        # Calculate dimensions to crop to the center
        new_dimension = min(width, height)
        left = (width - new_dimension) / 2
        top = (height - new_dimension) / 2
        right = (width + new_dimension) / 2
        bottom = (height + new_dimension) / 2

        # Crop and resize
        img = img.crop((left, top, right, bottom))
        img = img.resize(output_size)

        return img
