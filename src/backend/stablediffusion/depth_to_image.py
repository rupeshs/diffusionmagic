from time import time

import torch
from diffusers import StableDiffusionDepth2ImgPipeline
from PIL import Image

from backend.computing import Computing
from backend.stablediffusion.models.scheduler_types import SchedulerType
from backend.stablediffusion.models.setting import (
    StableDiffusionImageDepthToImageSetting,
)
from backend.stablediffusion.scheduler_mixin import SamplerMixin


class StableDiffusionDepthToImage(SamplerMixin):
    def __init__(self, compute: Computing):
        self.compute = compute
        self.device = self.compute.name
        super().__init__()

    def get_depth_to_image_pipleline(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-depth",
        low_vram_mode: bool = False,
        sampler: str = SchedulerType.DPMSolverMultistepScheduler.value,
    ):
        self.low_vram_mode = low_vram_mode
        print(f"StableDiffusion - {self.compute.name},{self.compute.datatype}")
        print(f"using model {model_id}")
        self.model_id = model_id
        self.default_sampler = self.find_sampler(
            sampler,
            self.model_id,
        )

        tic = time()
        self._load_model()
        delta = time() - tic
        print(f"Model loaded in {delta:.2f}s ")
        self._pipeline_to_device()

    def depth_to_image(
        self,
        setting: StableDiffusionImageDepthToImageSetting,
    ):
        if setting.scheduler is None:
            raise Exception("Scheduler cannot be  empty")
        print("Running depth to image pipeline")
        self.depth_pipeline.scheduler = self.find_sampler(
            setting.scheduler,
            self.model_id,
        )
        generator = None
        if setting.seed != -1 and setting.seed:
            print(f"Using seed {setting.seed}")
            generator = torch.Generator(self.device).manual_seed(setting.seed)

        if setting.attention_slicing:
            self.depth_pipeline.enable_attention_slicing()
        else:
            self.depth_pipeline.disable_attention_slicing()

        base_image = setting.image.convert("RGB").resize(
            (
                setting.image_width,
                setting.image_height,
            ),
            Image.Resampling.LANCZOS,
        )
        images = self.depth_pipeline(
            image=base_image,
            strength=setting.strength,
            prompt=setting.prompt,
            guidance_scale=setting.guidance_scale,
            num_inference_steps=setting.inference_steps,
            negative_prompt=setting.negative_prompt,
            num_images_per_prompt=setting.number_of_images,
            generator=generator,
        ).images
        return images

    def _pipeline_to_device(self):
        if self.low_vram_mode:
            print("Running in low VRAM mode,slower to generate images")
            self.depth_pipeline.enable_sequential_cpu_offload()
        else:
            if self.compute.name == "cuda":
                self.depth_pipeline = self.depth_pipeline.to("cuda")
            elif self.compute.name == "mps":
                self.depth_pipeline = self.depth_pipeline.to("mps")

    def _load_full_precision_model(self):
        self.depth_pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.compute.datatype,
            scheduler=self.default_sampler,
        )

    def _load_model(self):
        print("Loading model...")
        if self.compute.name == "cuda":
            try:
                self.depth_pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.compute.datatype,
                    scheduler=self.default_sampler,
                    revision="fp16",
                )
            except Exception as ex:
                print(
                    f" The fp16 of the model not found using full precision model,  {ex}"
                )
                self._load_full_precision_model()
        else:
            self._load_full_precision_model()
