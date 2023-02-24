from time import time

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image

from backend.computing import Computing
from backend.stablediffusion.scheduler_mixin import SamplerMixin
from backend.stablediffusion.models.scheduler_types import SchedulerType
from backend.stablediffusion.models.setting import (
    StableDiffusionImageInstructPixToPixSetting,
)


class StableDiffusionInstructPixToPix(SamplerMixin):
    def __init__(self, compute: Computing):
        self.compute = compute
        self.device = self.compute.name
        super().__init__()

    def get_instruct_pix_to_pix_pipleline(
        self,
        model_id: str = "timbrooks/instruct-pix2pix",
        low_vram_mode: bool = False,
        sampler: str = SchedulerType.DPMSolverMultistepScheduler.value,
    ):
        print(f"StableDiffusion - {self.compute.name},{self.compute.datatype}")
        print(f"using model {model_id}")
        self.model_id = model_id
        self.low_vram_mode = low_vram_mode
        self.default_sampler = self.find_sampler(
            sampler,
            self.model_id,
        )

        tic = time()
        self._load_model()
        delta = time() - tic
        print(f"Model loaded in {delta:.2f}s ")
        self._pipeline_to_device()

    def instruct_pix_to_pix(self, setting: StableDiffusionImageInstructPixToPixSetting):
        if setting.scheduler is None:
            raise Exception("Scheduler cannot be  empty")
        print("Running image to image pipeline")
        self.instruct_pix_pipeline.scheduler = self.find_sampler(
            setting.scheduler,
            self.model_id,
        )
        generator = None
        if setting.seed != -1 and setting.seed:
            print(f"Using seed {setting.seed}")
            generator = torch.Generator(self.device).manual_seed(setting.seed)

        if setting.attention_slicing:
            self.instruct_pix_pipeline.enable_attention_slicing()
        else:
            self.instruct_pix_pipeline.disable_attention_slicing()

        init_image = setting.image.resize(
            (
                setting.image_width,
                setting.image_height,
            ),
            Image.Resampling.LANCZOS,
        )

        images = self.instruct_pix_pipeline(
            image=init_image,
            image_guidance_scale=setting.image_guidance_scale,
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
            self.instruct_pix_pipeline.enable_sequential_cpu_offload()
        else:
            if self.compute.name == "cuda":
                self.instruct_pix_pipeline = self.instruct_pix_pipeline.to("cuda")
            elif self.compute.name == "mps":
                self.instruct_pix_pipeline = self.instruct_pix_pipeline.to("mps")

    def _load_full_precision_model(self):
        self.instruct_pix_pipeline = (
            StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.compute.datatype,
                scheduler=self.default_sampler,
            )
        )

    def _load_model(self):
        print("Loading model...")
        if self.compute.name == "cuda":
            try:
                self.instruct_pix_pipeline = (
                    StableDiffusionInstructPix2PixPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=self.compute.datatype,
                        scheduler=self.default_sampler,
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
