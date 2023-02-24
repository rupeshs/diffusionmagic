from time import time

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from backend.computing import Computing
from backend.stablediffusion.scheduler_mixin import SamplerMixin
from backend.stablediffusion.models.scheduler_types import SchedulerType
from backend.stablediffusion.models.setting import StableDiffusionImageInpaintingSetting


class StableDiffusionInpainting(SamplerMixin):
    def __init__(self, compute: Computing):
        self.compute = compute
        self.device = self.compute.name
        super().__init__()

    def get_inpainting_pipleline(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-inpainting",
        low_vram_mode: bool = False,
        sampler: str = SchedulerType.DPMSolverMultistepScheduler.value,
    ):
        self.model_id = model_id
        self.low_vram_mode = low_vram_mode
        print(f"StableDiffusion - {self.compute.name},{self.compute.datatype}")
        print(f"Using model {model_id}")
        self.default_sampler = self.find_sampler(
            sampler,
            self.model_id,
        )

        tic = time()
        self._load_model()
        delta = time() - tic
        print(f"Model loaded in {delta:.2f}s ")
        self._pipeline_to_device()

    def image_inpainting(self, setting: StableDiffusionImageInpaintingSetting):
        if setting.scheduler is None:
            raise Exception("Scheduler cannot be  empty")
        print("Running image inpainting pipeline")
        self.inpainting_pipeline.scheduler = self.find_sampler(
            setting.scheduler,
            self.model_id,
        )
        generator = None
        if setting.seed != -1 and setting.seed:
            print(f"Using seed {setting.seed}")
            generator = torch.Generator(self.device).manual_seed(setting.seed)

        if setting.attention_slicing:
            self.inpainting_pipeline.enable_attention_slicing()
        else:
            self.inpainting_pipeline.disable_attention_slicing()

        base_image = setting.image.convert("RGB").resize(
            (
                setting.image_width,
                setting.image_height,
            ),
            Image.Resampling.LANCZOS,
        )
        mask_image = setting.mask_image.convert("RGB").resize(
            (
                setting.image_width,
                setting.image_height,
            ),
            Image.Resampling.LANCZOS,
        )

        images = self.inpainting_pipeline(
            image=base_image,
            mask_image=mask_image,
            height=setting.image_height,
            width=setting.image_width,
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
            self.inpainting_pipeline.enable_sequential_cpu_offload()
        else:
            if self.compute.name == "cuda":
                self.inpainting_pipeline = self.inpainting_pipeline.to("cuda")
            elif self.compute.name == "mps":
                self.inpainting_pipeline = self.inpainting_pipeline.to("mps")

    def _load_full_precision_model(self):
        self.inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.compute.datatype,
            scheduler=self.default_sampler,
        )

    def _load_model(self):
        print("Loading model...")
        if self.compute.name == "cuda":
            try:
                self.inpainting_pipeline = (
                    StableDiffusionInpaintPipeline.from_pretrained(
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
