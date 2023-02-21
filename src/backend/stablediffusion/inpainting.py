import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from backend.computing import Computing
from backend.stablediffusion.models.samplers import SamplerMixin
from backend.stablediffusion.models.setting import StableDiffusionImageInpaintingSetting
from settings import AppSettings


class StableDiffusionInpainting(SamplerMixin):
    def __init__(self, compute: Computing):
        self.compute = compute
        self.device = self.compute.name
        self.app_settings = AppSettings().get_settings()
        super().__init__()

    def get_inpainting_pipleline(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-inpainting",
        vae_id: str = "stabilityai/sd-vae-ft-mse",
    ):
        model_id = self.app_settings.model_settings.model_id
        print(f"StableDiffusion - {self.compute.name},{self.compute.datatype}")
        print(f"using model {model_id}")
        self.load_samplers(model_id, vae_id)
        default_sampler = self.default_sampler()

        self.inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=self.compute.datatype,
            scheduler=default_sampler,
        )

        self._pipeline_to_device()

    def image_inpainting(self, setting: StableDiffusionImageInpaintingSetting):
        if setting.scheduler is None:
            raise Exception("Scheduler cannot be  empty")
        print("Running image inpainting pipeline")
        self.inpainting_pipeline.scheduler = self.find_sampler(setting.scheduler)
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
        if self.app_settings.low_memory_mode:
            self.inpainting_pipeline.enable_sequential_cpu_offload()
        else:
            if self.compute.name == "cuda":
                self.inpainting_pipeline = self.inpainting_pipeline.to("cuda")
            elif self.compute.name == "mps":
                self.inpainting_pipeline = self.inpainting_pipeline.to("mps")
