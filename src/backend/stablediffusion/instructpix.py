import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image

from backend.computing import Computing
from backend.stablediffusion.models.samplers import SamplerMixin
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
        vae_id: str = "stabilityai/sd-vae-ft-mse",
    ):
        print(f"StableDiffusion - {self.compute.name},{self.compute.datatype}")
        print(f"using model {model_id}")
        self.load_samplers(model_id, vae_id)
        default_sampler = self.default_sampler()

        self.instruct_pix_pipeline = (
            StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                scheduler=default_sampler,
            )
        )
        if self.compute.name == "cuda":
            self.instruct_pix_pipeline = self.instruct_pix_pipeline.to("cuda")
        elif self.compute.name == "mps":
            self.instruct_pix_pipeline = self.instruct_pix_pipeline.to("mps")

    def instruct_pix_to_pix(self, setting: StableDiffusionImageInstructPixToPixSetting):
        if setting.scheduler is None:
            raise Exception("Scheduler cannot be  empty")
        print("Running image to image pipeline")
        self.instruct_pix_pipeline.scheduler = self.find_sampler(setting.scheduler)
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
