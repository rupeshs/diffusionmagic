import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from backend.computing import Computing
from backend.stablediffusion.samplers import SamplerMixin
from backend.stablediffusion.setting import StableDiffusionImageInpaintingSetting


class StableDiffusionInpainting(SamplerMixin):
    def __init__(self, compute: Computing):
        self.compute = compute
        self.pipeline = None
        self.device = self.compute.name
        super().__init__()

    def get_inpainting_pipleline(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-inpainting",
        vae_id: str = "stabilityai/sd-vae-ft-mse",
    ):
        print(f"StableDiffusion - {self.compute.name},{self.compute.datatype}")
        print(f"using model {model_id}")
        self.load_samplers(model_id, vae_id)
        default_sampler = self.default_sampler()

        inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            scheduler=default_sampler,
        )
        if self.compute.name == "cuda":
            self.inpainting_pipeline = inpainting_pipeline.to("cuda")

    def image_inpainting(self, setting: StableDiffusionImageInpaintingSetting):
        print("Running image inpainting pipeline")
        self.inpainting_pipeline.scheduler = self.find_sampler(setting.scheduler)
        generator = None
        if setting.seed != -1:
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

        print(base_image.size)
        print(mask_image.size)

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
