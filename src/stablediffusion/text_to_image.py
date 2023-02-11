import torch
from diffusers import StableDiffusionPipeline

from computing import Computing
from stablediffusion.samplers import Sampler, SamplerMixin
from stablediffusion.setting import StableDiffusionSetting


class StableDiffusion(SamplerMixin):
    def __init__(self, compute: Computing):
        self.compute = compute
        self.pipeline = None
        super().__init__()


    def get_text_to_image_pipleline(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1-base",
        vae_id: str = "stabilityai/sd-vae-ft-mse",
    ):
        # Samplers
        print(f"StableDiffusion - {self.compute.name},{self.compute.datatype}")
        self.load_samplers(model_id, vae_id)
        default_sampler = self.default_sampler()
        print(default_sampler)

        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            scheduler=default_sampler,
        )
        if self.compute.name == "cuda":
            self.pipeline = pipeline.to("cuda")

    def text_to_image(self, setting: StableDiffusionSetting):
        print(setting.scheduler)
        self.pipeline.scheduler = self.find_sampler(setting.scheduler)
        generator = None
        if setting.seed != -1:
            print(f"Using seed {setting.seed}")
            generator = torch.Generator(self.compute.name).manual_seed(setting.seed)

        if setting.attention_slicing:
            self.pipeline.enable_attention_slicing()
        else:
            self.pipeline.disable_attention_slicing()

        if setting.vae_slicing:
            self.pipeline.enable_vae_slicing()
        else:
            self.pipeline.disable_vae_slicing()

        images = self.pipeline(
            setting.prompt,
            guidance_scale=setting.guidance_scale,
            num_inference_steps=setting.inference_steps,
            height=setting.image_height,
            width=setting.image_width,
            negative_prompt=setting.negative_prompt,
            num_images_per_prompt=setting.number_of_images,
            generator=generator,
        ).images
        return images
