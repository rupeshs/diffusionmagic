import torch
from diffusers import StableDiffusionDepth2ImgPipeline
from PIL import Image

from backend.computing import Computing
from backend.stablediffusion.samplers import SamplerMixin
from backend.stablediffusion.setting import StableDiffusionImageDepthToImageSetting


class StableDiffusionDepthToImage(SamplerMixin):
    def __init__(self, compute: Computing):
        self.compute = compute
        self.device = self.compute.name
        super().__init__()

    def get_depth_to_image_pipleline(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-depth",
        vae_id: str = "stabilityai/sd-vae-ft-mse",
    ):
        print(f"StableDiffusion - {self.compute.name},{self.compute.datatype}")
        print(f"using model {model_id}")
        self.load_samplers(model_id, vae_id)
        default_sampler = self.default_sampler()

        pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            scheduler=default_sampler,
        )
        if self.compute.name == "cuda":
            self.depth_pipeline = pipeline.to("cuda")
    
    def depth_to_image(self, setting: StableDiffusionImageDepthToImageSetting,):
        print("Running depth to image pipeline")
        self.depth_pipeline.scheduler = self.find_sampler(setting.scheduler)
        generator = None
        if setting.seed != -1:
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
            strength = setting.strength,
            prompt=setting.prompt,
            guidance_scale=setting.guidance_scale,
            num_inference_steps=setting.inference_steps,
            negative_prompt=setting.negative_prompt,
            num_images_per_prompt=setting.number_of_images,
            generator=generator,
        ).images
        return images

        