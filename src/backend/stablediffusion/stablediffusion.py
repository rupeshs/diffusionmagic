import torch
from diffusers import (StableDiffusionImg2ImgPipeline,
                       StableDiffusionPipeline,)
from PIL import Image

from backend.computing import Computing
from backend.stablediffusion.models.samplers import SamplerMixin
from backend.stablediffusion.models.setting import (
    StableDiffusionImageToImageSetting, StableDiffusionSetting,)

class StableDiffusion(SamplerMixin):
    def __init__(self, compute: Computing):
        self.compute = compute
        self.pipeline = None
        self.device = self.compute.name
        super().__init__()
        


    def get_text_to_image_pipleline(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1-base",
        vae_id: str = "stabilityai/sd-vae-ft-mse",
    ):
        # Samplers
        print(f"StableDiffusion - {self.compute.name},{self.compute.datatype}")
        print(f"using model {model_id}")
        self.load_samplers(model_id, vae_id)
        default_sampler = self.default_sampler()

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            scheduler=default_sampler,
        )
        
        if  self.pipeline is None:
            raise Exception("Text to image pipeline not initialized") 
        
        if self.compute.name == "cuda" :
            self.pipeline =  self.pipeline .to("cuda")
        elif self.compute.name == "mps":
            self.pipeline =  self.pipeline .to("mps")

        components = self.pipeline.components
        self.img_to_img_pipeline = StableDiffusionImg2ImgPipeline(**components) 

    def text_to_image(self, setting: StableDiffusionSetting):
        if self.pipeline is None:
            raise Exception("Text to image pipeline not initialized") 
        
        self.pipeline.scheduler = self.find_sampler(setting.scheduler)
        generator = None
        if setting.seed != -1:
            print(f"Using seed {setting.seed}")
            generator = torch.Generator(self.device).manual_seed(setting.seed)

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

    def image_to_image(self, setting: StableDiffusionImageToImageSetting):
        if setting.scheduler is None:
            raise Exception("Scheduler cannot be  empty")
            
        print("Running image to image pipeline")
        self.img_to_img_pipeline.scheduler = self.find_sampler(setting.scheduler)# type: ignore
        generator = None
        if setting.seed != -1 and  setting.seed:
            print(f"Using seed {setting.seed}")
            generator = torch.Generator(self.device).manual_seed(setting.seed)

        if setting.attention_slicing:
            self.img_to_img_pipeline.enable_attention_slicing() # type: ignore
        else:
            self.img_to_img_pipeline.disable_attention_slicing() # type: ignore

        init_image = setting.image.resize(
            (
                setting.image_width,
                setting.image_height,
            ),
            Image.Resampling.LANCZOS,
        )

        images = self.img_to_img_pipeline(  # type: ignore
            image=init_image,
            strength=setting.strength,
            prompt=setting.prompt,
            guidance_scale=setting.guidance_scale,
            num_inference_steps=setting.inference_steps,
            negative_prompt=setting.negative_prompt,
            num_images_per_prompt=setting.number_of_images,
            generator=generator,
        ).images 
        return images

    