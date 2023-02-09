import torch
from diffusers import StableDiffusionPipeline
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.models import AutoencoderKL
from computing import Computing
from stablediffusion.samplers import Sampler


class StableDiffusion:
    def __init__(self, compute: Computing):
        self.compute = compute
        self.pipeline = None
        self.samplers = {}

        print(f"StableDiffusion - {self.compute.name},{self.compute.datatype}")

    def get_text_to_image_pipleline(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1-base",
        vae_id: str = "stabilityai/sd-vae-ft-mse",
    ):

        # scheduler = DPMSolverMultistepScheduler.from_pretrained(
        #     model_id, vae=vae, subfolder="scheduler"
        # )
        self._load_schedulers(
            model_id,
            vae_id,
        )
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            scheduler=self.samplers[Sampler.DPMSolverMultistepScheduler.value],
        )
        if self.compute.name == "cuda":
            self.pipeline = pipeline.to("cuda")

        # return pipeline

    def text_to_image(
        self,
        prompt,
        neg_prompt,
        image_height,
        image_width,
        inference_steps,
        guidance_scale,
        num_images,
        attention_slicing,
        vae_slicing,
        seed,
        scheduler,
    ):
        print(scheduler)
        self.pipeline.scheduler = self.samplers[scheduler]
        generator = None
        if seed != -1:
            print(f"Using seed {seed}")
            generator = torch.Generator(self.compute.name).manual_seed(seed)

        if attention_slicing:
            self.pipeline.enable_attention_slicing()
        else:
            self.pipeline.disable_attention_slicing()

        if vae_slicing:
            self.pipeline.enable_vae_slicing()
        else:
            self.pipeline.disable_vae_slicing()

        images = self.pipeline(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=inference_steps,
            height=image_height,
            width=image_width,
            negative_prompt=neg_prompt,
            num_images_per_prompt=num_images,
            generator=generator,
        ).images

        return images

    def _load_schedulers(self, repo_id: str, vae_id: str):
        self.samplers = {}

        vae = AutoencoderKL.from_pretrained(vae_id)

        self.samplers[Sampler.DDIMScheduler.value] = DDIMScheduler.from_pretrained(
            repo_id, vae=vae, subfolder="scheduler"
        )
        self.samplers[Sampler.DDPMScheduler.value] = DDPMScheduler.from_pretrained(
            repo_id, vae=vae, subfolder="scheduler"
        )
        self.samplers[Sampler.PNDMScheduler.value] = PNDMScheduler.from_pretrained(
            repo_id, vae=vae, subfolder="scheduler"
        )
        self.samplers[
            Sampler.LMSDiscreteScheduler.value
        ] = LMSDiscreteScheduler.from_pretrained(
            repo_id, vae=vae, subfolder="scheduler"
        )
        self.samplers[
            Sampler.EulerAncestralDiscreteScheduler.value
        ] = EulerAncestralDiscreteScheduler.from_pretrained(
            repo_id, vae=vae, subfolder="scheduler"
        )
        self.samplers[
            Sampler.EulerDiscreteScheduler.value
        ] = EulerDiscreteScheduler.from_pretrained(
            repo_id, vae=vae, subfolder="scheduler"
        )
        self.samplers[
            Sampler.DPMSolverMultistepScheduler.value
        ] = DPMSolverMultistepScheduler.from_pretrained(
            repo_id, vae=vae, subfolder="scheduler"
        )
