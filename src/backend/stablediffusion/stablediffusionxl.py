from time import time

from torch import Generator
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from PIL import Image

from backend.computing import Computing
from backend.stablediffusion.modelmeta import ModelMeta
from backend.stablediffusion.models.scheduler_types import SchedulerType
from backend.stablediffusion.models.setting import (
    StableDiffusionImageToImageSetting,
    StableDiffusionSetting,
)
from backend.stablediffusion.scheduler_mixin import SamplerMixin


class StableDiffusionXl(SamplerMixin):
    def __init__(self, compute: Computing):
        self.compute = compute
        self.pipeline = None
        self.device = self.compute.name

        super().__init__()

    def get_text_to_image_xl_pipleline(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        low_vram_mode: bool = False,
        sampler: str = SchedulerType.DPMSolverMultistepScheduler.value,
    ):
        repo_id = model_id
        model_meta = ModelMeta(repo_id)
        is_lora_model = model_meta.is_loramodel()
        if is_lora_model:
            print("LoRA  model detected")
            self.model_id = model_meta.get_lora_base_model()
            print(f"LoRA  base model - {self.model_id}")
        else:
            self.model_id = model_id

        self.low_vram_mode = low_vram_mode
        print(f"StableDiffusion - {self.compute.name},{self.compute.datatype}")
        print(f"using model {model_id}")
        self.default_sampler = self.find_sampler(
            sampler,
            self.model_id,
        )
        tic = time()
        self._load_model()
        delta = time() - tic
        print(f"Model loaded in {delta:.2f}s ")

        if self.pipeline is None:
            raise Exception("Text to image pipeline not initialized")
        if is_lora_model:
            self.pipeline.unet.load_attn_procs(repo_id)
        self._pipeline_to_device()
        components = self.pipeline.components
        self.img_to_img_pipeline = StableDiffusionXLImg2ImgPipeline(**components)

    def text_to_image_xl(self, setting: StableDiffusionSetting):
        if self.pipeline is None:
            raise Exception("Text to image pipeline not initialized")

        self.pipeline.scheduler = self.find_sampler(
            setting.scheduler,
            self.model_id,
        )
        generator = None
        if setting.seed != -1:
            print(f"Using seed {setting.seed}")
            generator = Generator(self.device).manual_seed(setting.seed)

        # if setting.attention_slicing:
        #     self.pipeline.enable_attention_slicing()
        # else:
        #     self.pipeline.disable_attention_slicing()

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

        # self.pipeline.unet = torch.compile(
        #     self.pipeline.unet,
        #     mode="reduce-overhead",
        #     fullgraph=True,
        # )
        return images

    def _pipeline_to_device(self):
        if self.low_vram_mode:
            print("Running in low VRAM mode,slower to generate images")
            self.pipeline.enable_sequential_cpu_offload()
        else:
            if self.compute.name == "cuda":
                self.pipeline = self.pipeline.to("cuda")
            elif self.compute.name == "mps":
                self.pipeline = self.pipeline.to("mps")

    def _load_full_precision_model(self):
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.compute.datatype,
            scheduler=self.default_sampler,
        )

    def _load_model(self):
        if self.compute.name == "cuda":
            try:
                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.compute.datatype,
                    scheduler=self.default_sampler,
                    use_safetensors=True,
                    variant="fp16",
                )
            except Exception as ex:
                print(
                    f" The fp16 of the model not found using full precision model,  {ex}"
                )
                self._load_full_precision_model()
        else:
            self._load_full_precision_model()

    def image_to_image(self, setting: StableDiffusionImageToImageSetting):
        if setting.scheduler is None:
            raise Exception("Scheduler cannot be  empty")

        print("Running image to image pipeline")
        self.img_to_img_pipeline.scheduler = self.find_sampler(  # type: ignore
            setting.scheduler,
            self.model_id,
        )
        generator = None
        if setting.seed != -1 and setting.seed:
            print(f"Using seed {setting.seed}")
            generator = Generator(self.device).manual_seed(setting.seed)

        if setting.attention_slicing:
            self.img_to_img_pipeline.enable_attention_slicing()  # type: ignore
        else:
            self.img_to_img_pipeline.disable_attention_slicing()  # type: ignore

        if setting.vae_slicing:
            self.pipeline.enable_vae_slicing()  # type: ignore
        else:
            self.pipeline.disable_vae_slicing()  # type: ignore

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
