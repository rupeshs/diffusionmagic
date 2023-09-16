from time import time

from backend.computing import Computing
from backend.wuerstchen.models.setting import WurstchenSetting
from torch import Generator
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
from diffusers import AutoPipelineForText2Image


class Wuerstchen:
    def __init__(self, compute: Computing):
        self.compute = compute
        self.pipeline = None
        self.device = self.compute.name
        super().__init__()

    def get_text_to_image_wuerstchen_pipleline(
        self,
        model_id: str = "warp-ai/wuerstchen",
        low_vram_mode: bool = False,
    ):
        self.model_id = model_id

        self.low_vram_mode = low_vram_mode
        print(f"Wuerstchen - {self.compute.name},{self.compute.datatype}")
        print(f"using model {model_id}")
        tic = time()
        self._load_model()
        self._pipeline_to_device()
        delta = time() - tic
        print(f"Model loaded in {delta:.2f}s ")

    def text_to_image_wuerstchen(self, setting: WurstchenSetting):
        if self.pipeline is None:
            raise Exception("Text to image pipeline not initialized")

        generator = None
        if setting.seed != -1:
            print(f"Using seed {setting.seed}")
            generator = Generator(self.device).manual_seed(setting.seed)

        images = self.pipeline(
            setting.prompt,
            negative_prompt=setting.negative_prompt,
            height=setting.image_height,
            width=setting.image_width,
            prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
            prior_guidance_scale=setting.prior_guidance_scale,
            num_images_per_prompt=setting.number_of_images,
            generator=generator,
        ).images

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
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=self.compute.datatype,
        )

    def _load_model(self):
        if self.compute.name == "cuda":
            try:
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    self.model_id,
                    torch_dtype=self.compute.datatype,
                )
            except Exception as ex:
                print(
                    f" The fp16 of the model not found using full precision model,  {ex}"
                )
                self._load_full_precision_model()
        else:
            self._load_full_precision_model()
