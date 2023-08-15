import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline
from backend.stablediffusion.scheduler_mixin import SamplerMixin
from backend.computing import Computing
from backend.stablediffusion.models.setting import StableDiffusionSetting


class StableDiffusion(SamplerMixin):
    def __init__(self, compute: Computing):
        self.compute = compute
        self.pipeline = None
        self.params = None
        self.device = "tpu"

        super().__init__()

    def _create_key(self, seed=0):
        return jax.random.PRNGKey(seed)

    def get_text_to_image_pipleline(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1-base",
        # sampler: str = SchedulerType.DPMSolverMultistepScheduler.value,
    ):
        self.pipeline, self.params = FlaxStableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            revision="bf16",
            dtype=jnp.bfloat16,
        )

    def _get_prompt_ids(self, prompt: str, num_images: int):
        prompt = [prompt] * num_images
        prompt_ids = self.pipeline.prepare_inputs(prompt)
        prompt_ids = shard(prompt_ids)
        return prompt_ids

    def text_to_image(self, setting: StableDiffusionSetting):
        print("Starting text to image(TPU)")
        # As of now a little hack
        setting.number_of_images = jax.device_count()

        prompt_ids = self._get_prompt_ids(
            setting.prompt,
            setting.number_of_images,
        )
        negative_prompt_ids = self._get_prompt_ids(
            setting.negative_prompt,
            setting.number_of_images,
        )
        p_params = replicate(self.params)
        rng = self._create_key(0)
        rng = jax.random.split(
            rng,
            setting.number_of_images,
        )

        print("Starting pipeline")
        images = self.pipeline(
            prompt_ids=prompt_ids,
            neg_prompt_ids=negative_prompt_ids,
            num_inference_steps=setting.inference_steps,
            guidance_scale=setting.guidance_scale,
            params=p_params,
            prng_seed=rng,
            jit=True,
        )[0]
        images = images.reshape((images.shape[0],) + images.shape[-3:])
        images = self.pipeline.numpy_to_pil(images)
        print("Pipeline completed successfully")
        return images
