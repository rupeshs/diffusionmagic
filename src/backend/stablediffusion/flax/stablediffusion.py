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
            "CompVis/stable-diffusion-v1-4",
            revision="bf16",
            dtype=jnp.bfloat16,
        )

    def text_to_image(self, setting: StableDiffusionSetting):
        prompt = setting.prompt
        prompt = [prompt] * jax.device_count()
        prompt_ids = self.pipeline.prepare_inputs(prompt)
        p_params = replicate(self.params)
        prompt_ids = shard(prompt_ids)
        rng = self._create_key(0)
        rng = jax.random.split(rng, jax.device_count())
        images = self.pipeline(prompt_ids, p_params, rng, jit=True)[0]
        return images
