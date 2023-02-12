from enum import Enum

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.models import AutoencoderKL


class Sampler(Enum):
    """Diffuser schedulers"""

    DDPMScheduler = "DDPM"
    DPMSolverMultistepScheduler = "DPMSolverMultistep"
    DDIMScheduler = "DDIM"
    EulerDiscreteScheduler = "EulerDiscrete"
    EulerAncestralDiscreteScheduler = "EulerAncestralDiscrete"
    LMSDiscreteScheduler = "LMSDiscrete"
    PNDMScheduler = "PNDM"


class SamplerMixin:
    def __init__(self):
        self.samplers = {}

    def load_samplers(
        self,
        repo_id: str,
        vae_id: str,
    ) -> None:
        vae = AutoencoderKL.from_pretrained(vae_id)

        # https://github.com/huggingface/diffusers/issues/706
        self.samplers[Sampler.DDIMScheduler.value] = DDIMScheduler.from_pretrained(
            repo_id,
            vae=vae,
            subfolder="scheduler",
        )
        self.samplers[Sampler.DDPMScheduler.value] = DDPMScheduler.from_pretrained(
            repo_id,
            vae=vae,
            subfolder="scheduler",
        )
        self.samplers[Sampler.PNDMScheduler.value] = PNDMScheduler.from_pretrained(
            repo_id,
            vae=vae,
            subfolder="scheduler",
        )
        self.samplers[
            Sampler.LMSDiscreteScheduler.value
        ] = LMSDiscreteScheduler.from_pretrained(
            repo_id,
            vae=vae,
            subfolder="scheduler",
        )
        self.samplers[
            Sampler.EulerAncestralDiscreteScheduler.value
        ] = EulerAncestralDiscreteScheduler.from_pretrained(
            repo_id,
            vae=vae,
            subfolder="scheduler",
        )
        self.samplers[
            Sampler.EulerDiscreteScheduler.value
        ] = EulerDiscreteScheduler.from_pretrained(
            repo_id,
            vae=vae,
            subfolder="scheduler",
        )
        self.samplers[
            Sampler.DPMSolverMultistepScheduler.value
        ] = DPMSolverMultistepScheduler.from_pretrained(
            repo_id,
            vae=vae,
            subfolder="scheduler",
        )

    def find_sampler(
        self,
        scheduler_name: str,
    ):
        return self.samplers.get(scheduler_name)

    def default_sampler(self):
        return self.samplers.get(Sampler.DPMSolverMultistepScheduler.value)
