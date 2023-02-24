from enum import Enum
from typing import Union

from diffusers import (
    DDIMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)


class SchedulerType(Enum):
    """Diffuser schedulers"""

    # DDPMScheduler = "DDPM"
    DPMSolverMultistepScheduler = "DPMSolverMultistep"
    DDIMScheduler = "DDIM"
    EulerDiscreteScheduler = "EulerDiscrete"
    EulerAncestralDiscreteScheduler = "EulerAncestralDiscrete"
    LMSDiscreteScheduler = "LMSDiscrete"
    PNDMScheduler = "PNDM"
    DEISScheduler = "DEISMultistep"


Scheduler = Union[
    DDIMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
]


def get_sampler_names():
    samplers = [sampler.value for sampler in SchedulerType]
    return samplers
