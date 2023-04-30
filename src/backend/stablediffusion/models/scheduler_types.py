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
    UniPCMultistepScheduler,
    KDPM2DiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
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
    UniPCMultistepScheduler = "UniPCMultistep"
    KDPM2DiscreteScheduler = "KDPM2DiscreteScheduler"
    HeunDiscreteScheduler = "HeunDiscreteScheduler"
    KDPM2AncestralDiscreteScheduler = "KDPM2AncestralDiscreteScheduler"
    DPMSolverSinglestepScheduler = "DPMSolverSinglestepScheduler"


Scheduler = Union[
    DDIMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    KDPM2DiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
]


def get_sampler_names():
    samplers = [sampler.value for sampler in SchedulerType]
    return samplers
