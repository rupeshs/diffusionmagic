from enum import Enum


class Sampler(Enum):
    """Diffuser schedulers"""

    DDPMScheduler = "DDPM"
    DPMSolverMultistepScheduler = "DPMSolverMultistep"
    DDIMScheduler = "DDIM"
    EulerDiscreteScheduler = "EulerDiscrete"
    EulerAncestralDiscreteScheduler = "EulerAncestralDiscrete"
    LMSDiscreteScheduler = "LMSDiscrete"
    PNDMScheduler = "PNDM"
