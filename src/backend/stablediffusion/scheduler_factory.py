from diffusers import (
    DDIMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)

from backend.stablediffusion.models.scheduler_types import SchedulerType, Scheduler


class SchedulerFactory:
    def get_scheduler(
        self,
        scheduler_type: str,
        repo_id: str,
    ) -> Scheduler:
        if scheduler_type == SchedulerType.DDIMScheduler.value:
            return DDIMScheduler.from_pretrained(
                repo_id,
                subfolder="scheduler",
            )
        elif scheduler_type == SchedulerType.DEISScheduler.value:
            return DEISMultistepScheduler.from_pretrained(
                repo_id,
                subfolder="scheduler",
            )
        elif scheduler_type == SchedulerType.DPMSolverMultistepScheduler.value:
            return DPMSolverMultistepScheduler.from_pretrained(
                repo_id,
                subfolder="scheduler",
            )
        elif scheduler_type == SchedulerType.EulerAncestralDiscreteScheduler.value:
            return EulerAncestralDiscreteScheduler.from_pretrained(
                repo_id,
                subfolder="scheduler",
            )
        elif scheduler_type == SchedulerType.EulerDiscreteScheduler.value:
            return EulerDiscreteScheduler.from_pretrained(
                repo_id,
                subfolder="scheduler",
            )
        elif scheduler_type == SchedulerType.LMSDiscreteScheduler.value:
            return LMSDiscreteScheduler.from_pretrained(
                repo_id,
                subfolder="scheduler",
            )
        elif scheduler_type == SchedulerType.PNDMScheduler.value:
            return PNDMScheduler.from_pretrained(
                repo_id,
                subfolder="scheduler",
            )
        elif scheduler_type == SchedulerType.UniPCMultistepScheduler.value:
            return UniPCMultistepScheduler.from_pretrained(
                repo_id,
                subfolder="scheduler",
            )
        else:
            print(f"Scheduler {scheduler_type} not found")
            return None
