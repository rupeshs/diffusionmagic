from pydantic import BaseModel
from utils import DiffusionMagicPaths


class StableDiffusionModel(BaseModel):
    model_id: str = "stabilityai/stable-diffusion-2-1-base"
    use_local: bool = False


class OutputImages(BaseModel):
    path: str = DiffusionMagicPaths.get_results_path()
    format: str = "png"
    use_seperate_folders: bool = True


class DiffusionMagicSettings(BaseModel):
    model_settings: StableDiffusionModel = StableDiffusionModel()
    output_images: OutputImages = OutputImages()
    low_memory_mode: bool = False
