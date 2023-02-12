from pydantic import BaseModel


class StableDiffusionModel(BaseModel):
    model_id: str
    use_local: bool = False


class DiffusionMagicSettings(BaseModel):
    model_settings: StableDiffusionModel
