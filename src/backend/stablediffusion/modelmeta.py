from huggingface_hub import model_info


class ModelMeta:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def is_loramodel(self):
        self.info = model_info(self.model_path)
        return self.info.cardData.get("base_model") is not None

    def get_lora_base_model(self):
        return self.info.cardData.get("base_model")
