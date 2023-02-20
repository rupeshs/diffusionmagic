from utils import DiffusionMagicPaths


class StableDiffusionModels:
    def __init__(self):
        self.config_path = DiffusionMagicPaths().get_models_config_path()
        self.__models = []

    def load_hf_models_from_text_file(self):
        """Loads hugging face stable diffusion models"""
        with open(self.config_path, "r") as file:
            lines = file.readlines()
        for repo_id in lines:
            if repo_id.strip() != "":
                self.__models.append(repo_id.strip())

    def get_models(self):
        return self.__models
