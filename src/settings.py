import yaml

from models.settings import DiffusionMagicSettings
from utils import DiffusionMagicPaths


class AppSettings:
    __instance = None

    def __new__(cls):
        if AppSettings.__instance is None:
            AppSettings.__instance = super().__new__(cls)
        return AppSettings.__instance

    def __init__(self):
        self.config_path = DiffusionMagicPaths().get_app_settings_path()

    def load(self):
        try:
            with open(self.config_path) as file:
                settings_dict = yaml.safe_load(file)
                self.app_settings = DiffusionMagicSettings.parse_obj(settings_dict)
        except Exception as ex:
            print(f"Error in loading settings : {ex}")

    def get_settings(self):
        return self.app_settings

    def save(self):
        try:
            with open(self.config_path, "w") as file:
                yaml.dump(self.app_settings.dict(), file)
        except Exception as ex:
            print(f"Error in saving settings : {ex}")
