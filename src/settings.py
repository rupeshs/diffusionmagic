import yaml

from models.configs import DiffusionMagicSettings
from utils import DiffusionMagicPaths
import os


class AppSettings:
    __instance = None

    def __new__(cls):
        if AppSettings.__instance is None:
            AppSettings.__instance = super().__new__(cls)
        return AppSettings.__instance

    def __init__(self):
        self.config_path = DiffusionMagicPaths().get_app_settings_path()

    def load(self):
        if not os.path.exists(self.config_path):
            try:
                with open(self.config_path, "w") as file:
                    yaml.dump(
                        self.load_default(),
                        file,
                    )
            except Exception as ex:
                print(f"Error in creating settings : {ex}")
        try:
            with open(self.config_path) as file:
                settings_dict = yaml.safe_load(file)
                self._config = DiffusionMagicSettings.parse_obj(settings_dict)
        except Exception as ex:
            print(f"Error in loading settings : {ex}")

    def get_settings(self) -> DiffusionMagicSettings:
        return self._config

    def save(self):
        try:
            with open(self.config_path, "w") as file:
                yaml.dump(self._config.dict(), file)
        except Exception as ex:
            print(f"Error in saving settings : {ex}")

    def load_default(self):
        defult_config = DiffusionMagicSettings()
        return defult_config.dict()
