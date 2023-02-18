import os
import constants


def join_paths(
    first_path: str,
    second_path: str,
) -> str:
    return os.path.join(first_path, second_path)


def get_configs_path() -> str:
    app_dir = os.path.dirname(__file__)
    work_dir = os.path.dirname(app_dir)
    config_path = join_paths(work_dir, constants.CONFIG_DIRECTORY)
    return config_path


class DiffusionMagicPaths:
    @staticmethod
    def get_models_config_path():
        configs_path = get_configs_path()
        models_config_path = join_paths(
            configs_path,
            constants.STABLE_DIFFUSION_MODELS_FILE,
        )
        return models_config_path

    @staticmethod
    def get_app_settings_path():
        configs_path = get_configs_path()
        settings_path = join_paths(
            configs_path,
            constants.APP_SETTINGS_FILE,
        )
        return settings_path

    @staticmethod
    def get_css_path():
        app_dir = os.path.dirname(__file__)
        css_path = os.path.join(
            app_dir,
            "frontend",
            "web",
            "css",
            "style.css",
        )
        return css_path

    @staticmethod
    def get_results_path():
        app_dir = os.path.dirname(__file__)
        work_dir = os.path.dirname(app_dir)
        config_path = join_paths(work_dir, constants.RESULTS_DIRECTORY)
        return config_path
