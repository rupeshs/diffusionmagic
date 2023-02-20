from frontend.web.ui import diffusionmagic_web_ui
from settings import AppSettings

# mypy --ignore-missing-imports --explicit-package-bases .
if __name__ == "__main__":
    try:
        app_settings = AppSettings()
        app_settings.load()
    except Exception as ex:
        print(f"ERROR in loading application settings {ex}")
        print("Exiting...")
        exit()
    dm_web_ui = diffusionmagic_web_ui(app_settings.get_settings())
    dm_web_ui.launch()
