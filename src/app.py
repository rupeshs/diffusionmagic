from frontend.web.ui import diffusionmagic_web_ui
from settings import AppSettings

# model_id = "dreamlike-art/dreamlike-diffusion-1.0"

if __name__ == "__main__":
    app_settings = AppSettings()
    app_settings.load()
    dm_web_ui = diffusionmagic_web_ui(app_settings.get_settings())
    dm_web_ui.launch()
