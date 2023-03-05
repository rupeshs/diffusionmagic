import argparse

from backend.computing import Computing
from backend.generate import Generate
from frontend.web.ui import diffusionmagic_web_ui
from settings import AppSettings

# mypy --ignore-missing-imports --explicit-package-bases .
# flake8 --max-line-length=100 .
if __name__ == "__main__":
    try:
        app_settings = AppSettings()
        app_settings.load()
    except Exception as ex:
        print(f"ERROR in loading application settings {ex}")
        print("Exiting...")
        exit()
    parser = argparse.ArgumentParser(description="DiffusionMagic")
    parser.add_argument(
        "-s", "--share", help="Shareable link", action="store_true", default=False
    )
    args = parser.parse_args()
    compute = Computing()
    generate = Generate(compute)
    dm_web_ui = diffusionmagic_web_ui(generate)
    if args.share:
        dm_web_ui.launch(share=True)
    else:
        dm_web_ui.launch()
