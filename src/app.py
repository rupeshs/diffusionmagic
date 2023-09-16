import argparse

from backend.computing import Computing
from backend.generate import Generate
from frontend.web.ui import diffusionmagic_web_ui
from settings import AppSettings


def _get_model(model_id: str) -> str:
    if model_id == "":
        model_id = AppSettings().get_settings().model_settings.model_id
    return model_id


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
    parser.add_argument(
        "-m",
        "--model",
        help="Model identifier,E.g. runwayml/stable-diffusion-v1-5",
        default="",
    )
    args = parser.parse_args()
    compute = Computing()
    generate = Generate(compute)
    model_id = _get_model(args.model)

    print(f"Model : {model_id}")

    dm_web_ui = diffusionmagic_web_ui(
        generate,
        model_id,
    )
    if args.share:
        dm_web_ui.queue().launch(share=True)
    else:
        dm_web_ui.queue().launch()
