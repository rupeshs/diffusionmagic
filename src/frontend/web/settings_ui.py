from typing import Any

import gradio as gr
from hf_models import StableDiffusionModels
from settings import AppSettings


def save_app_settings(model_id):
    app_settings = AppSettings()
    app_settings.get_settings().model_settings.model_id = model_id
    app_settings.save()


def get_settings_ui() -> None:
    sd_models = StableDiffusionModels()
    sd_models.load_hf_models_from_text_file()

    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                model_id = gr.Dropdown(
                    sd_models.get_models(),
                    value="stable-diffusion-2-1-base",
                    label="Stable diffusion model ",
                )
                save_button = gr.Button("Save")
    save_button.click(
        fn=save_app_settings,
        inputs=[model_id],
    )
