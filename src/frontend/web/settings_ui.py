import gradio as gr
from hf_models import StableDiffusionModels
from settings import AppSettings
from datetime import datetime


def save_app_settings(
    model_id,
    result_path,
    img_format,
    use_seperate_folder,
    enable_low_vram,
):
    app_settings = AppSettings()
    app_settings.get_settings().model_settings.model_id = model_id
    app_settings.get_settings().output_images.format = img_format
    app_settings.get_settings().output_images.path = result_path
    app_settings.get_settings().output_images.use_seperate_folders = use_seperate_folder
    app_settings.get_settings().low_memory_mode = enable_low_vram
    app_settings.save()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return "Settings last saved at " + current_time + ""


def get_settings_ui() -> None:
    sd_models = StableDiffusionModels()
    sd_models.load_hf_models_from_text_file()
    app_settings = AppSettings().get_settings()

    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                save_status = gr.HTML(
                    "<center><p>Some setting changes requires app restarts!</p></center>",
                    elem_id="settings_header",
                )
                model_id = gr.Dropdown(
                    sd_models.get_models(),
                    value=app_settings.model_settings.model_id,
                    label="Stable diffusion model (Restart required)",
                )
                result_path = gr.TextArea(
                    label="Output folder (Generated images saved here)",
                    value=app_settings.output_images.path,
                    lines=1,
                )
                img_format = gr.Radio(
                    label="Output image format",
                    choices=["png", "jpeg"],
                    value=app_settings.output_images.format,
                )
                use_seperate_folder = gr.Checkbox(
                    label="Save in images in seperate folders (text2img,img2img etc)",
                    value=app_settings.output_images.use_seperate_folders,
                )
                vram_label = (
                    "Enable Low VRAM mode (GPUs with VRAM <4GB,"
                    "slower to generate images) (Restart required)"
                )
                enable_low_vram = gr.Checkbox(
                    label=vram_label,
                    value=app_settings.low_memory_mode,
                )
                with gr.Column():
                    save_button = gr.Button("Save", elem_id="save_button")
                    save_status = gr.HTML("")
    save_button.click(
        fn=save_app_settings,
        inputs=[
            model_id,
            result_path,
            img_format,
            use_seperate_folder,
            enable_low_vram,
        ],
        outputs=save_status,
    )
