import gradio as gr

from backend.generate import Generate
from backend.stablediffusion.stable_diffusion_types import (
    StableDiffusionType,
    get_diffusion_type,
)
from frontend.web.depth_to_image_ui import get_depth_to_image_ui
from frontend.web.image_inpainting_ui import get_image_inpainting_ui
from frontend.web.image_to_image_ui import get_image_to_image_ui
from frontend.web.image_variations_ui import get_image_variations_ui
from frontend.web.instruct_pix_to_pix_ui import get_instruct_pix_to_pix_ui
from frontend.web.settings_ui import get_settings_ui
from frontend.web.text_to_image_ui import get_text_to_image_ui
from frontend.web.controlnet.controlnet_image_ui import get_controlnet_to_image_ui
from settings import AppSettings
from utils import DiffusionMagicPaths
from constants import VERSION


def _get_footer_message() -> str:
    version = f"<center><p> v{VERSION} "
    footer_msg = version + (
        '  © 2023 <a href="https://github.com/rupeshs">'
        " Rupesh Sreeraman</a></p></center>"
    )
    return footer_msg


def diffusionmagic_web_ui(generate: Generate) -> gr.Blocks:
    model_id = AppSettings().get_settings().model_settings.model_id
    stable_diffusion_type = get_diffusion_type(model_id)
    with gr.Blocks(
        css=DiffusionMagicPaths.get_css_path(),
        title="DiffusionMagic",
    ) as diffusion_magic_ui:
        gr.HTML("<center><H3>DiffusionMagic 2.0 Beta</H3></center>")
        with gr.Tabs():
            if stable_diffusion_type == StableDiffusionType.base:
                with gr.TabItem("Text to Image"):
                    get_text_to_image_ui(generate.diffusion_text_to_image)
                with gr.TabItem("Image to Image"):
                    get_image_to_image_ui(generate.diffusion_image_to_image)
                with gr.TabItem("Image Variations"):
                    get_image_variations_ui(generate.diffusion_image_variations)
            elif stable_diffusion_type == StableDiffusionType.inpainting:
                with gr.TabItem("Image Inpainting"):
                    get_image_inpainting_ui(generate.diffusion_image_inpainting)
            elif stable_diffusion_type == StableDiffusionType.depth2img:
                with gr.TabItem("Depth to Image"):
                    get_depth_to_image_ui(generate.diffusion_depth_to_image)
            elif stable_diffusion_type == StableDiffusionType.instruct_pix2pix:
                with gr.TabItem("Instruct Pix to Pix"):
                    get_instruct_pix_to_pix_ui(generate.diffusion_pix_to_pix)
            elif stable_diffusion_type == StableDiffusionType.controlnet_canny:
                with gr.TabItem("Controlnet Edge"):
                    get_controlnet_to_image_ui(generate.diffusion_control_to_image)
            elif stable_diffusion_type == StableDiffusionType.controlnet_line:
                with gr.TabItem("Controlnet Lines"):
                    get_controlnet_to_image_ui(generate.diffusion_control_to_image)
            elif stable_diffusion_type == StableDiffusionType.controlnet_normal:
                with gr.TabItem("Controlnet Normal"):
                    get_controlnet_to_image_ui(generate.diffusion_control_to_image)
            elif stable_diffusion_type == StableDiffusionType.controlnet_hed:
                with gr.TabItem("Controlnet HED"):
                    get_controlnet_to_image_ui(generate.diffusion_control_to_image)
            elif stable_diffusion_type == StableDiffusionType.controlnet_pose:
                with gr.TabItem("Controlnet Pose"):
                    get_controlnet_to_image_ui(generate.diffusion_control_to_image)
            elif stable_diffusion_type == StableDiffusionType.controlnet_depth:
                with gr.TabItem("Controlnet Depth"):
                    get_controlnet_to_image_ui(generate.diffusion_control_to_image)
            elif stable_diffusion_type == StableDiffusionType.controlnet_scribble:
                with gr.TabItem("Controlnet Scribble"):
                    get_controlnet_to_image_ui(generate.diffusion_control_to_image)
            elif stable_diffusion_type == StableDiffusionType.controlnet_seg:
                with gr.TabItem("Controlnet Segmentation"):
                    get_controlnet_to_image_ui(generate.diffusion_control_to_image)
            with gr.TabItem("Settings"):
                get_settings_ui()

        gr.HTML(_get_footer_message())
    return diffusion_magic_ui
